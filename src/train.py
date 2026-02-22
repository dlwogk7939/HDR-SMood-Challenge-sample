import argparse
import os
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import (
    EventDataset,
    build_transforms,
    build_vocabs,
    collate_event_batch,
    detect_columns,
    group_event_indices,
    load_hf_split,
    make_index_map,
    split_events_group_kfold,
)
from utils import (
    EventGaussianRegressor,
    compute_regression_metrics,
    fit_sigma_scaling,
    gaussian_crps,
    gaussian_nll_loss,
    get_device,
    load_checkpoint,
    metrics_to_log_string,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train event-level Gaussian regressor")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--output_path", type=str, default="weights/model.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crps_weight", type=float, default=0.25)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--min_name_freq", type=int, default=1)
    parser.add_argument(
        "--pretrained_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--specimen_hidden_dim", type=int, default=512)
    parser.add_argument("--scientific_embed_dim", type=int, default=64)
    parser.add_argument("--domain_embed_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--name_dropout_prob", type=float, default=0.15)
    parser.add_argument("--domain_dropout_prob", type=float, default=0.25)
    parser.add_argument("--max_specimens_train", type=int, default=8)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--resume_state_path", type=str, default=None)
    parser.add_argument("--done_marker_path", type=str, default=None)

    return parser.parse_args()


def train_one_epoch(
    model: EventGaussianRegressor,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    crps_weight: float,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0

    for batch in loader:
        targets = batch["targets"]
        if targets is None:
            continue
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and device.type == "cuda":
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        elif amp_enabled and device.type == "mps":
            amp_ctx = torch.autocast(device_type="mps", dtype=torch.float16, enabled=True)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            mu, sigma = model(batch["events"])
            nll = gaussian_nll_loss(mu, sigma, targets, reduction="mean")
            if float(crps_weight) > 0.0:
                crps = gaussian_crps(mu, sigma, targets).mean()
                loss = nll + float(crps_weight) * crps
            else:
                loss = nll

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        running_loss += float(loss.item()) * batch_size
        seen += batch_size

    return running_loss / max(1, seen)


@torch.no_grad()
def evaluate(
    model: EventGaussianRegressor,
    loader: DataLoader,
    device: torch.device,
    sigma_scaling: torch.Tensor | None = None,
) -> Tuple[float, Dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    model.eval()
    running_loss = 0.0
    seen = 0

    all_mu = []
    all_sigma = []
    all_targets = []

    for batch in loader:
        targets = batch["targets"]
        if targets is None:
            continue
        targets = targets.to(device, non_blocking=True)

        mu, sigma = model(batch["events"])
        if sigma_scaling is not None:
            sigma = sigma * sigma_scaling.view(1, -1)

        loss = gaussian_nll_loss(mu, sigma, targets, reduction="mean")

        batch_size = targets.shape[0]
        running_loss += float(loss.item()) * batch_size
        seen += batch_size

        all_mu.append(mu.detach().cpu())
        all_sigma.append(sigma.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if seen == 0:
        raise RuntimeError("Validation loader produced zero labeled examples.")

    mu_tensor = torch.cat(all_mu, dim=0)
    sigma_tensor = torch.cat(all_sigma, dim=0)
    target_tensor = torch.cat(all_targets, dim=0)

    metrics = compute_regression_metrics(mu_tensor, sigma_tensor, target_tensor)
    return running_loss / seen, metrics, (mu_tensor, sigma_tensor, target_tensor)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    print("Loading Hugging Face dataset split...")
    hf_ds = load_hf_split(split=args.split, hf_token=args.hf_token, cache_dir=args.cache_dir)
    columns = detect_columns(hf_ds.column_names)

    print("Grouping records by eventID...")
    events = group_event_indices(hf_ds, columns)
    train_events, val_events, split_group_key = split_events_group_kfold(
        events,
        n_splits=args.n_splits,
        fold=args.fold,
        seed=args.seed,
        prefer_site=True,
    )

    if len(train_events) == 0 or len(val_events) == 0:
        raise RuntimeError(
            f"Split failed. train_events={len(train_events)}, val_events={len(val_events)}"
        )

    print(
        f"Events -> train: {len(train_events)}, val: {len(val_events)} "
        f"(grouped by {split_group_key})"
    )

    scientific_vocab, domain_vocab = build_vocabs(
        hf_ds,
        train_events,
        columns,
        min_name_freq=args.min_name_freq,
    )
    scientific_to_idx = make_index_map(scientific_vocab)
    domain_to_idx = make_index_map(domain_vocab)

    train_tf, eval_tf = build_transforms(args.image_size)

    train_dataset = EventDataset(
        dataset=hf_ds,
        events=train_events,
        columns=columns,
        scientific_to_idx=scientific_to_idx,
        domain_to_idx=domain_to_idx,
        transform=train_tf,
        include_targets=True,
        max_specimens_per_event=args.max_specimens_train,
    )
    val_dataset = EventDataset(
        dataset=hf_ds,
        events=val_events,
        columns=columns,
        scientific_to_idx=scientific_to_idx,
        domain_to_idx=domain_to_idx,
        transform=eval_tf,
        include_targets=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_event_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_event_batch,
    )

    model_config = {
        "specimen_hidden_dim": args.specimen_hidden_dim,
        "scientific_embed_dim": args.scientific_embed_dim,
        "domain_embed_dim": args.domain_embed_dim,
        "dropout": args.dropout,
        "image_size": args.image_size,
        "name_dropout_prob": args.name_dropout_prob,
        "domain_dropout_prob": args.domain_dropout_prob,
        "crps_weight": args.crps_weight,
    }

    model = EventGaussianRegressor(
        num_scientific_names=len(scientific_vocab),
        num_domain_ids=len(domain_vocab),
        specimen_hidden_dim=args.specimen_hidden_dim,
        scientific_embed_dim=args.scientific_embed_dim,
        domain_embed_dim=args.domain_embed_dim,
        dropout=args.dropout,
        pretrained_backbone=args.pretrained_backbone,
        name_dropout_prob=args.name_dropout_prob,
        domain_dropout_prob=args.domain_dropout_prob,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "mps":
        amp_enabled = bool(args.amp)
    scaler = (
        torch.cuda.amp.GradScaler(enabled=amp_enabled)
        if (amp_enabled and device.type == "cuda")
        else None
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    resume_state_path = args.resume_state_path or f"{args.output_path}.state"
    done_marker_path = args.done_marker_path or f"{args.output_path}.done"

    resume_state_dir = os.path.dirname(resume_state_path)
    if resume_state_dir:
        os.makedirs(resume_state_dir, exist_ok=True)
    done_marker_dir = os.path.dirname(done_marker_path)
    if done_marker_dir:
        os.makedirs(done_marker_dir, exist_ok=True)

    if os.path.exists(done_marker_path):
        os.remove(done_marker_path)

    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics = None
    start_epoch = 1

    if os.path.exists(args.output_path):
        try:
            existing_best = load_checkpoint(args.output_path, map_location="cpu")
            metrics = existing_best.get("metrics", {})
            best_epoch = int(metrics.get("best_epoch", best_epoch))
            best_val_loss = float(metrics.get("best_val_nll", best_val_loss))
            best_metrics = metrics.get("best_val_uncalibrated", best_metrics)
            print(
                f"Found existing best checkpoint: epoch={best_epoch} "
                f"val_nll={best_val_loss:.5f}"
            )
        except Exception:
            pass

    if args.resume_from and os.path.exists(args.resume_from):
        state = load_checkpoint(args.resume_from, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])

        start_epoch = int(state.get("epoch", 0)) + 1
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        best_epoch = int(state.get("best_epoch", best_epoch))
        best_metrics = state.get("best_metrics", best_metrics)
        print(
            f"Resumed from {args.resume_from} at epoch={start_epoch} "
            f"(best_epoch={best_epoch}, best_val_nll={best_val_loss:.5f})"
        )

    if start_epoch > args.epochs:
        print(
            f"Resume state already reached target epochs: start_epoch={start_epoch}, "
            f"epochs={args.epochs}. Skipping training loop."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            crps_weight=args.crps_weight,
        )

        val_loss, val_metrics, _ = evaluate(model=model, loader=val_loader, device=device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_obj={train_loss:.5f} val_nll={val_loss:.5f}"
        )
        print(metrics_to_log_string(val_metrics))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics

            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "scientific_vocab": scientific_vocab,
                "domain_vocab": domain_vocab,
                "sigma_calibration": [1.0, 1.0, 1.0],
                "config": model_config,
                "columns": columns,
                "split": {
                    "group_key": split_group_key,
                    "n_splits": args.n_splits,
                    "fold": args.fold,
                    "seed": args.seed,
                },
                "metrics": {
                    "best_val_uncalibrated": best_metrics,
                    "best_epoch": best_epoch,
                    "best_val_nll": best_val_loss,
                },
            }
            save_checkpoint(args.output_path, checkpoint_payload)

        train_state_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "config": model_config,
            "split": {
                "group_key": split_group_key,
                "n_splits": args.n_splits,
                "fold": args.fold,
                "seed": args.seed,
            },
        }
        save_checkpoint(resume_state_path, train_state_payload)

    if not os.path.exists(args.output_path):
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    best_checkpoint = load_checkpoint(args.output_path, map_location="cpu")
    best_epoch = int(best_checkpoint.get("metrics", {}).get("best_epoch", best_epoch))
    best_val_loss = float(best_checkpoint.get("metrics", {}).get("best_val_nll", best_val_loss))
    print(f"Best epoch={best_epoch}, val_nll={best_val_loss:.5f}")

    model.load_state_dict(best_checkpoint["model_state_dict"])
    _, uncali_metrics, (mu, sigma, target) = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        sigma_scaling=None,
    )
    sigma_scaling = fit_sigma_scaling(mu, sigma, target)

    _, cali_metrics, _ = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        sigma_scaling=sigma_scaling.to(device),
    )

    final_checkpoint = load_checkpoint(args.output_path, map_location="cpu")
    final_checkpoint["sigma_calibration"] = [
        float(x) for x in sigma_scaling.detach().cpu().tolist()
    ]
    final_checkpoint["metrics"]["best_val_uncalibrated"] = uncali_metrics
    final_checkpoint["metrics"]["best_val_calibrated"] = cali_metrics

    save_checkpoint(args.output_path, final_checkpoint)

    print("Final uncalibrated metrics:")
    print(metrics_to_log_string(uncali_metrics))
    print("Final calibrated metrics:")
    print(metrics_to_log_string(cali_metrics))
    print(f"Saved checkpoint to: {args.output_path}")

    done_payload = {
        "completed": True,
        "best_epoch": int(best_epoch),
        "best_val_nll": float(best_val_loss),
        "output_path": args.output_path,
        "resume_state_path": resume_state_path,
    }
    with open(done_marker_path, "w", encoding="utf-8") as f:
        f.write(str(done_payload))


if __name__ == "__main__":
    main()
