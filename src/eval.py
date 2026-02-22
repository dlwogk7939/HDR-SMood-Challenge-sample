import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import (
    EventDataset,
    build_transforms,
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
    gaussian_nll_loss,
    get_device,
    load_checkpoint,
    metrics_to_log_string,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained event-level model")
    parser.add_argument("--checkpoint", type=str, default="weights/model.pt")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_loader(
    model: EventGaussianRegressor,
    loader: DataLoader,
    device: torch.device,
    sigma_scaling: torch.Tensor | None = None,
):
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
        raise RuntimeError("No labeled events were evaluated.")

    mu = torch.cat(all_mu, dim=0)
    sigma = torch.cat(all_sigma, dim=0)
    target = torch.cat(all_targets, dim=0)

    metrics = compute_regression_metrics(mu, sigma, target)
    return running_loss / seen, metrics


def main() -> None:
    args = parse_args()
    device = get_device()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")

    config = checkpoint.get("config", {})
    scientific_vocab = checkpoint.get("scientific_vocab", ["__UNK__"])
    domain_vocab = checkpoint.get("domain_vocab", [-1])

    print("Loading Hugging Face dataset split for evaluation...")
    hf_ds = load_hf_split(split=args.split, hf_token=args.hf_token, cache_dir=args.cache_dir)
    columns = detect_columns(hf_ds.column_names)

    events = group_event_indices(hf_ds, columns)

    split_cfg = checkpoint.get("split", {})
    n_splits = args.n_splits if args.n_splits is not None else split_cfg.get("n_splits", 5)
    fold = args.fold if args.fold is not None else split_cfg.get("fold", 0)
    seed = args.seed if args.seed is not None else split_cfg.get("seed", 42)

    _, val_events, split_group_key = split_events_group_kfold(
        events,
        n_splits=int(n_splits),
        fold=int(fold),
        seed=int(seed),
        prefer_site=True,
    )

    if len(val_events) == 0:
        raise RuntimeError("Validation split is empty; cannot evaluate.")

    _, eval_tf = build_transforms(image_size=int(config.get("image_size", 224)))

    scientific_to_idx = make_index_map(scientific_vocab)
    domain_to_idx = make_index_map(domain_vocab)

    val_dataset = EventDataset(
        dataset=hf_ds,
        events=val_events,
        columns=columns,
        scientific_to_idx=scientific_to_idx,
        domain_to_idx=domain_to_idx,
        transform=eval_tf,
        include_targets=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_event_batch,
    )

    model = EventGaussianRegressor(
        num_scientific_names=len(scientific_vocab),
        num_domain_ids=len(domain_vocab),
        specimen_hidden_dim=int(config.get("specimen_hidden_dim", 512)),
        scientific_embed_dim=int(config.get("scientific_embed_dim", 64)),
        domain_embed_dim=int(config.get("domain_embed_dim", 16)),
        dropout=float(config.get("dropout", 0.2)),
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    uncal_nll, uncal_metrics = evaluate_loader(model, val_loader, device)

    sigma_scaling = torch.tensor(
        checkpoint.get("sigma_calibration", [1.0, 1.0, 1.0]), dtype=torch.float32
    )
    cal_nll, cal_metrics = evaluate_loader(
        model,
        val_loader,
        device,
        sigma_scaling=sigma_scaling.to(device),
    )

    print(
        f"Validation events: {len(val_events)} (grouped by {split_group_key}, fold={fold}/{n_splits})"
    )
    print(f"Uncalibrated mean NLL: {uncal_nll:.5f}")
    print(metrics_to_log_string(uncal_metrics))
    print(f"Calibrated mean NLL: {cal_nll:.5f}")
    print(metrics_to_log_string(cal_metrics))

    if args.output_json:
        payload = {
            "uncalibrated_mean_nll": uncal_nll,
            "uncalibrated": uncal_metrics,
            "calibrated_mean_nll": cal_nll,
            "calibrated": cal_metrics,
            "split": {
                "group_key": split_group_key,
                "n_splits": int(n_splits),
                "fold": int(fold),
                "seed": int(seed),
            },
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
