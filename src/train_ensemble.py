import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fold ensemble and build submission manifest")

    p.add_argument("--folds", type=str, default="0,1,2")
    p.add_argument("--output_dir", type=str, default="weights")

    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="train")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--crps_weight", type=float, default=0.25)

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--min_name_freq", type=int, default=1)
    p.add_argument(
        "--pretrained_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--specimen_hidden_dim", type=int, default=512)
    p.add_argument("--scientific_embed_dim", type=int, default=64)
    p.add_argument("--domain_embed_dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--name_dropout_prob", type=float, default=0.15)
    p.add_argument("--domain_dropout_prob", type=float, default=0.25)
    p.add_argument("--max_specimens_train", type=int, default=8)

    p.add_argument("--build_priors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_domain_count", type=int, default=8)
    p.add_argument("--min_scientific_count", type=int, default=20)
    p.add_argument("--max_scientific", type=int, default=5000)
    p.add_argument("--domain_smoothing", type=float, default=25.0)
    p.add_argument("--scientific_smoothing", type=float, default=50.0)

    p.add_argument("--alpha_model", type=float, default=0.88)
    p.add_argument("--alpha_prior_if_no_model", type=float, default=0.92)
    p.add_argument("--tta_hflip", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--skip_train", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--manifest_path", type=str, default="weights/model.pt")

    return p.parse_args()


def parse_folds(folds_arg: str) -> List[int]:
    folds = []
    for token in str(folds_arg).split(","):
        token = token.strip()
        if not token:
            continue
        folds.append(int(token))
    uniq = sorted(set(folds))
    if len(uniq) == 0:
        raise ValueError("No folds provided")
    return uniq


def run_cmd(cmd: List[str]) -> None:
    print("RUN:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)

    os.makedirs(args.output_dir, exist_ok=True)

    py = sys.executable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_py = os.path.join(project_root, "src", "train.py")
    prior_py = os.path.join(project_root, "src", "build_priors.py")

    if not args.skip_train:
        for fold in folds:
            out_path = os.path.join(args.output_dir, f"fold_{fold}.pt")
            state_path = os.path.join(args.output_dir, f"fold_{fold}.state")
            done_path = os.path.join(args.output_dir, f"fold_{fold}.done")

            if args.skip_completed and os.path.exists(done_path):
                print(f"SKIP fold {fold}: completed marker found ({done_path})")
                continue

            cmd = [
                py,
                train_py,
                "--output_path",
                out_path,
                "--resume_state_path",
                state_path,
                "--done_marker_path",
                done_path,
                "--split",
                args.split,
                "--seed",
                str(args.seed),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.weight_decay),
                "--n_splits",
                str(args.n_splits),
                "--fold",
                str(fold),
                "--image_size",
                str(args.image_size),
                "--min_name_freq",
                str(args.min_name_freq),
                "--specimen_hidden_dim",
                str(args.specimen_hidden_dim),
                "--scientific_embed_dim",
                str(args.scientific_embed_dim),
                "--domain_embed_dim",
                str(args.domain_embed_dim),
                "--dropout",
                str(args.dropout),
                "--crps_weight",
                str(args.crps_weight),
                "--name_dropout_prob",
                str(args.name_dropout_prob),
                "--domain_dropout_prob",
                str(args.domain_dropout_prob),
                "--max_specimens_train",
                str(args.max_specimens_train),
            ]

            if args.resume and os.path.exists(state_path):
                cmd.extend(["--resume_from", state_path])
                print(f"RESUME fold {fold} from {state_path}")

            if args.hf_token:
                cmd.extend(["--hf_token", args.hf_token])
            if args.cache_dir:
                cmd.extend(["--cache_dir", args.cache_dir])
            if args.amp:
                cmd.append("--amp")
            else:
                cmd.append("--no-amp")
            if args.pretrained_backbone:
                cmd.append("--pretrained_backbone")
            else:
                cmd.append("--no-pretrained_backbone")

            run_cmd(cmd)

    if args.build_priors:
        priors_path = os.path.join(args.output_dir, "priors.json")
        cmd = [
            py,
            prior_py,
            "--split",
            args.split,
            "--output",
            priors_path,
            "--min_domain_count",
            str(args.min_domain_count),
            "--min_scientific_count",
            str(args.min_scientific_count),
            "--max_scientific",
            str(args.max_scientific),
            "--domain_smoothing",
            str(args.domain_smoothing),
            "--scientific_smoothing",
            str(args.scientific_smoothing),
        ]
        if args.hf_token:
            cmd.extend(["--hf_token", args.hf_token])
        if args.cache_dir:
            cmd.extend(["--cache_dir", args.cache_dir])
        run_cmd(cmd)

    existing_folds = []
    for path in sorted(glob.glob(os.path.join(args.output_dir, "fold_*.pt"))):
        base = os.path.basename(path)
        if ".state." in base:
            continue
        rel = os.path.relpath(path, project_root).replace("\\", "/")
        existing_folds.append(rel)

    alpha_model_base = max(0.0, min(1.0, float(args.alpha_model)))
    alpha_prior_base = max(0.0, min(1.0, float(args.alpha_prior_if_no_model)))

    manifest = {
        "format_version": 2,
        "fold_paths": existing_folds,
        "prior_file": os.path.relpath(
            os.path.join(args.output_dir, "priors.json"), project_root
        ).replace("\\", "/"),
        "blend": {
            "alpha_model": [
                alpha_model_base,
                max(0.0, alpha_model_base - 0.03),
                max(0.0, alpha_model_base - 0.06),
            ],
            "alpha_prior_if_no_model": [
                alpha_prior_base,
                min(1.0, alpha_prior_base + 0.03),
                min(1.0, alpha_prior_base + 0.06),
            ],
        },
        "tta": {
            "hflip": bool(args.tta_hflip),
        },
        "post_calibration": {
            "base_sigma_scale": [1.02, 1.08, 1.20],
            "confidence_sigma_k": [0.20, 0.38, 0.78],
            "ood_sigma_boost": [1.08, 1.20, 1.56],
            "global_shrink": [0.10, 0.20, 0.34],
            "ood_shrink_extra": [0.04, 0.12, 0.24],
            "temporal_smooth": [0.18, 0.32],
            "sigma_min": [0.08, 0.10, 0.14],
            "long_horizon_pull": [0.18, 0.36],
        },
        "meta": {
            "folds_requested": folds,
            "folds_available": existing_folds,
        },
    }

    os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    print(f"Saved manifest: {args.manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
