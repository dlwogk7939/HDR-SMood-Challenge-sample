#!/usr/bin/env python3
"""
Google Colab training helper:
- optional Drive mount
- persistent cache/weights setup
- train ensemble (resume/skip-completed aware)
- build submission.zip
- copy artifacts to Drive
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd: Path | None = None) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def maybe_mount_drive(enable: bool) -> None:
    if not enable:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        print("Google Drive mounted at /content/drive")
    except Exception as exc:
        print(f"Drive mount skipped: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train challenge model in Colab (training phase)")
    p.add_argument("--repo_dir", type=str, default=".")
    p.add_argument("--mount_drive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drive_out", type=str, default="/content/drive/MyDrive/beetles_submission")
    p.add_argument(
        "--persistent_root",
        type=str,
        default="/content/drive/MyDrive/beetles_persist",
        help="Persistent root for cache/checkpoints on Drive.",
    )
    p.add_argument(
        "--link_weights_to_drive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store repo weights as a symlink to Drive for continuous checkpoint persistence.",
    )

    p.add_argument("--folds", type=str, default="0,1,2")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_specimens_train", type=int, default=8)
    p.add_argument("--specimen_hidden_dim", type=int, default=512)
    p.add_argument("--scientific_embed_dim", type=int, default=64)
    p.add_argument("--domain_embed_dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--crps_weight", type=float, default=0.25)
    p.add_argument("--name_dropout_prob", type=float, default=0.15)
    p.add_argument("--domain_dropout_prob", type=float, default=0.25)

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hf_token", type=str, default="")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--torch_cache_dir", type=str, default="")
    p.add_argument("--output_dir", type=str, default="weights")
    p.add_argument("--manifest_path", type=str, default="weights/model.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()

    if not repo_dir.exists():
        raise FileNotFoundError(f"repo_dir not found: {repo_dir}")

    maybe_mount_drive(args.mount_drive)

    persistent_root = Path(args.persistent_root).resolve()
    if args.mount_drive:
        persistent_root.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir
    if not cache_dir:
        cache_dir = (
            str((persistent_root / "hf_cache").resolve())
            if args.mount_drive
            else "/content/hf_cache"
        )
    torch_cache_dir = args.torch_cache_dir
    if not torch_cache_dir:
        torch_cache_dir = (
            str((persistent_root / "torch_cache").resolve())
            if args.mount_drive
            else "/content/torch_cache"
        )

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(torch_cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["TORCH_HOME"] = torch_cache_dir

    repo_weights = repo_dir / "weights"
    if args.mount_drive and args.link_weights_to_drive:
        drive_weights = persistent_root / "weights"
        drive_weights.mkdir(parents=True, exist_ok=True)

        if repo_weights.exists() and not repo_weights.is_symlink():
            if not any(drive_weights.iterdir()):
                shutil.copytree(repo_weights, drive_weights, dirs_exist_ok=True)
            shutil.rmtree(repo_weights)

        if repo_weights.is_symlink() and not repo_weights.exists():
            repo_weights.unlink()

        if not repo_weights.exists():
            repo_weights.symlink_to(drive_weights, target_is_directory=True)
            print(f"Linked {repo_weights} -> {drive_weights}")

    train_cmd = [
        sys.executable,
        "src/train_ensemble.py",
        "--folds",
        args.folds,
        "--n_splits",
        str(args.n_splits),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--image_size",
        str(args.image_size),
        "--max_specimens_train",
        str(args.max_specimens_train),
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
        "--cache_dir",
        cache_dir,
        "--output_dir",
        args.output_dir,
        "--manifest_path",
        args.manifest_path,
    ]

    if args.hf_token and args.hf_token != "YOUR_HF_TOKEN":
        train_cmd.extend(["--hf_token", args.hf_token])
    if args.resume:
        train_cmd.append("--resume")
    else:
        train_cmd.append("--no-resume")
    if args.skip_completed:
        train_cmd.append("--skip_completed")
    else:
        train_cmd.append("--no-skip_completed")

    run_cmd(train_cmd, cwd=repo_dir)

    # Make sure zip packs real files, not a symlinked weights dir.
    if repo_weights.is_symlink():
        weights_target = repo_weights.resolve()
        temp_weights = repo_dir / "weights_pack_tmp"
        if temp_weights.exists():
            shutil.rmtree(temp_weights)
        shutil.copytree(weights_target, temp_weights, dirs_exist_ok=True)
        repo_weights.unlink()
        temp_weights.rename(repo_weights)

    run_cmd(["bash", "scripts/make_submission.sh"], cwd=repo_dir)

    out_dir = Path(args.drive_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_zip = repo_dir / "submission.zip"
    dst_zip = out_dir / "submission.zip"
    shutil.copy2(src_zip, dst_zip)

    src_weights = repo_dir / "weights"
    dst_weights = out_dir / "weights"
    if dst_weights.exists():
        shutil.rmtree(dst_weights)
    shutil.copytree(src_weights, dst_weights)

    print(f"Saved submission zip to: {dst_zip}")
    print(f"Saved weights to: {dst_weights}")


if __name__ == "__main__":
    main()
