#!/usr/bin/env python3
"""
Google Colab prepare helper:
- optional Drive mount
- optional dependency install
- persistent cache setup (HF + torch)
- optional dataset/pretrained-weight pre-download
- optional persistent weights symlink
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


def maybe_install_deps(enable: bool) -> None:
    if not enable:
        return
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    # torch/torchvision are usually preinstalled on Colab GPU runtimes.
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "datasets", "Pillow", "numpy"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Colab environment for training")
    p.add_argument("--repo_dir", type=str, default=".")
    p.add_argument("--mount_drive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--install_deps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--persistent_root",
        type=str,
        default="/content/drive/MyDrive/beetles_persist",
        help="Persistent root for caches/checkpoints on Drive.",
    )
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--torch_cache_dir", type=str, default="")
    p.add_argument("--hf_token", type=str, default="")
    p.add_argument("--download_dataset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--download_backbone", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--link_weights_to_drive", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    persistent_root = Path(args.persistent_root).resolve()
    persistent_root.mkdir(parents=True, exist_ok=True)

    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else (persistent_root / "hf_cache").resolve()
    )
    torch_cache_dir = (
        Path(args.torch_cache_dir).resolve()
        if args.torch_cache_dir
        else (persistent_root / "torch_cache").resolve()
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, torch_cache_dir


def _maybe_link_weights(repo_dir: Path, persistent_root: Path, enable: bool) -> None:
    if not enable:
        return

    repo_weights = repo_dir / "weights"
    drive_weights = (persistent_root / "weights").resolve()
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


def _download_dataset(cache_dir: Path, split: str, hf_token: str) -> None:
    from datasets import load_dataset

    kwargs = {"split": split, "cache_dir": str(cache_dir)}
    if hf_token and hf_token != "YOUR_HF_TOKEN":
        kwargs["token"] = hf_token

    ds = load_dataset("imageomics/sentinel-beetles", **kwargs)
    print(f"Dataset ready: split={split}, rows={len(ds)}")


def _download_backbone() -> None:
    import torch  # noqa: F401
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    _ = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    print("EfficientNet-B0 pretrained weights ready")


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"repo_dir not found: {repo_dir}")

    maybe_mount_drive(args.mount_drive)
    maybe_install_deps(args.install_deps)

    persistent_root = Path(args.persistent_root).resolve()
    cache_dir, torch_cache_dir = _resolve_paths(args)

    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(torch_cache_dir)

    _maybe_link_weights(repo_dir, persistent_root, args.link_weights_to_drive)

    if args.download_dataset:
        _download_dataset(cache_dir, args.dataset_split, args.hf_token)
    if args.download_backbone:
        _download_backbone()

    print("Prepare complete.")
    print(f"HF_DATASETS_CACHE={cache_dir}")
    print(f"TORCH_HOME={torch_cache_dir}")


if __name__ == "__main__":
    main()
