# Beetles as Sentinel Taxa - End-to-End CodaBench Solution

This repository contains an event-level probabilistic regression pipeline for the CodaBench challenge:
**"Beetles as Sentinel Taxa: Predicting drought conditions from NEON specimen imagery"**.

The model predicts Gaussian parameters `(mu, sigma)` for:
- `SPEI_30d`
- `SPEI_1y`
- `SPEI_2y`

It trains with Gaussian NLL and supports:
- post-hoc sigma calibration
- site/domain-aware out-of-sample validation
- multi-fold ensemble inference with Gaussian moment fusion
- metadata priors (scientificName/domainID) blended with image predictions

## Project Structure

```
repo_root/
  README.md
  requirements.txt
  model.py
  weights/
    model.pt
  src/
    data.py
    train.py
    train_ensemble.py
    build_priors.py
    eval.py
    utils.py
  scripts/
    make_submission.sh
    sanity_check_submission.py
    colab_prepare.py
    colab_train.py
```

## Install

Submission-time dependencies:

```bash
pip install -r requirements.txt
```

Training/evaluation needs Hugging Face datasets:

```bash
pip install datasets
```

## Download Data

```bash
python - <<'PY'
from datasets import load_dataset
load_dataset("imageomics/sentinel-beetles", split="train")
print("download complete")
PY
```

Optional: provide `HF_TOKEN` or pass `--hf_token` to scripts if needed.

## Train (Single Fold)

```bash
python src/train.py \
  --output_path weights/model.pt \
  --epochs 12 \
  --batch_size 8 \
  --max_specimens_train 8 \
  --n_splits 5 \
  --fold 0 \
  --crps_weight 0.25 \
  --name_dropout_prob 0.15 \
  --domain_dropout_prob 0.25
```

Notes:
- Event-level grouping is by `eventID`.
- Validation is out-of-sample group split by `siteID` when available, otherwise `domainID`.
- Best checkpoint is saved to `weights/model.pt` (single-fold format).
- Sigma calibration factors are fitted on validation and stored in the checkpoint.

## Train (Ensemble + Priors)

This runs multiple folds, builds priors, and writes a submission manifest to `weights/model.pt`.

```bash
python src/train_ensemble.py \
  --folds 0,1,2 \
  --n_splits 5 \
  --epochs 12 \
  --batch_size 8 \
  --max_specimens_train 8 \
  --crps_weight 0.25 \
  --name_dropout_prob 0.15 \
  --domain_dropout_prob 0.25 \
  --output_dir weights \
  --manifest_path weights/model.pt \
  --resume \
  --skip_completed
```

Outputs:
- `weights/fold_0.pt`, `weights/fold_1.pt`, ...
- `weights/priors.json`
- `weights/model.pt` (manifest consumed by submission `model.py`)

Important:
- If `weights/model.pt` has `fold_paths: []`, submission runs in prior-only mode and score is usually lower.

Apple Silicon (MPS) memory tip:
- If you hit MPS OOM, lower `--batch_size` to `1-2` and `--max_specimens_train` to `4-6`.

## Google Colab

Run these cells in order:

```bash
!git clone https://github.com/Imageomics/HDR-SMood-Challenge-sample.git
%cd HDR-SMood-Challenge-sample
```

```bash
# Optional: if dataset access needs auth
import os
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
```

```bash
!python scripts/colab_prepare.py \
  --repo_dir . \
  --mount_drive \
  --install_deps \
  --persistent_root "/content/drive/MyDrive/beetles_persist" \
  --hf_token "$HF_TOKEN" \
  --download_dataset \
  --download_backbone \
  --link_weights_to_drive
```

```bash
!python scripts/colab_train.py \
  --repo_dir . \
  --mount_drive \
  --drive_out "/content/drive/MyDrive/beetles_submission" \
  --persistent_root "/content/drive/MyDrive/beetles_persist" \
  --folds 0,1,2 \
  --epochs 12 \
  --batch_size 8 \
  --num_workers 2 \
  --image_size 224 \
  --max_specimens_train 8 \
  --crps_weight 0.25 \
  --name_dropout_prob 0.15 \
  --domain_dropout_prob 0.25 \
  --hf_token "$HF_TOKEN" \
  --resume \
  --skip_completed
```

Colab OOM tip:
- Start with `--batch_size 4 --max_specimens_train 6`.
- If OOM persists, use `--batch_size 2 --image_size 192`.

Resume and avoid re-download:
- Keep `--persistent_root` on Google Drive. It stores:
  - `hf_cache` (dataset cache)
  - `torch_cache` (pretrained model cache)
  - `weights` (fold checkpoints, per-epoch resume states, done markers)
- Re-run the exact same command after disconnect/interruption.
  - Completed folds are skipped.
  - In-progress folds resume from `weights/fold_{k}.state`.

If you only want metadata priors quickly:

```bash
python src/build_priors.py --split train --output weights/priors.json
```

## Evaluate

```bash
python src/eval.py \
  --checkpoint weights/model.pt \
  --batch_size 8
```

Outputs include RMSE, NLL, and Gaussian CRPS for each target.

## Build Submission Zip

```bash
bash scripts/make_submission.sh
```

This creates `submission.zip` with zip-root contents:
- `model.py`
- `requirements.txt`
- `weights/` (entire folder, including `model.pt`, `priors.json`, and any `fold_*.pt`)

## Sanity Check Submission

```bash
python scripts/sanity_check_submission.py --zip submission.zip
```

This test:
1. Extracts archive to a temp directory.
2. Runs `import model; Model().load()`.
3. Calls `predict()` with a dummy event.
4. Verifies required output keys and positive `sigma`.

## Inference Contract (`model.py`)

`Model.predict(event_records)` expects a list for one sampling event where each item contains:
- `relative_img` (PIL.Image)
- `colorpicker_img` (PIL.Image, optional but supported)
- `scalebar_img` (PIL.Image, optional but supported)
- `scientificName` (str)
- `domainID` (int)

Returns:

```python
{
  "SPEI_30d": {"mu": float, "sigma": float},
  "SPEI_1y":  {"mu": float, "sigma": float},
  "SPEI_2y":  {"mu": float, "sigma": float},
}
```
