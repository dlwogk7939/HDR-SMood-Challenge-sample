"""
Submission model for CodaBench HDR Scientific Mood Beetles challenge.

Required interface:
- class Model
- load(self)
- predict(self, event_records)

This implementation supports:
- Fold ensemble (weights/fold_*.pt)
- Event-level Gaussian fusion across folds + TTA
- Metadata priors (weights/priors.json)
- Robust fallback behavior (never crash ingestion)
- Local debug mode: `python model.py --out_csv predictions.csv`
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageStat

TORCH_AVAILABLE = True
TORCH_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.models import efficientnet_b0
except Exception as exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc


TARGETS = ("SPEI_30d", "SPEI_1y", "SPEI_2y")
MIN_SIGMA = 1e-4
MAX_DEBUG_RECORDS = 4


def _utc_now_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = "[model.py {} UTC] {}".format(_utc_now_str(), msg)
    try:
        print(line, flush=True)
    except Exception:
        pass

    paths = ["/tmp/model_debug.log"]
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        paths.append(os.path.join(base, "model_debug.log"))
    except Exception:
        pass

    for p in paths:
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            continue


def _log_exc(where: str, exc: Exception) -> None:
    _log("ERROR at {}: {}: {}".format(where, type(exc).__name__, exc))
    try:
        _log(traceback.format_exc())
    except Exception:
        pass


if TORCH_AVAILABLE:

    class EventGaussianRegressor(nn.Module):
        def __init__(
            self,
            num_scientific_names: int,
            num_domain_ids: int,
            specimen_hidden_dim: int = 512,
            scientific_embed_dim: int = 64,
            domain_embed_dim: int = 16,
            dropout: float = 0.2,
            name_dropout_prob: float = 0.0,
            domain_dropout_prob: float = 0.0,
        ) -> None:
            super().__init__()

            self.backbone = efficientnet_b0(weights=None)
            self.image_feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

            self.scientific_embedding = nn.Embedding(
                num_embeddings=max(1, int(num_scientific_names)),
                embedding_dim=int(scientific_embed_dim),
                padding_idx=0,
            )
            self.domain_embedding = nn.Embedding(
                num_embeddings=max(1, int(num_domain_ids)),
                embedding_dim=int(domain_embed_dim),
                padding_idx=0,
            )

            specimen_input_dim = (
                self.image_feature_dim + int(scientific_embed_dim) + int(domain_embed_dim)
            )

            self.specimen_projection = nn.Sequential(
                nn.Linear(specimen_input_dim, int(specimen_hidden_dim)),
                nn.LayerNorm(int(specimen_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )

            self.attention = nn.Sequential(
                nn.Linear(int(specimen_hidden_dim), int(specimen_hidden_dim) // 2),
                nn.Tanh(),
                nn.Linear(int(specimen_hidden_dim) // 2, 1),
            )

            self.head = nn.Sequential(
                nn.Linear(int(specimen_hidden_dim), int(specimen_hidden_dim) // 2),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(specimen_hidden_dim) // 2, 6),
            )
            self.name_dropout_prob = float(max(0.0, min(1.0, name_dropout_prob)))
            self.domain_dropout_prob = float(max(0.0, min(1.0, domain_dropout_prob)))

        def _attention_pool(
            self, specimen_features: "torch.Tensor", event_lengths: Sequence[int]
        ) -> "torch.Tensor":
            event_vectors: List[torch.Tensor] = []
            start = 0
            for n_specimens in event_lengths:
                end = start + int(n_specimens)
                feats = specimen_features[start:end]
                scores = self.attention(feats).squeeze(-1)
                weights = torch.softmax(scores, dim=0)
                pooled = torch.sum(weights.unsqueeze(-1) * feats, dim=0)
                event_vectors.append(pooled)
                start = end
            return torch.stack(event_vectors, dim=0)

        def forward(
            self, events: Sequence[Dict[str, "torch.Tensor"]]
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            if len(events) == 0:
                raise ValueError("Empty event batch.")

            device = next(self.parameters()).device

            images = []
            scientific_indices = []
            domain_indices = []
            event_lengths: List[int] = []

            for event in events:
                n_specimens = int(event["images"].shape[0])
                if n_specimens <= 0:
                    continue
                images.append(event["images"])
                scientific_indices.append(event["scientific_idx"])
                domain_indices.append(event["domain_idx"])
                event_lengths.append(n_specimens)

            if len(event_lengths) == 0:
                raise ValueError("All events were empty.")

            flat_images = torch.cat(images, dim=0).to(device, non_blocking=True).float()
            flat_scientific = torch.cat(scientific_indices, dim=0).to(device, non_blocking=True)
            flat_domain = torch.cat(domain_indices, dim=0).to(device, non_blocking=True)

            if self.training and self.name_dropout_prob > 0.0:
                mask = torch.rand(flat_scientific.shape, device=device) < self.name_dropout_prob
                flat_scientific = torch.where(
                    mask,
                    torch.zeros_like(flat_scientific),
                    flat_scientific,
                )

            if self.training and self.domain_dropout_prob > 0.0:
                mask = torch.rand(flat_domain.shape, device=device) < self.domain_dropout_prob
                flat_domain = torch.where(
                    mask,
                    torch.zeros_like(flat_domain),
                    flat_domain,
                )

            image_features = self.backbone(flat_images)
            if image_features.ndim > 2:
                image_features = torch.flatten(image_features, start_dim=1)

            name_features = self.scientific_embedding(flat_scientific)
            domain_features = self.domain_embedding(flat_domain)
            specimen_features = torch.cat([image_features, name_features, domain_features], dim=-1)
            specimen_features = self.specimen_projection(specimen_features)

            event_features = self._attention_pool(specimen_features, event_lengths)
            raw = self.head(event_features)

            mu = raw[:, :3]
            sigma = F.softplus(raw[:, 3:]) + MIN_SIGMA
            return mu, sigma


def _safe_float(value: Any, default: float) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _as_rgb(img_value: Any) -> Image.Image:
    if isinstance(img_value, Image.Image):
        return img_value.convert("RGB")
    return Image.new("RGB", (128, 128), color=(127, 127, 127))


def _image_stats(img: Image.Image) -> Dict[str, float]:
    img = img.resize((128, 128)).convert("RGB")
    stat = ImageStat.Stat(img)

    means = [m / 255.0 for m in stat.mean[:3]]
    stds = [math.sqrt(max(v, 0.0)) / 255.0 for v in stat.var[:3]]

    return {
        "brightness": _clip(sum(means) / 3.0, 0.0, 1.0),
        "contrast": _clip(sum(stds) / 3.0, 0.0, 1.0),
        "redness": _clip(means[0] - 0.5 * (means[1] + means[2]), -1.0, 1.0),
    }


def _default_priors() -> Dict[str, Any]:
    return {
        "global": {"mu": [0.0, 0.0, 0.0], "sigma": [0.9, 0.85, 0.8], "count": 1},
        "domain": {},
        "scientific": {},
        "smoothing": {"domain": 25.0, "scientific": 50.0},
    }


def _default_manifest() -> Dict[str, Any]:
    return {
        "format_version": 2,
        "fold_paths": [],
        "prior_file": "weights/priors.json",
        "blend": {
            "alpha_model": [0.88, 0.85, 0.82],
            "alpha_prior_if_no_model": [0.92, 0.95, 0.98],
        },
        "tta": {"hflip": True},
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
    }


def _domain_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def _name_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _hash_name_bias(name: str) -> float:
    if not name:
        return 0.0
    digest = sum(ord(c) for c in name) % 997
    value = (digest / 996.0) - 0.5
    return _clip(value * 0.3, -0.15, 0.15)


def _heuristic_image_predict(records: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    if len(records) == 0:
        return [0.0, 0.0, 0.0], [0.9, 0.85, 0.8]

    rel_b = []
    rel_c = []
    rel_r = []
    name_bias = []
    dom_bias = []

    for rec in records:
        stats = _image_stats(_as_rgb(rec.get("relative_img")))
        rel_b.append(stats["brightness"])
        rel_c.append(stats["contrast"])
        rel_r.append(stats["redness"])
        name_bias.append(_hash_name_bias(_name_str(rec.get("scientificName"))))
        dom = _domain_int(rec.get("domainID"))
        dom_bias.append(_clip((((dom % 17) / 16.0) - 0.5) * 0.4, -0.2, 0.2))

    n = float(len(records))
    b = sum(rel_b) / n
    c = sum(rel_c) / n
    r = sum(rel_r) / n
    nb = sum(name_bias) / n
    db = sum(dom_bias) / n

    mu30 = 2.0 * (b - 0.5) - 0.9 * c + 0.7 * r + 0.6 * db + 0.4 * nb
    mu1y = 0.78 * mu30 + 0.22 * db
    mu2y = 0.62 * mu30 + 0.30 * db
    mu = [_clip(mu30, -4.0, 4.0), _clip(mu1y, -4.0, 4.0), _clip(mu2y, -4.0, 4.0)]

    n_factor = 1.0 / math.sqrt(max(1.0, n))
    base_sigma = 0.55 + 0.45 * n_factor
    sigma = [
        _clip(base_sigma + 0.15 * c, 0.10, 2.5),
        _clip(base_sigma * 0.95 + 0.12 * c, 0.10, 2.5),
        _clip(base_sigma * 0.90 + 0.10 * c, 0.10, 2.5),
    ]
    return mu, sigma


def _fuse_gaussians(predictions: List[Tuple[List[float], List[float]]]) -> Tuple[List[float], List[float]]:
    if len(predictions) == 0:
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    mu_out: List[float] = []
    sigma_out: List[float] = []
    n = float(len(predictions))

    for t in range(3):
        mus = [_safe_float(p[0][t], 0.0) for p in predictions]
        sig = [max(_safe_float(p[1][t], 1.0), MIN_SIGMA) for p in predictions]

        mu = sum(mus) / n
        second_moment = sum((s * s) + (m * m) for m, s in zip(mus, sig)) / n
        var = max(second_moment - mu * mu, MIN_SIGMA * MIN_SIGMA)

        mu_out.append(mu)
        sigma_out.append(math.sqrt(var))

    return mu_out, sigma_out


def _alpha_vector(alpha_value: Any, default: List[float]) -> List[float]:
    if isinstance(alpha_value, (int, float)):
        a = _clip(_safe_float(alpha_value, 0.5), 0.0, 1.0)
        return [a, a, a]

    if isinstance(alpha_value, (list, tuple)):
        out: List[float] = []
        for i in range(3):
            dv = default[i] if i < len(default) else 0.5
            av = alpha_value[i] if i < len(alpha_value) else dv
            out.append(_clip(_safe_float(av, dv), 0.0, 1.0))
        return out

    return [_clip(_safe_float(default[i], 0.5), 0.0, 1.0) for i in range(3)]


def _blend_predictions(
    mu_a: List[float],
    sigma_a: List[float],
    mu_b: List[float],
    sigma_b: List[float],
    alpha_a: Any,
    alpha_default: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    if alpha_default is None:
        alpha_default = [0.5, 0.5, 0.5]
    alpha_vec = _alpha_vector(alpha_a, alpha_default)

    mu = []
    sigma = []
    for i in range(3):
        a = alpha_vec[i]
        b = 1.0 - a
        m = a * _safe_float(mu_a[i], 0.0) + b * _safe_float(mu_b[i], 0.0)
        s1 = max(_safe_float(sigma_a[i], 1.0), MIN_SIGMA)
        s2 = max(_safe_float(sigma_b[i], 1.0), MIN_SIGMA)
        v = (a * s1) ** 2 + (b * s2) ** 2
        mu.append(_clip(m, -4.0, 4.0))
        sigma.append(max(math.sqrt(max(v, MIN_SIGMA * MIN_SIGMA)), MIN_SIGMA))
    return mu, sigma


def _format_output(mu: List[float], sigma: List[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, tgt in enumerate(TARGETS):
        out[tgt] = {
            "mu": float(_safe_float(mu[i] if i < len(mu) else 0.0, 0.0)),
            "sigma": float(max(_safe_float(sigma[i] if i < len(sigma) else 1.0, 1.0), MIN_SIGMA)),
        }
    return out


def _get_vector(cfg: Dict[str, Any], key: str, default: List[float]) -> List[float]:
    raw = cfg.get(key, default)
    if not isinstance(raw, list):
        return list(default)
    out = []
    for i in range(3):
        out.append(_safe_float(raw[i] if i < len(raw) else default[i], default[i]))
    return out


def _apply_post_calibration(
    mu: List[float],
    sigma: List[float],
    global_mu: List[float],
    confidence_meta: Dict[str, float],
    manifest: Dict[str, Any],
) -> Tuple[List[float], List[float]]:
    cfg = manifest.get("post_calibration", {}) if isinstance(manifest, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    base_sigma_scale = _get_vector(cfg, "base_sigma_scale", [1.02, 1.08, 1.20])
    confidence_sigma_k = _get_vector(cfg, "confidence_sigma_k", [0.20, 0.38, 0.78])
    ood_sigma_boost = _get_vector(cfg, "ood_sigma_boost", [1.08, 1.20, 1.56])
    global_shrink = _get_vector(cfg, "global_shrink", [0.10, 0.20, 0.34])
    ood_shrink_extra = _get_vector(cfg, "ood_shrink_extra", [0.04, 0.12, 0.24])
    temporal_smooth = _get_vector(cfg, "temporal_smooth", [0.18, 0.32, 0.32])
    sigma_min = _get_vector(cfg, "sigma_min", [0.08, 0.10, 0.14])
    long_horizon_pull = _get_vector(cfg, "long_horizon_pull", [0.18, 0.36, 0.36])

    confidence = _clip(_safe_float(confidence_meta.get("confidence", 0.5), 0.5), 0.0, 1.0)
    ood_score = _clip(_safe_float(confidence_meta.get("ood_score", 0.5), 0.5), 0.0, 1.0)
    domain_support = _clip(_safe_float(confidence_meta.get("domain_support", 0.0), 0.0), 0.0, 1.0)
    scientific_support = _clip(
        _safe_float(confidence_meta.get("scientific_support", 0.0), 0.0),
        0.0,
        1.0,
    )
    known_frac = _clip(
        _safe_float(confidence_meta.get("scientific_known_frac", 0.0), 0.0),
        0.0,
        1.0,
    )
    specimen_conf = _clip(
        math.sqrt(max(_safe_float(confidence_meta.get("n_specimens", 1.0), 1.0), 1.0)) / 3.0,
        0.0,
        1.0,
    )

    # Stabilize confidence with support-aware terms to avoid over-trusting sparse metadata.
    support_conf = _clip(
        0.40 * domain_support + 0.30 * scientific_support + 0.20 * known_frac + 0.10 * specimen_conf,
        0.0,
        1.0,
    )
    confidence = 0.6 * confidence + 0.4 * support_conf
    ood_score = _clip(0.65 * ood_score + 0.35 * (1.0 - support_conf), 0.0, 1.0)

    mu_out = [_safe_float(mu[i], 0.0) for i in range(3)]
    sigma_out = [max(_safe_float(sigma[i], 1.0), MIN_SIGMA) for i in range(3)]
    gmu = [_safe_float(global_mu[i] if i < len(global_mu) else 0.0, 0.0) for i in range(3)]

    for i in range(3):
        shrink = global_shrink[i] * (1.0 - confidence) + ood_shrink_extra[i] * ood_score
        shrink = _clip(shrink, 0.0, 0.55)
        mu_out[i] = (1.0 - shrink) * mu_out[i] + shrink * gmu[i]

        scale = base_sigma_scale[i]
        scale *= 1.0 + confidence_sigma_k[i] * (1.0 - confidence)
        scale *= 1.0 + (ood_sigma_boost[i] - 1.0) * ood_score
        sigma_out[i] = max(sigma_out[i] * scale, MIN_SIGMA)

    # Temporal coherence regularization, stronger for low-confidence/OOD cases.
    w1 = _clip(temporal_smooth[0] * (1.0 - confidence + 0.5 * ood_score), 0.0, 0.35)
    anchor1 = 0.75 * mu_out[0] + 0.25 * gmu[1]
    mu_out[1] = (1.0 - w1) * mu_out[1] + w1 * anchor1

    w2 = _clip(temporal_smooth[1] * (1.0 - confidence + 0.7 * ood_score), 0.0, 0.45)
    anchor2 = 0.65 * mu_out[1] + 0.35 * gmu[2]
    mu_out[2] = (1.0 - w2) * mu_out[2] + w2 * anchor2

    # Additional long-horizon stabilization for OOD drift.
    pull1 = _clip(long_horizon_pull[0] * (0.5 * (1.0 - confidence) + 0.5 * ood_score), 0.0, 0.35)
    pull2 = _clip(long_horizon_pull[1] * (0.35 + 0.65 * ood_score), 0.0, 0.55)
    mu_out[1] = (1.0 - pull1) * mu_out[1] + pull1 * (0.80 * mu_out[0] + 0.20 * gmu[1])
    mu_out[2] = (1.0 - pull2) * mu_out[2] + pull2 * (0.70 * mu_out[1] + 0.30 * gmu[2])

    for i in range(3):
        mu_out[i] = _clip(mu_out[i], -4.0, 4.0)
        sigma_floor = max(_safe_float(sigma_min[i], 0.08), MIN_SIGMA)
        sigma_out[i] = _clip(max(sigma_out[i], sigma_floor), sigma_floor, 3.0)

    return mu_out, sigma_out


class Model:
    def __init__(self):
        self.loaded = False
        self.device = None
        self.manifest = _default_manifest()
        self.priors = _default_priors()
        self.fold_runtimes: List[Dict[str, Any]] = []

        _log("Model.__init__")
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            _log("torch available, device={}".format(self.device))
        else:
            _log("torch unavailable, fallback-only mode: {}".format(TORCH_IMPORT_ERROR))

    def _load_manifest(self, weights_dir: str) -> None:
        manifest_path = os.path.join(weights_dir, "model.pt")
        manifest = _default_manifest()

        if os.path.exists(manifest_path):
            # First try JSON manifest.
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    manifest.update(loaded)
                    _log("Loaded JSON manifest from weights/model.pt")
                    self.manifest = manifest
                    return
            except Exception:
                pass

            # Next try single-fold torch checkpoint legacy mode.
            if TORCH_AVAILABLE:
                try:
                    ckpt = torch.load(manifest_path, map_location="cpu")
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        manifest["fold_paths"] = ["weights/model.pt"]
                        _log("weights/model.pt is a single-fold torch checkpoint")
                except Exception:
                    pass

        self.manifest = manifest

    def _load_priors(self, base_dir: str, weights_dir: str) -> None:
        priors = _default_priors()

        prior_file = self.manifest.get("prior_file", "weights/priors.json")
        prior_path = prior_file
        if not os.path.isabs(prior_path):
            prior_path = os.path.join(base_dir, prior_file)

        if os.path.exists(prior_path):
            try:
                with open(prior_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    priors.update(loaded)
                    _log("Loaded priors from {}".format(prior_path))
            except Exception as exc:
                _log_exc("load_priors_file", exc)

        # Also support priors packed inside torch checkpoint for convenience.
        if TORCH_AVAILABLE:
            model_pt = os.path.join(weights_dir, "model.pt")
            if os.path.exists(model_pt):
                try:
                    ckpt = torch.load(model_pt, map_location="cpu")
                    if isinstance(ckpt, dict) and isinstance(ckpt.get("prior_stats"), dict):
                        priors.update(ckpt["prior_stats"])
                        _log("Loaded priors from torch checkpoint payload")
                except Exception:
                    pass

        self.priors = priors

    def _build_eval_transform(self, image_size: int):
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(int(image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_fold_runtimes(self, base_dir: str, weights_dir: str) -> None:
        self.fold_runtimes = []

        if not TORCH_AVAILABLE:
            return

        fold_paths = self.manifest.get("fold_paths", [])
        resolved_paths: List[str] = []

        if isinstance(fold_paths, list) and len(fold_paths) > 0:
            for p in fold_paths:
                if not isinstance(p, str):
                    continue
                rp = p if os.path.isabs(p) else os.path.join(base_dir, p)
                if os.path.exists(rp):
                    resolved_paths.append(rp)

        if len(resolved_paths) == 0:
            globbed = sorted(glob.glob(os.path.join(weights_dir, "fold_*.pt")))
            resolved_paths.extend(globbed)

        for fold_path in resolved_paths:
            try:
                ckpt = torch.load(fold_path, map_location="cpu")
                if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
                    continue

                config = ckpt.get("config", {})
                scientific_vocab = ckpt.get("scientific_vocab", ["__UNK__"])
                domain_vocab = ckpt.get("domain_vocab", [-1])
                sigma_cal = ckpt.get("sigma_calibration", [1.0, 1.0, 1.0])

                model = EventGaussianRegressor(
                    num_scientific_names=len(scientific_vocab),
                    num_domain_ids=len(domain_vocab),
                    specimen_hidden_dim=int(config.get("specimen_hidden_dim", 512)),
                    scientific_embed_dim=int(config.get("scientific_embed_dim", 64)),
                    domain_embed_dim=int(config.get("domain_embed_dim", 16)),
                    dropout=float(config.get("dropout", 0.2)),
                    name_dropout_prob=float(config.get("name_dropout_prob", 0.0)),
                    domain_dropout_prob=float(config.get("domain_dropout_prob", 0.0)),
                )
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                model.to(self.device)
                model.eval()

                runtime = {
                    "path": fold_path,
                    "model": model,
                    "transform": self._build_eval_transform(int(config.get("image_size", 224))),
                    "scientific_to_idx": {str(x): i for i, x in enumerate(scientific_vocab)},
                    "domain_to_idx": {int(x): i for i, x in enumerate(domain_vocab)},
                    "sigma_cal": torch.tensor(sigma_cal, dtype=torch.float32, device=self.device),
                }

                self.fold_runtimes.append(runtime)
                _log("Loaded fold checkpoint: {}".format(fold_path))
            except Exception as exc:
                _log_exc("load_fold_runtime:{}".format(fold_path), exc)

        _log("Total loaded fold runtimes={}".format(len(self.fold_runtimes)))

    def load(self):
        _log("Model.load start")
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(base_dir, "weights")

            self._load_manifest(weights_dir)
            self._load_priors(base_dir, weights_dir)
            self._load_fold_runtimes(base_dir, weights_dir)

            self.loaded = True
            _log("Model.load success")
        except Exception as exc:
            _log_exc("load", exc)
            self.loaded = True
            _log("Model.load fallback success")

    def _prior_predict(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[float], Dict[str, float]]:
        priors = self.priors if isinstance(self.priors, dict) else _default_priors()

        global_stats = priors.get("global", {})
        global_mu = list(global_stats.get("mu", [0.0, 0.0, 0.0]))
        global_sigma = list(global_stats.get("sigma", [0.9, 0.85, 0.8]))

        mu = [_safe_float(global_mu[i] if i < len(global_mu) else 0.0, 0.0) for i in range(3)]
        sigma = [
            max(_safe_float(global_sigma[i] if i < len(global_sigma) else 1.0, 1.0), MIN_SIGMA)
            for i in range(3)
        ]

        meta = {
            "global_mu_0": _safe_float(global_mu[0] if len(global_mu) > 0 else 0.0, 0.0),
            "global_mu_1": _safe_float(global_mu[1] if len(global_mu) > 1 else 0.0, 0.0),
            "global_mu_2": _safe_float(global_mu[2] if len(global_mu) > 2 else 0.0, 0.0),
            "domain_known": 0.0,
            "scientific_known_frac": 0.0,
            "domain_support": 0.0,
            "scientific_support": 0.0,
            "confidence": 0.0,
            "ood_score": 1.0,
            "n_specimens": float(len(records)),
        }

        if len(records) == 0:
            return mu, sigma, meta

        smoothing = priors.get("smoothing", {})
        dom_smooth = max(_safe_float(smoothing.get("domain", 25.0), 25.0), 1.0)
        sci_smooth = max(_safe_float(smoothing.get("scientific", 50.0), 50.0), 1.0)

        domain_table = priors.get("domain", {})
        scientific_table = priors.get("scientific", {})

        # Domain prior (mode domain per event)
        domain_counts: Dict[int, int] = {}
        for rec in records:
            d = _domain_int(rec.get("domainID"))
            domain_counts[d] = domain_counts.get(d, 0) + 1

        if len(domain_counts) > 0:
            dom = max(domain_counts.items(), key=lambda x: x[1])[0]
            dom_key = str(dom)
            dom_stat = domain_table.get(dom_key)
            if isinstance(dom_stat, dict):
                meta["domain_known"] = 1.0
                dom_mu = dom_stat.get("mu", global_mu)
                dom_sigma = dom_stat.get("sigma", global_sigma)
                dom_count = max(_safe_float(dom_stat.get("count", 1.0), 1.0), 1.0)
                w = dom_count / (dom_count + dom_smooth)
                meta["domain_support"] = _clip(math.log1p(dom_count) / math.log1p(250.0), 0.0, 1.0)
                for i in range(3):
                    dmu = _safe_float(dom_mu[i] if i < len(dom_mu) else mu[i], mu[i])
                    dsig = max(
                        _safe_float(dom_sigma[i] if i < len(dom_sigma) else sigma[i], sigma[i]),
                        MIN_SIGMA,
                    )
                    mu[i] = mu[i] + w * (dmu - mu[i])
                    sigma[i] = (1.0 - w) * sigma[i] + w * dsig

        # Scientific-name prior (weighted over specimen names)
        name_terms = []
        named_total = 0
        for rec in records:
            name = _name_str(rec.get("scientificName"))
            if not name:
                continue
            named_total += 1
            stat = scientific_table.get(name)
            if isinstance(stat, dict):
                count = max(_safe_float(stat.get("count", 1.0), 1.0), 1.0)
                w = count / (count + sci_smooth)
                name_terms.append((w, stat))

        if len(name_terms) > 0:
            total_w = sum(w for w, _ in name_terms)
            norm_w = min(1.0, total_w / max(1.0, float(len(records))))

            for i in range(3):
                weighted_mu = 0.0
                weighted_sigma = 0.0
                for w, stat in name_terms:
                    mvec = stat.get("mu", global_mu)
                    svec = stat.get("sigma", global_sigma)
                    weighted_mu += w * _safe_float(mvec[i] if i < len(mvec) else mu[i], mu[i])
                    weighted_sigma += w * max(
                        _safe_float(svec[i] if i < len(svec) else sigma[i], sigma[i]),
                        MIN_SIGMA,
                    )
                weighted_mu /= max(total_w, 1e-8)
                weighted_sigma /= max(total_w, 1e-8)

                mu[i] = mu[i] + norm_w * (weighted_mu - mu[i])
                sigma[i] = (1.0 - norm_w) * sigma[i] + norm_w * weighted_sigma

            meta["scientific_known_frac"] = _clip(
                float(len(name_terms)) / max(1.0, float(named_total)),
                0.0,
                1.0,
            )
            avg_support = total_w / max(1.0, float(len(name_terms)))
            meta["scientific_support"] = _clip(avg_support, 0.0, 1.0)

        n_spec = max(1.0, float(len(records)))
        specimen_conf = min(1.0, n_spec / 4.0)
        meta["confidence"] = _clip(
            0.35 * meta["domain_support"]
            + 0.25 * meta["scientific_support"]
            + 0.25 * meta["scientific_known_frac"]
            + 0.15 * specimen_conf,
            0.0,
            1.0,
        )
        meta["ood_score"] = _clip(
            1.0 - (
                0.45 * meta["domain_support"]
                + 0.35 * meta["scientific_support"]
                + 0.20 * meta["scientific_known_frac"]
            ),
            0.0,
            1.0,
        )

        sigma = [max(s, MIN_SIGMA) for s in sigma]
        return mu, sigma, meta

    def _predict_fold_once(
        self,
        fold_runtime: Dict[str, Any],
        records: List[Dict[str, Any]],
        hflip: bool,
    ) -> Tuple[List[float], List[float]]:
        images = []
        sci_idx = []
        dom_idx = []

        for rec in records:
            img = _as_rgb(rec.get("relative_img"))
            if hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            images.append(fold_runtime["transform"](img))

            name = _name_str(rec.get("scientificName"))
            sci_idx.append(fold_runtime["scientific_to_idx"].get(name, 0))

            dom = _domain_int(rec.get("domainID"))
            dom_idx.append(fold_runtime["domain_to_idx"].get(dom, 0))

        if len(images) == 0:
            img = Image.new("RGB", (224, 224), color=(127, 127, 127))
            images = [fold_runtime["transform"](img)]
            sci_idx = [0]
            dom_idx = [0]

        event = {
            "images": torch.stack(images, dim=0),
            "scientific_idx": torch.tensor(sci_idx, dtype=torch.long),
            "domain_idx": torch.tensor(dom_idx, dtype=torch.long),
        }

        with torch.no_grad():
            mu, sigma = fold_runtime["model"]([event])
            sigma = sigma * fold_runtime["sigma_cal"].view(1, -1)

        mu_list = [float(x) for x in mu[0].detach().cpu().tolist()]
        sigma_list = [max(float(x), MIN_SIGMA) for x in sigma[0].detach().cpu().tolist()]
        return mu_list, sigma_list

    def _predict_with_models(self, records: List[Dict[str, Any]]) -> Optional[Tuple[List[float], List[float]]]:
        if not TORCH_AVAILABLE:
            return None
        if len(self.fold_runtimes) == 0:
            return None

        tta_cfg = self.manifest.get("tta", {})
        use_hflip = bool(tta_cfg.get("hflip", True))

        preds: List[Tuple[List[float], List[float]]] = []
        for fold_idx, runtime in enumerate(self.fold_runtimes):
            try:
                preds.append(self._predict_fold_once(runtime, records, hflip=False))
                if use_hflip:
                    preds.append(self._predict_fold_once(runtime, records, hflip=True))
            except Exception as exc:
                _log_exc("predict_fold_{}".format(fold_idx), exc)

        if len(preds) == 0:
            return None

        return _fuse_gaussians(preds)

    def predict(self, event_records):
        _log("Model.predict start")
        try:
            if not self.loaded:
                self.load()

            if event_records is None:
                records: List[Dict[str, Any]] = []
            elif isinstance(event_records, list):
                records = [x for x in event_records if isinstance(x, dict)]
            else:
                try:
                    records = [x for x in list(event_records) if isinstance(x, dict)]
                except Exception:
                    records = []

            _log("records_count={}".format(len(records)))
            for i, rec in enumerate(records[:MAX_DEBUG_RECORDS]):
                _log(
                    "record[{}] keys={} scientificName={} domainID={}".format(
                        i,
                        sorted(list(rec.keys())),
                        rec.get("scientificName"),
                        rec.get("domainID"),
                    )
                )

            prior_mu, prior_sigma, prior_meta = self._prior_predict(records)
            heur_mu, heur_sigma = _heuristic_image_predict(records)
            prior_mu, prior_sigma = _blend_predictions(
                prior_mu,
                prior_sigma,
                heur_mu,
                heur_sigma,
                alpha_a=self.manifest.get("blend", {}).get(
                    "alpha_prior_if_no_model",
                    [0.92, 0.95, 0.98],
                ),
                alpha_default=[0.92, 0.95, 0.98],
            )

            global_mu = [
                prior_meta.get("global_mu_0", 0.0),
                prior_meta.get("global_mu_1", 0.0),
                prior_meta.get("global_mu_2", 0.0),
            ]

            model_pred = self._predict_with_models(records)
            if model_pred is None:
                final_mu, final_sigma = _apply_post_calibration(
                    prior_mu,
                    prior_sigma,
                    global_mu,
                    prior_meta,
                    self.manifest,
                )
                out = _format_output(final_mu, final_sigma)
                _log("Model.predict success (prior-only mode)")
                _log("prior_meta={}".format(prior_meta))
                _log("prediction={}".format(out))
                return out

            model_mu, model_sigma = model_pred
            final_mu, final_sigma = _blend_predictions(
                model_mu,
                model_sigma,
                prior_mu,
                prior_sigma,
                alpha_a=self.manifest.get("blend", {}).get("alpha_model", [0.88, 0.85, 0.82]),
                alpha_default=[0.88, 0.85, 0.82],
            )
            final_mu, final_sigma = _apply_post_calibration(
                final_mu,
                final_sigma,
                global_mu,
                prior_meta,
                self.manifest,
            )

            out = _format_output(final_mu, final_sigma)
            _log("Model.predict success (ensemble+prior)")
            _log("prior_meta={}".format(prior_meta))
            _log("prediction={}".format(out))
            return out

        except Exception as exc:
            _log_exc("predict", exc)
            fallback = _format_output([0.0, 0.0, 0.0], [0.9, 0.85, 0.8])
            _log("Model.predict fallback prediction={}".format(fallback))
            return fallback


def _build_debug_events(num_events: int) -> Dict[str, List[Dict[str, Any]]]:
    events: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(max(1, int(num_events))):
        event_id = "debug_event_{:03d}".format(i + 1)
        n_specimens = 1 + (i % 3)
        recs: List[Dict[str, Any]] = []
        for j in range(n_specimens):
            color = (
                (80 + 31 * i + 13 * j) % 256,
                (70 + 27 * i + 9 * j) % 256,
                (60 + 22 * i + 7 * j) % 256,
            )
            img = Image.new("RGB", (128, 128), color=color)
            recs.append(
                {
                    "relative_img": img,
                    "colorpicker_img": img,
                    "scalebar_img": img,
                    "scientificName": "DebugSpecies_{}".format(i % 8),
                    "domainID": (i % 12) + 1,
                }
            )
        events[event_id] = recs
    return events


def _run_main() -> None:
    parser = argparse.ArgumentParser(description="Local debug runner for model.py")
    parser.add_argument("--out_csv", type=str, default="predictions.csv")
    parser.add_argument("--num_events", type=int, default=3)
    args = parser.parse_args()

    _log("__main__ start out_csv={} num_events={}".format(args.out_csv, args.num_events))

    model = Model()
    model.load()

    rows = []
    for event_id, records in _build_debug_events(args.num_events).items():
        pred = model.predict(records)
        rows.append(
            {
                "eventID": event_id,
                "SPEI_30d_mu": pred["SPEI_30d"]["mu"],
                "SPEI_30d_sigma": pred["SPEI_30d"]["sigma"],
                "SPEI_1y_mu": pred["SPEI_1y"]["mu"],
                "SPEI_1y_sigma": pred["SPEI_1y"]["sigma"],
                "SPEI_2y_mu": pred["SPEI_2y"]["mu"],
                "SPEI_2y_sigma": pred["SPEI_2y"]["sigma"],
            }
        )

    fields = [
        "eventID",
        "SPEI_30d_mu",
        "SPEI_30d_sigma",
        "SPEI_1y_mu",
        "SPEI_1y_sigma",
        "SPEI_2y_mu",
        "SPEI_2y_sigma",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    _log("__main__ wrote {}".format(os.path.abspath(args.out_csv)))
    print("Generated {}".format(os.path.abspath(args.out_csv)))


if __name__ == "__main__":
    _run_main()
