import math
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

TARGET_NAMES = ("SPEI_30d", "SPEI_1y", "SPEI_2y")
EPS = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


class EventGaussianRegressor(nn.Module):
    """Event-level distributional regressor with attention pooling."""

    def __init__(
        self,
        num_scientific_names: int,
        num_domain_ids: int,
        specimen_hidden_dim: int = 512,
        scientific_embed_dim: int = 64,
        domain_embed_dim: int = 16,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
        name_dropout_prob: float = 0.0,
        domain_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        self.backbone = efficientnet_b0(weights=weights)
        self.image_feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.scientific_embedding = nn.Embedding(
            num_embeddings=max(1, num_scientific_names),
            embedding_dim=scientific_embed_dim,
            padding_idx=0,
        )
        self.domain_embedding = nn.Embedding(
            num_embeddings=max(1, num_domain_ids),
            embedding_dim=domain_embed_dim,
            padding_idx=0,
        )

        specimen_input_dim = (
            self.image_feature_dim + scientific_embed_dim + domain_embed_dim
        )
        self.specimen_projection = nn.Sequential(
            nn.Linear(specimen_input_dim, specimen_hidden_dim),
            nn.LayerNorm(specimen_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention = nn.Sequential(
            nn.Linear(specimen_hidden_dim, specimen_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(specimen_hidden_dim // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(specimen_hidden_dim, specimen_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(specimen_hidden_dim // 2, 6),
        )

        self.name_dropout_prob = float(max(0.0, min(1.0, name_dropout_prob)))
        self.domain_dropout_prob = float(max(0.0, min(1.0, domain_dropout_prob)))

    def _attention_pool(
        self, specimen_features: torch.Tensor, event_lengths: Sequence[int]
    ) -> torch.Tensor:
        event_vectors: List[torch.Tensor] = []
        start = 0
        for n_specimens in event_lengths:
            end = start + n_specimens
            feats = specimen_features[start:end]
            scores = self.attention(feats).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            pooled = torch.sum(weights.unsqueeze(-1) * feats, dim=0)
            event_vectors.append(pooled)
            start = end

        return torch.stack(event_vectors, dim=0)

    def forward(
        self, events: Sequence[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(events) == 0:
            raise ValueError("Received an empty event batch.")

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
            raise ValueError("All events in the batch are empty.")

        flat_images = torch.cat(images, dim=0).to(device, non_blocking=True)
        flat_images = flat_images.float()
        flat_scientific = torch.cat(scientific_indices, dim=0).to(
            device, non_blocking=True
        )
        flat_domain = torch.cat(domain_indices, dim=0).to(device, non_blocking=True)

        if self.training and self.name_dropout_prob > 0.0:
            mask = torch.rand(flat_scientific.shape, device=device) < self.name_dropout_prob
            flat_scientific = torch.where(mask, torch.zeros_like(flat_scientific), flat_scientific)

        if self.training and self.domain_dropout_prob > 0.0:
            mask = torch.rand(flat_domain.shape, device=device) < self.domain_dropout_prob
            flat_domain = torch.where(mask, torch.zeros_like(flat_domain), flat_domain)

        image_features = self.backbone(flat_images)
        if image_features.ndim > 2:
            image_features = torch.flatten(image_features, start_dim=1)

        name_features = self.scientific_embedding(flat_scientific)
        domain_features = self.domain_embedding(flat_domain)

        specimen_features = torch.cat(
            [image_features, name_features, domain_features], dim=-1
        )
        specimen_features = self.specimen_projection(specimen_features)

        event_features = self._attention_pool(specimen_features, event_lengths)

        raw_outputs = self.head(event_features)
        mu = raw_outputs[:, :3]
        sigma = F.softplus(raw_outputs[:, 3:]) + EPS
        return mu, sigma


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    sigma = sigma.clamp_min(EPS)
    var = sigma * sigma
    per_target = 0.5 * torch.log(var) + 0.5 * (target - mu).pow(2) / var

    if reduction == "none":
        return per_target
    if reduction == "sum":
        return per_target.sum()
    return per_target.mean()


def gaussian_crps(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    sigma = sigma.clamp_min(EPS)
    z = (target - mu) / sigma
    phi = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps


def fit_sigma_scaling(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    min_scale: float = 0.05,
    max_scale: float = 10.0,
) -> torch.Tensor:
    sigma = sigma.clamp_min(EPS)
    standardized_error = (target - mu) / sigma
    scale = torch.sqrt(torch.mean(standardized_error.pow(2), dim=0)).clamp(
        min=min_scale,
        max=max_scale,
    )
    return scale


def compute_regression_metrics(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    diff = mu - target
    rmse = torch.sqrt(torch.mean(diff.pow(2), dim=0))
    nll = gaussian_nll_loss(mu, sigma, target, reduction="none").mean(dim=0)
    crps = gaussian_crps(mu, sigma, target).mean(dim=0)

    metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(TARGET_NAMES):
        metrics[name] = {
            "rmse": float(rmse[idx].item()),
            "nll": float(nll[idx].item()),
            "crps": float(crps[idx].item()),
        }

    metrics["overall"] = {
        "rmse_mean": float(rmse.mean().item()),
        "nll_mean": float(nll.mean().item()),
        "crps_mean": float(crps.mean().item()),
    }
    return metrics


def metrics_to_log_string(metrics: Dict[str, Dict[str, float]]) -> str:
    parts = []
    for name in TARGET_NAMES:
        m = metrics[name]
        parts.append(
            f"{name}: rmse={m['rmse']:.4f} nll={m['nll']:.4f} crps={m['crps']:.4f}"
        )
    o = metrics["overall"]
    parts.append(
        "overall: "
        f"rmse={o['rmse_mean']:.4f} nll={o['nll_mean']:.4f} crps={o['crps_mean']:.4f}"
    )
    return " | ".join(parts)


def save_checkpoint(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)
