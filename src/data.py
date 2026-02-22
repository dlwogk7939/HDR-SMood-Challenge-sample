import io
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

TARGET_COLUMNS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]
UNK_NAME = "__UNK__"
UNK_DOMAIN = -1


def load_hf_split(
    split: str = "train",
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {}
    if hf_token:
        kwargs["token"] = hf_token
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    return load_dataset("imageomics/sentinel-beetles", split=split, **kwargs)


def _find_column(
    column_names: Sequence[str],
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    for candidate in candidates:
        if candidate in column_names:
            return candidate

    if required:
        raise KeyError(
            f"Could not find any of {list(candidates)} in dataset columns: {list(column_names)}"
        )
    return None


def detect_columns(column_names: Sequence[str]) -> Dict[str, Optional[str]]:
    columns = list(column_names)
    mapping: Dict[str, Optional[str]] = {
        "event_id_col": _find_column(columns, ["eventID", "eventId", "event_id"]),
        "domain_id_col": _find_column(
            columns,
            ["domainID", "domainId", "domain_id"],
        ),
        "scientific_name_col": _find_column(
            columns,
            ["scientificName", "scientific_name"],
            required=False,
        ),
        "site_id_col": _find_column(
            columns,
            ["siteID", "siteId", "site_id"],
            required=False,
        ),
        "image_col": _find_column(
            columns,
            [
                "relative_img",
                "file_path",
                "image",
                "img",
                "relative_img_loc",
            ],
        ),
        "colorpicker_col": _find_column(
            columns,
            ["colorpicker_img", "colorpicker", "colorpicker_img_loc"],
            required=False,
        ),
        "scalebar_col": _find_column(
            columns,
            ["scalebar_img", "scalebar", "scalebar_img_loc"],
            required=False,
        ),
    }
    return mapping


def group_event_indices(
    dataset,
    columns: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    event_col = columns["event_id_col"]
    site_col = columns["site_id_col"]
    domain_col = columns["domain_id_col"]

    target_cols = [c for c in TARGET_COLUMNS if c in dataset.column_names]

    metadata_columns = [event_col, domain_col]
    if site_col:
        metadata_columns.append(site_col)
    for col in target_cols:
        if col not in metadata_columns:
            metadata_columns.append(col)

    metadata = dataset.select_columns(metadata_columns)

    grouped: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(metadata):
        event_id = str(row[event_col])
        domain_id = _safe_domain(row.get(domain_col, UNK_DOMAIN))
        site_id = row.get(site_col) if site_col else None

        if event_id not in grouped:
            target = None
            if len(target_cols) == len(TARGET_COLUMNS):
                target = [float(row[c]) for c in TARGET_COLUMNS]

            grouped[event_id] = {
                "event_id": event_id,
                "row_indices": [row_idx],
                "domain_id": domain_id,
                "site_id": site_id,
                "target": target,
            }
        else:
            grouped[event_id]["row_indices"].append(row_idx)

    events = [grouped[k] for k in sorted(grouped.keys())]
    return events


def choose_group_key(events: Sequence[Dict[str, Any]], prefer_site: bool = True) -> str:
    if prefer_site:
        site_ids = {
            str(event["site_id"])
            for event in events
            if event.get("site_id") not in (None, "", "None")
        }
        if len(site_ids) > 1:
            return "site_id"
    return "domain_id"


def split_events_group_kfold(
    events: Sequence[Dict[str, Any]],
    n_splits: int = 5,
    fold: int = 0,
    seed: int = 42,
    prefer_site: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    group_key = choose_group_key(events, prefer_site=prefer_site)
    group_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, event in enumerate(events):
        raw_group = event.get(group_key)
        if raw_group in (None, "", "None"):
            raw_group = event.get("domain_id", UNK_DOMAIN)
        group_id = str(raw_group)
        group_to_indices[group_id].append(idx)

    all_groups = list(group_to_indices.keys())
    if len(all_groups) == 0:
        return list(events), [], group_key

    if len(all_groups) == 1:
        # Degenerate case: only one group exists, so strict group-held-out split is impossible.
        # Fall back to an event-level 80/20 split to keep training/evaluation runnable.
        indices = list(range(len(events)))
        rng.shuffle(indices)
        cut = max(1, int(0.8 * len(indices)))
        train_idx = set(indices[:cut])
        train_events = [events[i] for i in range(len(events)) if i in train_idx]
        val_events = [events[i] for i in range(len(events)) if i not in train_idx]
        return train_events, val_events, group_key

    n_splits = max(2, min(n_splits, len(all_groups)))
    rng = random.Random(seed)
    rng.shuffle(all_groups)
    all_groups.sort(key=lambda g: len(group_to_indices[g]), reverse=True)

    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]

    for group in all_groups:
        min_size = min(fold_sizes)
        candidates = [i for i, size in enumerate(fold_sizes) if size == min_size]
        chosen_fold = rng.choice(candidates)
        fold_groups[chosen_fold].append(group)
        fold_sizes[chosen_fold] += len(group_to_indices[group])

    val_fold = fold % n_splits
    val_group_set = set(fold_groups[val_fold])

    train_events: List[Dict[str, Any]] = []
    val_events: List[Dict[str, Any]] = []

    for event in events:
        raw_group = event.get(group_key)
        if raw_group in (None, "", "None"):
            raw_group = event.get("domain_id", UNK_DOMAIN)
        group_id = str(raw_group)

        if group_id in val_group_set:
            val_events.append(event)
        else:
            train_events.append(event)

    return train_events, val_events, group_key


def build_vocabs(
    dataset,
    train_events: Sequence[Dict[str, Any]],
    columns: Dict[str, Optional[str]],
    min_name_freq: int = 1,
) -> Tuple[List[str], List[int]]:
    scientific_col = columns.get("scientific_name_col")
    domain_col = columns.get("domain_id_col")

    name_vocab = [UNK_NAME]
    domain_vocab = [UNK_DOMAIN]

    all_train_indices: List[int] = []
    for event in train_events:
        all_train_indices.extend([int(idx) for idx in event["row_indices"]])

    if scientific_col is not None:
        all_names = dataset[scientific_col]
        counter = Counter()
        for idx in all_train_indices:
            name = all_names[idx]
            if name is None:
                continue
            counter[str(name)] += 1

        filtered_names = [
            name for name, freq in counter.items() if int(freq) >= int(min_name_freq)
        ]
        filtered_names.sort()
        name_vocab.extend(filtered_names)

    if domain_col is not None:
        all_domains = dataset[domain_col]
        domain_set = set()
        for idx in all_train_indices:
            domain_set.add(_safe_domain(all_domains[idx]))

        for domain in sorted(domain_set):
            if domain == UNK_DOMAIN:
                continue
            domain_vocab.append(domain)

    return name_vocab, domain_vocab


def build_transforms(image_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(degrees=12),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, eval_tf


def make_index_map(vocab: Sequence[Any]) -> Dict[Any, int]:
    return {token: idx for idx, token in enumerate(vocab)}


def _safe_domain(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return UNK_DOMAIN


def _to_pil_rgb(value: Any, fallback_size: int = 224) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") and os.path.exists(value["path"]):
            return Image.open(value["path"]).convert("RGB")

    if isinstance(value, str) and os.path.exists(value):
        return Image.open(value).convert("RGB")

    return Image.new("RGB", (fallback_size, fallback_size), color=(127, 127, 127))


def _resolve_image(row: Dict[str, Any], columns: Dict[str, Optional[str]]) -> Image.Image:
    keys = [
        columns.get("image_col"),
        "relative_img",
        "file_path",
        "image",
        "img",
        "relative_img_loc",
    ]
    for key in keys:
        if key and key in row and row[key] is not None:
            return _to_pil_rgb(row[key])

    return Image.new("RGB", (224, 224), color=(127, 127, 127))


class EventDataset(Dataset):
    def __init__(
        self,
        dataset,
        events: Sequence[Dict[str, Any]],
        columns: Dict[str, Optional[str]],
        scientific_to_idx: Dict[str, int],
        domain_to_idx: Dict[int, int],
        transform,
        include_targets: bool = True,
        max_specimens_per_event: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.events = list(events)
        self.columns = columns
        self.scientific_to_idx = scientific_to_idx
        self.domain_to_idx = domain_to_idx
        self.transform = transform
        self.include_targets = include_targets
        self.max_specimens_per_event = max_specimens_per_event

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        event = self.events[index]

        images: List[torch.Tensor] = []
        scientific_idx: List[int] = []
        domain_idx: List[int] = []

        scientific_col = self.columns.get("scientific_name_col")
        domain_col = self.columns.get("domain_id_col")

        row_indices = list(event["row_indices"])
        if self.max_specimens_per_event is not None:
            max_n = max(1, int(self.max_specimens_per_event))
            if len(row_indices) > max_n:
                row_indices = random.sample(row_indices, max_n)

        for row_idx in row_indices:
            row = self.dataset[int(row_idx)]
            pil_img = _resolve_image(row, self.columns)
            images.append(self.transform(pil_img))

            if scientific_col is not None:
                name = row.get(scientific_col)
                name = UNK_NAME if name is None else str(name)
            else:
                name = UNK_NAME

            scientific_idx.append(self.scientific_to_idx.get(name, 0))

            raw_domain = row.get(domain_col, UNK_DOMAIN) if domain_col else UNK_DOMAIN
            domain = _safe_domain(raw_domain)
            domain_idx.append(self.domain_to_idx.get(domain, 0))

        if len(images) == 0:
            images = [self.transform(Image.new("RGB", (224, 224), color=(127, 127, 127)))]
            scientific_idx = [0]
            domain_idx = [0]

        target = None
        if self.include_targets and event.get("target") is not None:
            target = torch.tensor(event["target"], dtype=torch.float32)

        return {
            "event_id": event["event_id"],
            "images": torch.stack(images, dim=0),
            "scientific_idx": torch.tensor(scientific_idx, dtype=torch.long),
            "domain_idx": torch.tensor(domain_idx, dtype=torch.long),
            "target": target,
        }


def collate_event_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    events = []
    event_ids = []
    targets = []

    for item in batch:
        events.append(
            {
                "images": item["images"],
                "scientific_idx": item["scientific_idx"],
                "domain_idx": item["domain_idx"],
            }
        )
        event_ids.append(item["event_id"])
        if item["target"] is not None:
            targets.append(item["target"])

    target_tensor = None
    if len(targets) == len(batch):
        target_tensor = torch.stack(targets, dim=0)

    return {
        "events": events,
        "targets": target_tensor,
        "event_ids": event_ids,
    }
