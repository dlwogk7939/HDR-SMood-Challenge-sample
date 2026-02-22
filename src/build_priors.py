import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from data import detect_columns, group_event_indices, load_hf_split


TARGET_NAMES = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build metadata priors from train split")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output", type=str, default="weights/priors.json")
    p.add_argument("--min_domain_count", type=int, default=8)
    p.add_argument("--min_scientific_count", type=int, default=20)
    p.add_argument("--max_scientific", type=int, default=5000)
    p.add_argument("--domain_smoothing", type=float, default=25.0)
    p.add_argument("--scientific_smoothing", type=float, default=50.0)
    return p.parse_args()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _new_acc() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum": [0.0, 0.0, 0.0],
        "sumsq": [0.0, 0.0, 0.0],
    }


def _update_acc(acc: Dict[str, Any], y: List[float]) -> None:
    acc["count"] += 1
    for i in range(3):
        v = _safe_float(y[i], 0.0)
        acc["sum"][i] += v
        acc["sumsq"][i] += v * v


def _finalize_acc(acc: Dict[str, Any]) -> Dict[str, Any]:
    c = max(1, int(acc["count"]))
    mu = [acc["sum"][i] / c for i in range(3)]
    sigma = []
    for i in range(3):
        ex2 = acc["sumsq"][i] / c
        var = max(ex2 - (mu[i] * mu[i]), 1e-6)
        sigma.append(var ** 0.5)
    return {
        "mu": mu,
        "sigma": sigma,
        "count": int(acc["count"]),
    }


def _rank_by_count(stats: Dict[str, Dict[str, Any]]) -> List[Tuple[str, int]]:
    ranked = [(k, int(v.get("count", 0))) for k, v in stats.items()]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def main() -> None:
    args = parse_args()

    print("Loading dataset split for priors...")
    ds = load_hf_split(split=args.split, hf_token=args.hf_token, cache_dir=args.cache_dir)
    columns = detect_columns(ds.column_names)

    scientific_col = columns.get("scientific_name_col")
    domain_col = columns.get("domain_id_col")

    if scientific_col is None or domain_col is None:
        raise RuntimeError("Dataset missing scientific/domain columns required for priors")

    print("Grouping events...")
    events = group_event_indices(ds, columns)
    labeled_events = [e for e in events if e.get("target") is not None]

    if len(labeled_events) == 0:
        raise RuntimeError("No labeled events found in selected split")

    # Use metadata-only table to avoid touching image bytes.
    meta_cols = [scientific_col, domain_col]
    meta_ds = ds.select_columns(meta_cols)

    global_acc = _new_acc()
    domain_acc = defaultdict(_new_acc)
    scientific_acc = defaultdict(_new_acc)

    for event in labeled_events:
        y = [float(v) for v in event["target"]]
        _update_acc(global_acc, y)

        for idx in event["row_indices"]:
            row = meta_ds[int(idx)]
            dom = _safe_int(row.get(domain_col, -1), -1)
            name = row.get(scientific_col)
            name = "" if name is None else str(name)

            _update_acc(domain_acc[str(dom)], y)
            if name:
                _update_acc(scientific_acc[name], y)

    global_stats = _finalize_acc(global_acc)

    domain_stats: Dict[str, Dict[str, Any]] = {}
    for k, acc in domain_acc.items():
        if int(acc["count"]) < int(args.min_domain_count):
            continue
        domain_stats[k] = _finalize_acc(acc)

    scientific_stats: Dict[str, Dict[str, Any]] = {}
    for k, acc in scientific_acc.items():
        if int(acc["count"]) < int(args.min_scientific_count):
            continue
        scientific_stats[k] = _finalize_acc(acc)

    if int(args.max_scientific) > 0 and len(scientific_stats) > int(args.max_scientific):
        ranked = _rank_by_count(scientific_stats)
        keep = {k for k, _ in ranked[: int(args.max_scientific)]}
        scientific_stats = {k: v for k, v in scientific_stats.items() if k in keep}

    payload = {
        "global": global_stats,
        "domain": domain_stats,
        "scientific": scientific_stats,
        "smoothing": {
            "domain": float(args.domain_smoothing),
            "scientific": float(args.scientific_smoothing),
        },
        "meta": {
            "split": args.split,
            "num_labeled_events": len(labeled_events),
            "num_domains": len(domain_stats),
            "num_scientific": len(scientific_stats),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Saved priors to {args.output}")
    print(
        f"Global mu={payload['global']['mu']} sigma={payload['global']['sigma']} "
        f"domains={len(domain_stats)} scientific={len(scientific_stats)}"
    )


if __name__ == "__main__":
    main()
