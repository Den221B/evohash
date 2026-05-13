"""Shared metrics for EvoHash.

Naming convention:
- hash_l1: distance in hash space, returned by hash_fn.distance(...)
- pixel_l2: normalized pixel-space L2 distortion
- success: final_hash_l1 <= threshold
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np

_EPS = 1e-8


def to_float32(image: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] or float image to float32 [0,1]."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    return arr.clip(0.0, 1.0)


def compute_pixel_l2(x_source: np.ndarray, x_candidate: np.ndarray) -> float:
    """Normalized L2: ||x_candidate - x_source||_2 / sqrt(H*W*C)."""
    x0 = to_float32(x_source)
    x1 = to_float32(x_candidate)
    diff = (x1 - x0).reshape(-1)
    return float(np.linalg.norm(diff, ord=2) / np.sqrt(max(diff.size, 1)))


def compute_pixel_l2_raw(x_source: np.ndarray, x_candidate: np.ndarray) -> float:
    """Raw L2 kept only for comparability with old notebooks."""
    x0 = to_float32(x_source)
    x1 = to_float32(x_candidate)
    return float(np.linalg.norm((x1 - x0).reshape(-1), ord=2))


def compute_relative_improvement(
    initial_hash_l1: float,
    final_hash_l1: float,
    threshold: float,
) -> float:
    """Progress towards threshold, robust when initial is already near threshold."""
    denom = max(float(initial_hash_l1) - float(threshold), 1.0)
    return float((float(initial_hash_l1) - float(final_hash_l1)) / denom)


def score_asr_per_l2(asr: float, mean_pixel_l2: float) -> float:
    return float(asr / (mean_pixel_l2 + _EPS))


def aggregate_results(rows: Iterable[Any]) -> dict[str, float]:
    """Aggregate a list of EvalRow-like dataclasses or dicts."""
    dict_rows: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "__dataclass_fields__"):
            dict_rows.append(asdict(row))
        else:
            dict_rows.append(dict(row))

    if not dict_rows:
        return {}

    success = np.array([bool(r["success"]) for r in dict_rows], dtype=np.float32)
    pixel_l2 = np.array([float(r["pixel_l2"]) for r in dict_rows], dtype=np.float32)
    queries = np.array([float(r["queries"]) for r in dict_rows], dtype=np.float32)
    time_sec = np.array([float(r["time_sec"]) for r in dict_rows], dtype=np.float32)
    initial = np.array([float(r["initial_hash_l1"]) for r in dict_rows], dtype=np.float32)
    final = np.array([float(r["final_hash_l1"]) for r in dict_rows], dtype=np.float32)
    rel = np.array([float(r["relative_improvement"]) for r in dict_rows], dtype=np.float32)

    asr = float(success.mean())
    mean_pixel_l2 = float(pixel_l2.mean())

    return {
        "n_pairs": int(len(dict_rows)),
        "n_success": int(success.sum()),
        "ASR": asr,
        "mean_initial_hash_l1": float(initial.mean()),
        "mean_final_hash_l1": float(final.mean()),
        "mean_relative_improvement": float(rel.mean()),
        "improvement_rate": float(np.mean(final < initial)),
        "mean_pixel_l2": mean_pixel_l2,
        "median_pixel_l2": float(np.median(pixel_l2)),
        "mean_queries": float(queries.mean()),
        "median_queries": float(np.median(queries)),
        "mean_time_sec": float(time_sec.mean()),
        "score_asr_per_l2": score_asr_per_l2(asr, mean_pixel_l2),
    }


# ---------------------------------------------------------------------------
# Backwards-compatible names used by evohash.__init__ and older notebooks.
# ---------------------------------------------------------------------------

def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, key):
        return getattr(row, key)
    try:
        return row.get(key, default)
    except AttributeError:
        return default


def asr_over_l2(rows: Iterable[Any]) -> float:
    m = aggregate_results(rows)
    return float(m.get("score_asr_per_l2", 0.0))


def aggregate(rows: Iterable[Any]) -> dict[str, float]:
    return aggregate_results(rows)


def build_comparison_table(
    rows: Iterable[Any],
    hash_ids: list[str] | None = None,
    attack_ids: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    rows_list = list(rows)
    if hash_ids is None:
        hash_ids = sorted({str(_row_get(r, "hash_id")) for r in rows_list})
    if attack_ids is None:
        attack_ids = sorted({str(_row_get(r, "attack_id")) for r in rows_list})

    table: dict[str, dict[str, dict[str, float]]] = {}
    for hid in hash_ids:
        table[hid] = {}
        for aid in attack_ids:
            subset = [
                r for r in rows_list
                if str(_row_get(r, "hash_id")) == hid and str(_row_get(r, "attack_id")) == aid
            ]
            table[hid][aid] = aggregate_results(subset)
    return table


def print_comparison_table(table: dict[str, dict[str, dict[str, float]]]) -> None:
    for hash_id, attacks in table.items():
        print(f"\nHash: {hash_id}")
        print("-" * 90)
        print(
            f"{'attack':<16} {'ASR':>7} {'mean_hash_l1':>14} "
            f"{'mean_pixel_l2':>14} {'queries':>10} {'score':>10}"
        )
        for attack_id, m in attacks.items():
            if not m:
                print(f"{attack_id:<16} no data")
                continue
            print(
                f"{attack_id:<16} "
                f"{m.get('ASR', 0.0):>7.3f} "
                f"{m.get('mean_final_hash_l1', 0.0):>14.2f} "
                f"{m.get('mean_pixel_l2', 0.0):>14.6f} "
                f"{m.get('mean_queries', 0.0):>10.1f} "
                f"{m.get('score_asr_per_l2', 0.0):>10.3f}"
            )
