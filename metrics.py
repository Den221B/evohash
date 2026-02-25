"""Metrics and comparison utilities.

Primary metric: ASR / (mean_L2 + ε)
  ASR   = Attack Success Rate (fraction of pairs where attack succeeded)
  L2    = mean pixel-space RMS L2 distance between x_best and x0
  ε     = 1e-6 (numerical stability; avoids division by zero when L2 → 0)

This is the metric from the EvoHash project spec, with a small stabilisation
term so that attacks that make zero perturbation don't produce NaN.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from evohash.evaluator import EvalRow

_EPS = 1e-6


def asr_over_l2(rows: List[EvalRow]) -> float:
    """Compute ASR / (mean_L2 + ε) over a list of EvalRows."""
    if not rows:
        return 0.0
    asr = float(np.mean([r.success for r in rows]))
    mean_l2 = float(np.mean([r.d_img for r in rows]))
    return asr / (mean_l2 + _EPS)


def aggregate(rows: List[EvalRow]) -> Dict[str, float]:
    """Return a dict of aggregate metrics for a homogeneous list of rows."""
    if not rows:
        return {}
    successes = [r.success for r in rows]
    asr = float(np.mean(successes))
    mean_l2 = float(np.mean([r.d_img for r in rows]))
    mean_dhash = float(np.mean([r.d_hash for r in rows]))
    mean_queries = float(np.mean([r.queries_used for r in rows]))
    mean_time_s = float(np.mean([r.runtime_ms for r in rows])) / 1000.0
    fitness = asr / (mean_l2 + _EPS)

    return {
        "n": len(rows),
        "asr": round(asr, 4),
        "mean_l2": round(mean_l2, 6),
        "mean_d_hash": round(mean_dhash, 2),
        "mean_queries": round(mean_queries, 1),
        "mean_time_s": round(mean_time_s, 2),
        "fitness_asr_l2": round(fitness, 4),
    }


def build_comparison_table(
    rows: List[EvalRow],
    hash_ids: Optional[List[str]] = None,
    attack_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build a nested comparison table: table[hash_id][attack_id] = metrics.

    Returns a dict ready to be converted to a pandas DataFrame or printed.
    """
    if hash_ids is None:
        hash_ids = sorted({r.hash_id for r in rows})
    if attack_ids is None:
        attack_ids = sorted({r.attack_id for r in rows})

    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for hid in hash_ids:
        table[hid] = {}
        for aid in attack_ids:
            subset = [r for r in rows if r.hash_id == hid and r.attack_id == aid]
            table[hid][aid] = aggregate(subset)

    return table


def print_comparison_table(table: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Pretty-print the comparison table to stdout."""
    for hash_id, attacks in table.items():
        print(f"\n{'='*60}")
        print(f"Hash: {hash_id}")
        print(f"{'='*60}")
        header = f"{'Attack':<22} {'ASR':>6} {'L2':>8} {'d_hash':>8} {'Queries':>8} {'Time(s)':>8} {'Fitness':>10}"
        print(header)
        print("-" * 75)
        for attack_id, m in sorted(attacks.items(), key=lambda x: -x[1].get("fitness_asr_l2", 0)):
            if not m:
                print(f"  {attack_id:<20}  (no data)")
                continue
            print(
                f"  {attack_id:<20} "
                f"{m['asr']:>6.3f} "
                f"{m['mean_l2']:>8.5f} "
                f"{m['mean_d_hash']:>8.1f} "
                f"{m['mean_queries']:>8.0f} "
                f"{m['mean_time_s']:>8.2f} "
                f"{m['fitness_asr_l2']:>10.2f}"
            )
