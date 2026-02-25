"""Evaluator: runs attacks against hash functions and records results.

Design principles:
  - Evaluator never trusts AttackResult directly. It re-computes the final
    hash distance from scratch on x_best.
  - All results are plain dataclasses (EvalRow) — easy to serialize to JSON/CSV.
  - run_eval_on_ds() is the main entry point for bulk evaluation.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from evohash.attacks.base import AttackRegistry
from evohash.dataset import Dataset, PairSample
from evohash.hashes.base import HashRegistry
from evohash.oracle import BudgetSpec, ConstraintSpec, HashOracle
from evohash.utils import l2_img, to_float32


@dataclass
class EvalRow:
    pair_id: str
    hash_id: str
    attack_id: str
    seed: int
    success: bool
    d_hash: float       # hash distance of x_best from target hash
    d_img: float        # pixel L2 distance between x_best and x0
    queries_used: int
    runtime_ms: int
    stopped_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Evaluator:
    def __init__(
        self,
        hash_registry: HashRegistry,
        attack_registry: AttackRegistry,
        img_distance_fn: Callable[[np.ndarray, np.ndarray], float] = l2_img,
    ) -> None:
        self.hash_registry = hash_registry
        self.attack_registry = attack_registry
        self._img_dist = img_distance_fn

    def evaluate(
        self,
        sample: PairSample,
        hash_id: str,
        attack_id: str,
        budget: BudgetSpec,
        constraints: ConstraintSpec = ConstraintSpec(),
    ) -> EvalRow:
        hash_fn = self.hash_registry.get(hash_id)
        attack = self.attack_registry.get(attack_id)

        x0 = to_float32(sample.x)
        y = to_float32(sample.y)
        target_digest = hash_fn.compute(y)

        oracle = HashOracle(
            hash_fn=hash_fn,
            target_digest=target_digest,
            x0=x0,
            budget=budget,
            constraints=constraints,
            img_distance_fn=self._img_dist,
        )

        res = attack.run(x0, oracle, budget)

        # Independent re-evaluation — never trust attack's self-reported score
        x_best = to_float32(res.x_best)
        d_hash = float(hash_fn.distance(hash_fn.compute(x_best), target_digest))
        d_img = float(self._img_dist(x_best, x0))

        constraints_ok = (
            constraints.max_l2 is None or d_img <= constraints.max_l2
        )
        success = (d_hash <= hash_fn.spec.threshold_p) and constraints_ok

        return EvalRow(
            pair_id=sample.pair_id,
            hash_id=hash_id,
            attack_id=attack_id,
            seed=budget.seed,
            success=bool(success),
            d_hash=d_hash,
            d_img=d_img,
            queries_used=res.queries_used,
            runtime_ms=res.runtime_ms,
            stopped_reason=res.stopped_reason,
            extra={"constraints_ok": constraints_ok, **(res.extra or {})},
        )


def run_eval_on_ds(
    dataset: Dataset,
    evaluator: Evaluator,
    *,
    hash_ids: Optional[List[str]] = None,
    attack_ids: Optional[List[str]] = None,
    seeds: List[int] = [0],
    max_queries: int = 200,
    max_time_ms: int = 5_000,
    max_l2: Optional[float] = None,
    verbose: bool = True,
) -> List[EvalRow]:
    """Run all (hash, attack, seed) combinations over the dataset.

    Args:
        dataset:     Dataset to iterate over.
        evaluator:   Evaluator instance with registered hashes and attacks.
        hash_ids:    Hash IDs to evaluate (default: all registered).
        attack_ids:  Attack IDs to evaluate (default: all registered).
        seeds:       RNG seeds for repeated trials.
        max_queries: Query budget per trial.
        max_time_ms: Time budget per trial (ms).
        max_l2:      Optional pixel L2 constraint.
        verbose:     Print progress.

    Returns:
        List of EvalRow, one per (sample, hash, attack, seed) combination.
    """
    if hash_ids is None:
        hash_ids = evaluator.hash_registry.list_ids()
    if attack_ids is None:
        attack_ids = evaluator.attack_registry.list_ids()

    constraints = ConstraintSpec(max_l2=max_l2)
    rows: List[EvalRow] = []
    total = 0

    for sample in dataset:
        for hid in hash_ids:
            for aid in attack_ids:
                for seed in seeds:
                    budget = BudgetSpec(
                        max_queries=max_queries,
                        max_time_ms=max_time_ms,
                        seed=seed,
                    )
                    row = evaluator.evaluate(
                        sample=sample,
                        hash_id=hid,
                        attack_id=aid,
                        budget=budget,
                        constraints=constraints,
                    )
                    rows.append(row)
                    total += 1
                    if verbose and total % 10 == 0:
                        print(
                            f"  [{total}] {sample.pair_id} | {hid} | {aid} "
                            f"| success={row.success} "
                            f"d_hash={row.d_hash:.1f} "
                            f"d_img={row.d_img:.4f} "
                            f"q={row.queries_used}"
                        )

    return rows
