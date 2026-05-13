"""Evaluator: final source of truth for attack metrics."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from evohash.attacks.base import AttackRawResult, RegisteredAttack
from evohash.metrics import (
    compute_pixel_l2,
    compute_pixel_l2_raw,
    compute_relative_improvement,
    to_float32,
)
from evohash.oracle import BudgetSpec, ConstraintSpec, HashOracle
from evohash.preprocessing import ResizeSpec, apply_resize


@dataclass
class EvalRow:
    pair_id: str | None
    hash_id: str
    attack_id: str
    seed: int

    initial_hash_l1: float
    final_hash_l1: float
    best_hash_l1: float
    threshold: float
    success: bool
    improved: bool
    relative_improvement: float

    pixel_l2: float
    pixel_l2_raw: float
    queries: int
    budget: int | None
    time_sec: float

    history: list[float] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    x_best: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("x_best", None)
        return row


def _get_hash_id(hash_fn: Any) -> str:
    return str(getattr(getattr(hash_fn, "spec", None), "hash_id", "hash"))


def _get_threshold(hash_fn: Any) -> float:
    return float(getattr(getattr(hash_fn, "spec", None), "threshold_p"))


def _resolve_attack(attack: Any) -> tuple[str, Any, dict[str, Any]]:
    """Return (attack_id, run_attack_fn, default_params)."""
    if isinstance(attack, RegisteredAttack):
        return attack.attack_id, attack.run_attack, dict(attack.default_params)

    if hasattr(attack, "run_attack"):
        attack_id = getattr(attack, "ATTACK_ID", None) or getattr(attack, "attack_id", None)
        return str(attack_id or getattr(attack, "__name__", "attack").split(".")[-1]), attack.run_attack, dict(getattr(attack, "DEFAULT_PARAMS", {}))

    if callable(attack):
        return getattr(attack, "__name__", "attack"), attack, {}

    raise TypeError("attack must be a RegisteredAttack, module with run_attack, or callable")


def evaluate_attack(
    *,
    attack: Any,
    hash_fn: Any,
    x_source: np.ndarray,
    x_target: np.ndarray,
    params: Optional[dict[str, Any]] = None,
    budget: int | BudgetSpec | None = 10_000,
    seed: int = 0,
    pair_id: str | None = None,
    attack_id: str | None = None,
    constraints: ConstraintSpec | None = None,
    resize_size: int | None = None,
    resize_resample: str = "bilinear",
) -> EvalRow:
    """Run one attack on one pair and recompute all final metrics."""
    x0_original = to_float32(x_source)
    y_original = to_float32(x_target)

    resolved_attack_id, run_attack, default_params = _resolve_attack(attack)
    resolved_attack_id = attack_id or resolved_attack_id
    full_params = {**default_params, **(params or {})}
    if resize_size is None and "resize_size" in full_params:
        resize_size = int(full_params.pop("resize_size"))

    resize_spec = ResizeSpec(size=resize_size, resample=resize_resample)
    x0 = apply_resize(x0_original, resize_spec)
    y = apply_resize(y_original, resize_spec)

    if budget is None:
        budget_spec = BudgetSpec(max_queries=None, seed=seed)
    elif isinstance(budget, int):
        budget_spec = BudgetSpec(max_queries=int(budget), seed=seed)
    else:
        budget_spec = budget
        if getattr(budget_spec, "seed", seed) != seed:
            budget_spec = BudgetSpec(
                max_queries=budget_spec.max_queries,
                max_time_s=budget_spec.max_time_s,
                seed=seed,
            )

    target_hash = hash_fn.compute(y)
    source_hash = hash_fn.compute(x0)
    initial_hash_l1 = float(hash_fn.distance(source_hash, target_hash))
    threshold = _get_threshold(hash_fn)

    oracle = HashOracle(
        hash_fn=hash_fn,
        target_hash=target_hash,
        x_source=x0,
        budget=budget_spec,
        constraints=constraints,
    )

    t0 = time.perf_counter()
    raw: AttackRawResult = run_attack(
        x_source=x0,
        oracle=oracle,
        params=full_params,
        budget=budget_spec.max_queries,
        seed=seed,
    )
    time_sec = float(time.perf_counter() - t0)

    # Evaluator never trusts attack metrics. Recompute from x_best.
    x_best = to_float32(raw.x_best)
    final_hash_l1 = float(hash_fn.distance(hash_fn.compute(x_best), target_hash))
    best_hash_l1 = float(min([initial_hash_l1, final_hash_l1, oracle.best_hash_l1]))

    pixel_l2 = compute_pixel_l2(x0, x_best)
    pixel_l2_raw = compute_pixel_l2_raw(x0, x_best)
    success = bool(final_hash_l1 <= threshold)
    improved = bool(final_hash_l1 < initial_hash_l1)
    relative_improvement = compute_relative_improvement(
        initial_hash_l1=initial_hash_l1,
        final_hash_l1=final_hash_l1,
        threshold=threshold,
    )

    history = list(raw.history or oracle.state.best_history)

    return EvalRow(
        pair_id=pair_id,
        hash_id=_get_hash_id(hash_fn),
        attack_id=resolved_attack_id,
        seed=int(seed),
        initial_hash_l1=float(initial_hash_l1),
        final_hash_l1=float(final_hash_l1),
        best_hash_l1=float(best_hash_l1),
        threshold=float(threshold),
        success=success,
        improved=improved,
        relative_improvement=float(relative_improvement),
        pixel_l2=float(pixel_l2),
        pixel_l2_raw=float(pixel_l2_raw),
        queries=int(oracle.queries),
        budget=budget_spec.max_queries,
        time_sec=time_sec,
        history=history,
        params=dict(raw.params or full_params),
        extra={
            "original_shape": tuple(int(v) for v in x0_original.shape),
            "eval_shape": tuple(int(v) for v in x0.shape),
            "resize_size": resize_size,
            "resize_resample": resize_resample if resize_size is not None else None,
            "oracle_stopped_reason": oracle.state.stopped_reason,
            **(raw.extra or {}),
        },
        x_best=x_best,
    )


def evaluate_dataset(
    *,
    dataset: Iterable[Any],
    hash_fn: Any,
    attack: Any,
    params: Optional[dict[str, Any]] = None,
    budget: int = 10_000,
    seed: int = 0,
    max_pairs: Optional[int] = None,
    resize_size: int | None = None,
    resize_resample: str = "bilinear",
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for i, sample in enumerate(dataset):
        if max_pairs is not None and i >= max_pairs:
            break
        rows.append(
            evaluate_attack(
                attack=attack,
                hash_fn=hash_fn,
                x_source=sample.x,
                x_target=sample.y,
                params=params,
                budget=budget,
                seed=seed,
                pair_id=getattr(sample, "pair_id", str(i)),
                resize_size=resize_size,
                resize_resample=resize_resample,
            )
        )
    return rows


# Backwards-compatible light wrapper for older notebooks.
def run_eval_on_ds(
    dataset: Iterable[Any],
    evaluator: Any | None = None,
    *,
    hash_fn: Any | None = None,
    attack: Any | None = None,
    params: Optional[dict[str, Any]] = None,
    budget: int = 10_000,
    seed: int = 0,
    max_pairs: Optional[int] = None,
    resize_size: int | None = None,
    resize_resample: str = "bilinear",
    **_: Any,
) -> list[EvalRow]:
    if hash_fn is None or attack is None:
        raise ValueError("run_eval_on_ds now expects hash_fn=... and attack=...")
    return evaluate_dataset(
        dataset=dataset,
        hash_fn=hash_fn,
        attack=attack,
        params=params,
        budget=budget,
        seed=seed,
        max_pairs=max_pairs,
        resize_size=resize_size,
        resize_resample=resize_resample,
    )


class Evaluator:
    """Small compatibility facade around evaluate_attack().

    New code can call evaluate_attack(...) directly. This class exists so old
    imports from evohash.__init__ do not break.
    """

    def __init__(self, hash_registry: Any | None = None, attack_registry: Any | None = None) -> None:
        self.hash_registry = hash_registry
        self.attack_registry = attack_registry

    def evaluate(
        self,
        sample: Any,
        hash_id: str,
        attack_id: str = "nes",
        budget: int | BudgetSpec | None = 10_000,
        constraints: ConstraintSpec | None = None,
        params: Optional[dict[str, Any]] = None,
        seed: int = 0,
        resize_size: int | None = None,
        resize_resample: str = "bilinear",
    ) -> EvalRow:
        if self.hash_registry is None or self.attack_registry is None:
            raise ValueError("Evaluator requires hash_registry and attack_registry")
        hash_fn = self.hash_registry.get(hash_id)
        attack = self.attack_registry.get(attack_id)
        return evaluate_attack(
            attack=attack,
            hash_fn=hash_fn,
            x_source=sample.x,
            x_target=sample.y,
            params=params,
            budget=budget,
            seed=seed,
            pair_id=getattr(sample, "pair_id", None),
            constraints=constraints,
            resize_size=resize_size,
            resize_resample=resize_resample,
        )
