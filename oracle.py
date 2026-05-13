"""Hash oracle: the only interface attacks use to query a target hash."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from evohash.metrics import compute_pixel_l2, to_float32


@dataclass(frozen=True)
class BudgetSpec:
    max_queries: Optional[int] = 10_000
    max_time_s: Optional[float] = None
    seed: int = 0


@dataclass(frozen=True)
class ConstraintSpec:
    max_pixel_l2: Optional[float] = None
    clip_lo: float = 0.0
    clip_hi: float = 1.0


@dataclass
class OracleState:
    queries: int = 0
    started_at: float = field(default_factory=time.monotonic)
    best_hash_l1: float = float("inf")
    best_x: Optional[np.ndarray] = None
    history: list[float] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)
    stopped_reason: Optional[str] = None


class HashOracle:
    """Black-box oracle for one (hash_fn, target_hash, source image) task."""

    def __init__(
        self,
        *,
        hash_fn: Any,
        target_hash: Any | None = None,
        target_digest: Any | None = None,
        x_source: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        budget: BudgetSpec | int | None = None,
        constraints: ConstraintSpec | None = None,
    ) -> None:
        self.hash_fn = hash_fn
        self.target_hash = target_hash if target_hash is not None else target_digest
        if self.target_hash is None:
            raise ValueError("HashOracle requires target_hash or target_digest")

        src = x_source if x_source is not None else x0
        if src is None:
            raise ValueError("HashOracle requires x_source or x0")
        self.x_source = to_float32(src)

        if budget is None:
            self.budget = BudgetSpec()
        elif isinstance(budget, int):
            self.budget = BudgetSpec(max_queries=int(budget))
        else:
            self.budget = budget

        self.constraints = constraints or ConstraintSpec()
        self.state = OracleState()

        # Initial score is not counted as an attack query.
        initial = self._compute_hash_l1(self.x_source)
        self.state.best_hash_l1 = float(initial)
        self.state.best_x = self.x_source.copy()
        self.state.best_history.append(float(initial))

    @property
    def hash_id(self) -> str:
        return str(getattr(getattr(self.hash_fn, "spec", None), "hash_id", "hash"))

    @property
    def threshold(self) -> float:
        return float(getattr(getattr(self.hash_fn, "spec", None), "threshold_p"))

    @property
    def threshold_p(self) -> float:
        return self.threshold

    @property
    def queries(self) -> int:
        return int(self.state.queries)

    @property
    def queries_used(self) -> int:
        return self.queries

    @property
    def elapsed_s(self) -> float:
        return float(time.monotonic() - self.state.started_at)

    @property
    def best_x(self) -> np.ndarray:
        assert self.state.best_x is not None
        return self.state.best_x

    @property
    def best_hash_l1(self) -> float:
        return float(self.state.best_hash_l1)

    def budget_ok(self) -> bool:
        if self.budget.max_time_s is not None and self.elapsed_s >= self.budget.max_time_s:
            self.state.stopped_reason = self.state.stopped_reason or "max_time_s"
            return False
        if self.budget.max_queries is not None and self.queries >= self.budget.max_queries:
            self.state.stopped_reason = self.state.stopped_reason or "max_queries"
            return False
        return True

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(to_float32(x), self.constraints.clip_lo, self.constraints.clip_hi)

    def query(self, x: np.ndarray) -> float:
        """Return hash_l1 to target_hash and count one oracle query."""
        if not self.budget_ok():
            return self.best_hash_l1

        x = self.project(x)
        self.state.queries += 1
        hash_l1 = self._compute_hash_l1(x)

        self.state.history.append(float(hash_l1))
        if hash_l1 < self.state.best_hash_l1 and self._satisfies_constraints(x):
            self.state.best_hash_l1 = float(hash_l1)
            self.state.best_x = x.copy()

        self.state.best_history.append(float(self.state.best_hash_l1))
        return float(hash_l1)

    def score(self, x: np.ndarray) -> float:
        return self.query(x)

    def is_success(self, x: np.ndarray) -> bool:
        return self.query(x) <= self.threshold

    def is_success_from_distance(self, hash_l1: float) -> bool:
        return float(hash_l1) <= self.threshold

    def _compute_hash_l1(self, x: np.ndarray) -> float:
        h = self.hash_fn.compute(x)
        return float(self.hash_fn.distance(h, self.target_hash))

    def _satisfies_constraints(self, x: np.ndarray) -> bool:
        if self.constraints.max_pixel_l2 is None:
            return True
        return compute_pixel_l2(self.x_source, x) <= float(self.constraints.max_pixel_l2)
