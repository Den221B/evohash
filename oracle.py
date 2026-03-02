"""Oracle: the only interface attacks can use to query a hash function.

Attacks receive a HashOracle instance and call:
  oracle.query(x)       → float  (hash distance to target; lower = better)
  oracle.score(x)       → float  (alias for query)
  oracle.is_success(x)  → bool   (distance ≤ threshold)
  oracle.project(x)     → ndarray (clip + optional L2-ball projection)
  oracle.budget_ok()    → bool   (time budget not exhausted)

Properties:
  oracle.threshold      → float  (alias for threshold_p)
  oracle.threshold_p    → float  (hash distance threshold)
  oracle.queries_used   → int    (total queries so far)
  oracle.elapsed_s      → float  (seconds elapsed)

BudgetSpec:
  max_time_s  — hard wall-clock limit (primary). None = unlimited.
  max_queries — optional secondary cap. None = unlimited.
  seed        — RNG seed.

Design: time is the primary budget. Attacks stop when time runs out
OR on success. Query count is logged but not enforced by default.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from evohash.hashes.base import HashFunction


# ---------------------------------------------------------------------------
# Budget & Constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetSpec:
    max_time_s: Optional[float] = 300.0    # 5 min default; None = unlimited
    max_queries: Optional[int] = None       # None = no query cap
    seed: int = 0


@dataclass(frozen=True)
class ConstraintSpec:
    """Optional constraints on the perturbed image."""
    max_l2: Optional[float] = None
    clip_lo: float = 0.0
    clip_hi: float = 1.0


# ---------------------------------------------------------------------------
# State (mutable, one per oracle instance)
# ---------------------------------------------------------------------------

@dataclass
class OracleState:
    queries_used: int = 0
    started_at: float = 0.0        # time.monotonic()
    best_score: float = float("inf")
    best_x: Optional[np.ndarray] = None
    stopped_reason: Optional[str] = None
    # For logging: snapshot of best_score over time
    history_time: list = field(default_factory=list)   # elapsed_s at each query
    history_score: list = field(default_factory=list)  # best_score at each query


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class HashOracle:
    """Black-box query interface for a single hash function + target digest."""

    def __init__(
        self,
        hash_fn: HashFunction,
        target_digest: Any,
        x0: np.ndarray,
        budget: BudgetSpec,
        constraints: ConstraintSpec = ConstraintSpec(),
        img_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        from evohash.utils import l2_img

        self.hash_fn = hash_fn
        self.target_digest = target_digest
        self.budget = budget
        self.constraints = constraints
        self._img_distance = img_distance_fn or l2_img

        self.x0 = _to_float32(x0)
        self.state = OracleState(started_at=time.monotonic())

        # Score x0 once — does NOT count as a query
        initial_score = self._compute_score(self.x0)
        self.state.best_score = initial_score
        self.state.best_x = self.x0.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def threshold_p(self) -> float:
        return self.hash_fn.spec.threshold_p

    @property
    def threshold(self) -> float:
        return self.hash_fn.spec.threshold_p

    @property
    def queries_used(self) -> int:
        return self.state.queries_used

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.state.started_at

    def budget_ok(self) -> bool:
        """True if time and query budget are not exhausted."""
        # Time check (primary)
        if self.budget.max_time_s is not None:
            if self.elapsed_s >= self.budget.max_time_s:
                self.state.stopped_reason = self.state.stopped_reason or "max_time"
                return False
        # Query check (secondary, optional)
        if self.budget.max_queries is not None:
            if self.state.queries_used >= self.budget.max_queries:
                self.state.stopped_reason = self.state.stopped_reason or "max_queries"
                return False
        return True

    def query(self, x: np.ndarray) -> float:
        """Query hash distance. Counts as one query. Returns best_score if budget exhausted."""
        if not self.budget_ok():
            return float(self.state.best_score)

        self.state.queries_used += 1
        x = self.project(x)
        s = self._compute_score(x)

        # Log for later analysis
        self.state.history_time.append(self.elapsed_s)
        self.state.history_score.append(s)

        if s < self.state.best_score and self._satisfies_constraints(x):
            self.state.best_score = s
            self.state.best_x = x.copy()

        return s

    def score(self, x: np.ndarray) -> float:
        return self.query(x)

    def is_success(self, x: np.ndarray) -> bool:
        return self.query(x) <= self.threshold_p

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(_to_float32(x), self.constraints.clip_lo, self.constraints.clip_hi)

    def time_used_s(self) -> float:
        return self.elapsed_s

    def queries_remaining(self) -> Optional[int]:
        if self.budget.max_queries is None:
            return None
        return max(0, self.budget.max_queries - self.state.queries_used)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, x: np.ndarray) -> float:
        digest = self.hash_fn.compute(x)
        return float(self.hash_fn.distance(digest, self.target_digest))

    def _satisfies_constraints(self, x: np.ndarray) -> bool:
        if self.constraints.max_l2 is None:
            return True
        return self._img_distance(x, self.x0) <= self.constraints.max_l2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_float32(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return image.astype(np.float32).clip(0.0, 1.0)