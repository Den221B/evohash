"""Oracle: the only interface attacks can use to query a hash function.

Attacks receive a HashOracle instance and call:
  oracle.score(x)       → float  (hash distance to target; lower = better)
  oracle.is_success(x)  → bool   (distance ≤ threshold)
  oracle.project(x)     → ndarray (clip + optional L2-ball projection)
  oracle.budget_ok()    → bool   (queries / time budget not exhausted)
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
    max_queries: int = 200
    max_time_ms: int = 5_000
    seed: int = 0


@dataclass(frozen=True)
class ConstraintSpec:
    """Optional constraints on the perturbed image.

    max_l2: maximum pixel-space L2 distance between x_adv and x0.
            None means unconstrained.
    """
    max_l2: Optional[float] = None
    clip_lo: float = 0.0
    clip_hi: float = 1.0


# ---------------------------------------------------------------------------
# State (mutable, one per oracle instance)
# ---------------------------------------------------------------------------

@dataclass
class OracleState:
    queries_used: int = 0
    started_ms: int = 0
    best_score: float = float("inf")
    best_x: Optional[np.ndarray] = None
    stopped_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class HashOracle:
    """Black-box query interface for a single hash function + target digest.

    The attack receives x0 (source image) and queries oracle.score(x_candidate).
    It never sees the target image y directly — only hash(y) is stored here.
    """

    def __init__(
        self,
        hash_fn: HashFunction,
        target_digest: Any,
        x0: np.ndarray,
        budget: BudgetSpec,
        constraints: ConstraintSpec = ConstraintSpec(),
        img_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        from evohash.utils import l2_img  # avoid circular import at module level

        self.hash_fn = hash_fn
        self.target_digest = target_digest
        self.budget = budget
        self.constraints = constraints
        self._img_distance = img_distance_fn or l2_img

        self.x0 = _to_float32(x0)
        self.state = OracleState(
            started_ms=_now_ms(),
            best_x=self.x0.copy(),
        )
        # Score x0 once without counting against the budget
        initial_score = self._compute_score(self.x0)
        self.state.best_score = initial_score
        self.state.best_x = self.x0.copy()

    # ------------------------------------------------------------------
    # Public API for attacks
    # ------------------------------------------------------------------

    @property
    def threshold_p(self) -> float:
        return self.hash_fn.spec.threshold_p

    def budget_ok(self) -> bool:
        """Return True if there are still queries and time remaining."""
        if self.state.queries_used >= self.budget.max_queries:
            self.state.stopped_reason = self.state.stopped_reason or "max_queries"
            return False
        elapsed = _now_ms() - self.state.started_ms
        if elapsed >= self.budget.max_time_ms:
            self.state.stopped_reason = self.state.stopped_reason or "max_time"
            return False
        return True

    def score(self, x: np.ndarray) -> float:
        """Query hash distance of x to the target. Counts as one query."""
        if not self.budget_ok():
            return float(self.state.best_score)

        self.state.queries_used += 1
        x = self.project(x)
        s = self._compute_score(x)

        if s < self.state.best_score and self._satisfies_constraints(x):
            self.state.best_score = s
            self.state.best_x = x.copy()

        return s

    def is_success(self, x: np.ndarray) -> bool:
        """Return True if x produces a hash collision with the target."""
        return self.score(x) <= self.threshold_p

    def project(self, x: np.ndarray) -> np.ndarray:
        """Clip x to valid pixel range [clip_lo, clip_hi]."""
        return np.clip(_to_float32(x), self.constraints.clip_lo, self.constraints.clip_hi)

    def time_used_ms(self) -> int:
        return _now_ms() - self.state.started_ms

    def queries_remaining(self) -> int:
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


def _now_ms() -> int:
    return int(time.time() * 1000)
