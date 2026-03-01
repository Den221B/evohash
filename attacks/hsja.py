"""HopSkipJumpAttack (HSJA) — Decision-based boundary attack.

Uses only binary feedback: collision(x') = dist(hash(x'), hash(target)) < threshold.
Iteratively moves toward the source image while staying on the non-collision side
of the decision boundary, then estimates boundary gradient to step closer.

Algorithm:
    1. Start from a random image that is NOT a collision (or a provided init).
    2. Binary search along the line between current point and target to find boundary.
    3. Estimate gradient at boundary via Monte Carlo with binary decisions.
    4. Move from boundary in gradient direction toward target.
    5. Repeat.

Reference:
    Chen et al., "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
    IEEE S&P 2020.  https://arxiv.org/abs/1904.02144
    
    Adapted for perceptual hash collision (rather than classifier evasion):
    - "target class" → collision with hash(y)
    - decision boundary → threshold on hash distance
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from .base import AttackResult, AttackSpec


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


@dataclass
class HSJAAttack:
    """HopSkipJumpAttack adapted for perceptual hash collision.

    Parameters
    ----------
    max_iters : int
        Number of outer iterations (boundary search + gradient step).
    init_queries : int
        Queries for the initial binary search to find the decision boundary.
    grad_queries : int
        Monte Carlo samples for gradient estimation at each step.
    step_size : float
        Gradient step magnitude from boundary.
    binary_search_steps : int
        Steps for boundary binary search per iteration.
    threshold : float | None
        Hash distance threshold for collision.  If None, read from oracle.
    """
    max_iters: int = 40
    init_queries: int = 100
    grad_queries: int = 100
    step_size: float = 0.02
    binary_search_steps: int = 25
    threshold: Optional[float] = None

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="hsja",
            params=dict(
                max_iters=self.max_iters,
                init_queries=self.init_queries,
                grad_queries=self.grad_queries,
                step_size=self.step_size,
                binary_search_steps=self.binary_search_steps,
            ),
        )

    # ------------------------------------------------------------------

    def run(
        self,
        x0: np.ndarray,
        oracle: HashOracle,
        budget: BudgetSpec,
    ) -> AttackResult:
        t0 = time.monotonic()
        history: List[float] = []

        thr = self.threshold if self.threshold is not None else getattr(oracle, "threshold", 0.0)

        def is_collision(x: np.ndarray) -> bool:
            """Binary decision — True if hash(x) collides with target."""
            try:
                d = oracle.query(x)
                history.append(d)
                return d <= thr
            except Exception:
                return False

        # --- Step 0: find a collision starting point (random images) ---
        x_adv = self._find_initial_collision(x0, oracle, is_collision, thr, history)
        if x_adv is None:
            # could not find a collision start — return best random
            elapsed = int((time.monotonic() - t0) * 1000)
            dist = oracle.query(x0)
            return AttackResult(
                x_best=x0, best_score=dist, queries_used=oracle.queries_used,
                runtime_ms=elapsed, stopped_reason="no_init", history=history,
            )

        best_x = x_adv.copy()
        best_dist = oracle.query(x_adv)
        history.append(best_dist)

        # --- Main loop ---
        for iteration in range(self.max_iters):
            if oracle.queries_used >= getattr(budget, "max_queries", int(1e9)):
                break

            # Binary search to find boundary between x0 (no collision) and x_adv (collision)
            x_boundary = self._binary_search_boundary(x0, x_adv, is_collision)

            # Estimate gradient at boundary (sign of gradient toward collision side)
            grad = self._estimate_grad(x_boundary, x0, is_collision)

            # Adaptive step size: distance from x0 to boundary
            dist_to_boundary = _l2(x0, x_boundary)
            step = self.step_size * dist_to_boundary

            # Move from boundary in gradient direction (toward collision region)
            x_new = _clip(x_boundary + step * grad)

            if is_collision(x_new):
                x_adv = x_new
                d = oracle.query(x_new)
                history.append(d)
                if d < best_dist:
                    best_dist = d
                    best_x = x_new.copy()

        reason = "success" if best_dist <= thr else "budget"
        elapsed = int((time.monotonic() - t0) * 1000)
        return AttackResult(
            x_best=best_x,
            best_score=best_dist,
            queries_used=oracle.queries_used,
            runtime_ms=elapsed,
            stopped_reason=reason,
            history=history,
            extra={"l2": _l2(x0, best_x)},
        )

    # ------------------------------------------------------------------

    def _find_initial_collision(self, x0, oracle, is_collision, thr, history):
        """Try random images until we find one that collides with target."""
        for _ in range(self.init_queries):
            # interpolate towards fully random image
            alpha = np.random.uniform(0.5, 1.0)
            x_rand = _clip(alpha * np.random.rand(*x0.shape).astype(np.float32)
                           + (1 - alpha) * x0)
            if is_collision(x_rand):
                return x_rand
        return None

    def _binary_search_boundary(
        self,
        x_no: np.ndarray,   # not a collision
        x_yes: np.ndarray,  # is a collision
        is_collision,
    ) -> np.ndarray:
        """Binary search on the line segment [x_no, x_yes] for the boundary."""
        lo, hi = 0.0, 1.0
        for _ in range(self.binary_search_steps):
            mid = (lo + hi) / 2
            x_mid = _clip((1 - mid) * x_no + mid * x_yes)
            if is_collision(x_mid):
                hi = mid
            else:
                lo = mid
        return _clip((1 - hi) * x_no + hi * x_yes)

    def _estimate_grad(
        self,
        x_boundary: np.ndarray,
        x_source: np.ndarray,
        is_collision,
    ) -> np.ndarray:
        """Monte Carlo gradient estimation at boundary.

        Each sample p gets coefficient +1 if perturbing toward p is a collision,
        else -1.  Average gives the gradient direction.
        """
        grad = np.zeros_like(x_boundary)
        delta = _l2(x_source, x_boundary) * 0.01
        if delta < 1e-8:
            delta = 0.01

        for _ in range(self.grad_queries):
            p = np.random.randn(*x_boundary.shape).astype(np.float32)
            p /= np.linalg.norm(p) + 1e-12
            x_perturb = _clip(x_boundary + delta * p)
            coeff = 1.0 if is_collision(x_perturb) else -1.0
            grad += coeff * p

        grad /= (self.grad_queries + 1e-12)
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad /= norm
        return grad
