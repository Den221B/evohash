# attacks/hsja.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import AttackSpec, AttackResult


def _get_seed(budget) -> int:
    s = getattr(budget, "seed", None)
    if s is None:
        return 0
    try:
        return int(s) & 0xFFFFFFFF
    except Exception:
        return 0


def _as_float01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a.astype(np.float32) - b.astype(np.float32)).ravel()
    return float(np.sqrt(np.mean(d * d)))


def _l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(a.ravel(), ord=2))


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = _l2(v)
    if n < eps:
        return np.zeros_like(v)
    return v / (n + eps)


def _get_threshold(oracle):
    return getattr(oracle, "threshold_p", getattr(oracle, "threshold", None))


@dataclass
class HSJAAttack:
    """
    HSJA-like decision-based boundary refinement adapted to a "similarity set":
      S = { x : oracle.query(x) <= thr }

    Goal: find x in S that is as close as possible to x0 (measured by RMSE),
    while respecting oracle.project constraints and budgets.

    High-level steps:
      1) Find an initial point x_in in S (collision wrt x0).
      2) Binary search between x0 (outside) and x_in (inside) to get boundary point x_b in S.
      3) Estimate boundary normal/gradient direction via random probes around x_b.
      4) Take a step from x_b towards x0 along the estimated direction, then project back to boundary.
      5) Repeat until budget ends or success reached and no further RMSE improvements.

    Notes:
      - Deterministic RNG from budget.seed
      - Logs elapsed_ms each iteration and runtime_ms total
    """

    # Initialization
    init_max_tries: int = 200
    init_sigmas: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)  # progressively larger noise

    # Boundary search
    bin_search_steps: int = 12

    # Gradient estimation
    grad_queries: int = 40
    probe_delta_ratio: float = 0.01  # delta = probe_delta_ratio * current_dist_to_x0

    # Update
    step_ratio: float = 0.15  # step = step_ratio * dist_to_x0 (HSJA uses adaptive; this is a stable baseline)
    grayscale: bool = False

    spec: AttackSpec = AttackSpec(
        attack_id="hsja",
        name="HSJA",
        description="HSJA-like decision-based boundary refinement for hash-collision set; deterministic seed; logs time.",
    )

    def _decision(self, oracle, x: np.ndarray, thr: float) -> Tuple[bool, float]:
        """Returns (inside_set, dist). Consumes one oracle query."""
        d = float(oracle.query(x))
        return (d <= thr), d

    def _sample_noise(self, rng: np.random.Generator, shape) -> np.ndarray:
        u = rng.standard_normal(size=shape, dtype=np.float32)
        if self.grayscale and u.ndim == 3 and u.shape[-1] == 3:
            g = u.mean(axis=2, keepdims=True)
            u = np.repeat(g, 3, axis=2)
        return u

    def _find_initial_in_set(self, x0: np.ndarray, oracle, thr: float, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, float]]:
        """
        Find any x_in such that oracle.query(x_in) <= thr.
        Strategy: add Gaussian noise with increasing sigma; fallback to random uniform samples.
        """
        shape = x0.shape

        # Try noisy versions around x0 with increasing sigma
        for sigma in self.init_sigmas:
            for _ in range(self.init_max_tries // max(1, len(self.init_sigmas))):
                if not oracle.budget_ok():
                    return None
                u = self._sample_noise(rng, shape)
                x_try = oracle.project(x0 + sigma * u)
                inside, d = self._decision(oracle, x_try, thr)
                if inside:
                    return x_try, d

        # Fallback: random images in [0,1]
        for _ in range(max(20, self.init_max_tries // 4)):
            if not oracle.budget_ok():
                return None
            x_try = oracle.project(rng.random(size=shape, dtype=np.float32))
            inside, d = self._decision(oracle, x_try, thr)
            if inside:
                return x_try, d

        return None

    def _binary_search_to_boundary(self, x_out: np.ndarray, x_in: np.ndarray, oracle, thr: float) -> Tuple[np.ndarray, float]:
        """
        Assumes x_in is inside, x_out is outside (or at least not guaranteed inside).
        Returns x_b inside and close to boundary.
        """
        lo = x_out
        hi = x_in
        d_hi = float("inf")

        for _ in range(self.bin_search_steps):
            if not oracle.budget_ok():
                break
            mid = oracle.project((lo + hi) / 2.0)
            inside, d_mid = self._decision(oracle, mid, thr)
            if inside:
                hi = mid
                d_hi = d_mid
            else:
                lo = mid

        # Ensure returned point is inside (best effort)
        return hi, d_hi

    def _estimate_boundary_direction(self, x_b: np.ndarray, x0: np.ndarray, oracle, thr: float, rng: np.random.Generator) -> Optional[np.ndarray]:
        """
        Estimate a direction pointing (approximately) toward the inside region boundary normal.
        Uses random probes around x_b:
          sign = +1 if probe is inside, -1 otherwise
          grad ~ mean(sign * u)
        Then orient the direction to move toward x0.
        """
        if self.grad_queries <= 0:
            return None

        dist = _rmse(x_b, x0)
        delta = max(1e-4, float(self.probe_delta_ratio) * max(dist, 1e-6))

        g = np.zeros_like(x_b, dtype=np.float32)
        for _ in range(self.grad_queries):
            if not oracle.budget_ok():
                return None
            u = self._sample_noise(rng, x_b.shape)
            u = _normalize(u)
            x_probe = oracle.project(x_b + delta * u)
            inside, _ = self._decision(oracle, x_probe, thr)
            s = 1.0 if inside else -1.0
            g += (s * u)

        g = g / float(self.grad_queries)
        g = _normalize(g)

        # Orient g so that stepping along +g tends to move toward x0 (reduce distance to x0)
        # We want dot(g, x0 - x_b) > 0 ideally.
        if float(np.dot(g.ravel(), (x0 - x_b).ravel())) < 0.0:
            g = -g

        return g

    def run(self, x0: np.ndarray, oracle, budget) -> AttackResult:
        x0 = _as_float01(x0)
        seed = _get_seed(budget)
        rng = np.random.default_rng(seed)

        thr = _get_threshold(oracle)
        if thr is None:
            # No threshold => can't define decision set
            return AttackResult(
                success=False,
                best_dist=float(getattr(oracle.state, "best_dist", np.inf)),
                queries_used=int(getattr(oracle, "queries_used", 0)),
                best_x=None,
                history={},
                extra={"seed": seed, "error": "oracle has no threshold_p/threshold"},
            )
        thr = float(thr)

        t0 = perf_counter()

        history: Dict[str, Any] = {
            "iter": [],
            "elapsed_ms": [],
            "rmse": [],
            "best_rmse": [],
            "dist": [],       # oracle distance at current inside/boundary point
            "best_dist": [],  # oracle best (may track something else globally)
        }

        # 1) Find initial inside point
        init = self._find_initial_in_set(x0, oracle, thr, rng)
        if init is None:
            total_ms = (perf_counter() - t0) * 1000.0
            return AttackResult(
                success=False,
                best_dist=float(getattr(oracle.state, "best_dist", np.inf)),
                queries_used=int(getattr(oracle, "queries_used", 0)),
                best_x=None,
                history=history,
                extra={"seed": seed, "runtime_ms": float(total_ms), "stopped_reason": "no_initial_in_set"},
            )

        x_in, d_in = init

        # Track best by RMSE among inside points
        best_x = x_in
        best_rmse = _rmse(best_x, x0)
        best_inside_dist = float(d_in)

        # 2) Boundary point
        x_b, d_b = self._binary_search_to_boundary(x0, x_in, oracle, thr)
        if np.isfinite(d_b):
            x_in = x_b
            d_in = d_b

        it = 0
        while oracle.budget_ok():
            elapsed_ms = (perf_counter() - t0) * 1000.0

            # Update best
            cur_rmse = _rmse(x_in, x0)
            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                best_x = x_in
                best_inside_dist = float(d_in)

            history["iter"].append(it)
            history["elapsed_ms"].append(elapsed_ms)
            history["rmse"].append(cur_rmse)
            history["best_rmse"].append(best_rmse)
            history["dist"].append(float(d_in))
            history["best_dist"].append(float(getattr(oracle.state, "best_dist", d_in)))

            # Early stop on success (already inside set) + no budget => handled by loop
            # For hash attacks: "success" is being inside set; HSJA always maintains inside point once initialized.
            # We still stop if we cannot estimate grad.
            g = self._estimate_boundary_direction(x_in, x0, oracle, thr, rng)
            if g is None:
                break

            # Step length proportional to current distance to x0 (stable baseline)
            step = float(self.step_ratio) * max(cur_rmse, 1e-6)
            x_try = oracle.project(x_in + step * g)

            # If x_try is outside, project back to boundary between (x_in inside) and (x_try outside)
            inside, d_try = self._decision(oracle, x_try, thr)
            if not oracle.budget_ok():
                break

            if inside:
                # Now refine boundary between x0 (outside) and x_try (inside)
                x_in = x_try
                d_in = float(d_try)
                x_in, d_in = self._binary_search_to_boundary(x0, x_in, oracle, thr)
            else:
                # Bring it back to boundary between current inside and outside candidate
                x_in, d_in = self._binary_search_to_boundary(x_try, x_in, oracle, thr)

            it += 1

        total_ms = (perf_counter() - t0) * 1000.0

        # Determine success: best_x is inside set by construction (if we had init)
        success = True if best_x is not None else False

        return AttackResult(
            success=bool(success),
            best_dist=float(best_inside_dist),
            queries_used=int(getattr(oracle, "queries_used", 0)),
            best_x=best_x,
            history=history,
            extra={
                "seed": seed,
                "threshold": thr,
                "init_max_tries": int(self.init_max_tries),
                "init_sigmas": list(self.init_sigmas),
                "bin_search_steps": int(self.bin_search_steps),
                "grad_queries": int(self.grad_queries),
                "probe_delta_ratio": float(self.probe_delta_ratio),
                "step_ratio": float(self.step_ratio),
                "grayscale": bool(self.grayscale),
                "best_rmse": float(best_rmse),
                "runtime_ms": float(total_ms),
            },
        )