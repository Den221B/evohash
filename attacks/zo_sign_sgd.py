# attacks/zo_sign_sgd.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

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


@dataclass
class ZOSignSGDAttack:
    """
    ZO-signSGD (ICLR'19) adapted to minimize oracle distance (e.g., hash distance).

    Update:
      delta <- delta - lr * sign(g_hat)

    Gradient estimator (random directions u ~ N(0, I)):
      forward:  g_hat = (1/q) * sum_j ((f(x+mu*u_j) - f(x)) / mu) * u_j
      central:  g_hat = (1/q) * sum_j ((f(x+mu*u_j) - f(x-mu*u_j)) / (2*mu)) * u_j

    Notes for your project:
      - Use oracle.project() for constraints (instead of local clip)
      - Deterministic RNG with budget.seed
      - Log elapsed_ms each step for time-based analysis
    """

    mu: float = 0.1
    lr: float = 0.01
    n_samples: int = 20

    estimator: str = "forward"   # "forward" or "central"
    momentum: float = 0.0        # 0 => no momentum; else m <- beta*m + (1-beta)*sign(g)
    grayscale: bool = False      # if True, enforce same perturbation per RGB channel

    spec: AttackSpec = AttackSpec(
        attack_id="zo_sign_sgd",
        name="ZO-signSGD",
        description="ZO-signSGD with Gaussian-direction gradient estimator; deterministic seed; logs time.",
    )

    def _sample_direction(self, rng: np.random.Generator, shape) -> np.ndarray:
        u = rng.standard_normal(size=shape, dtype=np.float32)
        if self.grayscale and u.ndim == 3 and u.shape[-1] == 3:
            g = u.mean(axis=2, keepdims=True)
            u = np.repeat(g, 3, axis=2)
        return u

    def run(self, x0: np.ndarray, oracle, budget) -> AttackResult:
        x0 = _as_float01(x0)
        seed = _get_seed(budget)
        rng = np.random.default_rng(seed)

        threshold = getattr(oracle, "threshold_p", getattr(oracle, "threshold", None))
        t0 = perf_counter()

        delta = np.zeros_like(x0, dtype=np.float32)
        m = np.zeros_like(x0, dtype=np.float32)

        history: Dict[str, Any] = {
            "iter": [],
            "elapsed_ms": [],
            "dist": [],
            "best_dist": [],
            "grad_l1": [],     # proxy magnitude of sign-gradient
            "accepted": [],    # always True for signSGD step (kept for consistency)
        }

        # initial query
        x_cur = oracle.project(x0 + delta)
        f0 = float(oracle.query(x_cur))

        best_dist = float(getattr(oracle.state, "best_dist", f0))
        best_x = getattr(oracle.state, "best_x", x_cur)

        it = 0
        while oracle.budget_ok():
            elapsed_ms = (perf_counter() - t0) * 1000.0

            # success check
            best_dist = float(getattr(oracle.state, "best_dist", best_dist))
            if threshold is not None and best_dist <= float(threshold):
                break

            # recompute baseline at current point (more correct than reusing best_dist)
            x_cur = oracle.project(x0 + delta)
            base = float(oracle.query(x_cur))
            if not oracle.budget_ok():
                break

            g_hat = self._estimate_grad(x_cur, oracle, base, rng)
            if g_hat is None:
                break

            s = np.sign(g_hat).astype(np.float32)
            if self.momentum and self.momentum > 0.0:
                beta = float(self.momentum)
                m = beta * m + (1.0 - beta) * s
                step_dir = np.sign(m)
            else:
                step_dir = s

            delta = delta - float(self.lr) * step_dir

            # optional: evaluate after update to refresh best_x quickly
            if not oracle.budget_ok():
                break
            x_new = oracle.project(x0 + delta)
            f_new = float(oracle.query(x_new))

            best_dist = float(getattr(oracle.state, "best_dist", min(best_dist, f_new)))
            best_x = getattr(oracle.state, "best_x", best_x)

            history["iter"].append(it)
            history["elapsed_ms"].append(elapsed_ms)
            history["dist"].append(f_new)
            history["best_dist"].append(best_dist)
            history["grad_l1"].append(float(np.mean(np.abs(step_dir))))
            history["accepted"].append(True)

            it += 1

        total_ms = (perf_counter() - t0) * 1000.0
        best_dist = float(getattr(oracle.state, "best_dist", best_dist))
        best_x = getattr(oracle.state, "best_x", best_x)
        success = (threshold is not None and best_dist <= float(threshold))

        return AttackResult(
            success=bool(success),
            best_dist=best_dist,
            queries_used=int(getattr(oracle, "queries_used", 0)),
            best_x=best_x,
            history=history,
            extra={
                "seed": seed,
                "mu": float(self.mu),
                "lr": float(self.lr),
                "n_samples": int(self.n_samples),
                "estimator": self.estimator,
                "momentum": float(self.momentum),
                "grayscale": bool(self.grayscale),
                "runtime_ms": float(total_ms),
            },
        )

    def _estimate_grad(
        self,
        x_cur: np.ndarray,
        oracle,
        base: float,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        mu = float(self.mu)
        if mu <= 0:
            raise ValueError("mu must be > 0")

        q = int(self.n_samples)
        if q <= 0:
            raise ValueError("n_samples must be > 0")

        grad = np.zeros_like(x_cur, dtype=np.float32)

        central = (self.estimator == "central")
        if self.estimator not in ("forward", "central"):
            raise ValueError("estimator must be 'forward' or 'central'")

        for _ in range(q):
            if not oracle.budget_ok():
                return None
            u = self._sample_direction(rng, x_cur.shape)

            if central:
                x_p = oracle.project(x_cur + mu * u)
                fp = float(oracle.query(x_p))
                if not oracle.budget_ok():
                    return None
                x_n = oracle.project(x_cur - mu * u)
                fn = float(oracle.query(x_n))
                grad += ((fp - fn) / (2.0 * mu)) * u
            else:
                x_p = oracle.project(x_cur + mu * u)
                fp = float(oracle.query(x_p))
                grad += ((fp - base) / mu) * u

        grad /= float(q)
        return grad