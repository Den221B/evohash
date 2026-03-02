"""Natural Evolution Strategies (NES) and Prokos variant.

NES (baseline):
    Monte Carlo gradient estimation via antithetic sampling.
    Optimises delta (perturbation from x0): x_adv = clip(x0 + delta).

Prokos (USENIX Security 2023):
    NES + momentum (rho=0.5) + grayscale noise.

Key design:
  - Loop runs until oracle.budget_ok() returns False (time limit) or success.
    max_iters removed — time is the only budget.
  - Gradient estimated from x_cur = clip(x0 + delta), not from x0.
    This is correct: we probe around the current best perturbation.
  - After gradient step, delta is updated; x_new = clip(x0 + delta) is queried.
    This is a separate query from the gradient probes — needed for accurate tracking.
  - L2 metric: RMS (sqrt(mean(...))), consistent with utils.l2_img.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from .base import AttackResult, AttackSpec


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    """RMS L2 — consistent with utils.l2_img."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass
class NESAttack:
    """NES with antithetic sampling.

    Parameters
    ----------
    sigma : float
        Noise std for gradient estimation. In [0,1] image scale.
        Too small → no signal through hash's quantization.
        Too large → perturbation too far from current x.
        Typical: 0.05..0.2. For pHash/PDQ: 0.1..0.3.
    lr : float
        Step size for delta update after each gradient estimate.
        Typical: 0.01..0.1.
    n_samples : int
        Antithetic pairs per gradient estimate.
        More → lower variance gradient, but more queries per step.
        Typical: 5..20.
    grayscale : bool
        If True, identical noise across RGB channels (Prokos trick).
    normalize_grad : bool
        If True, normalize gradient to unit norm before lr step.
        Helps with stability when landscape is flat.
    """
    sigma: float = 0.1
    lr: float = 0.01
    n_samples: int = 10
    grayscale: bool = False
    normalize_grad: bool = True

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="nes",
            params=dict(
                sigma=self.sigma, lr=self.lr,
                n_samples=self.n_samples,
                grayscale=self.grayscale,
                normalize_grad=self.normalize_grad,
            ),
        )

    def run(self, x0: np.ndarray, oracle: HashOracle, budget: BudgetSpec) -> AttackResult:
        t0 = time.monotonic()
        x0 = x0.astype(np.float32)
        delta = np.zeros_like(x0)
        history: List[float] = []

        best_dist = oracle.query(x0)
        best_x = x0.copy()
        history.append(best_dist)

        threshold = oracle.threshold

        while oracle.budget_ok() and best_dist > threshold:
            grad = self._estimate_grad(x0, delta, oracle)
            if grad is None:
                break  # budget exhausted during grad estimation

            # Gradient step on delta
            if self.normalize_grad:
                norm = np.linalg.norm(grad)
                if norm > 1e-12:
                    delta = delta - self.lr * grad / norm
                else:
                    # Flat landscape — random restart nudge
                    delta = delta + self._sample_noise(delta.shape) * self.sigma * 0.1
            else:
                delta = delta - self.lr * grad

            if not oracle.budget_ok():
                break

            x_new = _clip(x0 + delta)
            dist = oracle.query(x_new)
            history.append(dist)

            if dist < best_dist:
                best_dist = dist
                best_x = x_new.copy()

        reason = "success" if best_dist <= threshold else "budget"
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

    def _sample_noise(self, shape) -> np.ndarray:
        if self.grayscale and len(shape) == 3 and shape[2] == 3:
            h, w, _ = shape
            n = np.random.randn(h, w, 1).astype(np.float32)
            return np.repeat(n, 3, axis=2)
        return np.random.randn(*shape).astype(np.float32)

    def _estimate_grad(self, x0: np.ndarray, delta: np.ndarray, oracle: HashOracle):
        """Antithetic NES gradient estimate w.r.t. delta.

        Probes: f(clip(x0 + delta + sigma*p)) and f(clip(x0 + delta - sigma*p))
        Returns gradient array or None if budget exhausted mid-estimation.
        """
        grad = np.zeros_like(delta)
        x_cur = _clip(x0 + delta)

        for _ in range(self.n_samples):
            if not oracle.budget_ok():
                return None
            p = self._sample_noise(delta.shape)
            xp = _clip(x_cur + self.sigma * p)
            xn = _clip(x_cur - self.sigma * p)
            dp = oracle.query(xp)
            dn = oracle.query(xn)
            grad += (dp - dn) * p

        grad /= (2 * self.n_samples * self.sigma)
        return grad


@dataclass
class ProkosAttack(NESAttack):
    """Prokos et al. USENIX 2023: NES + momentum + grayscale.

    Parameters
    ----------
    rho : float
        Momentum coefficient: m = rho*m + (1-rho)*grad.
    """
    rho: float = 0.5

    def __post_init__(self) -> None:
        self.grayscale = True
        self.spec = AttackSpec(
            attack_id="prokos",
            params=dict(
                sigma=self.sigma, lr=self.lr,
                n_samples=self.n_samples, rho=self.rho,
            ),
        )

    def run(self, x0: np.ndarray, oracle: HashOracle, budget: BudgetSpec) -> AttackResult:
        t0 = time.monotonic()
        x0 = x0.astype(np.float32)
        delta = np.zeros_like(x0)
        momentum = np.zeros_like(x0)
        history: List[float] = []

        best_dist = oracle.query(x0)
        best_x = x0.copy()
        history.append(best_dist)

        threshold = oracle.threshold

        while oracle.budget_ok() and best_dist > threshold:
            grad = self._estimate_grad(x0, delta, oracle)
            if grad is None:
                break

            norm = np.linalg.norm(grad)
            g = grad / norm if norm > 1e-12 else self._sample_noise(grad.shape)

            momentum = self.rho * momentum + (1 - self.rho) * g
            delta = delta - self.lr * momentum

            if not oracle.budget_ok():
                break

            x_new = _clip(x0 + delta)
            dist = oracle.query(x_new)
            history.append(dist)

            if dist < best_dist:
                best_dist = dist
                best_x = x_new.copy()

        reason = "success" if best_dist <= threshold else "budget"
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