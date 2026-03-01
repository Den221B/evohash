"""Natural Evolution Strategies (NES) and Prokos variant.

NES (baseline):
    Monte Carlo gradient estimation via antithetic sampling.
    Optimises a perturbation delta added to x0, so x = clip(x0 + delta).
    This is the correct formulation: we never lose track of the base image.

Prokos (USENIX Security 2023):
    NES + momentum (rho=0.5) + grayscale noise constraint.

Reference:
    Prokos et al., "Squint Hard Enough: Attacking Perceptual Hashing
    with Adversarial Machine Learning", USENIX Security 2023.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from .base import AttackResult, AttackSpec


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass
class NESAttack:
    """NES with antithetic sampling.

    Key design choices:
    - Optimises delta (perturbation from x0), not x directly.
      x_adv = clip(x0 + delta). This avoids drift and keeps L2 meaningful.
    - sigma is in the perturbation domain (fraction of [0,1]), not image scale.
      Good starting values: sigma=0.05..0.1 for most hashes.
    - After each gradient step, only accept x if it improves the score.
      (Greedy descent — avoids random walks when gradient is noisy.)

    Parameters
    ----------
    sigma : float
        Noise std for gradient estimation. In [0,1] image scale.
        Typical: 0.05 (fine), 0.1 (coarse), 0.2 (very coarse for PDQ).
    lr : float
        Step size for delta update after each gradient estimate.
    n_samples : int
        Antithetic pairs per gradient estimate (total queries = 2 * n_samples).
    max_iters : int
        Max gradient steps.
    grayscale : bool
        If True, use identical noise across RGB channels.
    """
    sigma: float = 0.05
    lr: float = 0.01
    n_samples: int = 10
    max_iters: int = 3000
    grayscale: bool = False

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="nes",
            params=dict(
                sigma=self.sigma, lr=self.lr,
                n_samples=self.n_samples, max_iters=self.max_iters,
                grayscale=self.grayscale,
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

        for _ in range(self.max_iters):
            if best_dist <= threshold or not oracle.budget_ok():
                break

            grad = self._estimate_grad(x0, delta, oracle)
            if grad is None:
                break

            # normalised gradient step on delta
            norm = np.linalg.norm(grad)
            if norm > 1e-12:
                delta = delta - self.lr * grad / norm
            else:
                delta = delta + np.random.randn(*delta.shape).astype(np.float32) * self.sigma * 0.1

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

        Evaluates f(clip(x0 + delta + sigma*p)) and f(clip(x0 + delta - sigma*p)).
        Returns gradient array or None if budget exhausted.
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
        Momentum coefficient. delta_update = rho * prev + (1-rho) * grad.
    """
    rho: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "grayscale", True)
        self.spec = AttackSpec(
            attack_id="prokos",
            params=dict(
                sigma=self.sigma, lr=self.lr,
                n_samples=self.n_samples, max_iters=self.max_iters,
                rho=self.rho,
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

        for _ in range(self.max_iters):
            if best_dist <= threshold or not oracle.budget_ok():
                break

            grad = self._estimate_grad(x0, delta, oracle)
            if grad is None:
                break

            norm = np.linalg.norm(grad)
            g = grad / norm if norm > 1e-12 else np.random.randn(*grad.shape).astype(np.float32)

            # momentum update
            momentum = self.rho * momentum + (1 - self.rho) * g
            delta = delta - self.lr * momentum

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