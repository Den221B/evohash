"""ZO-Sign-SGD — Zeroth-Order Sign SGD.

Optimises delta (perturbation from x0): x_adv = clip(x0 + delta).

References:
    Liu et al., "Signsgd via zeroth-order oracle", ICLR 2019.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from .base import AttackResult, AttackSpec


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


@dataclass
class ZOSignSGDAttack:
    """Zeroth-order sign SGD.

    Parameters
    ----------
    mu : float
        Finite-difference step. Should be small enough not to jump over
        hash boundaries, but large enough to get signal: 0.01..0.05.
    lr : float
        Step size applied to sign(grad). In [0,1] scale: 0.001..0.01.
    n_samples : int
        Random directions per gradient estimate.
    max_iters : int
        Maximum steps.
    """
    mu: float = 0.02
    lr: float = 0.005
    n_samples: int = 20
    max_iters: int = 3000

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="zo_sign_sgd",
            params=dict(mu=self.mu, lr=self.lr,
                        n_samples=self.n_samples, max_iters=self.max_iters),
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

            x_cur = _clip(x0 + delta)
            f0 = best_dist   # use best as baseline (cheaper than extra query)

            grad = self._estimate_grad(x_cur, oracle, f0)
            if grad is None:
                break

            delta = delta - self.lr * np.sign(grad)

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

    def _estimate_grad(self, x_cur: np.ndarray, oracle: HashOracle, f0: float):
        grad = np.zeros_like(x_cur)
        for _ in range(self.n_samples):
            if not oracle.budget_ok():
                return None
            p = np.random.randn(*x_cur.shape).astype(np.float32)
            fp = oracle.query(_clip(x_cur + self.mu * p))
            grad += (fp - f0) / self.mu * p
        return grad / self.n_samples