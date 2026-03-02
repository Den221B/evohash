from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from .base import AttackSpec, AttackResult


def _get_seed(budget) -> int:
    # Robust seed retrieval (works even if budget doesn't have seed)
    s = getattr(budget, "seed", None)
    if s is None:
        return 0
    try:
        return int(s) & 0xFFFFFFFF
    except Exception:
        return 0


def _as_float01(x):
    # Ensure float32 in [0,1]
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    # If it's likely in [0,255], rescale
    mx = float(np.max(x)) if x.size else 0.0
    if mx > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _apply_grayscale_noise(p: np.ndarray) -> np.ndarray:
    # Make noise identical across channels
    if p.ndim == 3 and p.shape[-1] == 3:
        g = p.mean(axis=2, keepdims=True)
        return np.repeat(g, 3, axis=2)
    return p


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v.ravel(), ord=2))
    if n < eps:
        return np.zeros_like(v)
    return v / (n + eps)


@dataclass
class NESAttack:
    """
    NES-like gradient-free attack using antithetic sampling.

    Minimizes oracle distance: f(x) = oracle.query(x) (or score).
    Uses symmetric difference: (f(x+σp) - f(x-σp)) * p / (2σ).
    """
    sigma: float = 0.05
    lr: float = 0.02
    n_samples: int = 20
    antithetic: bool = True
    grayscale: bool = False
    normalize_grad: bool = True

    spec: AttackSpec = AttackSpec(
        attack_id="nes",
        name="NES",
        description="NES-style gradient estimator with (optional) antithetic sampling and grad normalization.",
    )

    def _sample_noise(self, rng: np.random.Generator, shape) -> np.ndarray:
        p = rng.standard_normal(size=shape, dtype=np.float32)
        if self.grayscale:
            p = _apply_grayscale_noise(p)
        return p

    def run(self, x0, oracle, budget) -> AttackResult:
        x0 = _as_float01(x0)
        seed = _get_seed(budget)
        rng = np.random.default_rng(seed)

        threshold = getattr(oracle, "threshold_p", None)
        eps = 1e-12

        delta = np.zeros_like(x0, dtype=np.float32)
        best_dist = float(getattr(oracle.state, "best_dist", np.inf))
        best_x = getattr(oracle.state, "best_x", None)

        hist: Dict[str, Any] = {
            "dist": [],
            "best_dist": [],
        }

        # Ensure oracle has baseline (optional)
        # We operate using oracle.query which should increment queries.
        while oracle.budget_ok():
            x_cur = oracle.project(x0 + delta)
            base = oracle.query(x_cur)

            # Track best
            best_dist = float(getattr(oracle.state, "best_dist", base))
            best_x = getattr(oracle.state, "best_x", x_cur)

            hist["dist"].append(float(base))
            hist["best_dist"].append(best_dist)

            # Early success stop
            if threshold is not None and best_dist <= float(threshold):
                break

            # Estimate gradient
            grad = np.zeros_like(delta, dtype=np.float32)

            # Each sample may consume 2 queries (antithetic) or 1 (one-sided)
            for _ in range(self.n_samples):
                if not oracle.budget_ok():
                    break
                p = self._sample_noise(rng, delta.shape)

                if self.antithetic:
                    x_p = oracle.project(x0 + (delta + self.sigma * p))
                    dp = oracle.query(x_p)
                    if not oracle.budget_ok():
                        break
                    x_n = oracle.project(x0 + (delta - self.sigma * p))
                    dn = oracle.query(x_n)
                    grad += (float(dp) - float(dn)) * p
                else:
                    x_p = oracle.project(x0 + (delta + self.sigma * p))
                    dp = oracle.query(x_p)
                    grad += (float(dp) - float(base)) * p

            denom = (2.0 * self.n_samples * self.sigma) if self.antithetic else (self.n_samples * self.sigma)
            if denom > 0:
                grad = grad / float(denom)

            if self.normalize_grad:
                grad = _normalize(grad, eps=eps)

            # Gradient descent step on delta (minimize distance)
            delta = delta - float(self.lr) * grad

        # Final result from oracle state
        best_dist = float(getattr(oracle.state, "best_dist", best_dist))
        best_x = getattr(oracle.state, "best_x", best_x)
        success = (threshold is not None and best_dist <= float(threshold))

        return AttackResult(
            success=bool(success),
            best_dist=best_dist,
            queries_used=int(getattr(oracle, "queries_used", 0)),
            best_x=best_x,
            history=hist,
            extra={
                "seed": seed,
                "sigma": self.sigma,
                "lr": self.lr,
                "n_samples": self.n_samples,
                "antithetic": self.antithetic,
                "grayscale": self.grayscale,
                "normalize_grad": self.normalize_grad,
            },
        )


@dataclass
class ProkosAttack:
    """
    Prokos-style MC gradient estimator (USENIX'23):
      base = f(x)
      for j in 1..q: p~N(0,1), c = f(x+σp) - base, grad += c * p
      g = Norm(grad/q)
      with momentum: m = ρ m + (1-ρ) g
      step: delta -= lr * m

    Optional double-sample uses both +p and -p (extra queries).
    """
    sigma: float = 0.05
    lr: float = 0.02
    n_samples: int = 20
    rho: float = 0.5
    double_sample: bool = False
    grayscale: bool = False
    normalize_grad: bool = True

    spec: AttackSpec = AttackSpec(
        attack_id="prokos",
        name="Prokos",
        description="Prokos-style MC gradient estimator with Norm(grad) and momentum; optional double-sample.",
    )

    def _sample_noise(self, rng: np.random.Generator, shape) -> np.ndarray:
        p = rng.standard_normal(size=shape, dtype=np.float32)
        if self.grayscale:
            p = _apply_grayscale_noise(p)
        return p

    def run(self, x0, oracle, budget) -> AttackResult:
        x0 = _as_float01(x0)
        seed = _get_seed(budget)
        rng = np.random.default_rng(seed)

        threshold = getattr(oracle, "threshold_p", None)
        eps = 1e-12

        delta = np.zeros_like(x0, dtype=np.float32)
        mom = np.zeros_like(x0, dtype=np.float32)

        hist: Dict[str, Any] = {
            "dist": [],
            "best_dist": [],
            "grad_norm": [],
        }

        while oracle.budget_ok():
            x_cur = oracle.project(x0 + delta)
            base = oracle.query(x_cur)

            best_dist = float(getattr(oracle.state, "best_dist", base))
            best_x = getattr(oracle.state, "best_x", x_cur)

            hist["dist"].append(float(base))
            hist["best_dist"].append(best_dist)

            if threshold is not None and best_dist <= float(threshold):
                break

            grad = np.zeros_like(delta, dtype=np.float32)

            # Prokos one-sided estimator
            # Each sample consumes 1 query (or 2 if double_sample)
            for _ in range(self.n_samples):
                if not oracle.budget_ok():
                    break
                p = self._sample_noise(rng, delta.shape)

                x_p = oracle.project(x0 + (delta + self.sigma * p))
                fp = oracle.query(x_p)
                c = float(fp) - float(base)
                grad += c * p

                if self.double_sample:
                    if not oracle.budget_ok():
                        break
                    x_n = oracle.project(x0 + (delta - self.sigma * p))
                    fn = oracle.query(x_n)
                    c2 = float(fn) - float(base)
                    grad += c2 * (-p)

            grad = grad / float(max(self.n_samples, 1))

            # Norm(g) from the paper
            if self.normalize_grad:
                g = _normalize(grad, eps=eps)
            else:
                g = grad

            hist["grad_norm"].append(float(np.linalg.norm(g.ravel(), ord=2)))

            # Momentum from the paper
            rho = float(self.rho)
            mom = rho * mom + (1.0 - rho) * g

            # Step
            delta = delta - float(self.lr) * mom

        best_dist = float(getattr(oracle.state, "best_dist", np.inf))
        best_x = getattr(oracle.state, "best_x", None)
        success = (threshold is not None and best_dist <= float(threshold))

        return AttackResult(
            success=bool(success),
            best_dist=best_dist,
            queries_used=int(getattr(oracle, "queries_used", 0)),
            best_x=best_x,
            history=hist,
            extra={
                "seed": seed,
                "sigma": self.sigma,
                "lr": self.lr,
                "n_samples": self.n_samples,
                "rho": self.rho,
                "double_sample": self.double_sample,
                "grayscale": self.grayscale,
                "normalize_grad": self.normalize_grad,
            },
        )