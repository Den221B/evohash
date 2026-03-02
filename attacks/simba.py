"""SimBa — Simple Black-box Adversarial Attack.

Greedy coordinate descent on hash distance.
Optimises delta (perturbation from x0): x_adv = clip(x0 + delta).

Key fix vs original:
  - Loop is time-based (oracle.budget_ok()), not max_iters.
  - DCT basis: freq_dims now properly controls the low-frequency window.
    For pHash/PDQ (DCT-based hashes), using low-freq DCT basis vectors
    is the correct choice — they survive the hash's internal resize.

Reference:
    Guo et al., "Simple Black-box Adversarial Attacks", ICML 2019.
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
    """RMS L2 — consistent with utils.l2_img."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _idct2(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import idct
        return idct(idct(x, axis=0, norm="ortho"), axis=1, norm="ortho")
    except ImportError:
        return x


@dataclass
class SimBaAttack:
    """SimBa in pixel or DCT basis.

    Parameters
    ----------
    epsilon : float
        Step size for each basis update. In [0,1] image scale.
        For pixel basis: 0.01..0.1.
        For DCT basis: 0.05..0.3 (DCT basis vectors are normalized).
    basis : str
        'pixel' — single random pixel/channel.
        'dct'   — random DCT frequency (low-freq for pHash/PDQ).
    freq_dims : int | None
        Restrict DCT to top-left freq_dims×freq_dims block (low frequencies).
        For pHash: set to 8 (matches 8x8 DCT submatrix).
        For PDQ:   set to 16 (matches 16x16 DCT submatrix).
        None → min(h, w) (all frequencies).
    """
    epsilon: float = 0.1
    basis: str = "dct"
    freq_dims: Optional[int] = None

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="simba",
            params=dict(
                epsilon=self.epsilon,
                basis=self.basis,
                freq_dims=self.freq_dims,
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
        h, w, c = x0.shape
        freq = self.freq_dims or min(h, w)

        while oracle.budget_ok() and best_dist > threshold:
            q = self._sample_direction(h, w, c, freq)

            # Try positive step
            d_pos = oracle.query(_clip(x0 + delta + self.epsilon * q))
            history.append(d_pos)
            if d_pos < best_dist:
                delta = delta + self.epsilon * q
                best_dist = d_pos
                best_x = _clip(x0 + delta).copy()
                continue

            if not oracle.budget_ok():
                break

            # Try negative step
            d_neg = oracle.query(_clip(x0 + delta - self.epsilon * q))
            history.append(d_neg)
            if d_neg < best_dist:
                delta = delta - self.epsilon * q
                best_dist = d_neg
                best_x = _clip(x0 + delta).copy()

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

    def _sample_direction(self, h: int, w: int, c: int, freq: int) -> np.ndarray:
        if self.basis == "dct":
            return self._dct_direction(h, w, c, freq)
        return self._pixel_direction(h, w, c)

    @staticmethod
    def _pixel_direction(h: int, w: int, c: int) -> np.ndarray:
        q = np.zeros((h, w, c), dtype=np.float32)
        q[np.random.randint(h), np.random.randint(w), np.random.randint(c)] = 1.0
        return q

    @staticmethod
    def _dct_direction(h: int, w: int, c: int, freq: int) -> np.ndarray:
        fi = np.random.randint(freq)
        fj = np.random.randint(freq)
        ci = np.random.randint(c)
        coeff = np.zeros((h, w), dtype=np.float32)
        coeff[fi, fj] = 1.0
        basis_ch = _idct2(coeff).astype(np.float32)
        n = np.linalg.norm(basis_ch)
        if n > 1e-12:
            basis_ch /= n
        q = np.zeros((h, w, c), dtype=np.float32)
        q[:, :, ci] = basis_ch
        return qы