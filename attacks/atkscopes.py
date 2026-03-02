"""Atkscopes — Multiresolution Adversarial Perturbation.

Perturbation is applied in a transformed domain (DCT or pixel) matched
to the scale at which the target hash extracts features:
  - 'global'  (pHash, PDQ)    → DCT over the full image
  - 'mid'     (PhotoDNA)      → DCT over overlapping mid-scale patches
  - 'pixel'   (NeuralHash)    → pixel domain (standard)

Optimisation: random coordinate descent with Adam per-coordinate.

Key fixes vs original:
  - best_dist is updated from oracle.query(x_after_adam_step), NOT from
    min(fp, fn). The probe points are only used for gradient estimation;
    the actual x after the Adam step gets its own query for accurate tracking.
  - oracle.budget_ok() checked at every iteration in all modes.
  - L2 metric consistent: RMS (sqrt(mean(...))), same as utils.l2_img.

Reference:
    Zhang et al., "Atkscopes: Multiresolution Adversarial Perturbation as
    a Unified Attack on Perceptual Hashing and Beyond",
    USENIX Security 2025.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from .base import AttackResult, AttackSpec


# ---------------------------------------------------------------------------
# DCT helpers
# ---------------------------------------------------------------------------

def _dct2(block: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import dct
        return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
    except ImportError:
        return block


def _idct2(block: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import idct
        return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
    except ImportError:
        return block


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    """RMS L2 — consistent with utils.l2_img."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


# ---------------------------------------------------------------------------
# Adam state (per-coordinate)
# ---------------------------------------------------------------------------

class _Adam:
    __slots__ = ("m", "v", "t", "beta1", "beta2", "eps")

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = 0.0
        self.v = 0.0
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, g: float, lr: float) -> float:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return -lr * m_hat / (v_hat ** 0.5 + self.eps)


# ---------------------------------------------------------------------------
# Patch grid for mid-scale (PhotoDNA)
# ---------------------------------------------------------------------------

def _patch_coords(h: int, w: int, k: int) -> List[Tuple[int, int, int, int]]:
    coords = []
    for r in range(0, h - k + 1, k):
        for c in range(0, w - k + 1, k):
            coords.append((r, r + k, c, c + k))
    return coords


# ---------------------------------------------------------------------------
# Atkscopes
# ---------------------------------------------------------------------------

@dataclass
class AtkScopesAttack:
    """Multiresolution coordinate-descent attack (Atkscopes).

    Parameters
    ----------
    scale : str
        'global' — DCT over full image (pHash, PDQ).
        'mid'    — DCT over mid-scale patches (PhotoDNA).
        'pixel'  — direct pixel perturbation (NeuralHash).
    lr : float
        Adam learning rate.
    a : float
        Finite-difference step for gradient estimation.
        Should be large enough to change at least one hash bit.
        Typical: 0.05..0.2 for pixel/global, 1.0+ for DCT coefficients.
    patch_size : int | None
        Patch size for 'mid' scale. None → image_height // 4.
    """
    scale: str = "global"
    lr: float = 0.05
    a: float = 0.1
    patch_size: Optional[int] = None

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="atkscopes",
            params=dict(scale=self.scale, lr=self.lr, a=self.a, patch_size=self.patch_size),
        )

    def run(self, x0: np.ndarray, oracle: HashOracle, budget: BudgetSpec) -> AttackResult:
        t0 = time.monotonic()
        x0 = x0.astype(np.float32)
        history: List[float] = []

        best_dist = oracle.query(x0)
        best_x = x0.copy()
        history.append(best_dist)

        threshold = oracle.threshold
        h, w, c = x0.shape
        k = self.patch_size or max(h // 4, 8)

        if self.scale == "pixel":
            best_x, best_dist, history = self._run_pixel(
                x0, oracle, threshold, h, w, c, history, best_dist
            )
        elif self.scale == "global":
            best_x, best_dist, history = self._run_dct_global(
                x0, oracle, threshold, h, w, c, history, best_dist
            )
        else:  # "mid"
            best_x, best_dist, history = self._run_dct_mid(
                x0, oracle, threshold, h, w, c, k, history, best_dist
            )

        reason = "success" if best_dist <= threshold else "budget"
        elapsed = int((time.monotonic() - t0) * 1000)
        return AttackResult(
            x_best=best_x,
            best_score=best_dist,
            queries_used=oracle.queries_used,
            runtime_ms=elapsed,
            stopped_reason=reason,
            history=history,
            extra={"l2": _l2(x0, best_x), "scale": self.scale},
        )

    # ------------------------------------------------------------------
    # Pixel mode
    # ------------------------------------------------------------------

    def _run_pixel(self, x0, oracle, threshold, h, w, c, history, best_dist):
        delta = np.zeros_like(x0)
        adam_states = {}
        best_x = x0.copy()

        while oracle.budget_ok() and best_dist > threshold:
            hi = np.random.randint(h)
            wi = np.random.randint(w)
            ci = np.random.randint(c)
            key = (hi, wi, ci)

            # FD gradient: probe ±a at single coordinate
            e = np.zeros_like(delta)
            e[hi, wi, ci] = self.a

            x_cur = _clip(x0 + delta)
            fp = oracle.query(_clip(x_cur + e))
            if not oracle.budget_ok():
                break
            fn = oracle.query(_clip(x_cur - e))
            history.extend([fp, fn])

            g = (fp - fn) / (2 * self.a)
            adam = adam_states.setdefault(key, _Adam())
            delta[hi, wi, ci] += adam.step(g, self.lr)

            # Query the actual updated point — this is the fix
            x_new = _clip(x0 + delta)
            dist = oracle.query(x_new)
            history.append(dist)

            if dist < best_dist:
                best_dist = dist
                best_x = x_new.copy()

        return best_x, best_dist, history

    # ------------------------------------------------------------------
    # DCT-global mode
    # ------------------------------------------------------------------

    def _run_dct_global(self, x0, oracle, threshold, h, w, c, history, best_dist):
        coeffs = [_dct2(x0[:, :, ch]) for ch in range(c)]
        delta = [np.zeros((h, w), dtype=np.float32) for _ in range(c)]
        adam_states = {}
        best_x = x0.copy()

        while oracle.budget_ok() and best_dist > threshold:
            fi = np.random.randint(h)
            fj = np.random.randint(w)
            ci = np.random.randint(c)
            key = (fi, fj, ci)

            e = np.zeros((h, w), dtype=np.float32)
            e[fi, fj] = self.a

            def _build(sign):
                d = [delta[ch].copy() for ch in range(c)]
                d[ci] = d[ci] + sign * e
                chs = [_idct2(coeffs[ch] + d[ch]).astype(np.float32) for ch in range(c)]
                return _clip(np.stack(chs, axis=2))

            xp = _build(+1)
            xn = _build(-1)
            fp = oracle.query(xp)
            if not oracle.budget_ok():
                break
            fn = oracle.query(xn)
            history.extend([fp, fn])

            g = (fp - fn) / (2 * self.a)
            adam = adam_states.setdefault(key, _Adam())
            delta[ci][fi, fj] += adam.step(g, self.lr)

            # Query actual updated point — the fix
            chs = [_idct2(coeffs[ch] + delta[ch]).astype(np.float32) for ch in range(c)]
            x_new = _clip(np.stack(chs, axis=2))
            dist = oracle.query(x_new)
            history.append(dist)

            if dist < best_dist:
                best_dist = dist
                best_x = x_new.copy()

        return best_x, best_dist, history

    # ------------------------------------------------------------------
    # DCT-mid mode (patch-level)
    # ------------------------------------------------------------------

    def _run_dct_mid(self, x0, oracle, threshold, h, w, c, k, history, best_dist):
        patches = _patch_coords(h, w, k)
        if not patches:
            return self._run_dct_global(x0, oracle, threshold, h, w, c, history, best_dist)

        delta = np.zeros_like(x0)
        adam_states = {}
        best_x = x0.copy()

        while oracle.budget_ok() and best_dist > threshold:
            pi = np.random.randint(len(patches))
            r0, r1, c0, c1 = patches[pi]
            ph, pw = r1 - r0, c1 - c0
            fi = np.random.randint(ph)
            fj = np.random.randint(pw)
            ci = np.random.randint(c)
            key = (r0, r1, c0, c1, fi, fj, ci)

            e_coeff = np.zeros((ph, pw), dtype=np.float32)
            e_coeff[fi, fj] = self.a
            d_pixel = _idct2(e_coeff).astype(np.float32)

            def _build_v2(sign):
                xc = _clip(x0 + delta)
                patch = xc[r0:r1, c0:c1, ci]
                xc = xc.copy()
                xc[r0:r1, c0:c1, ci] = np.clip(patch + sign * self.a * d_pixel, 0.0, 1.0)
                return xc

            fp = oracle.query(_build_v2(+1))
            if not oracle.budget_ok():
                break
            fn = oracle.query(_build_v2(-1))
            history.extend([fp, fn])

            g = (fp - fn) / (2 * self.a)
            adam = adam_states.setdefault(key, _Adam())
            delta[r0:r1, c0:c1, ci] += adam.step(g, self.lr) * d_pixel

            # Query actual updated point — the fix
            x_new = _clip(x0 + delta)
            dist = oracle.query(x_new)
            history.append(dist)

            if dist < best_dist:
                best_dist = dist
                best_x = x_new.copy()

        return best_x, best_dist, history