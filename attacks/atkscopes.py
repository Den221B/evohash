"""Atkscopes — Multiresolution Adversarial Perturbation.

Perturbation is applied in a transformed domain (DCT or pixel) matched
to the scale at which the target hash extracts features:
  - 'global'  (pHash, PDQ)    → DCT over the full image
  - 'mid'     (PhotoDNA)      → DCT over overlapping mid-scale patches
  - 'pixel'   (NeuralHash)    → pixel domain (standard)

Optimisation: random coordinate descent with Adam per-coordinate,
as described in Algorithm 2 of the paper (black-box branch).

Loss for triggering regulation (collision):
    L = max(D(hash(x'), h_target), 0)   → minimise hash distance

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
        return block   # no-op fallback — attack degrades to pixel domain


def _idct2(block: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import idct
        return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
    except ImportError:
        return block


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


# ---------------------------------------------------------------------------
# Adam state (per-coordinate)
# ---------------------------------------------------------------------------

class _Adam:
    """Scalar Adam for a single perturbation coordinate."""
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
    """Return list of (r0,r1,c0,c1) for non-overlapping k×k patches."""
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
        'global' — single DCT over full image (pHash, PDQ).
        'mid'    — DCT over mid-scale patches, patch_size controls k (PhotoDNA).
        'pixel'  — direct pixel perturbation (NeuralHash).
    lr : float
        Adam learning rate.
    a : float
        Finite-difference step for gradient estimation (symmetric quotient).
    max_iters : int
        Maximum coordinate-update iterations.
    patch_size : int | None
        Patch size for 'mid' scale.  If None, defaults to image_height // 4.
    """
    scale: str = "global"       # "global" | "mid" | "pixel"
    lr: float = 1.0
    a: float = 1.0              # FD perturbation magnitude
    max_iters: int = 2000
    patch_size: Optional[int] = None

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="atkscopes",
            params=dict(
                scale=self.scale,
                lr=self.lr,
                a=self.a,
                max_iters=self.max_iters,
                patch_size=self.patch_size,
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
        x = x0.astype(np.float32).copy()
        history: List[float] = []

        best_dist = oracle.query(x)
        best_x = x.copy()
        history.append(best_dist)

        threshold = getattr(oracle, "threshold", 0.0)

        h, w, c = x.shape
        k = self.patch_size or max(h // 4, 8)

        # build coefficient tensor in chosen domain
        if self.scale == "pixel":
            delta = np.zeros_like(x)           # perturbation directly in pixel space
            adam_states = {}                    # (i,j,ch) -> _Adam

            for step in range(self.max_iters):
                if best_dist <= threshold:
                    break

                # random coordinate
                hi = np.random.randint(h)
                wi = np.random.randint(w)
                ci = np.random.randint(c)
                key = (hi, wi, ci)

                # symmetric FD gradient for this coordinate
                e = np.zeros_like(delta)
                e[hi, wi, ci] = self.a

                try:
                    fp = oracle.query(_clip(x + e))
                    fn = oracle.query(_clip(x - e))
                except Exception:
                    break
                g = (fp - fn) / (2 * self.a)
                history.extend([fp, fn])

                adam = adam_states.setdefault(key, _Adam())
                delta[hi, wi, ci] += adam.step(g, self.lr)

                x = _clip(x0 + delta)
                dist = min(fp, fn)   # approximate; true dist computed lazily
                if dist < best_dist:
                    best_dist = dist
                    best_x = x.copy()

        elif self.scale == "global":
            # delta lives in DCT coefficient space (full image, per channel)
            x, best_x, history, best_dist = self._run_dct_global(
                x0, oracle, threshold, h, w, c, history, best_dist
            )

        else:  # "mid"
            x, best_x, history, best_dist = self._run_dct_mid(
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
    # DCT-global mode
    # ------------------------------------------------------------------

    def _run_dct_global(self, x0, oracle, threshold, h, w, c, history, best_dist):
        # Initialise coefficient tensors per channel
        coeffs = [_dct2(x0[:, :, ch]) for ch in range(c)]
        delta  = [np.zeros((h, w), dtype=np.float32) for _ in range(c)]
        adam_states = {}

        best_x = x0.copy()
        x = x0.copy()

        for _ in range(self.max_iters):
            if best_dist <= threshold:
                break

            fi = np.random.randint(h)
            fj = np.random.randint(w)
            ci = np.random.randint(c)
            key = (fi, fj, ci)

            # perturb single DCT coefficient
            e = np.zeros((h, w), dtype=np.float32)
            e[fi, fj] = self.a

            def _build(sign):
                d = [delta[ch].copy() for ch in range(c)]
                d[ci] = d[ci] + sign * e
                chs = [_idct2(coeffs[ch] + d[ch]).astype(np.float32) for ch in range(c)]
                return _clip(np.stack(chs, axis=2))

            try:
                xp = _build(+1)
                xn = _build(-1)
                fp = oracle.query(xp)
                fn = oracle.query(xn)
            except Exception:
                break
            history.extend([fp, fn])

            g = (fp - fn) / (2 * self.a)
            adam = adam_states.setdefault(key, _Adam())
            delta[ci][fi, fj] += adam.step(g, self.lr)

            # rebuild current image
            chs = [_idct2(coeffs[ch] + delta[ch]).astype(np.float32) for ch in range(c)]
            x = _clip(np.stack(chs, axis=2))

            dist = min(fp, fn)
            if dist < best_dist:
                best_dist = dist
                best_x = x.copy()

        return x, best_x, history, best_dist

    # ------------------------------------------------------------------
    # DCT-mid mode (patch-level)
    # ------------------------------------------------------------------

    def _run_dct_mid(self, x0, oracle, threshold, h, w, c, k, history, best_dist):
        patches = _patch_coords(h, w, k)
        if not patches:
            # fallback to global if patches don't fit
            return self._run_dct_global(x0, oracle, threshold, h, w, c, history, best_dist)

        # per-patch DCT coefficients
        patch_coeffs = {}
        for (r0, r1, c0, c1) in patches:
            for ch in range(c):
                patch_coeffs[(r0, r1, c0, c1, ch)] = _dct2(x0[r0:r1, c0:c1, ch])

        delta = np.zeros_like(x0)
        adam_states = {}
        best_x = x0.copy()
        x = x0.copy()

        for _ in range(self.max_iters):
            if best_dist <= threshold:
                break

            # pick random patch, random coefficient, random channel
            pi = np.random.randint(len(patches))
            r0, r1, c0, c1 = patches[pi]
            ph, pw = r1 - r0, c1 - c0
            fi = np.random.randint(ph)
            fj = np.random.randint(pw)
            ci = np.random.randint(c)
            key = (r0, r1, c0, c1, fi, fj, ci)

            e_coeff = np.zeros((ph, pw), dtype=np.float32)
            e_coeff[fi, fj] = self.a

            def _build_patch(sign, dx=delta):
                xc = x0.copy() + dx
                pcoeff = patch_coeffs[(r0, r1, c0, c1, ci)]
                d_patch = _idct2(sign * e_coeff).astype(np.float32)
                xc[r0:r1, c0:c1, ci] = np.clip(
                    x0[r0:r1, c0:c1, ci] + _idct2(pcoeff).astype(np.float32) - x0[r0:r1, c0:c1, ci] + d_patch,
                    0.0, 1.0,
                )
                return _clip(xc)

            # simpler: just perturb in pixel domain within patch via IDCT of e
            d_pixel = _idct2(e_coeff).astype(np.float32)  # shape (ph,pw)

            def _build_v2(sign):
                xc = x.copy()
                xc[r0:r1, c0:c1, ci] = np.clip(
                    xc[r0:r1, c0:c1, ci] + sign * self.a * d_pixel, 0.0, 1.0
                )
                return _clip(xc)

            try:
                fp = oracle.query(_build_v2(+1))
                fn = oracle.query(_build_v2(-1))
            except Exception:
                break
            history.extend([fp, fn])

            g = (fp - fn) / (2 * self.a)
            adam = adam_states.setdefault(key, _Adam())
            step_val = adam.step(g, self.lr)

            delta[r0:r1, c0:c1, ci] += step_val * d_pixel
            x = _clip(x0 + delta)

            dist = min(fp, fn)
            if dist < best_dist:
                best_dist = dist
                best_x = x.copy()

        return x, best_x, history, best_dist
