# attacks/atkscopes.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import AttackSpec, AttackResult


# -----------------------------
# Utilities
# -----------------------------
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


def _get_threshold(oracle):
    return getattr(oracle, "threshold_p", getattr(oracle, "threshold", None))


# -----------------------------
# DCT helpers (orthonormal)
# -----------------------------
def _dct2(block: np.ndarray) -> np.ndarray:
    # Optional SciPy; if absent, fallback to identity
    try:
        from scipy.fft import dct
        return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
    except Exception:
        return block


def _idct2(block: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import idct
        return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
    except Exception:
        return block


def _patch_coords(h: int, w: int, k: int) -> np.ndarray:
    # non-overlapping grid (simple & fast); can be extended to overlapping if needed
    coords = []
    for r in range(0, h - k + 1, k):
        for c in range(0, w - k + 1, k):
            coords.append((r, r + k, c, c + k))
    return np.array(coords, dtype=np.int32)


# -----------------------------
# Per-coordinate Adam (as in Alg.2)
# β1=0.9, β2=0.999, ε=1e-8  (paper)
# -----------------------------
class _Adam1D:
    __slots__ = ("m", "v", "t", "beta1", "beta2", "eps")

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.m = 0.0
        self.v = 0.0
        self.t = 0
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def step(self, g: float, lr: float) -> float:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)
        m_hat = self.m / (1.0 - (self.beta1 ** self.t))
        v_hat = self.v / (1.0 - (self.beta2 ** self.t))
        return -float(lr) * m_hat / ((v_hat ** 0.5) + self.eps)


@dataclass
class AtkScopesAttack:
    """
    ATKSCOPES-style multiresolution coordinate descent with per-coordinate Adam.

    We adapt it to your oracle setting:
      - oracle.query(x) returns a distance to target (lower is better)
      - success when best_dist <= threshold

    Modes:
      - scale="pixel": update one pixel-channel coordinate in image space (NeuralHash scale)
      - scale="global": update one DCT coefficient over full image (pHash/PDQ scale)
      - scale="mid": update one local (patch) DCT coefficient mapped back to pixels (PhotoDNA-ish scale)

    Paper alignment:
      - pick a coordinate i uniformly at random each round (Alg.2 line 3)
      - estimate gradient by symmetric difference quotient (Alg.2 line 5)
      - update that coordinate via Adam with β1=0.9, β2=0.999, ε=1e-8 (Alg.2)
    """
    scale: str = "global"          # "pixel" | "global" | "mid"
    lr: float = 0.05               # γ in Alg.2 (learning rate)
    a: float = 0.1                 # finite-diff step a (Alg.2)
    patch_size: Optional[int] = None  # for "mid": default h//4 per paper heuristic

    # Adam hyperparams (paper)
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    spec: AttackSpec = AttackSpec(
        attack_id="atkscopes",
        name="ATKSCOPES",
        description="Multiresolution coordinate-descent (random coordinate + symmetric FD + per-coordinate Adam).",
    )

    def run(self, x0: np.ndarray, oracle, budget) -> AttackResult:
        x0 = _as_float01(x0)
        seed = _get_seed(budget)
        rng = np.random.default_rng(seed)

        thr = _get_threshold(oracle)
        if thr is None:
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

        H, W, C = x0.shape
        k = self.patch_size or max(H // 4, 8)  # paper suggests k ~ n/4 for PhotoDNA setting :contentReference[oaicite:5]{index=5}

        # current state
        best_x = oracle.project(x0)
        best_dist = float(oracle.query(best_x))

        # internal representation of perturbation
        if self.scale == "pixel":
            state = _PixelState(delta=np.zeros_like(x0, dtype=np.float32))
        elif self.scale == "global":
            state = _GlobalDCTState.from_x0(x0)
        elif self.scale == "mid":
            state = _MidDCTState.from_x0(x0, k=k)
        else:
            raise ValueError("scale must be one of: 'pixel', 'global', 'mid'")

        # per-coordinate Adam states
        adam = {}  # key -> _Adam1D

        history: Dict[str, Any] = {
            "iter": [],
            "elapsed_ms": [],
            "dist": [],
            "best_dist": [],
            "rmse": [],
            "coord": [],          # coordinate identifier (varies by scale)
            "fp": [],             # probe f(δ+ae_i)
            "fn": [],             # probe f(δ-ae_i)
        }

        it = 0
        while oracle.budget_ok() and best_dist > thr:
            elapsed_ms = (perf_counter() - t0) * 1000.0

            # --- pick random coordinate (Alg.2 line 3) ---
            key, plus_img, minus_img, apply_step = state.make_probes(
                x0=x0,
                oracle=oracle,
                rng=rng,
                a=float(self.a),
                keyspace=adam,  # just to allow key compatibility; not required
            )

            # --- symmetric FD gradient (Alg.2 line 5) ---
            fp = float(oracle.query(plus_img))
            if not oracle.budget_ok():
                break
            fn = float(oracle.query(minus_img))

            g = (fp - fn) / (2.0 * float(self.a))

            opt = adam.get(key)
            if opt is None:
                opt = _Adam1D(beta1=self.beta1, beta2=self.beta2, eps=self.eps)
                adam[key] = opt

            step = opt.step(g, self.lr)
            # apply coordinate update
            x_new = apply_step(step)
            # query actual updated point (important fix for correct tracking)
            dist_new = float(oracle.query(x_new))

            if dist_new < best_dist:
                best_dist = dist_new
                best_x = x_new.copy()

            history["iter"].append(it)
            history["elapsed_ms"].append(float(elapsed_ms))
            history["dist"].append(dist_new)
            history["best_dist"].append(best_dist)
            history["rmse"].append(_rmse(x0, best_x))
            history["coord"].append(key)
            history["fp"].append(fp)
            history["fn"].append(fn)

            it += 1

        total_ms = (perf_counter() - t0) * 1000.0
        success = bool(best_dist <= thr)

        return AttackResult(
            success=success,
            best_dist=float(best_dist),
            queries_used=int(getattr(oracle, "queries_used", 0)),
            best_x=best_x,
            history=history,
            extra={
                "seed": seed,
                "threshold": thr,
                "scale": self.scale,
                "lr": float(self.lr),
                "a": float(self.a),
                "patch_size": int(k),
                "beta1": float(self.beta1),
                "beta2": float(self.beta2),
                "eps": float(self.eps),
                "runtime_ms": float(total_ms),
                "best_rmse": float(_rmse(x0, best_x)),
            },
        )


# -----------------------------
# Internal states per scale
# -----------------------------
class _PixelState:
    __slots__ = ("delta",)

    def __init__(self, delta: np.ndarray):
        self.delta = delta

    def make_probes(self, x0, oracle, rng, a: float, keyspace):
        H, W, C = x0.shape
        i = int(rng.integers(0, H))
        j = int(rng.integers(0, W))
        ch = int(rng.integers(0, C))
        key = ("px", i, j, ch)

        x_cur = oracle.project(x0 + self.delta)

        e = np.zeros_like(self.delta)
        e[i, j, ch] = a

        plus_img = oracle.project(x_cur + e)
        minus_img = oracle.project(x_cur - e)

        def apply_step(step: float):
            self.delta[i, j, ch] += float(step)
            return oracle.project(x0 + self.delta)

        return key, plus_img, minus_img, apply_step


class _GlobalDCTState:
    __slots__ = ("coeffs", "delta")

    def __init__(self, coeffs, delta):
        self.coeffs = coeffs  # list[C] of (H,W)
        self.delta = delta    # list[C] of (H,W)

    @classmethod
    def from_x0(cls, x0: np.ndarray) -> "_GlobalDCTState":
        H, W, C = x0.shape
        coeffs = [_dct2(x0[:, :, ch]) for ch in range(C)]
        delta = [np.zeros((H, W), dtype=np.float32) for _ in range(C)]
        return cls(coeffs, delta)

    def _build(self, oracle, ch: int) -> np.ndarray:
        # reconstruct full image from current coeffs+delta
        C = len(self.coeffs)
        chs = [_idct2(self.coeffs[k] + self.delta[k]).astype(np.float32) for k in range(C)]
        return oracle.project(np.stack(chs, axis=2))

    def make_probes(self, x0, oracle, rng, a: float, keyspace):
        H, W, C = x0.shape
        fi = int(rng.integers(0, H))
        fj = int(rng.integers(0, W))
        ch = int(rng.integers(0, C))
        key = ("dctG", fi, fj, ch)

        # probes modify only one coefficient in one channel
        # build images by temporarily adding +/- a at that coefficient
        old = float(self.delta[ch][fi, fj])

        self.delta[ch][fi, fj] = old + a
        plus_img = self._build(oracle, ch)
        self.delta[ch][fi, fj] = old - a
        minus_img = self._build(oracle, ch)
        self.delta[ch][fi, fj] = old  # restore

        def apply_step(step: float):
            self.delta[ch][fi, fj] += float(step)
            return self._build(oracle, ch)

        return key, plus_img, minus_img, apply_step


class _MidDCTState:
    __slots__ = ("k", "patches", "delta")

    def __init__(self, k: int, patches: np.ndarray, delta: np.ndarray):
        self.k = int(k)
        self.patches = patches  # (P,4) list of (r0,r1,c0,c1)
        self.delta = delta      # pixel-space accumulated perturbation (H,W,C)

    @classmethod
    def from_x0(cls, x0: np.ndarray, k: int) -> "_MidDCTState":
        H, W, _ = x0.shape
        patches = _patch_coords(H, W, k)
        if patches.size == 0:
            # fallback to single patch = full image
            patches = np.array([[0, H, 0, W]], dtype=np.int32)
        delta = np.zeros_like(x0, dtype=np.float32)
        return cls(k=k, patches=patches, delta=delta)

    def make_probes(self, x0, oracle, rng, a: float, keyspace):
        H, W, C = x0.shape
        pidx = int(rng.integers(0, len(self.patches)))
        r0, r1, c0, c1 = [int(v) for v in self.patches[pidx]]
        ph, pw = (r1 - r0), (c1 - c0)

        fi = int(rng.integers(0, ph))
        fj = int(rng.integers(0, pw))
        ch = int(rng.integers(0, C))
        key = ("dctM", pidx, fi, fj, ch)

        # Create pixel-space basis from a single local DCT coefficient
        coeff = np.zeros((ph, pw), dtype=np.float32)
        coeff[fi, fj] = 1.0
        basis = _idct2(coeff).astype(np.float32)
        # normalize basis so that 'a' behaves as a consistent probe size
        nrm = float(np.linalg.norm(basis.ravel(), ord=2))
        if nrm > 1e-12:
            basis = basis / nrm

        x_cur = oracle.project(x0 + self.delta)

        def _apply(sign: float) -> np.ndarray:
            x = x_cur.copy()
            patch = x[r0:r1, c0:c1, ch]
            x[r0:r1, c0:c1, ch] = np.clip(patch + sign * a * basis, 0.0, 1.0)
            return oracle.project(x)

        plus_img = _apply(+1.0)
        minus_img = _apply(-1.0)

        def apply_step(step: float):
            # Apply step in the same basis direction to the stored delta
            self.delta[r0:r1, c0:c1, ch] += float(step) * basis
            return oracle.project(x0 + self.delta)

        return key, plus_img, minus_img, apply_step