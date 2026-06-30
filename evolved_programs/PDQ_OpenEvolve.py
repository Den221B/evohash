# Inlined EvoHash seed attack: atkscopes
# Generated for evolution so the LLM can mutate the actual attack logic.
# The evaluator calls run_attack(x_source, oracle, params, budget, seed).
"""Fast ATKScopes-style coordinate attacks in [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.fft import dct, idct

from evohash.attacks.base import AttackRawResult, resolve_max_iters
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = 'atkscopes'

DEFAULT_PARAMS: dict[str, Any] = {
    'a': 4.498432179443098,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-08,
    'lr': 2.2240806832096887,
    'max_freq': 17,
    'scale': 'global',
    'refine_budget': 250,
}


def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _raw_l2_01(a: np.ndarray, b: np.ndarray) -> float:
    diff = (to_float32(a) - to_float32(b)).astype(np.float32).ravel()
    return float(np.linalg.norm(diff, ord=2))


def _patch_coords(height: int, width: int, patch_size: int) -> list[tuple[int, int, int, int]]:
    coords: list[tuple[int, int, int, int]] = []
    for row in range(0, height - patch_size + 1, patch_size):
        for col in range(0, width - patch_size + 1, patch_size):
            coords.append((row, row + patch_size, col, col + patch_size))
    return coords


class Adam1D:
    __slots__ = ("m", "v", "t", "beta1", "beta2", "eps")

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.m = 0.0
        self.v = 0.0
        self.t = 0
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def step(self, grad: float, lr: float) -> float:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return float(-lr * m_hat / (np.sqrt(v_hat) + self.eps))


class GlobalDCTState:
    """Optimize only the green channel (highest PDQ luminance weight: 0.587).

    PDQ converts to grayscale before DCT, so per-channel RGB optimization wastes
    queries and inflates L2.  Constraining to green keeps R/B untouched (lower L2)
    while still capturing >58% of the luminance signal.
    """

    def __init__(self, x0: np.ndarray, max_freq: int) -> None:
        self.height, self.width, self.channels = x0.shape
        self.max_freq_h = max(2, min(int(max_freq), self.height))
        self.max_freq_w = max(2, min(int(max_freq), self.width))
        self.base = x0.astype(np.float32)
        # Single-channel DCT on green (highest luminance weight)
        self.coeffs = _dct2(x0[:, :, 1]).astype(np.float32)
        self.delta = np.zeros((self.height, self.width), dtype=np.float32)
        # Coordinate importance: EMA of |grad| per (freq_i, freq_j) for sampling bias
        self.coord_importance: dict[tuple[int, int], float] = {}
        # Pre-compute valid sampling keys once (skip DC to avoid trivial brightness shifts)
        self._sample_keys = [
            (i, j) for i in range(self.max_freq_h)
            for j in range(self.max_freq_w)
            if i != 0 or j != 0
        ]

    def build(self) -> np.ndarray:
        green = _idct2(self.coeffs + self.delta).astype(np.float32)
        green = np.clip(green, 0.0, 1.0)
        result = self.base.copy()
        result[:, :, 1] = green
        return result

    def _sample_coord(self, rng: np.random.Generator) -> tuple[int, int]:
        """Importance-weighted sampling biased toward coordinates with high historical |grad|."""
        keys = self._sample_keys
        if not keys:
            while True:
                fi = int(rng.integers(self.max_freq_h))
                fj = int(rng.integers(self.max_freq_w))
                if fi != 0 or fj != 0:
                    break
            return (fi, fj)

        importances = np.array([self.coord_importance.get(k, 1.0) for k in keys], dtype=np.float32)
        # Numerically stable softmax with temperature for exploration/exploitation balance
        temp = 0.5
        importances = importances / temp
        importances = importances - importances.max()  # shift for numerical stability
        importances = np.exp(importances)
        probs = importances / (importances.sum() + 1e-12)
        idx = int(rng.choice(len(keys), p=probs))
        return keys[idx]

    def record_importance(self, key: tuple[int, int], grad_mag: float) -> None:
        # EMA update for coordinate importance
        alpha = 0.3
        old = self.coord_importance.get(key, 1.0)
        self.coord_importance[key] = alpha * grad_mag + (1.0 - alpha) * old

    def make_probes(self, rng: np.random.Generator, a: float):
        freq_i, freq_j = self._sample_coord(rng)
        key = ("dctG", freq_i, freq_j)
        old = float(self.delta[freq_i, freq_j])

        self.delta[freq_i, freq_j] = old + a
        plus = self.build()
        self.delta[freq_i, freq_j] = old - a
        minus = self.build()
        self.delta[freq_i, freq_j] = old

        def apply_step(step: float) -> np.ndarray:
            self.delta[freq_i, freq_j] += step
            return self.build()

        return key, plus, minus, apply_step


class PixelState:
    def __init__(self, x0: np.ndarray) -> None:
        self.x0 = x0.astype(np.float32)
        self.delta = np.zeros_like(self.x0, dtype=np.float32)

    def build(self) -> np.ndarray:
        return np.clip(self.x0 + self.delta, 0.0, 1.0)

    def make_probes(self, rng: np.random.Generator, a: float):
        height, width, channels = self.x0.shape
        i = int(rng.integers(height))
        j = int(rng.integers(width))
        channel = int(rng.integers(channels))
        key = ("px", i, j, channel)
        current = self.build()
        plus = current.copy()
        minus = current.copy()
        plus[i, j, channel] = float(np.clip(plus[i, j, channel] + a, 0.0, 1.0))
        minus[i, j, channel] = float(np.clip(minus[i, j, channel] - a, 0.0, 1.0))

        def apply_step(step: float) -> np.ndarray:
            self.delta[i, j, channel] += step
            return self.build()

        return key, plus, minus, apply_step


class MidDCTState:
    """Patch DCT state for PhotoDNA-style mid-scale ATKScopes."""

    def __init__(self, x0: np.ndarray, patch_size: int, max_freq: int) -> None:
        height, width, _ = x0.shape
        self.x0 = x0.astype(np.float32)
        self.patch_size = int(patch_size)
        self.max_freq = max(1, int(max_freq))
        self.patches = _patch_coords(height, width, self.patch_size)
        if not self.patches:
            self.patches = [(0, height, 0, width)]
        self.delta = np.zeros_like(self.x0, dtype=np.float32)

    def make_probes(self, rng: np.random.Generator, a: float):
        _, _, channels = self.x0.shape
        patch_idx = int(rng.integers(len(self.patches)))
        row0, row1, col0, col1 = self.patches[patch_idx]
        patch_h = row1 - row0
        patch_w = col1 - col0
        freq_h = min(self.max_freq, patch_h)
        freq_w = min(self.max_freq, patch_w)
        freq_i = int(rng.integers(freq_h))
        freq_j = int(rng.integers(freq_w))
        channel = int(rng.integers(channels))
        key = ("dctM", patch_idx, freq_i, freq_j, channel)

        coeff = np.zeros((patch_h, patch_w), dtype=np.float32)
        coeff[freq_i, freq_j] = 1.0
        basis = _idct2(coeff).astype(np.float32)
        norm = float(np.linalg.norm(basis.ravel()))
        if norm > 1e-12:
            basis /= norm

        current = np.clip(self.x0 + self.delta, 0.0, 1.0)
        plus = current.copy()
        minus = current.copy()
        plus[row0:row1, col0:col1, channel] = np.clip(
            plus[row0:row1, col0:col1, channel] + a * basis,
            0.0,
            1.0,
        )
        minus[row0:row1, col0:col1, channel] = np.clip(
            minus[row0:row1, col0:col1, channel] - a * basis,
            0.0,
            1.0,
        )

        def apply_step(step: float) -> np.ndarray:
            self.delta[row0:row1, col0:col1, channel] += step * basis
            return np.clip(self.x0 + self.delta, 0.0, 1.0)

        return key, plus, minus, apply_step


@dataclass
class AttackTrace:
    iters: list[int] = field(default_factory=list)
    best_dist: list[float] = field(default_factory=list)
    cur_dist: list[float] = field(default_factory=list)
    grads: list[float] = field(default_factory=list)
    steps: list[float] = field(default_factory=list)
    l2: list[float] = field(default_factory=list)
    queries: list[int] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)


def _make_state(scale: str, x0: np.ndarray, max_freq: int, patch_size: int | None):
    if scale == "global":
        return GlobalDCTState(x0, max_freq=max_freq)
    if scale == "pixel":
        return PixelState(x0)
    if scale == "mid":
        patch = int(patch_size or max(x0.shape[0] // 4, 8))
        return MidDCTState(x0, patch_size=patch, max_freq=max_freq)
    raise ValueError(f"Unsupported ATKScopes scale={scale!r}; expected 'global', 'mid', or 'pixel'")


def run_attack(
    x_source: np.ndarray,
    oracle: HashOracle,
    params: Optional[dict[str, Any]] = None,
    budget: int | None = 10_000,
    seed: int = 0,
) -> AttackRawResult:
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    scale = str(cfg.get("scale", "global"))
    a = float(cfg.get("a", 4.498432179443098))
    lr = float(cfg.get("lr", 2.2240806832096887))
    max_freq = int(cfg.get("max_freq", 17))
    patch_size_raw = cfg.get("patch_size", None)
    patch_size = None if patch_size_raw is None else int(patch_size_raw)
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("eps", 1e-8))
    log_every = int(cfg.get("log_every", 50))
    max_iters = resolve_max_iters(cfg, budget=budget, queries_per_iter=3)

    rng = np.random.default_rng(int(cfg.get("seed", seed)))
    x0 = to_float32(x_source).astype(np.float32)
    if x0.ndim == 2:
        x0 = np.stack([x0, x0, x0], axis=-1)
    if x0.shape[-1] == 4:
        x0 = x0[..., :3]

    state = _make_state(scale, x0, max_freq=max_freq, patch_size=patch_size)
    adam: dict[tuple[Any, ...], Adam1D] = {}
    best = x0.copy()
    best_dist = float(oracle.query(x0))
    history = [best_dist]
    trace = AttackTrace()
    n_steps = 0
    n_grad_zero = 0
    n_improved = 0

    for it in range(1, max_iters + 1):
        if best_dist <= oracle.threshold or not oracle.budget_ok():
            break

        key, plus, minus, apply_step = state.make_probes(rng, a)
        f_plus = float(oracle.query(plus))
        if not oracle.budget_ok():
            break
        f_minus = float(oracle.query(minus))
        grad = (f_plus - f_minus) / (2.0 * a + 1e-12)

        if grad == 0.0:
            n_grad_zero += 1
        else:
            # Record coordinate importance for future sampling bias
            if isinstance(state, GlobalDCTState):
                coord_key = (key[1], key[2])  # (freq_i, freq_j)
                state.record_importance(coord_key, abs(grad))

        if grad == 0.0:
            continue

        opt = adam.get(key)
        if opt is None:
            opt = Adam1D(beta1=beta1, beta2=beta2, eps=eps)
            adam[key] = opt

        step = opt.step(grad, lr)
        x_new = apply_step(step)
        new_dist = float(oracle.query(x_new))
        n_steps += 1

        if new_dist < best_dist:
            best = x_new.copy()
            best_dist = new_dist
            n_improved += 1
            # Early stop as soon as threshold is met to save queries
            if best_dist <= oracle.threshold:
                break

        if log_every > 0 and (it % log_every == 0 or it == max_iters):
            trace.iters.append(it)
            trace.best_dist.append(best_dist)
            trace.cur_dist.append(new_dist)
            trace.grads.append(float(grad))
            trace.steps.append(float(step))
            trace.l2.append(_raw_l2_01(best, x0))
            trace.queries.append(int(oracle.queries))
            trace.keys.append(str(key))

        history.append(best_dist)

    # Post-threshold L2 refinement: global scaling first, then greedy zeroing + halving
    # Order matters: global scaling shrinks all coefficients, making subsequent zeroing/halving
    # more likely to succeed. Fewer binary search iterations save queries while still finding
    # a near-minimal scale factor.
    if best_dist <= oracle.threshold and isinstance(state, GlobalDCTState):
        refine_budget = int(cfg.get("refine_budget", 250))
        delta_ch = state.delta  # single green-channel delta

        # Phase 1: Binary search for minimum uniform scaling of delta (do this first)
        if oracle.budget_ok():
            saved_delta = delta_ch.copy()  # snapshot before scaling
            lo, hi = 0.0, 1.0
            for _ in range(8):  # 8 iterations for ~1.5% precision; better L2 reduction
                if refine_budget <= 0 or not oracle.budget_ok():
                    break
                mid = (lo + hi) * 0.5
                delta_ch[:] = saved_delta * mid
                candidate = state.build()
                cand_dist = float(oracle.query(candidate))
                refine_budget -= 1
                if cand_dist <= oracle.threshold:
                    best = candidate.copy()
                    best_dist = cand_dist
                    hi = mid
                else:
                    lo = mid
            # Commit to hi: smallest valid scale gives strictly lower L2 than midpoint
            delta_ch[:] = saved_delta * hi
            best = state.build()

        # Phase 2: Greedy zeroing of smallest coefficients (easier after global scaling)
        flat_idx = np.argsort(np.abs(delta_ch.ravel()))  # smallest magnitude first
        for idx in flat_idx:
            if refine_budget <= 0 or not oracle.budget_ok():
                break
            freq_i, freq_j = np.unravel_index(idx, delta_ch.shape)
            if delta_ch[freq_i, freq_j] == 0.0:
                continue
            saved = float(delta_ch[freq_i, freq_j])
            delta_ch[freq_i, freq_j] = 0.0
            candidate = state.build()
            cand_dist = float(oracle.query(candidate))
            refine_budget -= 1
            if cand_dist <= oracle.threshold:
                best = candidate.copy()
                best_dist = cand_dist
                # keep coefficient zeroed
            else:
                delta_ch[freq_i, freq_j] = saved  # restore

        # Phase 3: Try halving remaining non-zero coefficients (conservative L2 reduction)
        if oracle.budget_ok():
            flat_idx = np.argsort(np.abs(delta_ch.ravel()))  # re-sort after zeroing
            for idx in flat_idx:
                if refine_budget <= 0 or not oracle.budget_ok():
                    break
                freq_i, freq_j = np.unravel_index(idx, delta_ch.shape)
                val = float(delta_ch[freq_i, freq_j])
                if val == 0.0:
                    continue
                saved = val
                delta_ch[freq_i, freq_j] = val * 0.5  # halve instead of zero
                candidate = state.build()
                cand_dist = float(oracle.query(candidate))
                refine_budget -= 1
                if cand_dist <= oracle.threshold:
                    best = candidate.copy()
                    best_dist = cand_dist
                    # keep halved value
                else:
                    delta_ch[freq_i, freq_j] = saved  # restore

        # Phase 4: Zero out numerically tiny coefficients post-all-refinement (free L2)
        mask = np.abs(delta_ch) < 1e-3
        delta_ch[mask] = 0.0
        best = state.build()

        # Phase 5: Redistribute perturbation across RGB channels to minimize L2.
        # PDQ grayscale: 0.299*R + 0.587*G + 0.114*B
        # Optimal L2 distribution for fixed grayscale effect: delta_c proportional to weight_c.
        # This can reduce L2 by ~12% without changing the hash (ignoring clipping effects).
        if refine_budget > 0 and oracle.budget_ok():
            delta_rgb = best - x0
            w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            # Grayscale effect from current perturbation
            delta_gray = (delta_rgb * w[np.newaxis, np.newaxis, :]).sum(axis=-1)
            sum_w_sq = float(np.sum(w ** 2))
            # Optimal redistribution: delta_c = delta_gray * w_c / sum(w^2)
            new_delta = delta_gray[:, :, np.newaxis] * w[np.newaxis, np.newaxis, :] / sum_w_sq
            best_redist = np.clip(x0 + new_delta, 0.0, 1.0).astype(np.float32)
            cand_dist = float(oracle.query(best_redist))
            refine_budget -= 1
            if cand_dist <= oracle.threshold and _raw_l2_01(best_redist, x0) < _raw_l2_01(best, x0):
                best = best_redist
                best_dist = cand_dist

        # Phase 6: Pixel-space linear interpolation toward source for additional L2 reduction.
        # After DCT-domain refinement, probing fixed interpolation steps is more robust
        # than binary search near clipping boundaries and uses fewer queries.
        if refine_budget > 0 and oracle.budget_ok():
            delta_px = best - x0
            for alpha in (0.9, 0.85, 0.8):
                if refine_budget <= 0 or not oracle.budget_ok():
                    break
                x_try = np.clip(x0 + alpha * delta_px, 0.0, 1.0).astype(np.float32)
                cand_dist = float(oracle.query(x_try))
                refine_budget -= 1
                if cand_dist <= oracle.threshold:
                    best = x_try
                    best_dist = cand_dist
                else:
                    break  # monotonic assumption: smaller alpha will also fail

    history_rows = [
        {
            "iter": it,
            "best_hash_l1": dist,
            "hash_l1": cur,
            "grad": grad,
            "step": step,
            "L2": l2,
            "queries": queries,
            "key": key,
        }
        for it, dist, cur, grad, step, l2, queries, key in zip(
            trace.iters,
            trace.best_dist,
            trace.cur_dist,
            trace.grads,
            trace.steps,
            trace.l2,
            trace.queries,
            trace.keys,
        )
    ]

    return AttackRawResult(
        x_best=best.astype(np.float32),
        history=history,
        queries=oracle.queries,
        params=cfg,
        extra={
            "history_rows": history_rows,
            "iters_steps_taken": n_steps,
            "iters_grad_zero": n_grad_zero,
            "iters_improved": n_improved,
            "L2": _raw_l2_01(best, x0),
            "Linf": float(np.max(np.abs(best - x0))),
            "stopped_reason": oracle.state.stopped_reason,
        },
    )
