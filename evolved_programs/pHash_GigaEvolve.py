from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.fft import dct, idct

from evohash.attacks.base import AttackRawResult, resolve_max_iters
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = 'atkscopes_v4_deep_refine'

DEFAULT_PARAMS: dict[str, Any] = {
    'a': 15.0,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-08,
    'lr': 3.0071889520628323,
    'max_freq': 8,
    'scale': 'global',
    'decay': 0.998,
    'bs_steps': 4,
    'refinement_budget': 600,
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
    """Image = base DCT coefficients + optimized DCT delta.

    Modified: grayscale-aware (all channels equal), round-robin coordinate scheduling.
    """

    def __init__(self, x0: np.ndarray, max_freq: int) -> None:
        self.height, self.width, self.channels = x0.shape
        self.max_freq_h = max(2, min(int(max_freq), self.height))
        self.max_freq_w = max(2, min(int(max_freq), self.width))
        self.coeffs = [_dct2(x0[:, :, ch]).astype(np.float32) for ch in range(self.channels)]
        self.delta = [np.zeros((self.height, self.width), dtype=np.float32) for _ in range(self.channels)]
        self._coord_idx = 0  # Round-robin counter

    def build(self) -> np.ndarray:
        channels = [
            _idct2(self.coeffs[ch] + self.delta[ch]).astype(np.float32)
            for ch in range(self.channels)
        ]
        return np.clip(np.stack(channels, axis=2), 0.0, 1.0)

    def make_probes(self, a: float):
        # Round-robin over low-frequency coefficients (skip DC)
        total = self.max_freq_h * self.max_freq_w
        idx = self._coord_idx % total
        self._coord_idx += 1
        freq_i = idx // self.max_freq_w
        freq_j = idx % self.max_freq_w

        # Skip DC coefficient (0, 0)
        if freq_i == 0 and freq_j == 0:
            idx = self._coord_idx % total
            self._coord_idx += 1
            freq_i = idx // self.max_freq_w
            freq_j = idx % self.max_freq_w

        # Apply to ALL channels equally (grayscale-aware: pHash converts to grayscale)
        key = ("dctG", freq_i, freq_j)
        old = [float(self.delta[ch][freq_i, freq_j]) for ch in range(self.channels)]

        # Plus probe
        for ch in range(self.channels):
            self.delta[ch][freq_i, freq_j] = old[ch] + a
        plus = self.build()

        # Minus probe
        for ch in range(self.channels):
            self.delta[ch][freq_i, freq_j] = old[ch] - a
        minus = self.build()

        # Restore
        for ch in range(self.channels):
            self.delta[ch][freq_i, freq_j] = old[ch]

        def apply_step(step: float) -> np.ndarray:
            for ch in range(self.channels):
                self.delta[ch][freq_i, freq_j] += step
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
    a = float(cfg.get("a", 15.0))
    lr = float(cfg.get("lr", 1.0))
    max_freq = int(cfg.get("max_freq", 16))
    patch_size_raw = cfg.get("patch_size", None)
    patch_size = None if patch_size_raw is None else int(patch_size_raw)
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("eps", 1e-8))
    log_every = int(cfg.get("log_every", 50))
    grad_clip = float(cfg.get("grad_clip", 5.0))
    decay = float(cfg.get("decay", 0.998))
    bs_steps = int(cfg.get("bs_steps", 4))
    refinement_budget = int(cfg.get("refinement_budget", 600))
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
    # Track delta corresponding to best for L2 refinement
    best_delta = None
    if hasattr(state, 'delta') and isinstance(state.delta, list):
        best_delta = [np.zeros_like(state.delta[ch]) for ch in range(len(state.delta))]
    history = [best_dist]
    trace = AttackTrace()
    n_steps = 0
    n_grad_zero = 0
    n_improved = 0

    for it in range(1, max_iters + 1):
        if best_dist <= oracle.threshold or oracle.queries >= budget:
            break

        # Anneal step size and learning rate
        a_current = a * (decay ** it)
        lr_current = lr * (decay ** it)

        if scale == "global":
            key, plus, minus, apply_step = state.make_probes(a_current)
        else:
            key, plus, minus, apply_step = state.make_probes(rng, a_current)

        f_plus = float(oracle.query(plus))
        if oracle.queries >= budget:
            break
        f_minus = float(oracle.query(minus))
        grad = (f_plus - f_minus) / (2.0 * a_current + 1e-12)

        # Clip gradient to prevent Adam drift on sparse discrete signals
        grad = float(np.clip(grad, -grad_clip, grad_clip))

        if grad == 0.0:
            n_grad_zero += 1
            continue

        opt = adam.get(key)
        if opt is None:
            opt = Adam1D(beta1=beta1, beta2=beta2, eps=eps)
            adam[key] = opt

        step = opt.step(grad, lr_current)
        x_new = apply_step(step)
        new_dist = float(oracle.query(x_new))
        n_steps += 1

        # Early stop: break immediately if threshold is reached
        if new_dist <= oracle.threshold:
            best = x_new.copy()
            best_dist = new_dist
            if best_delta is not None:
                best_delta = [d.copy() for d in state.delta]
            break

        if new_dist < best_dist:
            best = x_new.copy()
            best_dist = new_dist
            if best_delta is not None:
                best_delta = [d.copy() for d in state.delta]
            n_improved += 1

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

    # L2 refinement phase: per-coefficient binary search with safe ordering
    # DESCENDING sort: largest L2 contribution first (most distortion to reduce)
    if best_dist <= oracle.threshold and oracle.queries < budget and best_delta is not None:
        # Restore the delta that corresponds to best
        for ch in range(len(state.delta)):
            state.delta[ch] = best_delta[ch].copy()

        # Collect non-zero coefficients sorted by L2 contribution (LARGEST first)
        contributions = []
        for fi in range(state.max_freq_h):
            for fj in range(state.max_freq_w):
                if fi == 0 and fj == 0:
                    continue
                val = state.delta[0][fi, fj]
                if abs(val) > 1e-6:
                    l2_contrib = val ** 2 * len(state.delta)
                    contributions.append((fi, fj, val, l2_contrib))

        # Sort DESCENDING: largest L2 contribution first (biggest distortion to reduce)
        contributions.sort(key=lambda x: x[3], reverse=True)

        queries_at_start = oracle.queries

        for fi, fj, orig_val, _ in contributions:
            if oracle.queries >= budget:
                break
            if oracle.queries - queries_at_start >= refinement_budget:
                break

            sign = np.sign(orig_val)
            low = 0.0
            high = abs(orig_val)

            for _ in range(bs_steps):
                if oracle.queries >= budget:
                    break
                if oracle.queries - queries_at_start >= refinement_budget:
                    break

                mid = (low + high) / 2.0
                if mid < 1e-10:
                    break

                for ch in range(len(state.delta)):
                    state.delta[ch][fi, fj] = sign * mid
                x_test = state.build()
                d = float(oracle.query(x_test))

                if d <= oracle.threshold:
                    high = mid
                else:
                    low = mid

            # Set coefficient to the smallest safe value
            for ch in range(len(state.delta)):
                state.delta[ch][fi, fj] = sign * high

            # Per-coefficient verification: check if hash still holds after this reduction
            x_test = state.build()
            d = float(oracle.query(x_test))
            if d > oracle.threshold:
                # Revert this coefficient to original value and continue
                for ch in range(len(state.delta)):
                    state.delta[ch][fi, fj] = orig_val

        # Final safety check
        x_test = state.build()
        d = float(oracle.query(x_test))
        if d <= oracle.threshold:
            best = x_test.copy()
            best_dist = d
        else:
            # Restore original best_delta
            for ch in range(len(state.delta)):
                state.delta[ch] = best_delta[ch].copy()
            best = state.build()
            best_dist = float(oracle.query(best))

        history.append(best_dist)

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


from evohash.stage3_gigaevolve.runtime import evaluate_entrypoint_attack


def entrypoint(context: dict) -> dict:
    return evaluate_entrypoint_attack(
        run_attack,
        context,
        params=DEFAULT_PARAMS,
        attack_id=ATTACK_ID,
    )