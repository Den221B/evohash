# Inlined EvoHash seed attack: atkscopes
# Generated for evolution so the LLM can mutate the actual attack logic.
# The evaluator calls run_attack(x_source, oracle, params, budget, seed).
"""Fast ATKScopes-style DCT coordinate attacks in [0, 1]."""
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
    'a': 22.70446859111565,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-08,
    'lr': 3.0071889520628323,
    'max_freq': 8,
    'scale': 'global',
    'refine_iters': 10,
    'gray_refine_iters': 5,
}


def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _raw_l2_01(a: np.ndarray, b: np.ndarray) -> float:
    diff = (to_float32(a) - to_float32(b)).astype(np.float32).ravel()
    return float(np.linalg.norm(diff, ord=2))


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

    pHash uses max_freq=8 low-frequency coefficients from 32x32 grayscale DCT.
    """

    def __init__(self, x0: np.ndarray, max_freq: int) -> None:
        self.height, self.width, self.channels = x0.shape
        self.max_freq_h = max(2, min(int(max_freq), self.height))
        self.max_freq_w = max(2, min(int(max_freq), self.width))
        self.coeffs = [_dct2(x0[:, :, ch]).astype(np.float32) for ch in range(self.channels)]
        self.delta = [np.zeros((self.height, self.width), dtype=np.float32) for _ in range(self.channels)]

    def build(self) -> np.ndarray:
        channels = [
            _idct2(self.coeffs[ch] + self.delta[ch]).astype(np.float32)
            for ch in range(self.channels)
        ]
        return np.clip(np.stack(channels, axis=2), 0.0, 1.0)

    def make_probes(self, rng: np.random.Generator, a: float):
        while True:
            freq_i = int(rng.integers(self.max_freq_h))
            freq_j = int(rng.integers(self.max_freq_w))
            if freq_i != 0 or freq_j != 0:
                break
        channel = int(rng.integers(self.channels))
        key = ("dctG", freq_i, freq_j, channel)
        old = float(self.delta[channel][freq_i, freq_j])

        self.delta[channel][freq_i, freq_j] = old + a
        plus = self.build()
        self.delta[channel][freq_i, freq_j] = old - a
        minus = self.build()
        self.delta[channel][freq_i, freq_j] = old

        def apply_step(step: float) -> np.ndarray:
            self.delta[channel][freq_i, freq_j] += step
            return self.build()

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


def run_attack(
    x_source: np.ndarray,
    oracle: HashOracle,
    params: Optional[dict[str, Any]] = None,
    budget: int | None = 10_000,
    seed: int = 0,
) -> AttackRawResult:
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    a = float(cfg.get("a", 1.5))
    lr = float(cfg.get("lr", 1.0))
    max_freq = int(cfg.get("max_freq", 16))
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("eps", 1e-8))
    log_every = int(cfg.get("log_every", 50))
    refine_iters = int(cfg.get("refine_iters", 10))
    gray_refine_iters = int(cfg.get("gray_refine_iters", 5))
    max_iters = resolve_max_iters(cfg, budget=budget, queries_per_iter=3)

    rng = np.random.default_rng(int(cfg.get("seed", seed)))
    x0 = to_float32(x_source).astype(np.float32)
    if x0.ndim == 2:
        x0 = np.stack([x0, x0, x0], axis=-1)
    if x0.shape[-1] == 4:
        x0 = x0[..., :3]

    state = GlobalDCTState(x0, max_freq=max_freq)
    adam: dict[tuple[Any, ...], Adam1D] = {}
    best = x0.copy()
    best_dist = float(oracle.query(x0))
    history = [best_dist]
    trace = AttackTrace()
    n_steps = 0
    n_grad_zero = 0
    n_improved = 0
    threshold_solutions: list[tuple[np.ndarray, float]] = []

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

        # Track valid threshold solutions for later L2-optimized rollback
        if best_dist <= oracle.threshold:
            threshold_solutions.append((best.copy(), best_dist))

        # Early stop within iteration if threshold reached after step query
        if best_dist <= oracle.threshold:
            history.append(best_dist)
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

    # Post-threshold L2 refinement: binary search from min-L2 solution toward source
    if threshold_solutions and oracle.budget_ok():
        source = to_float32(x_source)
        # Start refinement from the minimum-L2 valid solution, not the last one
        best_x, best_dist = min(
            threshold_solutions, key=lambda item: np.linalg.norm(item[0] - source)
        )
        delta_to_source = source - best_x
        delta_norm = float(np.linalg.norm(delta_to_source))
        if delta_norm > 1e-12:
            lo, hi = 0.0, 1.0
            for _ in range(refine_iters):
                if not oracle.budget_ok():
                    break
                mid = (lo + hi) / 2.0
                candidate = np.clip(best_x + mid * delta_to_source, 0.0, 1.0)
                dist = float(oracle.query(candidate))
                if dist <= oracle.threshold:
                    best_x = candidate.copy()
                    best_dist = dist
                    lo = mid  # Valid: move further toward source
                else:
                    hi = mid  # Invalid: shrink step toward best_x
                if hi - lo < 1e-6:
                    break
            threshold_solutions.append((best_x.copy(), best_dist))

        # Phase 2: DCT-coordinate L2 shrink — reduce delta coefficients toward zero
        # Process largest coefficients first (they contribute most to L2 distortion).
        # Cascading shrink: each successful reduction enables better baseline for next.
        if isinstance(state, GlobalDCTState):
            # Collect all non-DC coefficients with their magnitudes, sort descending
            coeff_list: list[tuple[float, int, int, int]] = []  # (abs_val, ch, fi, fj)
            for ch in range(state.channels):
                for fi in range(state.max_freq_h):
                    for fj in range(state.max_freq_w):
                        if fi == 0 and fj == 0:
                            continue
                        val = float(state.delta[ch][fi, fj])
                        if abs(val) > 1e-12:
                            coeff_list.append((abs(val), ch, fi, fj))
            coeff_list.sort(reverse=True)  # Largest magnitude first

            for _, ch, fi, fj in coeff_list:
                if not oracle.budget_ok():
                    break
                old_val = float(state.delta[ch][fi, fj])
                if abs(old_val) < 1e-12:
                    continue
                # Try 10% reduction first (very aggressive for L2 savings)
                state.delta[ch][fi, fj] = old_val * 0.1
                candidate = state.build()
                dist = float(oracle.query(candidate))
                if dist <= oracle.threshold:
                    # 10% succeeded — keep it
                    best_x = candidate.copy()
                    best_dist = dist
                else:
                    # Fall back to 50% reduction
                    state.delta[ch][fi, fj] = old_val * 0.5
                    candidate = state.build()
                    dist = float(oracle.query(candidate))
                    if dist <= oracle.threshold:
                        best_x = candidate.copy()
                        best_dist = dist
                    else:
                        # Neither worked — restore original
                        state.delta[ch][fi, fj] = old_val
            threshold_solutions.append((best_x.copy(), best_dist))

        # Phase 3: Grayscale-preserving L2 reduction with binary search
        # Project source onto the grayscale hyperplane of best_x, then binary search
        # along the chroma-only direction to find minimum-L2 valid point.
        if threshold_solutions and oracle.budget_ok():
            source = to_float32(x_source)
            best_x, best_dist = min(
                threshold_solutions, key=lambda item: np.linalg.norm(item[0] - source)
            )
            # ITU-R BT.601 luminance weights (common in pHash implementations)
            _w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            _w_norm_sq = float(np.sum(_w ** 2))
            gray_best = np.sum(best_x * _w[np.newaxis, np.newaxis, :], axis=2, keepdims=True)
            gray_src = np.sum(source * _w[np.newaxis, np.newaxis, :], axis=2, keepdims=True)
            _diff = gray_best - gray_src
            x_gray_proj = source + (_diff / _w_norm_sq) * _w[np.newaxis, np.newaxis, :]
            x_gray_proj = np.clip(x_gray_proj, 0.0, 1.0)
            # Binary search along the direction from best_x toward grayscale projection
            lo, hi = 0.0, 1.0
            for _ in range(gray_refine_iters):
                if not oracle.budget_ok():
                    break
                mid = (lo + hi) / 2.0
                candidate = np.clip(best_x + mid * (x_gray_proj - best_x), 0.0, 1.0)
                dist = float(oracle.query(candidate))
                if dist <= oracle.threshold:
                    best_x = candidate.copy()
                    best_dist = dist
                    lo = mid  # Valid: move further toward projection
                else:
                    hi = mid  # Invalid: shrink step toward best_x
                if hi - lo < 1e-6:
                    break
            threshold_solutions.append((best_x.copy(), best_dist))

    # Rollback to the valid solution with minimum L2 distortion
    if threshold_solutions:
        source = to_float32(x_source)
        best, best_dist = min(
            threshold_solutions, key=lambda item: np.linalg.norm(item[0] - source)
        )

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
