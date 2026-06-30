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
    'a': 36.272652825354406,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 8.243022130911872e-09,
    'lr': 5.195703445581074,
    'max_freq': 10,
    'scale': 'global',
    'patience': 60,
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

    The working notebooks restrict coordinate sampling to a low-frequency
    window: pHash uses max_freq=8, PDQ uses max_freq=16.
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
    a = float(cfg.get("a", 1.5))
    lr = float(cfg.get("lr", 1.0))
    max_freq = int(cfg.get("max_freq", 16))
    patch_size_raw = cfg.get("patch_size", None)
    patch_size = None if patch_size_raw is None else int(patch_size_raw)
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("eps", 1e-8))
    log_every = int(cfg.get("log_every", 50))
    patience = int(cfg.get("patience", 60))
    # 3 queries per iteration: plus, minus, and updated point
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
    no_improve = 0

    for it in range(1, max_iters + 1):
        if best_dist <= oracle.threshold or not oracle.budget_ok():
            break

        # Adaptive probe step: shrink as we approach threshold to avoid oscillation
        gap = max(best_dist - oracle.threshold, 0.0)
        a_cur = a * max(0.1, min(1.0, 0.3 + 0.7 * gap / max(oracle.threshold, 1.0)))

        key, plus, minus, apply_step = state.make_probes(rng, a_cur)
        f_plus = float(oracle.query(plus))
        if not oracle.budget_ok():
            break
        f_minus = float(oracle.query(minus))
        # Use a_cur (actual probe step) instead of a (base step) for correct gradient scaling
        grad = (f_plus - f_minus) / (2.0 * a_cur + 1e-12)

        if grad == 0.0:
            n_grad_zero += 1
            no_improve += 1
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
            no_improve = 0

        # Early exit once threshold is reliably hit; skip further refinement to save queries
        if best_dist <= oracle.threshold:
            break

        if new_dist >= best_dist:
            no_improve += 1

        # Exit main loop early when near threshold to preserve budget for finishing phase.
        # The multi-coordinate finishing phase is better suited for flipping the last few
        # entangled bits than single-coordinate gradient descent.
        if budget is not None and best_dist <= oracle.threshold + 6 and oracle.queries > budget * 0.45:
            break

        # Restart Adam state when optimization stalls to escape local plateaus
        if no_improve >= patience:
            # Near threshold: try multi-coefficient random restart to escape plateaus
            # where single-coordinate search cannot flip the last few bits.
            # Perturb the CURRENT delta to preserve accumulated progress.
            if (
                best_dist <= oracle.threshold + 12
                and oracle.budget_ok()
            ):
                n_restart = 6
                best_restart = None
                best_restart_dist = best_dist
                restart_a = a * 0.35
                for _ in range(n_restart):
                    if not oracle.budget_ok():
                        break
                    n_coeffs = int(rng.integers(2, 8))
                    # Perturb current delta instead of resetting to zero
                    restart_delta = [
                        state.delta[ch].copy() for ch in range(state.channels)
                    ]
                    for _ in range(n_coeffs):
                        ch = int(rng.integers(state.channels))
                        fi = int(rng.integers(state.max_freq_h))
                        fj = int(rng.integers(state.max_freq_w))
                        restart_delta[ch][fi, fj] += float(rng.standard_normal()) * restart_a
                    channels_restart = [
                        _idct2(state.coeffs[ch] + restart_delta[ch]).astype(np.float32)
                        for ch in range(state.channels)
                    ]
                    x_restart = np.clip(
                        np.stack(channels_restart, axis=2), 0.0, 1.0
                    )
                    d_restart = float(oracle.query(x_restart))
                    if d_restart < best_restart_dist:
                        best_restart_dist = d_restart
                        best_restart = restart_delta
                if best_restart is not None:
                    state.delta = best_restart
                    best = np.clip(
                        np.stack([
                            _idct2(state.coeffs[ch] + state.delta[ch]).astype(np.float32)
                            for ch in range(state.channels)
                        ], axis=2),
                        0.0,
                        1.0,
                    )
                    best_dist = best_restart_dist
                    adam.clear()
                    no_improve = 0
                else:
                    adam.clear()
                    no_improve = 0
            else:
                adam.clear()
                no_improve = 0

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

    # Finishing phase: accumulating multi-coordinate DCT perturbations for remaining gap.
    # Successful perturbations persist in an accumulator, allowing the search to
    # compound progress toward threshold. Failed perturbations are reverted.
    # This is more effective than revert-on-failure for NeuralHash entangled bits.
    if best_dist > oracle.threshold and hasattr(state, "coeffs"):
        remaining = (budget - oracle.queries) if budget is not None else 500
        if remaining > 30:
            gap = best_dist - oracle.threshold
            # Scale sigma by gap: larger when far, smaller when close to threshold.
            # Higher floor when gap is small to ensure perturbations can flip stubborn bits.
            finish_sigma = a * 0.4 * max(0.3, min(1.5, gap / max(oracle.threshold, 1.0)))
            finish_n = min(max(1, remaining // 2), 600)
            finish_no_improve = 0
            finish_patience = 60  # Allow more attempts when very close to threshold
            # Accumulate delta separately so accepted perturbations persist
            accum_delta = [state.delta[ch].copy() for ch in range(state.channels)]
            for _ in range(finish_n):
                if not oracle.budget_ok() or best_dist <= oracle.threshold:
                    break
                if finish_no_improve >= finish_patience:
                    break
                # Perturb more coefficients when gap is tiny to increase chance of flipping last bits
                if gap <= 3:
                    n_c = int(rng.integers(3, 7))  # 3-6 coefficients for stubborn bits
                else:
                    n_c = int(rng.integers(2, 6))  # 2-5 coefficients otherwise
                perturbations = []
                for _ in range(n_c):
                    fi = int(rng.integers(state.max_freq_h))
                    fj = int(rng.integers(state.max_freq_w))
                    ch = int(rng.integers(state.channels))
                    delta_val = float(finish_sigma * rng.standard_normal())
                    accum_delta[ch][fi, fj] += delta_val
                    perturbations.append((ch, fi, fj, delta_val))
                # Build candidate from accumulated delta
                channels = [
                    _idct2(state.coeffs[ch] + accum_delta[ch]).astype(np.float32)
                    for ch in range(state.channels)
                ]
                cand = np.clip(np.stack(channels, axis=2), 0.0, 1.0)
                if not oracle.budget_ok():
                    for ch, fi, fj, delta_val in perturbations:
                        accum_delta[ch][fi, fj] -= delta_val
                    break
                d = float(oracle.query(cand))
                if d < best_dist:
                    best = cand.copy()
                    best_dist = d
                    # Keep perturbations: accumulate progress toward threshold
                    gap = best_dist - oracle.threshold
                    finish_no_improve = 0
                    # Anneal sigma on success for finer control near threshold
                    # Higher floor when gap is small to maintain bit-flipping power
                    finish_sigma = max(6.0 if gap <= 3 else 3.0, finish_sigma * 0.94)
                else:
                    # Revert only failed perturbations
                    for ch, fi, fj, delta_val in perturbations:
                        accum_delta[ch][fi, fj] -= delta_val
                    finish_no_improve += 1

    # Post-threshold L2 refinement: shrink delta coefficients to reduce distortion
    # while maintaining hash collision. Try reducing largest deltas first.
    if best_dist <= oracle.threshold and oracle.budget_ok() and hasattr(state, "coeffs"):
        shrink_budget = min(300, max(1, (budget - oracle.queries) // 2))
        shrink_iters = 0
        # Collect all modified AC coefficients sorted by |delta| descending
        modified_coords = []
        for ch in range(state.channels):
            for fi in range(1, state.max_freq_h):
                for fj in range(1, state.max_freq_w):
                    val = float(state.delta[ch][fi, fj])
                    if abs(val) > 1e-8:
                        modified_coords.append((abs(val), ch, fi, fj, val))
        modified_coords.sort(reverse=True)  # try shrinking largest deltas first

        for _, ch, fi, fj, val in modified_coords:
            if not oracle.budget_ok() or shrink_iters >= shrink_budget:
                break
            # Try reducing this coefficient by 15%
            old_val = state.delta[ch][fi, fj]
            state.delta[ch][fi, fj] = old_val * 0.85
            candidate = state.build()
            d = float(oracle.query(candidate))
            shrink_iters += 1
            if d <= oracle.threshold:
                # Keep the shrinkage; try more aggressive reduction
                for _ in range(4):
                    if not oracle.budget_ok():
                        break
                    state.delta[ch][fi, fj] *= 0.85
                    candidate = state.build()
                    d2 = float(oracle.query(candidate))
                    shrink_iters += 1
                    if d2 <= oracle.threshold:
                        best = candidate.copy()
                        best_dist = d2
                    else:
                        state.delta[ch][fi, fj] /= 0.85  # revert last
                        break
            else:
                state.delta[ch][fi, fj] = old_val  # revert

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
