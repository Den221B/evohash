from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.fft import dct, idct

from evohash.attacks.base import AttackRawResult, resolve_max_iters
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = 'atkscopes_centered_adam_restart_v2'

DEFAULT_PARAMS: dict[str, Any] = {
    'a': 4.443566509056172,
    'beta1': 0.9,
    'beta2': 0.999,
    'lr': 3.855169125841428,
    'max_freq': 5,
    'patch_size': 128,
    'scale': 'mid',
    'stall_threshold': 150,
    'restart_delta_scale': 0.3,
}


def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _raw_l2_01(a: np.ndarray, b: np.ndarray) -> float:
    diff = (to_float32(a) - to_float32(b)).astype(np.float32).ravel()
    return float(np.linalg.norm(diff, ord=2))


def _centered_square_bounds(height: int, width: int) -> tuple[int, int, int, int]:
    """Compute bounds of the largest centered square crop (PhotoDNA preprocessing)."""
    side = min(height, width)
    row0 = (height - side) // 2
    col0 = (width - side) // 2
    return row0, row0 + side, col0, col0 + side


def _centered_patch_coords(height: int, width: int, patch_size: int) -> list[tuple[int, int, int, int]]:
    """Generate patch coordinates restricted to the centered square crop region."""
    row0, row1, col0, col1 = _centered_square_bounds(height, width)
    coords: list[tuple[int, int, int, int]] = []
    for r in range(row0, row1 - patch_size + 1, patch_size):
        for c in range(col0, col1 - patch_size + 1, patch_size):
            coords.append((r, r + patch_size, c, c + patch_size))
    if not coords:
        coords.append((row0, row1, col0, col1))
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


class CenteredMidDCTState:
    """Patch DCT state restricted to PhotoDNA's centered square crop region."""

    def __init__(self, x0: np.ndarray, patch_size: int, max_freq: int) -> None:
        height, width, _ = x0.shape
        self.x0 = x0.astype(np.float32)
        self.patch_size = int(patch_size)
        self.max_freq = max(1, int(max_freq))
        self.patches = _centered_patch_coords(height, width, self.patch_size)
        self.delta = np.zeros_like(self.x0, dtype=np.float32)
        self._last_applied: Optional[tuple] = None

    def build(self) -> np.ndarray:
        return np.clip(self.x0 + self.delta, 0.0, 1.0)

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

        current = self.build()
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

        # Store last applied info for safe undo
        self._last_applied = (patch_idx, freq_i, freq_j, channel, basis)

        def apply_step(step: float) -> np.ndarray:
            self.delta[row0:row1, col0:col1, channel] += step * basis
            return np.clip(self.x0 + self.delta, 0.0, 1.0)

        return key, plus, minus, apply_step

    def restart(self, rng: np.random.Generator, scale: float) -> None:
        """Reset deltas partially and preserve Adam state by only scaling deltas."""
        self.delta[:] *= scale


class GlobalDCTState:
    """Image = base DCT coefficients + optimized DCT delta (full image)."""

    def __init__(self, x0: np.ndarray, max_freq: int) -> None:
        self.height, self.width, self.channels = x0.shape
        self.max_freq_h = max(2, min(int(max_freq), self.height))
        self.max_freq_w = max(2, min(int(max_freq), self.width))
        self.coeffs = [_dct2(x0[:, :, ch]).astype(np.float32) for ch in range(self.channels)]
        self.delta = [np.zeros((self.height, self.width), dtype=np.float32) for _ in range(self.channels)]
        self._last_applied: Optional[tuple] = None

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

        self._last_applied = (channel, freq_i, freq_j, old)

        def apply_step(step: float) -> np.ndarray:
            self.delta[channel][freq_i, freq_j] += step
            return self.build()

        return key, plus, minus, apply_step

    def restart(self, rng: np.random.Generator, scale: float) -> None:
        """Shrink deltas toward zero for restart."""
        for ch in range(self.channels):
            self.delta[ch] *= scale


def _make_state(scale: str, x0: np.ndarray, max_freq: int, patch_size: int | None):
    if scale == "global":
        return GlobalDCTState(x0, max_freq=max_freq)
    if scale == "mid":
        patch = int(patch_size or max(x0.shape[0] // 4, 8))
        return CenteredMidDCTState(x0, patch_size=patch, max_freq=max_freq)
    raise ValueError(f"Unsupported ATKScopes scale={scale!r}; expected 'global' or 'mid'")


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
    scale = str(cfg.get("scale", "mid"))
    a = float(cfg.get("a", 4.443566509056172))
    lr = float(cfg.get("lr", 3.855169125841428))
    max_freq = int(cfg.get("max_freq", 5))
    patch_size_raw = cfg.get("patch_size", None)
    patch_size = None if patch_size_raw is None else int(patch_size_raw)
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("eps", 1e-8))
    log_every = int(cfg.get("log_every", 50))
    stall_threshold = int(cfg.get("stall_threshold", 150))
    restart_delta_scale = float(cfg.get("restart_delta_scale", 0.3))
    # FIX: queries_per_iter=3 to match central difference (f_plus, f_minus) + evaluation (x_new)
    max_iters = resolve_max_iters(cfg, budget=budget, queries_per_iter=3)

    rng = np.random.default_rng(int(cfg.get("seed", seed)))
    x0 = to_float32(x_source).astype(np.float32)
    if x0.ndim == 2:
        x0 = np.stack([x0, x0, x0], axis=-1)
    if x0.shape[-1] == 4:
        x0 = x0[..., :3]

    state = _make_state(scale, x0, max_freq=max_freq, patch_size=patch_size)
    # Adam state is now preserved across restarts for momentum continuity
    adam: dict[tuple[Any, ...], Adam1D] = {}
    best = x0.copy()
    best_dist = float(oracle.query(x0))
    history = [best_dist]
    trace = AttackTrace()
    n_steps = 0
    n_improved = 0
    n_stalled = 0
    threshold_reached = False
    iters_since_improvement = 0

    # Main optimization loop
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
            iters_since_improvement += 1
            continue

        opt = adam.get(key)
        if opt is None:
            opt = Adam1D(beta1=beta1, beta2=beta2, eps=eps)
            adam[key] = opt

        # Threshold-proximity boost: increase lr when close to threshold to push through
        effective_lr = lr
        if best_dist - oracle.threshold < 1500.0:
            effective_lr = lr * 2.0

        step = opt.step(grad, effective_lr)
        x_new = apply_step(step)
        new_dist = float(oracle.query(x_new))
        n_steps += 1

        if new_dist < best_dist:
            best = x_new.copy()
            best_dist = new_dist
            n_improved += 1
            iters_since_improvement = 0
            if best_dist <= oracle.threshold:
                threshold_reached = True
        else:
            # Undo the specific step, do NOT wipe all state
            _ = apply_step(-step)  # undo
            iters_since_improvement += 1

        # Stall detection with restart - preserve Adam momentum
        if iters_since_improvement >= stall_threshold:
            state.restart(rng, restart_delta_scale)
            # ADDED: Do NOT clear Adam state - preserve learned gradient directions
            # adam.clear()  # Removed: clearing kills momentum that helps push through threshold
            n_stalled += 1
            iters_since_improvement = 0
            # Re-query current state after restart
            current = state.build()
            current_dist = float(oracle.query(current))
            if current_dist < best_dist:
                best = current.copy()
                best_dist = current_dist

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

        if threshold_reached:
            break

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
            "iters_improved": n_improved,
            "iters_stalled": n_stalled,
            "L2": _raw_l2_01(best, x0),
            "Linf": float(np.max(np.abs(best - x0))),
            "stopped_reason": oracle.state.stopped_reason,
            "threshold_reached": threshold_reached,
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