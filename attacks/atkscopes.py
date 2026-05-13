"""Fast ATKScopes-style coordinate attacks in [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.fft import dct, idct

from evohash.attacks.base import AttackRawResult
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = "atkscopes"

DEFAULT_PARAMS: dict[str, Any] = {
    "scale": "global",
    "max_iters": 4000,
    "a": 1.5,
    "lr": 1.0,
    "max_freq": 16,
    "patch_size": None,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "log_every": 50,
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
    max_iters = int(cfg.get("max_iters", cfg.get("n_iter", 4000)))
    a = float(cfg.get("a", 1.5))
    lr = float(cfg.get("lr", 1.0))
    max_freq = int(cfg.get("max_freq", 16))
    patch_size_raw = cfg.get("patch_size", None)
    patch_size = None if patch_size_raw is None else int(patch_size_raw)
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("eps", 1e-8))
    log_every = int(cfg.get("log_every", 50))

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


def run(context: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    hash_fn = context["hash_fn"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]
    params = {**DEFAULT_PARAMS, **kwargs}
    max_queries = params.pop("max_queries", None)
    seed = int(params.pop("seed", 0))

    attacked_images: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    for i, (x, target_hash) in enumerate(zip(sources, target_hashes)):
        x0 = to_float32(np.asarray(x))
        oracle = HashOracle(
            hash_fn=hash_fn,
            target_hash=target_hash,
            x_source=x0,
            budget=BudgetSpec(max_queries=max_queries, seed=seed + i),
        )
        raw = run_attack(x0, oracle, params=params, budget=max_queries, seed=seed + i)
        attacked_images.append(raw.x_best)
        metrics.append({
            "success": bool(raw.history[-1] <= oracle.threshold),
            "final_hash_l1": float(raw.history[-1]),
            "pixel_l2": compute_pixel_l2(x0, raw.x_best),
            "L2": _raw_l2_01(raw.x_best, x0),
            "queries": int(oracle.queries),
            **raw.extra,
        })
    return {"attacked_images": attacked_images, "metrics": metrics}
