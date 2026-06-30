from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.fft import dct, idct

from evohash.attacks.base import AttackRawResult, resolve_max_iters
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = 'atkscopes_grayscale_unified'

DEFAULT_PARAMS: dict[str, Any] = {
    'a': 4.498432179443098,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-08,
    'lr': 2.2240806832096887,
    'max_freq': 16,
    'scale': 'global',
    'refine_max_iters': 10,
    'refine_tol': 1e-4,
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
    """Grayscale DCT optimization: single delta applied to all RGB channels with 1/sqrt(3) scaling.

    PDQ converts RGB to grayscale before DCT, so optimizing a single grayscale DCT plane
    is 3x more query-efficient than per-channel optimization. The 1/sqrt(3) factor preserves
    raw L2 distortion since L2^2 = 3 * (delta/sqrt(3))^2 = delta^2.
    """

    def __init__(self, x0: np.ndarray, max_freq: int) -> None:
        self.height, self.width, self.channels = x0.shape
        # Clamp to PDQ's 16x16 DCT window to avoid wasted probes
        self.max_freq_h = max(2, min(int(max_freq), 16))
        self.max_freq_w = max(2, min(int(max_freq), 16))

        # Convert to grayscale using PDQ luminance weights
        gray = (0.299 * x0[:, :, 0] + 0.587 * x0[:, :, 1] + 0.114 * x0[:, :, 2]).astype(np.float32)
        self.gray_dct = _dct2(gray)
        self.delta = np.zeros((self.height, self.width), dtype=np.float32)
        self.x0_rgb = x0.copy()

    def build(self) -> np.ndarray:
        # IDCT of delta gives spatial grayscale perturbation
        gray_perturbation = _idct2(self.delta)
        # Apply to all channels with 1/sqrt(3) scaling to preserve L2
        scale = 1.0 / np.sqrt(3.0)
        channels = [
            np.clip(self.x0_rgb[:, :, ch] + gray_perturbation * scale, 0.0, 1.0)
            for ch in range(self.channels)
        ]
        return np.stack(channels, axis=2)

    def make_probes(self, rng: np.random.Generator, a: float):
        while True:
            freq_i = int(rng.integers(self.max_freq_h))
            freq_j = int(rng.integers(self.max_freq_w))
            if freq_i != 0 or freq_j != 0:
                break
        key = ("dctG", freq_i, freq_j)
        old = float(self.delta[freq_i, freq_j])

        # Adaptive step: scale by coefficient magnitude for better convergence
        a_scaled = max(a, 0.1 * abs(old))

        self.delta[freq_i, freq_j] = old + a_scaled
        plus = self.build()
        self.delta[freq_i, freq_j] = old - a_scaled
        minus = self.build()
        self.delta[freq_i, freq_j] = old

        def apply_step(step: float) -> np.ndarray:
            self.delta[freq_i, freq_j] += step
            return self.build()

        return key, plus, minus, apply_step, a_scaled


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
    refine_max_iters = int(cfg.get("refine_max_iters", 10))
    refine_tol = float(cfg.get("refine_tol", 1e-4))
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

    for it in range(1, max_iters + 1):
        if best_dist <= oracle.threshold or not oracle.budget_ok():
            break

        key, plus, minus, apply_step, a_scaled = state.make_probes(rng, a)
        f_plus = float(oracle.query(plus))
        if not oracle.budget_ok():
            break
        f_minus = float(oracle.query(minus))
        grad = (f_plus - f_minus) / (2.0 * a_scaled + 1e-12)

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

    # Post-threshold L2 refinement via spatial scaling with binary search
    if best_dist <= oracle.threshold and oracle.budget_ok():
        delta_best = best - x0  # perturbation from source to best
        lo, hi = 0.0, 1.0
        for _ in range(refine_max_iters):
            if not oracle.budget_ok():
                break
            if hi - lo < refine_tol:
                break
            mid = (lo + hi) * 0.5
            x_refined = np.clip(x0 + delta_best * mid, 0.0, 1.0)
            d_refined = float(oracle.query(x_refined))
            if d_refined <= oracle.threshold:
                best = x_refined.copy()
                best_dist = d_refined
                hi = mid
            else:
                lo = mid

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