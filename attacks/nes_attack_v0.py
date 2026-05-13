"""Working NES/Prokos-style baseline with image-space delta accumulation."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evohash.attacks.base import AttackRawResult
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = "nes_attack_v0"

DEFAULT_PARAMS: dict[str, Any] = {
    "max_iters": 300,
    "n_samples": 20,
    "sigma": 0.5,
    "lr": 10.0,
    "momentum": 0.5,
    "grayscale_noise": True,
}


def _gaussian_direction(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def _grayscale_gaussian_direction(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 3 and shape[2] == 3:
        p = rng.standard_normal(shape[:2]).astype(np.float32)
        return np.stack([p, p, p], axis=-1)
    return _gaussian_direction(rng, shape)


def run_attack(
    x_source: np.ndarray,
    oracle: HashOracle,
    params: Optional[dict[str, Any]] = None,
    budget: int | None = 10_000,
    seed: int = 0,
) -> AttackRawResult:
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    max_iters = int(cfg.get("max_iters", cfg.get("n_iter", 300)))
    n_samples = int(cfg.get("n_samples", 20))
    sigma = float(cfg.get("sigma", 0.5))
    lr = float(cfg.get("lr", 10.0))
    momentum = float(cfg.get("momentum", 0.5))
    grayscale_noise = bool(cfg.get("grayscale_noise", cfg.get("grayscale", True)))

    rng = np.random.default_rng(int(cfg.get("seed", seed)))
    sample_dir = _grayscale_gaussian_direction if grayscale_noise else _gaussian_direction
    x0 = to_float32(x_source).astype(np.float32)
    delta = np.zeros_like(x0, dtype=np.float32)
    mom = np.zeros_like(x0, dtype=np.float32)
    best_dist = float(oracle.query(x0))
    best_x = x0.copy()
    history = [best_dist]

    for it in range(1, max_iters + 1):
        if best_dist <= oracle.threshold or not oracle.budget_ok():
            break

        grad = np.zeros_like(x0, dtype=np.float32)
        used = 0
        for _ in range(n_samples):
            if not oracle.budget_ok():
                break
            p = sample_dir(rng, x0.shape)
            f_pos = float(oracle.query(np.clip(x0 + delta + sigma * p, 0.0, 1.0)))
            if not oracle.budget_ok():
                break
            f_neg = float(oracle.query(np.clip(x0 + delta - sigma * p, 0.0, 1.0)))
            grad += (f_pos - f_neg) * p
            used += 1

        if used == 0:
            break

        grad /= float(used)
        norm = float(np.linalg.norm(grad))
        if norm < 1e-10:
            history.append(best_dist)
            continue

        g_unit = grad / norm
        mom = momentum * mom + (1.0 - momentum) * g_unit
        delta = delta - lr * mom
        cand = np.clip(x0 + delta, 0.0, 1.0)
        cand_dist = float(oracle.query(cand))
        if cand_dist < best_dist:
            best_dist = cand_dist
            best_x = cand.copy()

        history.append(best_dist)

    return AttackRawResult(
        x_best=best_x.astype(np.float32),
        history=history,
        queries=oracle.queries,
        params=cfg,
        extra={"stopped_reason": oracle.state.stopped_reason},
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
            "queries": int(oracle.queries),
        })
    return {"attacked_images": attacked_images, "metrics": metrics}
