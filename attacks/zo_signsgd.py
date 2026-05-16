"""Clean ZO-signSGD attack in float32 [0, 1] image space."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evohash.attacks.base import AttackRawResult
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = "zo_signsgd"

DEFAULT_PARAMS: dict[str, Any] = {
    "estimator": "central",
    "direction_dist": "gaussian",
    "q": 16,
    "mu": 16.0 / 255.0,
    "lr": 12.0 / 255.0,
    "max_iters": 606,
    "eval_updated_point": True,
}


def _sample_direction(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    *,
    distribution: str = "gaussian",
) -> np.ndarray:
    direction = rng.standard_normal(shape).astype(np.float32)
    if distribution == "sphere":
        norm = float(np.linalg.norm(direction.ravel()))
        if norm > 0.0:
            direction /= norm
    elif distribution == "gaussian":
        pass # Useless
    else:
        raise ValueError("direction_dist must be 'gaussian' or 'sphere'")
    return direction


def run_attack(
    x_source: np.ndarray,
    oracle: HashOracle,
    params: Optional[dict[str, Any]] = None,
    budget: int | None = 10_000,
    seed: int = 0,
) -> AttackRawResult:
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    estimator = str(cfg.get("estimator", "central"))
    direction_dist = str(cfg.get("direction_dist", "gaussian"))
    q = int(cfg.get("q", 16))
    mu = float(cfg.get("mu", 16.0 / 255.0))
    lr = float(cfg.get("lr", 12.0 / 255.0))
    max_iters = int(cfg.get("max_iters", cfg.get("n_iter", 606)))
    eval_updated_point = bool(cfg.get("eval_updated_point", True))
    log_every = int(cfg.get("log_every", 0))

    rng = np.random.default_rng(int(cfg.get("seed", seed)))
    current = to_float32(x_source).astype(np.float32)
    if current.ndim == 2:
        current = np.stack([current, current, current], axis=-1)
    if current.shape[-1] == 4:
        current = current[..., :3]

    current_dist = float(oracle.query(current))
    best_dist = current_dist
    history = [best_dist]
    history_rows: list[dict[str, Any]] = []
    iters_done = 0
    dim = int(np.prod(current.shape))

    for iters_done in range(1, max_iters + 1):
        if best_dist <= oracle.threshold or not oracle.budget_ok():
            break

        grad = np.zeros_like(current, dtype=np.float32)
        used = 0

        if estimator in {"forward", "majority"}:
            if not oracle.budget_ok():
                break
            f_x = float(oracle.query(current))
            for _ in range(q):
                if not oracle.budget_ok():
                    break

                direction = _sample_direction(current.shape, rng, distribution=direction_dist)
                x_plus = np.clip(current + mu * direction, 0.0, 1.0)
                f_plus = float(oracle.query(x_plus))
                indiv = ((f_plus - f_x) / max(mu, 1e-12)) * direction
                if direction_dist == "sphere":
                    indiv *= dim
                if estimator == "majority":
                    grad += np.sign(indiv)
                else:
                    grad += indiv
                used += 1

        elif estimator == "central":
            for _ in range(q):
                if not oracle.budget_ok():
                    break

                direction = _sample_direction(current.shape, rng, distribution=direction_dist)
                x_plus = np.clip(current + mu * direction, 0.0, 1.0)
                f_plus = float(oracle.query(x_plus))

                if not oracle.budget_ok():
                    break

                x_minus = np.clip(current - mu * direction, 0.0, 1.0)
                f_minus = float(oracle.query(x_minus))
                indiv = ((f_plus - f_minus) / max(2.0 * mu, 1e-12)) * direction
                if direction_dist == "sphere":
                    indiv *= dim
                grad += indiv
                used += 1

        else:
            raise ValueError("estimator must be 'forward', 'central', or 'majority'")

        if used == 0:
            break

        grad /= float(used)
        current = np.clip(current - lr * np.sign(grad), 0.0, 1.0)

        if eval_updated_point and oracle.budget_ok():
            current_dist = float(oracle.query(current))
        else:
            current_dist = float(oracle.best_hash_l1)

        best_dist = float(oracle.best_hash_l1)
        history.append(best_dist)

        if log_every > 0 and (iters_done % log_every == 0 or best_dist <= oracle.threshold):
            history_rows.append({
                "iter": int(iters_done),
                "queries": int(oracle.queries),
                "best_hash_l1": best_dist,
                "hash_l1": float(current_dist),
                "estimator": estimator,
                "direction_dist": direction_dist,
            })

    return AttackRawResult(
        x_best=oracle.best_x.astype(np.float32),
        history=history,
        queries=oracle.queries,
        params=cfg,
        extra={
            "history_rows": history_rows,
            "iters": int(iters_done),
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
    for index, (image, target_hash) in enumerate(zip(sources, target_hashes)):
        x0 = to_float32(np.asarray(image))
        oracle = HashOracle(
            hash_fn=hash_fn,
            target_hash=target_hash,
            x_source=x0,
            budget=BudgetSpec(max_queries=max_queries, seed=seed + index),
        )
        raw = run_attack(x0, oracle, params=params, budget=max_queries, seed=seed + index)
        attacked_images.append(raw.x_best)
        metrics.append({
            "success": bool(raw.history[-1] <= oracle.threshold),
            "final_hash_l1": float(raw.history[-1]),
            "pixel_l2": compute_pixel_l2(x0, raw.x_best),
            "queries": int(oracle.queries),
        })
    return {"attacked_images": attacked_images, "metrics": metrics}
