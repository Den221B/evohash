"""Clean SimBA-DCT attack in float32 [0, 1] image space."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evohash.attacks.base import AttackRawResult, resolve_max_iters
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = "simba"

DEFAULT_PARAMS: dict[str, Any] = {
    "basis": "dct",
    "order": "strided",
    "freq_dims": 10,
    "stride": 1,
    "epsilon": 1900.0 / 255.0,
}

_DCT_BASIS_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}
_MAX_DCT_CACHE_PIXELS = 64 * 64


def _dct_basis_2d(height: int, width: int, freq_row: int, freq_col: int) -> np.ndarray:
    use_cache = (height * width) <= _MAX_DCT_CACHE_PIXELS
    key = (height, width, int(freq_row), int(freq_col))
    if use_cache and key in _DCT_BASIS_CACHE:
        return _DCT_BASIS_CACHE[key]

    rows = np.arange(height, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(width, dtype=np.float32).reshape(1, -1)
    basis = (
        np.cos(np.pi * freq_row * (2 * rows + 1) / (2 * height))
        * np.cos(np.pi * freq_col * (2 * cols + 1) / (2 * width))
    ).astype(np.float32)

    norm = float(np.linalg.norm(basis))
    if norm > 0.0:
        basis /= norm

    if use_cache:
        _DCT_BASIS_CACHE[key] = basis
    return basis


def _flat_indices_to_coords(indices: np.ndarray, image_size: int) -> list[tuple[int, int, int]]:
    flat = np.asarray(indices, dtype=np.int64)
    channels = flat // (image_size * image_size)
    remainder = flat % (image_size * image_size)
    freq_rows = remainder // image_size
    freq_cols = remainder % image_size
    return list(zip(channels.astype(int), freq_rows.astype(int), freq_cols.astype(int)))


def _block_order(
    *,
    image_size: int,
    channels: int,
    initial_size: int,
    stride: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    initial_size = int(min(max(1, initial_size), image_size))
    stride = int(max(1, stride))

    order = np.zeros((channels, image_size, image_size), dtype=np.float64)
    total_elems = channels * initial_size * initial_size
    perm = rng.permutation(total_elems)
    order[:, :initial_size, :initial_size] = perm.reshape(channels, initial_size, initial_size)

    for index in range(initial_size, image_size, stride):
        end = min(index + stride, image_size)
        strip = end - index
        num_elems = channels * (2 * strip * index + strip * strip)
        perm = rng.permutation(num_elems) + total_elems
        num_first = channels * strip * (strip + index)
        order[:, :end, index:end] = perm[:num_first].reshape(channels, -1, strip)
        order[:, index:end, :index] = perm[num_first:].reshape(channels, strip, -1)
        total_elems += num_elems

    return np.argsort(order.reshape(-1))


def _make_simba_coords(
    height: int,
    width: int,
    *,
    channels: int,
    freq_dims: int,
    stride: int,
    seed: int,
    max_iters: int,
) -> list[tuple[int, int, int]]:
    if height != width:
        raise ValueError(f"SimBA-DCT expects square image, got {height}x{width}")
    indices = _block_order(
        image_size=int(height),
        channels=int(channels),
        initial_size=int(freq_dims),
        stride=int(stride),
        seed=int(seed),
    )
    return _flat_indices_to_coords(indices[: int(max_iters)], int(height))


def _simba_direction(shape: tuple[int, int, int], coord: tuple[int, int, int]) -> np.ndarray:
    height, width, channels = shape
    channel, freq_row, freq_col = coord
    if channel >= channels:
        raise ValueError(f"Coordinate channel={channel} is out of range for shape={shape}")
    direction = np.zeros(shape, dtype=np.float32)
    direction[:, :, channel] = _dct_basis_2d(height, width, freq_row, freq_col)
    return direction


def run_attack(
    x_source: np.ndarray,
    oracle: HashOracle,
    params: Optional[dict[str, Any]] = None,
    budget: int | None = 10_000,
    seed: int = 0,
) -> AttackRawResult:
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    freq_dims = int(cfg.get("freq_dims", 10))
    stride = int(cfg.get("stride", 1))
    epsilon = float(cfg.get("epsilon", 1900.0 / 255.0))
    max_iters = resolve_max_iters(cfg, budget=budget, queries_per_iter=1)
    log_every = int(cfg.get("log_every", 0))

    x0 = to_float32(x_source).astype(np.float32)
    if x0.ndim == 2:
        x0 = np.stack([x0, x0, x0], axis=-1)
    if x0.shape[-1] == 4:
        x0 = x0[..., :3]

    height, width, channels = x0.shape
    coords = _make_simba_coords(
        height,
        width,
        channels=channels,
        freq_dims=freq_dims,
        stride=stride,
        seed=int(cfg.get("seed", seed)),
        max_iters=max_iters,
    )

    current = x0.copy()
    current_dist = float(oracle.query(current))
    best_dist = current_dist
    history = [best_dist]
    history_rows: list[dict[str, Any]] = []
    iters_done = 0

    for iters_done, coord in enumerate(coords, start=1):
        if best_dist <= oracle.threshold or not oracle.budget_ok():
            break

        direction = _simba_direction(current.shape, coord)
        x_neg = np.clip(current - epsilon * direction, 0.0, 1.0)
        d_neg = float(oracle.query(x_neg))

        if d_neg < current_dist:
            current = x_neg
            current_dist = d_neg
        elif oracle.budget_ok():
            x_pos = np.clip(current + epsilon * direction, 0.0, 1.0)
            d_pos = float(oracle.query(x_pos))
            if d_pos < current_dist:
                current = x_pos
                current_dist = d_pos

        best_dist = float(oracle.best_hash_l1)
        history.append(best_dist)

        if log_every > 0 and (iters_done % log_every == 0 or best_dist <= oracle.threshold):
            history_rows.append({
                "iter": int(iters_done),
                "queries": int(oracle.queries),
                "best_hash_l1": best_dist,
                "hash_l1": float(current_dist),
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
