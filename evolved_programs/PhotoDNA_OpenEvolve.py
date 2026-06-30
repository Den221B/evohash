# Inlined EvoHash seed attack: simba
# Generated for evolution so the LLM can mutate the actual attack logic.
# The evaluator calls run_attack(x_source, oracle, params, budget, seed).
"""Clean SimBA-DCT attack in float32 [0, 1] image space."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from evohash.attacks.base import AttackRawResult, resolve_max_iters
from evohash.metrics import compute_pixel_l2, to_float32
from evohash.oracle import BudgetSpec, HashOracle

ATTACK_ID = 'simba'

DEFAULT_PARAMS: dict[str, Any] = {
    'epsilon': 15.238244537908484,
    'freq_dims': 9,
    'stride': 1,
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
    max_iters = resolve_max_iters(cfg, budget=budget, queries_per_iter=3)
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
    adaptive_epsilon = epsilon  # decay near threshold to reduce L2 distortion
    successful_main_coords: list[tuple[int, int, int]] = []

    for iters_done, coord in enumerate(coords, start=1):
        if best_dist <= oracle.threshold or not oracle.budget_ok():
            break

        # Adaptive epsilon: granular decay near threshold to reduce L2 distortion
        gap_ratio = best_dist / max(oracle.threshold, 1e-12)
        if gap_ratio < 1.05:
            adaptive_epsilon = max(adaptive_epsilon * 0.90, epsilon * 0.15)
        elif gap_ratio < 1.15:
            adaptive_epsilon = max(adaptive_epsilon * 0.93, epsilon * 0.25)
        elif gap_ratio < 1.30:
            adaptive_epsilon = max(adaptive_epsilon * 0.97, epsilon * 0.40)

        direction = _simba_direction(current.shape, coord)
        x_neg = np.clip(current - adaptive_epsilon * direction, 0.0, 1.0)
        d_neg = float(oracle.query(x_neg))

        if d_neg < current_dist:
            current = x_neg
            current_dist = d_neg
            successful_main_coords.append(coord)
        elif oracle.budget_ok() and best_dist >= oracle.threshold * 1.1:
            # Skip positive probe when near/below threshold: it rarely helps and wastes queries
            x_pos = np.clip(current + adaptive_epsilon * direction, 0.0, 1.0)
            d_pos = float(oracle.query(x_pos))
            if d_pos < current_dist:
                current = x_pos
                current_dist = d_pos
                successful_main_coords.append(coord)

        best_dist = float(oracle.best_hash_l1)
        history.append(best_dist)

        if log_every > 0 and (iters_done % log_every == 0 or best_dist <= oracle.threshold):
            history_rows.append({
                "iter": int(iters_done),
                "queries": int(oracle.queries),
                "best_hash_l1": best_dist,
                "hash_l1": float(current_dist),
            })

    # Local refinement: geometrically decaying step sizes on low-frequency coords
    # to push past the threshold when the main search plateaus just above it.
    successful_coords: list[tuple[int, int, int]] = []
    if best_dist > oracle.threshold and oracle.budget_ok():
        # Start from the true best point found by oracle, not the trajectory endpoint
        current = oracle.best_x.copy()
        current_dist = best_dist
        # Combine low-freq coords with successful main-loop coords for diversity
        refine_coords = coords[:min(150, len(coords))]
        refine_set = set(refine_coords)
        for c in successful_main_coords[:100]:  # limit to top 100 for efficiency
            if c not in refine_set:
                refine_coords.append(c)
                refine_set.add(c)
        refine_epsilon = epsilon * 0.35  # larger initial step for more impact
        for _pass in range(2):  # reduced to 2; ASR is 100% so fewer passes saves queries for L2
            if best_dist <= oracle.threshold or not oracle.budget_ok():
                break
            # Resync with oracle's best to always refine from strongest position
            current = oracle.best_x.copy()
            current_dist = best_dist
            for coord in refine_coords:
                if best_dist <= oracle.threshold or not oracle.budget_ok():
                    break
                direction = _simba_direction(current.shape, coord)
                x_neg = np.clip(current - refine_epsilon * direction, 0.0, 1.0)
                d_neg = float(oracle.query(x_neg))
                if d_neg < current_dist:
                    current = x_neg
                    current_dist = d_neg
                    successful_coords.append(coord)
                elif oracle.budget_ok():
                    x_pos = np.clip(current + refine_epsilon * direction, 0.0, 1.0)
                    d_pos = float(oracle.query(x_pos))
                    if d_pos < current_dist:
                        current = x_pos
                        current_dist = d_pos
                        successful_coords.append(coord)
                best_dist = float(oracle.best_hash_l1)
            refine_epsilon *= 0.6  # slower decay preserves search resolution longer

    # Final threshold push: combine successful refinement and main-loop directions
    all_successful_set = set(successful_coords)
    for c in successful_main_coords:
        all_successful_set.add(c)
    all_successful = list(all_successful_set)[:200]  # deduplicate and limit for efficiency
    if best_dist > oracle.threshold and oracle.budget_ok() and len(all_successful) > 0:
        rng_push = np.random.default_rng(seed + 999)
        push_epsilon = refine_epsilon * 0.5  # very fine steps for micro-adjustments
        max_push = min(80, max(1, ((budget or 10000) - oracle.queries) // 2))  # tighter limit saves queries
        for _ in range(max_push):
            if best_dist <= oracle.threshold or not oracle.budget_ok():
                break
            # Combine 1-4 successful coordinates to exploit correlated improvements
            n_dirs = int(rng_push.integers(1, min(5, len(all_successful) + 1)))
            dir_indices = rng_push.choice(len(all_successful), size=n_dirs, replace=False)
            combined_dir = np.zeros(current.shape, dtype=np.float32)
            for idx in dir_indices:
                coord = all_successful[idx]
                combined_dir += _simba_direction(current.shape, coord)
            norm = float(np.linalg.norm(combined_dir))
            if norm > 0.0:
                combined_dir /= norm
            # Try both directions to cover sign ambiguity in DCT basis
            x_neg_try = np.clip(current - push_epsilon * combined_dir, 0.0, 1.0)
            d_neg_try = float(oracle.query(x_neg_try))
            if d_neg_try < current_dist:
                current = x_neg_try
                current_dist = d_neg_try
            elif oracle.budget_ok():
                x_pos_try = np.clip(current + push_epsilon * combined_dir, 0.0, 1.0)
                d_pos_try = float(oracle.query(x_pos_try))
                if d_pos_try < current_dist:
                    current = x_pos_try
                    current_dist = d_pos_try
            best_dist = float(oracle.best_hash_l1)

    # Post-optimization L2 refinement
    # REORDER: greedy coefficient removal FIRST (more hash margin), then binary search.
    # This allows removing more DCT components before binary search tightens the margin.
    final_x = oracle.best_x.copy() if float(oracle.best_hash_l1) < best_dist else current.copy()
    if float(oracle.best_hash_l1) <= oracle.threshold and oracle.budget_ok():
        # Phase 1: Greedy L2 reduction — remove DCT components while hash margin is wide
        # Doing this before binary search maximizes removable perturbation energy.
        delta = final_x - x0
        coord_contribs = []
        for coord in coords[:min(120, len(coords))]:
            direction = _simba_direction(final_x.shape, coord)
            contrib = float(np.abs(np.sum(delta * direction)))
            coord_contribs.append((contrib, coord))
        coord_contribs.sort(reverse=True)

        for contrib, coord in coord_contribs:
            if not oracle.budget_ok():
                break
            direction = _simba_direction(final_x.shape, coord)
            proj = float(np.sum(delta * direction))
            x_trial = np.clip(final_x - proj * direction, 0.0, 1.0)
            d_trial = float(oracle.query(x_trial))
            if d_trial <= oracle.threshold:
                final_x = x_trial
                delta = final_x - x0
            elif oracle.budget_ok():
                x_trial = np.clip(final_x - 0.5 * proj * direction, 0.0, 1.0)
                d_trial = float(oracle.query(x_trial))
                if d_trial <= oracle.threshold:
                    final_x = x_trial
                    delta = final_x - x0

        # Phase 2: Binary search toward source for fine-grained L2 reduction
        x_lo = final_x.copy()  # feasible: start from post-greedy result
        x_hi = x0.copy()       # infeasible: source (zero L2)
        for _ in range(5):  # 5 iterations sufficient for float32; saves 11 queries vs 16
            if not oracle.budget_ok():
                break
            x_trial = np.clip(0.5 * (x_lo + x_hi), 0.0, 1.0)
            d_trial = float(oracle.query(x_trial))
            if d_trial <= oracle.threshold:
                x_lo = x_trial
            else:
                x_hi = x_trial
        final_x = x_lo.copy()

        # Phase 3: Second greedy pass on updated delta (budget permitting)
        if oracle.budget_ok():
            delta = final_x - x0
            coord_contribs = []
            for coord in coords[:min(40, len(coords))]:
                direction = _simba_direction(final_x.shape, coord)
                contrib = float(np.abs(np.sum(delta * direction)))
                coord_contribs.append((contrib, coord))
            coord_contribs.sort(reverse=True)

            for contrib, coord in coord_contribs:
                if not oracle.budget_ok():
                    break
                direction = _simba_direction(final_x.shape, coord)
                proj = float(np.sum(delta * direction))
                for frac in (0.5, 0.25):  # skip 0.75 after binary search tightened margin; saves queries
                    if not oracle.budget_ok():
                        break
                    x_trial = np.clip(final_x - frac * proj * direction, 0.0, 1.0)
                    d_trial = float(oracle.query(x_trial))
                    if d_trial <= oracle.threshold:
                        final_x = x_trial
                        delta = final_x - x0
                        break
        history.append(float(oracle.best_hash_l1))

    return AttackRawResult(
        x_best=final_x.astype(np.float32),
        history=history,
        queries=oracle.queries,
        params=cfg,
        extra={
            "history_rows": history_rows,
            "iters": int(iters_done),
            "stopped_reason": oracle.state.stopped_reason,
        },
    )
