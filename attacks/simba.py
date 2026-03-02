# attacks/simba.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.fftpack import idct
from time import perf_counter

from .base import AttackSpec, AttackResult


def _get_seed(budget) -> int:
    s = getattr(budget, "seed", None)
    if s is None:
        return 0
    try:
        return int(s) & 0xFFFFFFFF
    except Exception:
        return 0


def _as_float01(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v.ravel(), ord=2))
    if n < eps:
        return np.zeros_like(v)
    return v / (n + eps)


def _idct2(x: np.ndarray) -> np.ndarray:
    # Orthonormal 2D inverse DCT
    return idct(idct(x, axis=0, norm="ortho"), axis=1, norm="ortho")


@dataclass
class SimBAAttack:
    """
    SimBA (Simple Black-box Attack) adapted to minimize oracle distance (e.g., hash distance).

    Key properties (as in the original SimBA):
      - greedy coordinate descent over an orthonormal basis Q
      - choose directions WITHOUT replacement (we permute indices once)
      - at each step, try +/- epsilon along the chosen direction and accept if improves objective

    Basis:
      - "pixel": one-hot per pixel/channel coordinate
      - "dct": low-frequency DCT basis (unit-norm in image space), similar to SimBA-DCT

    We record history BOTH by iteration and by elapsed time (ms).
    """

    epsilon: float = 0.2
    basis: str = "dct"          # "pixel" or "dct"
    freq_dims: int = 28         # used for dct basis (low-frequency window)
    order: str = "strided"      # "rand" or "strided" (used to generate index order)
    stride: int = 7             # used if order="strided"
    grayscale: bool = False     # if True, enforce same perturbation across RGB channels
    normalize_direction: bool = True  # keep each basis direction unit-norm in image space

    spec: AttackSpec = AttackSpec(
        attack_id="simba",
        name="SimBA",
        description="SimBA coordinate descent over pixel or low-freq DCT basis; deterministic with seed; logs time.",
    )

    # ----------------------------
    # Direction index generation
    # ----------------------------
    def _make_index_order(
        self, rng: np.random.Generator, H: int, W: int, C: int
    ) -> List[Tuple[int, int, int]]:
        """
        Returns an ordered list of indices for the basis directions WITHOUT replacement.
        For pixel: indices are (i,j,ch) over H*W*C.
        For dct: indices are (fi,fj,ch) over freq_dims*freq_dims*C.
        Supports:
          - order="rand": random permutation
          - order="strided": the strided traversal from the reference repo (approximation)
        """
        if self.basis == "pixel":
            ii, jj, cc = np.meshgrid(
                np.arange(H, dtype=np.int32),
                np.arange(W, dtype=np.int32),
                np.arange(C, dtype=np.int32),
                indexing="ij",
            )
            idx = np.stack([ii.ravel(), jj.ravel(), cc.ravel()], axis=1)
            perm = rng.permutation(idx.shape[0])
            out = idx[perm]
            return [(int(a), int(b), int(c)) for a, b, c in out]

        if self.basis != "dct":
            raise ValueError(f"Unknown basis='{self.basis}'. Use 'pixel' or 'dct'.")

        fd = int(self.freq_dims)
        if fd <= 0:
            raise ValueError("freq_dims must be positive for dct basis.")

        # all low-frequency coords
        fi, fj, cc = np.meshgrid(
            np.arange(fd, dtype=np.int32),
            np.arange(fd, dtype=np.int32),
            np.arange(C, dtype=np.int32),
            indexing="ij",
        )
        idx = np.stack([fi.ravel(), fj.ravel(), cc.ravel()], axis=1)

        if self.order == "rand":
            perm = rng.permutation(idx.shape[0])
            out = idx[perm]
            return [(int(a), int(b), int(c)) for a, b, c in out]

        if self.order == "strided":
            # Strided order inspired by the authors' code: walk in blocks with stride in both dims.
            # We implement deterministic "strided scan" then apply a seed-based permutation of blocks
            # to avoid always same early coords across all runs.
            s = max(1, int(self.stride))
            coords = []
            # build blocks by offset within stride
            blocks = []
            for oi in range(s):
                for oj in range(s):
                    block = []
                    for a in range(oi, fd, s):
                        for b in range(oj, fd, s):
                            for c in range(C):
                                block.append((a, b, c))
                    if block:
                        blocks.append(block)
            # shuffle blocks deterministically
            block_perm = rng.permutation(len(blocks))
            for bi in block_perm:
                coords.extend(blocks[int(bi)])
            return coords

        raise ValueError(f"Unknown order='{self.order}'. Use 'rand' or 'strided'.")

    # ----------------------------
    # Basis direction construction
    # ----------------------------
    def _pixel_direction(self, shape: Tuple[int, int, int], idx: Tuple[int, int, int]) -> np.ndarray:
        H, W, C = shape
        i, j, ch = idx
        q = np.zeros((H, W, C), dtype=np.float32)
        if self.grayscale and C == 3:
            q[i, j, :] = 1.0
        else:
            q[i, j, ch] = 1.0
        # pixel basis already unit in L2
        return q

    def _dct_direction(self, shape: Tuple[int, int, int], idx: Tuple[int, int, int]) -> np.ndarray:
        H, W, C = shape
        fi, fj, ch = idx

        # Build a single-channel DCT coefficient matrix and inverse DCT to image space
        coeff = np.zeros((H, W), dtype=np.float32)
        if fi < H and fj < W:
            coeff[fi, fj] = 1.0

        basis_2d = _idct2(coeff).astype(np.float32)  # (H, W)

        q = np.zeros((H, W, C), dtype=np.float32)
        if self.grayscale and C == 3:
            q[:, :, :] = basis_2d[:, :, None]
        else:
            q[:, :, ch] = basis_2d

        if self.normalize_direction:
            q = _normalize(q)

        return q

    def _direction(self, shape: Tuple[int, int, int], idx: Tuple[int, int, int]) -> np.ndarray:
        if self.basis == "pixel":
            return self._pixel_direction(shape, idx)
        return self._dct_direction(shape, idx)

    # ----------------------------
    # Main attack loop
    # ----------------------------
    def run(self, x0, oracle, budget) -> AttackResult:
        x0 = _as_float01(x0)
        seed = _get_seed(budget)
        rng = np.random.default_rng(seed)

        threshold = getattr(oracle, "threshold_p", getattr(oracle, "threshold", None))
        t0 = perf_counter()

        H, W, C = x0.shape
        idx_order = self._make_index_order(rng, H, W, C)

        delta = np.zeros_like(x0, dtype=np.float32)

        hist: Dict[str, Any] = {
            "iter": [],
            "elapsed_ms": [],
            "dist": [],
            "best_dist": [],
            "accepted": [],
            "sign": [],          # +1 or -1 for accepted move; 0 if none
            "idx": [],           # (a,b,c) chosen
        }

        # initialize
        x_cur = oracle.project(x0 + delta)
        base = float(oracle.query(x_cur))
        best_dist = float(getattr(oracle.state, "best_dist", base))
        best_x = getattr(oracle.state, "best_x", x_cur)

        it = 0
        for idx in idx_order:
            if not oracle.budget_ok():
                break

            # record at start of iteration (before queries for +/-)
            elapsed_ms = (perf_counter() - t0) * 1000.0

            q = self._direction((H, W, C), idx)

            # Try +epsilon
            x_plus = oracle.project(x0 + (delta + self.epsilon * q))
            d_plus = float(oracle.query(x_plus))

            # Early update after plus query
            best_dist = float(getattr(oracle.state, "best_dist", min(best_dist, d_plus)))
            best_x = getattr(oracle.state, "best_x", best_x)

            if threshold is not None and best_dist <= float(threshold):
                # log iteration summary and stop
                hist["iter"].append(it)
                hist["elapsed_ms"].append(elapsed_ms)
                hist["dist"].append(base)
                hist["best_dist"].append(best_dist)
                hist["accepted"].append(True)
                hist["sign"].append(+1)
                hist["idx"].append(idx)
                delta = delta + self.epsilon * q
                base = d_plus
                break

            # Try -epsilon
            if not oracle.budget_ok():
                # budget ended before second query; log and stop
                hist["iter"].append(it)
                hist["elapsed_ms"].append(elapsed_ms)
                hist["dist"].append(base)
                hist["best_dist"].append(best_dist)
                hist["accepted"].append(False)
                hist["sign"].append(0)
                hist["idx"].append(idx)
                break

            x_minus = oracle.project(x0 + (delta - self.epsilon * q))
            d_minus = float(oracle.query(x_minus))

            best_dist = float(getattr(oracle.state, "best_dist", min(best_dist, d_minus)))
            best_x = getattr(oracle.state, "best_x", best_x)

            # Greedy accept
            accepted = False
            sgn = 0
            if d_plus < base and d_plus <= d_minus:
                delta = delta + self.epsilon * q
                base = d_plus
                accepted = True
                sgn = +1
            elif d_minus < base and d_minus < d_plus:
                delta = delta - self.epsilon * q
                base = d_minus
                accepted = True
                sgn = -1

            # log
            hist["iter"].append(it)
            hist["elapsed_ms"].append(elapsed_ms)
            hist["dist"].append(base)
            hist["best_dist"].append(float(getattr(oracle.state, "best_dist", best_dist)))
            hist["accepted"].append(accepted)
            hist["sign"].append(sgn)
            hist["idx"].append(idx)

            # early success stop
            if threshold is not None and float(getattr(oracle.state, "best_dist", best_dist)) <= float(threshold):
                break

            it += 1

        # finalize
        total_ms = (perf_counter() - t0) * 1000.0
        best_dist = float(getattr(oracle.state, "best_dist", best_dist))
        best_x = getattr(oracle.state, "best_x", best_x)
        success = (threshold is not None and best_dist <= float(threshold))

        return AttackResult(
            success=bool(success),
            best_dist=best_dist,
            queries_used=int(getattr(oracle, "queries_used", 0)),
            best_x=best_x,
            history=hist,
            extra={
                "seed": seed,
                "epsilon": float(self.epsilon),
                "basis": self.basis,
                "freq_dims": int(self.freq_dims),
                "order": self.order,
                "stride": int(self.stride),
                "grayscale": bool(self.grayscale),
                "normalize_direction": bool(self.normalize_direction),
                "runtime_ms": float(total_ms),
            },
        )