"""PDQ hash wrapper using the `pdqhash` library.

Install: pip install pdqhash
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import HashSpec, HashFunction


class PDQWrapper:
    """PDQ perceptual hash (Facebook / Meta).

    Digest type  : np.ndarray  (bit vector, typically 256 bits as uint8 array)
    Distance     : Hamming
    Threshold    : 92
    """

    def __init__(self, threshold_p: float = 92.0) -> None:
        try:
            import pdqhash  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pdqhash not installed. Run: pip install pdqhash"
            ) from e
        import pdqhash as _pdq
        self._pdq = _pdq
        self.spec = HashSpec(
            hash_id="pdq",
            threshold_p=float(threshold_p),
            distance_name="hamming",
        )

    def compute(self, image: np.ndarray) -> np.ndarray:
        img = _to_uint8_rgb(image)
        out = self._pdq.compute(img)
        # Some versions return (digest, quality), some just digest
        digest = out[0] if isinstance(out, tuple) else out
        return np.asarray(digest)

    def distance(self, d1: Any, d2: Any) -> float:
        # Try library's own hamming first
        if hasattr(self._pdq, "hamming_distance"):
            try:
                return float(self._pdq.hamming_distance(d1, d2))
            except Exception:
                pass
        return _hamming_ndarrays(np.asarray(d1), np.asarray(d2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        arr = image
    else:
        arr = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _hamming_ndarrays(a: np.ndarray, b: np.ndarray) -> float:
    """Robust Hamming distance for various ndarray digest formats."""
    if a.shape != b.shape:
        raise ValueError(f"PDQ digest shape mismatch: {a.shape} vs {b.shape}")

    if a.dtype == np.bool_ or b.dtype == np.bool_:
        return float(np.count_nonzero(np.logical_xor(a, b)))

    if np.issubdtype(a.dtype, np.integer):
        a_vals = (int(a.min()), int(a.max()))
        b_vals = (int(b.min()), int(b.max()))
        # Bit-vector (0/1 per element)
        if a_vals[0] >= 0 and a_vals[1] <= 1 and b_vals[0] >= 0 and b_vals[1] <= 1:
            return float(np.count_nonzero(a ^ b))
        # Byte/int — popcount via XOR
        x = np.bitwise_xor(a, b)
        if x.dtype == np.uint8:
            lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint32)
            return float(lut[x].sum())
        return float(sum(int(v).bit_count() for v in x.reshape(-1)))

    # Float fallback (rare) — L2
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))
