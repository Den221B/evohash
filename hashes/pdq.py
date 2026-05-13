"""PDQ hash wrapper.

Dependency:
    pip install pdqhash

Digest:
    np.ndarray uint8, shape (256,), values {0, 1}
Distance:
    hash_l1 over the bit vector; equivalent to Hamming distance.
Threshold:
    92 by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import HashSpec, binary_l1, to_uint8_rgb


class PDQWrapper:
    def __init__(self, threshold_p: float = 92.0) -> None:
        try:
            import pdqhash
        except ImportError as exc:
            raise ImportError("pdqhash is not installed. Run: pip install pdqhash") from exc

        self._pdq = pdqhash
        self.spec = HashSpec(
            hash_id="pdq",
            threshold_p=float(threshold_p),
            distance_name="hash_l1",
        )

    @property
    def hash_id(self) -> str:
        return self.spec.hash_id

    @property
    def threshold(self) -> float:
        return self.spec.threshold_p

    def compute(self, image: np.ndarray) -> np.ndarray:
        img = to_uint8_rgb(image)
        out = self._pdq.compute(img)
        # Different pdqhash builds may return digest or (digest, quality).
        digest = out[0] if isinstance(out, tuple) else out
        return _to_pdq_bits(digest)

    def distance(self, d1: Any, d2: Any) -> float:
        return binary_l1(_to_pdq_bits(d1), _to_pdq_bits(d2))


def _to_pdq_bits(digest: Any) -> np.ndarray:
    """Normalize common PDQ digest formats to a 256-bit 0/1 vector."""
    if isinstance(digest, str):
        s = digest.strip().lower()
        if s.startswith("0x"):
            s = s[2:]
        if len(s) % 2 != 0:
            s = "0" + s
        raw = bytes.fromhex(s)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        return bits.astype(np.uint8)

    arr = np.asarray(digest)

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8).reshape(-1)

    arr = arr.reshape(-1)

    # Already a 0/1 bit vector.
    if arr.size > 0:
        mn = int(np.nanmin(arr))
        mx = int(np.nanmax(arr))
        if arr.size == 256 and mn >= 0 and mx <= 1:
            return arr.astype(np.uint8)
        # Some implementations use signed {-1, +1} values.
        if arr.size == 256 and mn >= -1 and mx <= 1:
            return (arr > 0).astype(np.uint8)

    # Packed 32-byte representation of a 256-bit hash.
    if arr.size == 32 and np.issubdtype(arr.dtype, np.integer):
        return np.unpackbits(arr.astype(np.uint8)).astype(np.uint8)

    # Single Python integer / numpy scalar representation.
    if arr.size == 1 and np.issubdtype(arr.dtype, np.integer):
        value = int(arr[0])
        if value < 0:
            raise ValueError("PDQ integer digest must be non-negative")
        return np.array([(value >> (255 - i)) & 1 for i in range(256)], dtype=np.uint8)

    raise ValueError(
        "Unsupported PDQ digest format. Expected 256 bits, 32 packed bytes, "
        f"hex string, or integer; got shape={np.asarray(digest).shape}, dtype={np.asarray(digest).dtype}."
    )
