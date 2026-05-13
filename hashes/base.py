"""Base interfaces and helpers for perceptual hash wrappers.

Project convention
------------------
Every ``distance(...)`` method returns ``hash_l1``.

For binary hashes this is equivalent to Hamming distance, because the digest is
stored as a 0/1 vector and ``L1(bits_a, bits_b) == count(bits_a != bits_b)``.
For PhotoDNA this is the standard L1 distance over its 144-byte digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class HashSpec:
    hash_id: str
    threshold_p: float
    distance_name: str = "hash_l1"


@runtime_checkable
class HashFunction(Protocol):
    spec: HashSpec

    def compute(self, image: np.ndarray) -> Any:
        """Return digest for image.

        Implementations should accept uint8 [0,255] or float [0,1] images in
        HxW, HxWx3, or HxWx4 format where possible.
        """
        ...

    def distance(self, d1: Any, d2: Any) -> float:
        """Return ``hash_l1`` between two digests."""
        ...


class HashRegistry:
    def __init__(self) -> None:
        self._reg: Dict[str, HashFunction] = {}

    def register(self, hf: HashFunction, *, overwrite: bool = False) -> None:
        hid = hf.spec.hash_id
        if hid in self._reg and not overwrite:
            raise ValueError(
                f"HashFunction '{hid}' already registered. "
                "Use overwrite=True to replace."
            )
        self._reg[hid] = hf

    def get(self, hash_id: str) -> HashFunction:
        if hash_id not in self._reg:
            raise KeyError(
                f"Unknown hash_id '{hash_id}'. Registered: {self.list_ids()}"
            )
        return self._reg[hash_id]

    def list_ids(self) -> List[str]:
        return sorted(self._reg.keys())

    def items(self):
        return self._reg.items()

    def values(self):
        return self._reg.values()

    def __iter__(self) -> Iterator[HashFunction]:
        return iter(self._reg.values())

    def __contains__(self, hash_id: str) -> bool:
        return hash_id in self._reg

    def __len__(self) -> int:
        return len(self._reg)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to contiguous uint8 RGB HxWx3."""
    arr = np.asarray(image)

    if arr.dtype == np.uint8:
        out = arr.copy()
    else:
        arr = arr.astype(np.float32)
        if arr.size and float(np.nanmax(arr)) > 1.5:
            arr = arr / 255.0
        # Match the working attack notebooks and NumPy/PIL-style uint8 casts:
        # float values are clipped and truncated, not rounded to nearest.
        out = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    elif out.ndim == 3 and out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=-1)
    elif out.ndim == 3 and out.shape[-1] == 4:
        out = out[..., :3]

    if out.ndim != 3 or out.shape[-1] != 3:
        raise ValueError(f"Expected image shape HxW, HxWx1, HxWx3, or HxWx4; got {out.shape}")

    return np.ascontiguousarray(out)


def binary_l1(d1: Any, d2: Any) -> float:
    """L1 distance for binary 0/1 digests."""
    a = np.asarray(d1, dtype=np.int16).reshape(-1)
    b = np.asarray(d2, dtype=np.int16).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Digest shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).sum())


def numeric_l1(d1: Any, d2: Any) -> float:
    """L1 distance for numeric digests."""
    a = np.asarray(d1, dtype=np.int32).reshape(-1)
    b = np.asarray(d2, dtype=np.int32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Digest shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).sum())
