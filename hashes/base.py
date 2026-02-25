"""Base types for hash functions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class HashSpec:
    hash_id: str
    threshold_p: float
    distance_name: str  # "hamming" | "l2" | etc.


@runtime_checkable
class HashFunction(Protocol):
    spec: HashSpec

    def compute(self, image: np.ndarray) -> Any:
        """Return digest for image (uint8 HxWx3 or float32 [0,1] HxWx3)."""
        ...

    def distance(self, d1: Any, d2: Any) -> float:
        """Return scalar distance between two digests."""
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
                f"Unknown hash_id '{hash_id}'. "
                f"Registered: {self.list_ids()}"
            )
        return self._reg[hash_id]

    def list_ids(self) -> List[str]:
        return sorted(self._reg.keys())

    def __contains__(self, hash_id: str) -> bool:
        return hash_id in self._reg
