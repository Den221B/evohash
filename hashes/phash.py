"""pHash wrapper.

Dependency:
    pip install ImageHash

Digest:
    np.ndarray uint8, shape (64,), values {0, 1}
Distance:
    hash_l1 over the bit vector; equivalent to Hamming distance.
Threshold:
    12 by default.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import HashSpec, binary_l1, to_uint8_rgb


class PHashWrapper:
    def __init__(self, threshold_p: float = 12.0) -> None:
        try:
            import imagehash
        except ImportError as exc:
            raise ImportError("imagehash is not installed. Run: pip install ImageHash") from exc

        self._imagehash = imagehash
        self.spec = HashSpec(
            hash_id="phash",
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
        pil = Image.fromarray(img).convert("RGB")
        digest = self._imagehash.phash(pil)
        # imagehash.ImageHash.hash is a boolean matrix, normally 8x8.
        return np.asarray(digest.hash, dtype=np.uint8).reshape(-1)

    def distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        return binary_l1(d1, d2)
