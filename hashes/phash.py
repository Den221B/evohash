"""pHash wrapper using the `imagehash` library.

Install: pip install ImageHash
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .base import HashSpec, HashFunction


class PHashWrapper:
    """Perceptual hash via DCT (pHash).

    Digest type  : imagehash.ImageHash  (supports - operator → Hamming distance)
    Distance     : Hamming
    Threshold    : 12 (images with distance ≤ 12 considered similar)
    """

    def __init__(self, threshold_p: float = 12.0) -> None:
        try:
            import imagehash  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "imagehash not installed. Run: pip install ImageHash"
            ) from e
        import imagehash as _ih
        self._ih = _ih
        self.spec = HashSpec(
            hash_id="phash",
            threshold_p=float(threshold_p),
            distance_name="hamming",
        )

    def compute(self, image: np.ndarray) -> Any:
        img = _to_uint8(image)
        pil = Image.fromarray(img).convert("RGB")
        return self._ih.phash(pil)

    def distance(self, d1: Any, d2: Any) -> float:
        return float(d1 - d2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Accept uint8 [0,255] or float32 [0,1]; always return uint8 HxWx3."""
    if image.dtype == np.uint8:
        arr = image
    else:
        arr = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr
