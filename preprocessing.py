"""Image preprocessing shared by evaluators and notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from evohash.metrics import to_float32


@dataclass(frozen=True)
class ResizeSpec:
    """Optional square RGB resize in float32 [0, 1] space."""

    size: Optional[int] = None
    resample: str = "bilinear"


_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def resize_rgb01(image: np.ndarray, size: int, *, resample: str = "bilinear") -> np.ndarray:
    """Return RGB image resized to ``size x size`` as float32 [0, 1]."""
    if size <= 0:
        raise ValueError(f"resize size must be positive, got {size}")

    arr = to_float32(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxW, HxWx1, HxWx3, or HxWx4 image; got {arr.shape}")

    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.float32, copy=True)

    pil = Image.fromarray(np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    resized = pil.resize((int(size), int(size)), _RESAMPLE.get(resample, Image.Resampling.BILINEAR))
    return (np.asarray(resized, dtype=np.float32) / 255.0).clip(0.0, 1.0)


def apply_resize(image: np.ndarray, spec: ResizeSpec | None) -> np.ndarray:
    """Convert to float32 [0, 1] and apply optional resize."""
    if spec is None or spec.size is None:
        return to_float32(image).astype(np.float32, copy=True)
    return resize_rgb01(image, int(spec.size), resample=spec.resample)
