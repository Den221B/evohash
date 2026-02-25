"""Shared image utilities used across evohash modules."""
from __future__ import annotations

import numpy as np


def to_float32(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 [0, 1].  Accepts uint8 [0,255] or float."""
    if image.dtype == np.uint8:
        return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return image.astype(np.float32).clip(0.0, 1.0)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 [0, 255]. Accepts float32 [0,1] or uint8."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)


def l2_img(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel-space RMS L2 distance (normalised to [0,1] range)."""
    a = to_float32(a)
    b = to_float32(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))
