"""Input-transformation ("purification") defense for perceptual hashing.

A :class:`DefendedHash` wraps any existing hash function and applies a fixed
image transformation *before* hashing. Because it keeps the same
``compute`` / ``distance`` / ``spec`` contract (and the same ``hash_id`` and
threshold as the inner hash), it drops into the whole stack unchanged --
oracle, attacks, evaluator, optimizer, ``parallel_eval``, ``score_evolved`` --
so the *same* attacks can be evaluated against the hardened mechanism.

Deployment semantics: the transform is part of the hashing mechanism, so both
the enrolled/target hash and every queried image are purified identically. On a
clean image the transform is near-identity, so the target barely moves; on an
adversarially crafted image it tends to destroy the fragile perturbation that
landed the collision.

The transforms double as *benign* edits (JPEG, resize, blur, brightness) for
measuring the legitimate-match cost of a defense.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from evohash.metrics import to_float32

from .base import HashSpec, to_uint8_rgb


# --------------------------------------------------------------------------- #
# Low-level image transforms (operate on float32 [0,1] HxWx3, return the same) #
# --------------------------------------------------------------------------- #

def _pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(to_uint8_rgb(x))


def _from_pil(pil: Image.Image) -> np.ndarray:
    return np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0


def jpeg(x: np.ndarray, quality: float) -> np.ndarray:
    """JPEG re-compression at the given quality (1..95). Lower = stronger."""
    buf = io.BytesIO()
    _pil(x).save(buf, format="JPEG", quality=int(round(quality)))
    buf.seek(0)
    with Image.open(buf) as out:
        return _from_pil(out)


def gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with the given radius (sigma)."""
    return _from_pil(_pil(x).filter(ImageFilter.GaussianBlur(radius=float(sigma))))


def median(x: np.ndarray, size: float) -> np.ndarray:
    """Median filter; ``size`` is the (odd) window size."""
    k = int(round(size))
    if k % 2 == 0:
        k += 1
    k = max(1, k)
    return _from_pil(_pil(x).filter(ImageFilter.MedianFilter(size=k)))


def quantize(x: np.ndarray, bits: float) -> np.ndarray:
    """Reduce colour bit-depth to ``bits`` bits per channel."""
    b = max(1, int(round(bits)))
    levels = float(2 ** b - 1)
    z = to_float32(x)
    return (np.round(np.clip(z, 0.0, 1.0) * levels) / levels).astype(np.float32)


def resize_roundtrip(x: np.ndarray, scale: float) -> np.ndarray:
    """Downscale by ``scale`` (0<scale<=1) then upscale back to the original size."""
    pil = _pil(x)
    w, h = pil.size
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = pil.resize((nw, nh), Image.BILINEAR)
    return _from_pil(small.resize((w, h), Image.BILINEAR))


def brightness(x: np.ndarray, factor: float) -> np.ndarray:
    """Scale brightness by ``factor`` (1.0 = identity)."""
    return _from_pil(ImageEnhance.Brightness(_pil(x)).enhance(float(factor)))


_TRANSFORMS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "jpeg": jpeg,
    "blur": gaussian_blur,
    "median": median,
    "quantize": quantize,
    "resize": resize_roundtrip,
    "brightness": brightness,
}


@dataclass(frozen=True)
class Defense:
    """A named, single-parameter image transform (picklable, stateless).

    ``kind="none"`` is the identity (the undefended baseline). Use as either a
    hashing-time defense or a benign edit.
    """

    kind: str = "none"
    strength: float = 0.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return to_float32(image)
        try:
            fn = _TRANSFORMS[self.kind]
        except KeyError as exc:
            raise ValueError(
                f"Unknown defense kind {self.kind!r}. "
                f"Available: {['none', *_TRANSFORMS]}"
            ) from exc
        return fn(to_float32(image), self.strength)

    @property
    def label(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}={self.strength:g}"


# --------------------------------------------------------------------------- #
# The defended hash wrapper                                                    #
# --------------------------------------------------------------------------- #

class DefendedHash:
    """Wrap a hash function so it purifies the image before hashing.

    Keeps the inner hash's ``spec`` (same ``hash_id`` and threshold), so it is a
    drop-in replacement everywhere a ``HashFunction`` is expected.
    """

    def __init__(self, inner: Any, defense: Defense) -> None:
        self.inner = inner
        self.defense = defense
        self.spec = HashSpec(
            hash_id=inner.spec.hash_id,
            threshold_p=float(inner.spec.threshold_p),
            distance_name=getattr(inner.spec, "distance_name", "hash_l1"),
        )

    @property
    def hash_id(self) -> str:
        return self.spec.hash_id

    @property
    def threshold(self) -> float:
        return self.spec.threshold_p

    def compute(self, image: np.ndarray) -> Any:
        return self.inner.compute(self.defense(image))

    def distance(self, d1: Any, d2: Any) -> float:
        return self.inner.distance(d1, d2)

    def __repr__(self) -> str:
        return f"DefendedHash({self.spec.hash_id!r}, defense={self.defense.label!r})"


def make_defended(inner: Any, defense: Defense) -> DefendedHash:
    """Build a defended hash from an inner hash function and a defense."""
    return DefendedHash(inner, defense)


class RethresholdHash:
    """Delegate to an inner hash but report a different ``threshold_p``.

    Used to disable an attack's early-stop-on-success: attacks break when
    ``best_dist <= oracle.threshold``; setting the threshold below the metric's
    minimum (e.g. ``-1``) forces the attack to spend the whole query budget and
    push the hash distance to its achievable minimum.
    """

    def __init__(self, inner: Any, threshold_p: float) -> None:
        self.inner = inner
        self.spec = HashSpec(
            hash_id=inner.spec.hash_id,
            threshold_p=float(threshold_p),
            distance_name=getattr(inner.spec, "distance_name", "hash_l1"),
        )

    @property
    def threshold(self) -> float:
        return self.spec.threshold_p

    def compute(self, image: np.ndarray) -> Any:
        return self.inner.compute(image)

    def distance(self, d1: Any, d2: Any) -> float:
        return self.inner.distance(d1, d2)


def with_threshold(inner: Any, threshold_p: float) -> RethresholdHash:
    """Return a view of ``inner`` that reports ``threshold_p`` (compute unchanged)."""
    return RethresholdHash(inner, threshold_p)


def defense_grid(kind: str, strengths: list[float]) -> list[Defense]:
    """Convenience: a ``none`` baseline plus one defense kind swept over strengths."""
    return [Defense("none"), *[Defense(kind, float(s)) for s in strengths]]


__all__ = [
    "Defense",
    "DefendedHash",
    "RethresholdHash",
    "make_defended",
    "with_threshold",
    "defense_grid",
    "jpeg",
    "gaussian_blur",
    "median",
    "quantize",
    "resize_roundtrip",
    "brightness",
]
