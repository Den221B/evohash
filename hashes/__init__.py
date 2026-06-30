"""Hash functions registry and implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import HashFunction, HashRegistry, HashSpec
from .neuralhash import NeuralHashWrapper
from .pdq import PDQWrapper
from .phash import PHashWrapper
from .photodna import PhotoDNAWrapper, setup_photodna

__all__ = [
    "HashSpec",
    "HashFunction",
    "HashRegistry",
    "PHashWrapper",
    "PDQWrapper",
    "NeuralHashWrapper",
    "PhotoDNAWrapper",
    "setup_photodna",
    "build_default_registry",
    "build_hash_fn",
]


def build_hash_fn(
    hash_id: str,
    *,
    thresholds: dict[str, float] | None = None,
    dll_path: str = "",
    model_dir: str = "",
) -> HashFunction:
    """Construct a single HashFunction by hash_id and optional config.

    ``thresholds`` is a mapping like ``{"phash": 12.0, "pdq": 92.0, ...}``.
    This is the single source of truth used by parallel_eval workers,
    the optimizer, and stage3 runtimes — no module needs to re-implement it.

    Example::

        hash_fn = build_hash_fn("phash", thresholds=config.get("thresholds", {}))
    """
    t: dict[str, Any] = thresholds or {}

    if hash_id == "pdq":
        return PDQWrapper(threshold_p=float(t.get("pdq", 92.0)))

    if hash_id == "phash":
        return PHashWrapper(threshold_p=float(t.get("phash", 12.0)))

    if hash_id == "neuralhash":
        kw: dict[str, Any] = {"threshold_p": float(t.get("neuralhash", 17.0))}
        if model_dir:
            kw["model_dir"] = model_dir
        return NeuralHashWrapper(**kw)

    if hash_id == "photodna":
        kw = {"threshold_p": float(t.get("photodna", 3855.0))}
        _dll = dll_path or str(
            Path(__file__).resolve().parent / "dll" / "PhotoDNAx64.dll"
        )
        if Path(_dll).exists():
            kw["dll_path"] = _dll
        return PhotoDNAWrapper(**kw)

    raise ValueError(
        f"Unknown hash_id {hash_id!r}. "
        f"Supported: 'phash', 'pdq', 'neuralhash', 'photodna'."
    )


def build_default_registry(
    *,
    phash: bool = True,
    pdq: bool = True,
    neuralhash: bool = False,
    photodna: bool = False,
    neuralhash_model_dir: str = "",
    photodna_work_dir: str = "",
    photodna_dll_path: str = "",
) -> HashRegistry:
    """Build registry with selected hash wrappers.

    pHash and PDQ are enabled by default. NeuralHash and PhotoDNA are optional
    because they need model/DLL setup.
    """
    reg = HashRegistry()

    if phash:
        reg.register(PHashWrapper())
    if pdq:
        reg.register(PDQWrapper())
    if neuralhash:
        kwargs = {"model_dir": neuralhash_model_dir} if neuralhash_model_dir else {}
        reg.register(NeuralHashWrapper(**kwargs))
    if photodna:
        kwargs = {}
        if photodna_work_dir:
            kwargs["work_dir"] = photodna_work_dir
        if photodna_dll_path:
            kwargs["dll_path"] = photodna_dll_path
        reg.register(PhotoDNAWrapper(**kwargs))

    return reg
