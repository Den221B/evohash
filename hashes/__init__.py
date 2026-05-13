"""Hash functions registry and implementations."""

from __future__ import annotations

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
]


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
