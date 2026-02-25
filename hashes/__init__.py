"""Hash functions registry and implementations."""
from .base import HashSpec, HashFunction, HashRegistry
from .phash import PHashWrapper
from .pdq import PDQWrapper
from .neuralhash import NeuralHashWrapper
from .photodna import PhotoDNAWrapper

__all__ = [
    "HashSpec",
    "HashFunction",
    "HashRegistry",
    "PHashWrapper",
    "PDQWrapper",
    "NeuralHashWrapper",
    "PhotoDNAWrapper",
    "build_default_registry",
]


def build_default_registry(
    *,
    phash: bool = True,
    pdq: bool = True,
    neuralhash: bool = False,
    photodna: bool = False,
    neuralhash_model_dir: str = "",   # "" → uses NeuralHashWrapper default cache
    photodna_work_dir: str = "",      # "" → uses PhotoDNAWrapper default cache
) -> HashRegistry:
    """Build a HashRegistry with the requested hash functions.

    pHash and PDQ are enabled by default (require imagehash / pdqhash).
    NeuralHash and PhotoDNA require additional one-time setup — see their
    respective setup functions before enabling.

    Parameters
    ----------
    neuralhash_model_dir : str
        Override cache directory for NeuralHash ONNX/dat files.
        Leave empty to use the default (~/.cache/neuralhash).
    photodna_work_dir : str
        Override work directory for PhotoDNA DLL and Wine Python.
        Leave empty to use the default (~/.cache/photodna).
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
        kwargs = {"work_dir": photodna_work_dir} if photodna_work_dir else {}
        reg.register(PhotoDNAWrapper(**kwargs))
    return reg