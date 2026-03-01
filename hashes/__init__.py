"""Hash functions registry and implementations."""
from .base import HashSpec, HashFunction, HashRegistry
from .phash import PHashWrapper
from .pdq import PDQWrapper
from .neuralhash import NeuralHashWrapper
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
    neuralhash_model_dir: str = "",  # "" → evohash/hashes/model/ (bundled)
    photodna_work_dir: str = "",     # "" → ~/.cache/photodna
) -> HashRegistry:
    """Build a HashRegistry with the requested hash functions.

    pHash and PDQ are enabled by default.
    NeuralHash requires model files in evohash/hashes/model/.
    PhotoDNA requires setup_photodna() to have been called first.

    Parameters
    ----------
    neuralhash_model_dir : str
        Override directory for model.onnx / model.dat.
        Leave empty to use bundled evohash/hashes/model/.
    photodna_work_dir : str
        Override work dir for PhotoDNA (DLL + Wine Python).
        Leave empty to use ~/.cache/photodna.
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