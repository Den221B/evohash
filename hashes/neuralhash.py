"""NeuralHash — Apple's perceptual hash via ONNX.

Model files must be placed in evohash/hashes/model/ (committed to repo):
    neuralhash_128x96_seed1.onnx
    neuralhash_128x96_seed1.dat

Source: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX

Pipeline (all steps run locally):
    1. Convert image to RGB
    2. Resize to 360x360
    3. Normalise pixel values to [-1, 1]  (arr * 2.0 - 1.0)
    4. ONNX model inference → 128-dim embedding
    5. Dot product: seed (96x128) @ embedding (128,) → 96 floats
    6. Binarise via sign → {0,1}^96
    7. Hamming distance on binarised bits

Digest    : np.ndarray uint8, shape (96,)  — binarised bits {0, 1}
Distance  : Hamming on binarised bits
Threshold : 17
"""
from __future__ import annotations

import os
import numpy as np
from PIL import Image

from .base import HashSpec, HashFunction

# ---------------------------------------------------------------------------
# Model file resolution
# Priority:
#   1. NEURALHASH_MODEL_DIR env var  (override for custom location)
#   2. evohash/hashes/model/         (bundled in repo — default)
# ---------------------------------------------------------------------------

_BUNDLED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

_MODEL_DIR = (
    os.environ.get("NEURALHASH_MODEL_DIR") or _BUNDLED_MODEL_DIR
)

_ONNX_FILENAME = "model.onnx"
_SEED_FILENAME  = "model.dat"


def _check_model_files(model_dir: str) -> None:
    """Raise a clear error if model files are missing."""
    for fname in (_ONNX_FILENAME, _SEED_FILENAME):
        path = os.path.join(model_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[NeuralHash] Model file not found: {path}\n"
                f"Place both files in: {model_dir}\n"
                f"  {_ONNX_FILENAME}\n"
                f"  {_SEED_FILENAME}\n"
                "Then commit to repo:  git add hashes/model/ && git push"
            )


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image: np.ndarray) -> np.ndarray:
    """Steps 1-3: ensure uint8 RGB → resize 360x360 → normalise to [-1, 1].

    Matches exactly the reference nnhash.py implementation:
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0

    Returns float32 array of shape (1, 3, 360, 360).
    """
    if image.dtype != np.uint8:
        image = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    pil = Image.fromarray(image).convert("RGB").resize((360, 360))
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0                          # → [-1, 1]
    return arr.transpose(2, 0, 1)[np.newaxis]       # → (1, 3, 360, 360)


# ---------------------------------------------------------------------------
# GPU provider selection
# ---------------------------------------------------------------------------

def _get_providers(ort) -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        print("[NeuralHash] Using GPU (CUDAExecutionProvider)")
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider",
        ]
    print("[NeuralHash] Using CPU")
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class NeuralHashWrapper:
    """Apple NeuralHash perceptual hash.

    Model files must be committed to evohash/hashes/model/ before use.
    Loaded eagerly on construction by default (eager_load=True).

    Parameters
    ----------
    threshold_p : float
        Maximum Hamming distance for similarity (default 17).
    model_dir : str
        Directory containing ONNX and dat files.
        Defaults to evohash/hashes/model/ (bundled in repo).
    eager_load : bool
        Load model immediately on construction (default True).
    """

    def __init__(
        self,
        threshold_p: float = 17.0,
        model_dir: str = _MODEL_DIR,
        eager_load: bool = True,
    ) -> None:
        self.spec = HashSpec(
            hash_id="neuralhash",
            threshold_p=float(threshold_p),
            distance_name="hamming",
        )
        self._model_dir   = model_dir
        self._session     = None
        self._seed        = None   # np.ndarray (96, 128)
        self._input_name  = None

        if eager_load:
            self.warmup()

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------

    def warmup(self) -> "NeuralHashWrapper":
        """Load ONNX session and seed matrix (idempotent).

        Returns self for chaining:
            register_or_replace_hash(hashes, NeuralHashWrapper().warmup())
        """
        if self._session is not None:
            return self

        _check_model_files(self._model_dir)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime not installed.\n"
                "GPU:  pip install onnxruntime-gpu\n"
                "CPU:  pip install onnxruntime"
            ) from exc

        model_path = os.path.join(self._model_dir, _ONNX_FILENAME)
        seed_path  = os.path.join(self._model_dir, _SEED_FILENAME)

        self._session    = ort.InferenceSession(model_path, providers=_get_providers(ort))
        self._input_name = self._session.get_inputs()[0].name

        # Matches reference: read()[128:], reshape([96, 128])
        raw = open(seed_path, "rb").read()[128:]
        self._seed = np.frombuffer(raw, dtype=np.float32).reshape([96, 128])

        print(f"[NeuralHash] Loaded — providers: {self._session.get_providers()}")
        return self

    # ------------------------------------------------------------------
    # HashFunction interface
    # ------------------------------------------------------------------

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Return binarised 96-bit hash as uint8 array of 0/1 values.

        Steps performed:
            1-3  _preprocess()  — RGB / resize / normalise [-1,1]
            4    ONNX inference — → 128-dim embedding
            5    seed.dot()     — (96,128) @ (128,) → (96,) floats
            6    binarise       — value >= 0 → 1, else → 0
        """
        if self._session is None:
            self.warmup()

        arr       = _preprocess(image)
        out       = self._session.run(None, {self._input_name: arr})
        embedding = out[0].flatten()                               # (128,)
        floats    = self._seed.dot(embedding)                      # (96,)
        return (floats >= 0).astype(np.uint8)                      # (96,) bits

    def distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """Hamming distance on binarised 96-bit hashes."""
        return float(np.count_nonzero(
            d1.astype(np.uint8) != d2.astype(np.uint8)
        ))