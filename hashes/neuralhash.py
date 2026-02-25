"""NeuralHash — Apple's perceptual hash via ONNX.

Model files are fetched automatically from HuggingFace the first time.
No macOS required.

Source: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX

Pipeline (all steps run locally):
    1. Convert image to RGB
    2. Resize to 360×360
    3. Normalise pixel values to [-1, 1]
    4. ONNX model inference → 128-dim embedding
    5. Dot product with seed matrix (128×96) → 96 floats
    6. Binarise via sign → {0, 1}^96
    7. Hamming distance on binarised bits

Digest    : np.ndarray float32, shape (96,)  — step-5 output (pre-binarisation)
Distance  : Hamming on binarised bits
Threshold : 17
"""
from __future__ import annotations

import os
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

from .base import HashSpec, HashFunction

# ---------------------------------------------------------------------------
# HuggingFace mirror — model files extracted from iOS 14.8 IPSW,
# published in the public domain by the research community.
# ---------------------------------------------------------------------------
_HF_BASE = "https://huggingface.co/QualiaSystems/neural-hash-onnx/resolve/main"
_FILES = {
    "neuralhash_128x96_seed1.onnx": f"{_HF_BASE}/neuralhash_128x96_seed1.onnx",
    "neuralhash_128x96_seed1.dat":  f"{_HF_BASE}/neuralhash_128x96_seed1.dat",
}

_DEFAULT_CACHE_DIR = os.environ.get(
    "NEURALHASH_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "neuralhash"),
)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download_models(cache_dir: str) -> None:
    """Download model files to cache_dir if not already present."""
    os.makedirs(cache_dir, exist_ok=True)
    for filename, url in _FILES.items():
        dest = os.path.join(cache_dir, filename)
        if os.path.isfile(dest):
            continue
        print(f"[NeuralHash] Downloading {filename} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[NeuralHash] Saved -> {dest}")
        except Exception as exc:
            if os.path.isfile(dest):
                os.remove(dest)
            raise RuntimeError(
                f"Failed to download {filename} from {url}\n"
                "Set NEURALHASH_MODEL_DIR to a directory with manually placed files."
            ) from exc


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image: np.ndarray) -> np.ndarray:
    """Steps 1-3: RGB -> resize 360x360 -> normalise to [-1, 1].

    Returns float32 array of shape (1, 3, 360, 360).
    """
    # Step 1: ensure uint8 RGB
    if image.dtype != np.uint8:
        image = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    # Step 2: resize to 360x360
    pil = Image.fromarray(image).convert("RGB").resize((360, 360), Image.BILINEAR)

    # Step 3: normalise to [-1, 1]
    arr = (np.array(pil).astype(np.float32) / 127.5) - 1.0

    # NCHW layout expected by the model
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 360, 360)


# ---------------------------------------------------------------------------
# GPU / provider selection
# ---------------------------------------------------------------------------

def _get_providers(ort_module) -> list:
    """Return provider list: CUDA first if available, CPU as fallback.

    ONNX Runtime silently falls back to CPU if CUDA initialisation fails,
    so listing both is always safe.
    """
    available = ort_module.get_available_providers()
    if "CUDAExecutionProvider" in available:
        print("[NeuralHash] CUDAExecutionProvider detected -- GPU will be used")
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider",
        ]
    print("[NeuralHash] CUDAExecutionProvider not available -- using CPU")
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class NeuralHashWrapper:
    """Apple NeuralHash perceptual hash.

    Call warmup() once after construction (or it will happen automatically
    on the first compute() call).  After warmup the session stays loaded
    in memory for the lifetime of the object -- compute() is just inference.

    Parameters
    ----------
    threshold_p : float
        Maximum Hamming distance to consider two images similar (default 17).
    model_dir : str
        Directory for ONNX / dat files.  Downloaded automatically if absent.
    """

    def __init__(
        self,
        threshold_p: float = 17.0,
        model_dir: str = _DEFAULT_CACHE_DIR,
        eager_load: bool = True,
    ) -> None:
        self.spec = HashSpec(
            hash_id="neuralhash",
            threshold_p=float(threshold_p),
            distance_name="hamming",
        )
        self._model_dir = model_dir
        self._session = None   # loaded once in warmup()
        self._seed: np.ndarray | None = None
        self._input_name: str | None = None

        if eager_load:
            self.warmup()

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------

    def warmup(self) -> "NeuralHashWrapper":
        """Load model into memory (idempotent).

        Returns self for chaining:
            register_or_replace_hash(hashes, NeuralHashWrapper().warmup())
        """
        if self._session is not None:
            return self

        _download_models(self._model_dir)

        model_path = os.path.join(self._model_dir, "neuralhash_128x96_seed1.onnx")
        seed_path  = os.path.join(self._model_dir, "neuralhash_128x96_seed1.dat")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime not installed.\n"
                "CPU:  pip install onnxruntime\n"
                "GPU:  pip install onnxruntime-gpu"
            ) from exc

        providers = _get_providers(ort)
        self._session = ort.InferenceSession(model_path, providers=providers)
        active = self._session.get_providers()
        print(f"[NeuralHash] Session providers: {active}")

        # Cache input name (usually 'image' but may differ by ONNX export)
        self._input_name = self._session.get_inputs()[0].name

        # Seed matrix: (128, 96) float32 — used in step 5
        self._seed = (
            np.frombuffer(open(seed_path, "rb").read(), dtype=np.float32)
            .reshape((128, 96))
        )
        print("[NeuralHash] Model loaded")
        return self

    # ------------------------------------------------------------------
    # HashFunction interface
    # ------------------------------------------------------------------

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Return 96-dim float32 embedding (pre-binarisation, step-5 output).

        Steps performed here:
            1-3  _preprocess()     -- RGB / resize 360x360 / normalise [-1,1]
            4    ONNX inference    -- model -> 128-dim embedding
            5    seed dot product  -- embedding @ seed -> 96 floats
        """
        if self._session is None:
            self.warmup()

        arr = _preprocess(image)                          # (1, 3, 360, 360)
        outputs = self._session.run(None, {self._input_name: arr})
        embedding = outputs[0].flatten()                  # (128,)
        hash_bits = self._seed.T @ embedding              # (96,)
        return hash_bits.astype(np.float32)

    def distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """Steps 6-7: binarise via sign, count differing bits (Hamming)."""
        b1 = (np.sign(d1) > 0).astype(np.int8)
        b2 = (np.sign(d2) > 0).astype(np.int8)
        return float(np.count_nonzero(b1 != b2))
