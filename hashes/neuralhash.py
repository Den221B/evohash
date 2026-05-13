"""Apple NeuralHash wrapper via ONNX.

Expected model files by default:
    evohash/hashes/model/model.onnx
    evohash/hashes/model/model.dat

Environment override:
    NEURALHASH_MODEL_DIR=/path/to/model_dir

Digest:
    np.ndarray uint8, shape (96,), values {0, 1}
Distance:
    hash_l1 over the bit vector; equivalent to Hamming distance.
Threshold:
    17 by default.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from .base import HashSpec, binary_l1, to_uint8_rgb

_BUNDLED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
_MODEL_DIR = os.environ.get("NEURALHASH_MODEL_DIR") or _BUNDLED_MODEL_DIR
_ONNX_FILENAME = "model.onnx"
_SEED_FILENAME = "model.dat"


def _check_model_files(model_dir: str) -> None:
    missing = []
    for fname in (_ONNX_FILENAME, _SEED_FILENAME):
        path = os.path.join(model_dir, fname)
        if not os.path.isfile(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "[NeuralHash] Missing model files:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\nPlace {_ONNX_FILENAME} and {_SEED_FILENAME} into {model_dir} "
              "or set NEURALHASH_MODEL_DIR."
        )


def _preprocess(image: np.ndarray) -> np.ndarray:
    img = to_uint8_rgb(image)
    pil = Image.fromarray(img).convert("RGB").resize((360, 360))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def _get_providers(ort) -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


class NeuralHashWrapper:
    def __init__(
        self,
        threshold_p: float = 17.0,
        model_dir: str = _MODEL_DIR,
        eager_load: bool = True,
    ) -> None:
        self.spec = HashSpec(
            hash_id="neuralhash",
            threshold_p=float(threshold_p),
            distance_name="hash_l1",
        )
        self._model_dir = model_dir
        self._session = None
        self._seed: Optional[np.ndarray] = None
        self._input_name: Optional[str] = None
        if eager_load:
            self.warmup()

    @property
    def hash_id(self) -> str:
        return self.spec.hash_id

    @property
    def threshold(self) -> float:
        return self.spec.threshold_p

    def warmup(self) -> "NeuralHashWrapper":
        if self._session is not None:
            return self

        _check_model_files(self._model_dir)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Run one of:\n"
                "  pip install onnxruntime\n"
                "  pip install onnxruntime-gpu"
            ) from exc

        model_path = os.path.join(self._model_dir, _ONNX_FILENAME)
        seed_path = os.path.join(self._model_dir, _SEED_FILENAME)

        self._session = ort.InferenceSession(model_path, providers=_get_providers(ort))
        self._input_name = self._session.get_inputs()[0].name

        raw = open(seed_path, "rb").read()[128:]
        self._seed = np.frombuffer(raw, dtype=np.float32).reshape([96, 128])
        return self

    def compute(self, image: np.ndarray) -> np.ndarray:
        if self._session is None:
            self.warmup()
        assert self._session is not None
        assert self._seed is not None
        assert self._input_name is not None

        arr = _preprocess(image)
        out = self._session.run(None, {self._input_name: arr})
        embedding = np.asarray(out[0]).reshape(-1)
        floats = self._seed.dot(embedding)
        return (floats >= 0).astype(np.uint8)

    def distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        return binary_l1(d1, d2)
