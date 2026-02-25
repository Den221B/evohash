"""Dataset: reads image pairs from a zip archive produced by the dataset builder.

Zip layout (public):
    manifest.jsonl          # one JSON line per pair
    images/src_public_p0000.jpg
    images/tgt_public_p0000.jpg
    ...

Zip layout (full):
    manifest_public.jsonl
    manifest_secret.jsonl
    images/public/src_public_p0000.jpg
    images/secret/src_secret_p0000.jpg
    ...
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PairSample:
    pair_id: str
    x: np.ndarray           # source image, uint8 HxWx3
    y: np.ndarray           # target image, uint8 HxWx3
    meta: Dict[str, Any] = field(default_factory=dict)


class Dataset:
    """Iterate over image pairs from a zip archive.

    Args:
        zip_path:  Path to the dataset zip file.
        split:     "public" or "secret".
        max_pairs: Optional cap on number of pairs (useful for quick tests).
    """

    def __init__(
        self,
        zip_path: str,
        split: str = "public",
        max_pairs: Optional[int] = None,
    ) -> None:
        self.zip_path = zip_path
        self.split = split
        self.max_pairs = max_pairs

    def __iter__(self) -> Iterator[PairSample]:
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            manifest_name = _find_manifest(zf, self.split)
            lines = _read_text(zf, manifest_name).strip().splitlines()

            if self.max_pairs is not None:
                lines = lines[: self.max_pairs]

            for line in lines:
                row = json.loads(line)
                x = _read_image(zf, row["src_rel"])
                y = _read_image(zf, row["tgt_rel"])
                meta = dict(row.get("meta", {}))
                meta["split"] = row.get("split", self.split)
                meta["src_rel"] = row["src_rel"]
                meta["tgt_rel"] = row["tgt_rel"]
                yield PairSample(pair_id=row["pair_id"], x=x, y=y, meta=meta)

    def __len__(self) -> Optional[int]:
        """Return number of pairs (reads manifest only, not images)."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            manifest_name = _find_manifest(zf, self.split)
            lines = _read_text(zf, manifest_name).strip().splitlines()
        n = len(lines)
        return min(n, self.max_pairs) if self.max_pairs is not None else n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_manifest(zf: zipfile.ZipFile, split: str) -> str:
    names = set(zf.namelist())
    for candidate in (
        "manifest.jsonl",
        f"manifest_{split}.jsonl",
    ):
        if candidate in names:
            return candidate
    raise FileNotFoundError(
        f"No manifest found for split='{split}' in zip. "
        f"Available files: {sorted(n for n in names if 'manifest' in n)}"
    )


def _read_text(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as f:
        return f.read().decode("utf-8")


def _read_image(zf: zipfile.ZipFile, path: str) -> np.ndarray:
    with zf.open(path) as f:
        data = f.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(pil, dtype=np.uint8)
