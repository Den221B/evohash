"""PhotoDNA — Microsoft PhotoDNA via pyPhotoDNA (Wine).

Everything is handled inside this file:
  - Wine + cabextract + genisoimage installation
  - PhotoDNAx64.dll extraction from FTK ISO
  - Minimal Wine Python 3.9 download
  - Hash computation via subprocess call to Wine Python

Digest  : np.ndarray uint16, shape (144,)
Distance: L1 norm
Threshold: 3855

Performance note:
  Each compute() call spawns a new wine64 process (~1-3 sec overhead on first
  call per Colab session due to Wine prefix init; subsequent calls are ~0.3-0.5
  sec each).  For large eval runs use compute_batch() — one process for N images.

Usage (in notebook):
    from evohash.hashes.photodna import PhotoDNAWrapper, setup_photodna
    setup_photodna()                 # one-time, ~5 min
    register_or_replace_hash(hashes, PhotoDNAWrapper())
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, List, Tuple

import numpy as np
from PIL import Image

from .base import HashSpec, HashFunction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_WORK_DIR = os.environ.get(
    "PHOTODNA_WORK_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "photodna"),
)

_WINE_PYTHON_SUBPATH = "wine_python_39/python-3.9.12-embed-amd64/python.exe"

# ---------------------------------------------------------------------------
# generateHashes.py — zipped into this file, written to disk on first setup
# ---------------------------------------------------------------------------

_GENERATE_HASHES_SRC = r'''"""
PhotoDNA hash generator for Wine Python.
Usage (single):  python generateHashes.py <image_path>
Usage (batch):   python generateHashes.py <img1> <img2> ...
Output: one line per image — comma-separated uint16 values (144 numbers).
"""
import sys, os, ctypes, struct
from PIL import Image

DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PhotoDNAx64.dll")

_lib = ctypes.CDLL(DLL_PATH)
_lib.PhotoDNAHashImage.restype  = ctypes.c_int
_lib.PhotoDNAHashImage.argtypes = [
    ctypes.c_char_p,  # image data (RGB24, row-major)
    ctypes.c_int,     # width
    ctypes.c_int,     # height
    ctypes.c_int,     # stride = width * 3
    ctypes.c_char_p,  # output buffer (288 bytes = 144 x uint16 LE)
]

HASH_BYTES = 288  # 144 * sizeof(uint16)


def compute(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    raw = img.tobytes()
    buf = ctypes.create_string_buffer(HASH_BYTES)
    ret = _lib.PhotoDNAHashImage(raw, w, h, w * 3, buf)
    if ret != 0:
        raise RuntimeError(f"PhotoDNAHashImage returned {ret} for {image_path}")
    values = struct.unpack_from(f"<{HASH_BYTES // 2}H", buf.raw)
    return ",".join(str(v) for v in values)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generateHashes.py <image_path> [image_path ...]", file=sys.stderr)
        sys.exit(1)
    for path in sys.argv[1:]:
        print(compute(path))
'''

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True)


def _apt_install(*packages: str) -> None:
    """Install apt packages only if not already present."""
    missing = [p for p in packages if not shutil.which(p)]
    if not missing:
        return
    print(f"[PhotoDNA] apt-get install {' '.join(missing)} …")
    _run("apt-get update -qq")
    _run(f"apt-get install -y -q {' '.join(missing)}")


def setup_photodna(work_dir: str = _DEFAULT_WORK_DIR, force: bool = False) -> str:
    """Full one-time setup. Returns path to Wine Python executable.

    Steps:
        1. Install system deps (wine64, cabextract, genisoimage) via apt.
        2. Download FTK ISO (3.3 GB) and extract PhotoDNAx64.dll.
        3. Download minimal Wine Python 3.9.
        4. Write generateHashes.py vendor script.

    Parameters
    ----------
    work_dir : str
        Where to place PhotoDNAx64.dll and Wine Python.
    force : bool
        Re-run even if already set up.

    Returns
    -------
    str
        Full path to the Wine Python executable.
    """
    wine_python = os.path.join(work_dir, _WINE_PYTHON_SUBPATH)
    dll_path    = os.path.join(work_dir, "PhotoDNAx64.dll")

    if os.path.isfile(wine_python) and os.path.isfile(dll_path) and not force:
        print(f"[PhotoDNA] Already set up at {work_dir} ✓")
        _ensure_vendor_script(work_dir)
        return wine_python

    os.makedirs(work_dir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(work_dir)

    try:
        # 1. System dependencies
        _apt_install("wine64", "cabextract", "genisoimage", "curl")

        # 2. Extract PhotoDNAx64.dll from FTK ISO
        if not os.path.isfile(dll_path):
            print("[PhotoDNA] Downloading FTK ISO (~3.3 GB) — this takes a while …")
            _run(
                "curl -L --progress-bar -o AD_FTK_7.0.0.iso "
                "https://d1kpmuwb7gvu1i.cloudfront.net/AD_FTK_7.0.0.iso"
            )
            print("[PhotoDNA] Extracting DLL …")
            _run(
                "isoinfo -i AD_FTK_7.0.0.iso -x /FTK/FTK/X64/_8A89F09/DATA1.CAB "
                "> Data1.cab"
            )
            os.makedirs("tmp_cab", exist_ok=True)
            _run("cabextract -d tmp_cab -q Data1.cab")

            found = None
            for root, _, files in os.walk("tmp_cab"):
                for f in files:
                    if "photodna" in f.lower() and f.endswith(".dll"):
                        found = os.path.join(root, f)
                        break

            if not found:
                raise RuntimeError(
                    "PhotoDNAx64.dll not found in FTK ISO — "
                    "the ISO internal structure may have changed."
                )
            shutil.copy(found, dll_path)
            print(f"[PhotoDNA] DLL saved → {dll_path}")

            for f in ("AD_FTK_7.0.0.iso", "Data1.cab"):
                if os.path.isfile(f):
                    os.remove(f)
            shutil.rmtree("tmp_cab", ignore_errors=True)

        # 3. Wine Python 3.9
        if not os.path.isfile(wine_python):
            print("[PhotoDNA] Downloading minimal Wine Python 3.9 …")
            _run(
                "curl -L --progress-bar -o wine_python_39.tar.gz "
                "https://github.com/jankais3r/pyPhotoDNA/releases/download/"
                "wine_python_39/wine_python_39.tar.gz"
            )
            _run("tar -xf wine_python_39.tar.gz")
            os.remove("wine_python_39.tar.gz")
            print(f"[PhotoDNA] Wine Python → {wine_python}")

        # 4. Vendor script
        _ensure_vendor_script(work_dir)

        print(f"\n[PhotoDNA] Setup complete ✓")
        return wine_python

    finally:
        os.chdir(orig_dir)


def _ensure_vendor_script(work_dir: str) -> None:
    """Write generateHashes.py and symlink DLL into work_dir/vendor/."""
    vendor_dir  = os.path.join(work_dir, "vendor")
    os.makedirs(vendor_dir, exist_ok=True)

    script_path = os.path.join(vendor_dir, "generateHashes.py")
    with open(script_path, "w") as f:
        f.write(_GENERATE_HASHES_SRC)

    dll_src = os.path.join(work_dir, "PhotoDNAx64.dll")
    dll_lnk = os.path.join(vendor_dir, "PhotoDNAx64.dll")
    if os.path.isfile(dll_src) and not os.path.exists(dll_lnk):
        os.symlink(dll_src, dll_lnk)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image


def _parse_hash_line(line: str) -> np.ndarray:
    values = [int(v.strip()) for v in line.strip().split(",") if v.strip()]
    if len(values) != 144:
        raise RuntimeError(f"Expected 144 hash values, got {len(values)}: {line[:120]}")
    return np.array(values, dtype=np.uint16)


def _wine_run(wine_python: str, script: str, args: List[str], timeout: int = 120) -> str:
    """Run Wine Python script and return stdout."""
    result = subprocess.run(
        ["wine64", wine_python, script, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "WINEDEBUG": "-all"},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PhotoDNA subprocess failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------


class PhotoDNAWrapper:
    """Microsoft PhotoDNA perceptual hash.

    Call ``setup_photodna()`` once before use (or on import via the notebook
    section).  After that, paths are resolved lazily on first compute() call.

    Parameters
    ----------
    threshold_p : float
        Maximum L1 distance for similarity (default 3855).
    work_dir : str
        Directory used by ``setup_photodna``.
    """

    def __init__(
        self,
        threshold_p: float = 3855.0,
        work_dir: str = _DEFAULT_WORK_DIR,
    ) -> None:
        self.spec = HashSpec(
            hash_id="photodna",
            threshold_p=float(threshold_p),
            distance_name="l1",
        )
        self._work_dir    = work_dir
        self._wine_python: str | None = None
        self._script:      str | None = None

    def _resolve(self) -> None:
        """Resolve and validate paths once."""
        if self._wine_python is not None:
            return
        wp     = os.path.join(self._work_dir, _WINE_PYTHON_SUBPATH)
        script = os.path.join(self._work_dir, "vendor", "generateHashes.py")
        if not os.path.isfile(wp) or not os.path.isfile(script):
            raise RuntimeError(
                "[PhotoDNA] Not set up.\n"
                "Run: from evohash.hashes.photodna import setup_photodna; setup_photodna()"
            )
        self._wine_python = wp
        self._script      = script

    # ------------------------------------------------------------------
    # HashFunction interface
    # ------------------------------------------------------------------

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute PhotoDNA hash for a single image.

        Note: spawns a wine64 process each call.  For many images prefer
        compute_batch() to amortise Wine startup cost.
        """
        self._resolve()
        img_u8 = _to_uint8_rgb(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
            Image.fromarray(img_u8).save(tmp)
        try:
            stdout = _wine_run(self._wine_python, self._script, [tmp])
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
        return _parse_hash_line(stdout.splitlines()[0])

    def compute_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Compute PhotoDNA hashes for multiple images in one Wine process.

        Significantly faster than calling compute() in a loop.
        """
        self._resolve()
        tmps: List[str] = []
        try:
            for img in images:
                img_u8 = _to_uint8_rgb(img)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmps.append(f.name)
                    Image.fromarray(img_u8).save(f.name)
            stdout = _wine_run(self._wine_python, self._script, tmps)
        finally:
            for t in tmps:
                if os.path.isfile(t):
                    os.remove(t)

        lines = [l for l in stdout.splitlines() if l.strip()]
        if len(lines) != len(images):
            raise RuntimeError(
                f"Expected {len(images)} hash lines, got {len(lines)}"
            )
        return [_parse_hash_line(l) for l in lines]

    def distance(self, d1: Any, d2: Any) -> float:
        a = np.asarray(d1, dtype=np.int32)
        b = np.asarray(d2, dtype=np.int32)
        return float(np.abs(a - b).sum())
