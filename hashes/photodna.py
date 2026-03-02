"""PhotoDNA — Microsoft PhotoDNA, cross-platform.

Platform behaviour
------------------
Windows  : DLL loaded directly via ctypes — no Wine, no subprocess.
           Fast (~1-5 ms per image after first load).

Linux / macOS : DLL called via a Wine Python subprocess (original approach).
                Requires wine64 + the bundled Wine Python 3.9 environment.
                ~0.3–1 s per image; use compute_batch() to amortise startup.

Digest   : np.ndarray uint8, shape (144,)
Distance : L1 norm
Threshold: 3855

──────────────────────────────────────────────────
Windows setup (one-time, ~30 sec):
──────────────────────────────────────────────────
1. Obtain PhotoDNAx64.dll (extract from FTK ISO with 7-Zip, or source elsewhere).
2. Place it anywhere; pass the path at construction or set env var:

    set PHOTODNA_DLL=C:\\path\\to\\PhotoDNAx64.dll
    # or
    wrapper = PhotoDNAWrapper(dll_path=r"C:\\path\\to\\PhotoDNAx64.dll")

3. No further setup needed — ctypes loads the DLL natively.

──────────────────────────────────────────────────
Linux / Colab setup (one-time, ~5–10 min):
──────────────────────────────────────────────────
    from evohash.hashes.photodna import setup_photodna
    setup_photodna()   # downloads FTK ISO, extracts DLL, fetches Wine Python
    wrapper = PhotoDNAWrapper()

──────────────────────────────────────────────────
Usage (both platforms):
──────────────────────────────────────────────────
    from evohash.hashes.photodna import PhotoDNAWrapper, setup_photodna
    # Windows: setup_photodna() is optional if dll_path is set
    wrapper = PhotoDNAWrapper()
    digest  = wrapper.compute(image_uint8)   # np.ndarray uint8 (144,)
    dist    = wrapper.distance(digest_a, digest_b)
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from .base import HashSpec, HashFunction

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_WORK_DIR = (
    os.environ.get("PHOTODNA_WORK_DIR")
    or os.path.join(os.path.expanduser("~"), ".cache", "photodna")
)

# Windows: path to the DLL itself
_DEFAULT_DLL_PATH = os.environ.get("PHOTODNA_DLL", "")

# Linux: subpath to Wine Python inside work_dir
_WINE_PYTHON_SUBPATH = "python-3.9.12-embed-amd64/python.exe"

# ---------------------------------------------------------------------------
# generateHashes.py — used by the Wine subprocess on Linux
# ---------------------------------------------------------------------------

_GENERATE_HASHES_SRC = r'''"""
PhotoDNA hash generator — runs inside Wine Python on Linux/macOS.
Usage: python generateHashes.py <img1> [img2 ...]
Output: one CSV line per image (144 uint8 values).
"""
import sys, os, ctypes
from ctypes import c_char_p, c_int, c_ubyte, POINTER
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PhotoDNAx64.dll")

_lib = ctypes.cdll.LoadLibrary(DLL_PATH)
_fn  = _lib.ComputeRobustHash
_fn.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
_fn.restype  = c_ubyte


def compute(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    buf = (c_ubyte * 144)()
    _fn(c_char_p(img.tobytes()), img.width, img.height, 0, buf, 0)
    return ",".join(str(v) for v in buf)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generateHashes.py <image_path> [...]", file=sys.stderr)
        sys.exit(1)
    for path in sys.argv[1:]:
        print(compute(path))
'''

# ---------------------------------------------------------------------------
# Shared image helper
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
        raise RuntimeError(
            f"Expected 144 hash values, got {len(values)}: {line[:120]}"
        )
    return np.array(values, dtype=np.uint8)


# ===========================================================================
# Windows: direct ctypes backend
# ===========================================================================

class _WindowsBackend:
    """Calls PhotoDNAx64.dll directly via ctypes — no Wine needed."""

    def __init__(self, dll_path: str) -> None:
        if not os.path.isfile(dll_path):
            raise FileNotFoundError(
                f"[PhotoDNA] DLL not found: {dll_path}\n"
                "Set PHOTODNA_DLL env var or pass dll_path= to PhotoDNAWrapper().\n"
                "Obtain the DLL by extracting it from the FTK ISO with 7-Zip:\n"
                "  https://d1kpmuwb7gvu1i.cloudfront.net/AD_FTK_7.0.0.iso"
            )
        try:
            lib = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"[PhotoDNA] Failed to load DLL '{dll_path}': {e}") from e

        fn = lib.ComputeRobustHash
        fn.argtypes = [
            ctypes.c_char_p,               # image bytes (RGB24, row-major)
            ctypes.c_int,                  # width
            ctypes.c_int,                  # height
            ctypes.c_int,                  # mode (0 = default)
            ctypes.POINTER(ctypes.c_ubyte),# output buffer (144 bytes)
            ctypes.c_int,                  # reserved (0)
        ]
        fn.restype = ctypes.c_ubyte
        self._fn = fn
        print(f"[PhotoDNA] Loaded DLL (Windows native): {dll_path}")

    def compute(self, image: np.ndarray) -> np.ndarray:
        img = _to_uint8_rgb(image)
        pil = Image.fromarray(img)
        buf = (ctypes.c_ubyte * 144)()
        self._fn(
            ctypes.c_char_p(pil.tobytes()),
            pil.width, pil.height,
            0, buf, 0,
        )
        return np.array(list(buf), dtype=np.uint8)

    def compute_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.compute(img) for img in images]


# ===========================================================================
# Linux / macOS: Wine subprocess backend
# ===========================================================================

class _WineBackend:
    """Calls PhotoDNAx64.dll through a Wine Python subprocess."""

    def __init__(self, work_dir: str) -> None:
        wine_python = os.path.join(work_dir, _WINE_PYTHON_SUBPATH)
        script      = os.path.join(work_dir, "vendor", "generateHashes.py")

        if not os.path.isfile(wine_python) or not os.path.isfile(script):
            raise RuntimeError(
                "[PhotoDNA] Not set up for Linux.\n"
                "Run: from evohash.hashes.photodna import setup_photodna; setup_photodna()"
            )
        self._wine_python = wine_python
        self._script      = script

    def _run(self, args: List[str], timeout: int = 120) -> str:
        result = subprocess.run(
            ["wine64", self._wine_python, self._script, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "WINEDEBUG": "-all"},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PhotoDNA Wine subprocess failed (rc={result.returncode}):\n"
                f"{result.stderr}"
            )
        return result.stdout

    def compute(self, image: np.ndarray) -> np.ndarray:
        img_u8 = _to_uint8_rgb(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
            Image.fromarray(img_u8).save(tmp)
        try:
            stdout = self._run([tmp])
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
        return _parse_hash_line(stdout.splitlines()[0])

    def compute_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        tmps: List[str] = []
        try:
            for img in images:
                img_u8 = _to_uint8_rgb(img)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmps.append(f.name)
                    Image.fromarray(img_u8).save(f.name)
            stdout = self._run(tmps)
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


# ===========================================================================
# Setup helpers (Linux only — Windows needs no automated setup)
# ===========================================================================

def _shell(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True)


def _apt_install(*packages: str) -> None:
    missing = [p for p in packages if not shutil.which(p)]
    if not missing:
        return
    print(f"[PhotoDNA] apt-get install {' '.join(missing)} …")
    _shell("apt-get update -qq")
    _shell(f"apt-get install -y -q {' '.join(missing)}")


def _ensure_vendor_script(work_dir: str) -> None:
    """Write generateHashes.py and symlink DLL into work_dir/vendor/."""
    vendor_dir = os.path.join(work_dir, "vendor")
    os.makedirs(vendor_dir, exist_ok=True)

    script_path = os.path.join(vendor_dir, "generateHashes.py")
    with open(script_path, "w") as f:
        f.write(_GENERATE_HASHES_SRC)

    dll_src = os.path.join(work_dir, "PhotoDNAx64.dll")
    dll_lnk = os.path.join(vendor_dir, "PhotoDNAx64.dll")
    if os.path.isfile(dll_src) and not os.path.exists(dll_lnk):
        os.symlink(dll_src, dll_lnk)


def setup_photodna(work_dir: str = _DEFAULT_WORK_DIR, force: bool = False) -> str:
    """One-time setup.

    Windows : prints guidance on where to place the DLL; no automated steps.
    Linux   : installs Wine, downloads FTK ISO, extracts DLL, fetches Wine Python.

    Returns
    -------
    str  Path to the executable used to run PhotoDNA
         (the DLL path on Windows, or Wine Python path on Linux).
    """
    if _IS_WINDOWS:
        dll = _DEFAULT_DLL_PATH or os.path.join(work_dir, "PhotoDNAx64.dll")
        if os.path.isfile(dll):
            print(f"[PhotoDNA] Windows — DLL found: {dll} ✓")
            return dll
        print(
            "[PhotoDNA] Windows setup:\n"
            "  1. Extract PhotoDNAx64.dll from the FTK ISO using 7-Zip:\n"
            "       https://d1kpmuwb7gvu1i.cloudfront.net/AD_FTK_7.0.0.iso\n"
            "       Path inside ISO: FTK/FTK/X64/_8A89F09/DATA1.CAB → photodnax64.*.dll\n"
            "  2. Rename it to PhotoDNAx64.dll and place it anywhere, then either:\n"
            f"     a) Put it in: {work_dir}\n"
            "     b) Set env var: set PHOTODNA_DLL=C:\\path\\to\\PhotoDNAx64.dll\n"
            "     c) Pass dll_path= to PhotoDNAWrapper()\n"
            "  No further steps needed — ctypes loads it natively."
        )
        return ""

    # ---- Linux / macOS ----
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
        _apt_install("wine64", "cabextract", "genisoimage", "curl")

        if not os.path.isfile(dll_path):
            print("[PhotoDNA] Downloading FTK ISO (~3.3 GB) …")
            _shell(
                "curl -L --progress-bar -o AD_FTK_7.0.0.iso "
                "https://d1kpmuwb7gvu1i.cloudfront.net/AD_FTK_7.0.0.iso"
            )
            _shell(
                "isoinfo -i AD_FTK_7.0.0.iso -x /FTK/FTK/X64/_8A89F09/DATA1.CAB "
                "> Data1.cab"
            )
            os.makedirs("tmp_cab", exist_ok=True)
            _shell("cabextract -d tmp_cab -q Data1.cab")

            found = None
            for root, _, files in os.walk("tmp_cab"):
                for fname in files:
                    if "photodna" in fname.lower() and fname.endswith(".dll"):
                        found = os.path.join(root, fname)
                        break

            if not found:
                raise RuntimeError(
                    "PhotoDNAx64.dll not found in FTK ISO — "
                    "the ISO internal path may have changed."
                )
            shutil.copy(found, dll_path)
            print(f"[PhotoDNA] DLL saved → {dll_path}")

            for fname in ("AD_FTK_7.0.0.iso", "Data1.cab"):
                if os.path.isfile(fname):
                    os.remove(fname)
            shutil.rmtree("tmp_cab", ignore_errors=True)

        if not os.path.isfile(wine_python):
            print("[PhotoDNA] Downloading minimal Wine Python 3.9 …")
            _shell(
                "curl -L --progress-bar -o wine_python_39.tar.gz "
                "https://github.com/jankais3r/pyPhotoDNA/releases/download/"
                "wine_python_39/wine_python_39.tar.gz"
            )
            _shell("tar -xf wine_python_39.tar.gz")
            os.remove("wine_python_39.tar.gz")

        _ensure_vendor_script(work_dir)
        print("[PhotoDNA] Setup complete ✓")
        return wine_python

    finally:
        os.chdir(orig_dir)


# ===========================================================================
# Public wrapper — platform-transparent
# ===========================================================================

class PhotoDNAWrapper:
    """Microsoft PhotoDNA perceptual hash (Windows + Linux).

    Windows : loads PhotoDNAx64.dll directly via ctypes.
    Linux   : calls DLL through a Wine Python subprocess.

    Backend is selected automatically based on the current platform and
    resolved lazily on the first compute() call.

    Parameters
    ----------
    threshold_p : float
        Maximum L1 distance for similarity (default 3855).
    dll_path : str
        Windows only — explicit path to PhotoDNAx64.dll.
        Falls back to PHOTODNA_DLL env var, then work_dir/PhotoDNAx64.dll.
    work_dir : str
        Directory used by setup_photodna() (Linux) or containing the DLL
        as a fallback (Windows).
    """

    def __init__(
        self,
        threshold_p: float = 3855.0,
        dll_path: str = _DEFAULT_DLL_PATH,
        work_dir: str = _DEFAULT_WORK_DIR,
    ) -> None:
        self.spec = HashSpec(
            hash_id="photodna",
            threshold_p=float(threshold_p),
            distance_name="l1",
        )
        self._dll_path = dll_path
        self._work_dir = work_dir
        self._backend: Optional[_WindowsBackend | _WineBackend] = None

    def _resolve(self) -> None:
        if self._backend is not None:
            return
        if _IS_WINDOWS:
            # Priority: explicit arg > env var (read lazily!) > work_dir fallback
            dll = (
                self._dll_path
                or os.environ.get("PHOTODNA_DLL", "")
                or os.path.join(self._work_dir, "PhotoDNAx64.dll")
            )
            self._backend = _WindowsBackend(dll)
        else:
            self._backend = _WineBackend(self._work_dir)

    # ------------------------------------------------------------------
    # HashFunction interface
    # ------------------------------------------------------------------

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute PhotoDNA hash for a single image.

        Returns np.ndarray uint8, shape (144,).

        Linux note: spawns a Wine process each call; prefer compute_batch()
        for multiple images.
        """
        self._resolve()
        return self._backend.compute(image)

    def compute_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Compute hashes for multiple images.

        On Linux, all images are processed in one Wine process invocation —
        significantly faster than calling compute() in a loop.
        On Windows, this is equivalent to calling compute() per image
        (ctypes has no subprocess overhead).
        """
        self._resolve()
        return self._backend.compute_batch(images)

    def distance(self, d1: Any, d2: Any) -> float:
        a = np.asarray(d1, dtype=np.int32)
        b = np.asarray(d2, dtype=np.int32)
        return float(np.abs(a - b).sum())