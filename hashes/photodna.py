"""Microsoft PhotoDNA wrapper, Windows + Linux/macOS.

Windows backend:
    Direct ctypes call into PhotoDNAx64.dll. No Wine is needed.

Linux/macOS backend:
    Wine subprocess that runs a small Python script and calls the same DLL.
    Run ``setup_photodna()`` once to prepare ~/.cache/photodna.

Digest:
    np.ndarray uint8, shape (144,)
Distance:
    hash_l1 over the 144-byte digest.
Threshold:
    3855 by default.
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

from .base import HashSpec, numeric_l1, to_uint8_rgb

_IS_WINDOWS = platform.system() == "Windows"
_IS_LINUX = platform.system() == "Linux"

_DEFAULT_WORK_DIR = os.environ.get("PHOTODNA_WORK_DIR") or os.path.join(
    os.path.expanduser("~"), ".cache", "photodna"
)
_DEFAULT_DLL_PATH = os.environ.get("PHOTODNA_DLL", "")
_WINE_PYTHON_SUBPATH = "python-3.9.12-embed-amd64/python.exe"

_GENERATE_HASHES_SRC = r'''
import ctypes
import os
import sys
from ctypes import POINTER, c_char_p, c_int, c_ubyte
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PhotoDNAx64.dll")
_lib = ctypes.cdll.LoadLibrary(DLL_PATH)
_fn = _lib.ComputeRobustHash
_fn.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_ubyte), c_int]
_fn.restype = c_ubyte


def compute(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    data = ctypes.create_string_buffer(img.tobytes())
    buf = (c_ubyte * 144)()
    _fn(ctypes.cast(data, c_char_p), img.width, img.height, 0, buf, 0)
    return ",".join(str(int(v)) for v in buf)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generateHashes.py <img1> [img2 ...]", file=sys.stderr)
        sys.exit(1)
    for path in sys.argv[1:]:
        print(compute(path))
'''


def _parse_hash_line(line: str) -> np.ndarray:
    values = [int(v.strip()) for v in line.strip().split(",") if v.strip()]
    if len(values) != 144:
        raise RuntimeError(f"Expected 144 PhotoDNA values, got {len(values)}: {line[:120]}")
    return np.asarray(values, dtype=np.uint8)


class _WindowsBackend:
    def __init__(self, dll_path: str) -> None:
        if not os.path.isfile(dll_path):
            raise FileNotFoundError(
                f"[PhotoDNA] DLL not found: {dll_path}\n"
                "Set PHOTODNA_DLL or pass dll_path=PhotoDNAWrapper(...)."
            )
        try:
            lib = ctypes.CDLL(dll_path)
        except OSError as exc:
            raise RuntimeError(f"[PhotoDNA] Failed to load DLL '{dll_path}': {exc}") from exc

        fn = lib.ComputeRobustHash
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_ubyte
        self._fn = fn

    def compute(self, image: np.ndarray) -> np.ndarray:
        img = to_uint8_rgb(image)
        pil = Image.fromarray(img).convert("RGB")
        data = ctypes.create_string_buffer(pil.tobytes())
        buf = (ctypes.c_ubyte * 144)()
        self._fn(
            ctypes.cast(data, ctypes.c_char_p),
            pil.width,
            pil.height,
            0,
            buf,
            0,
        )
        return np.asarray(list(buf), dtype=np.uint8)

    def compute_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.compute(img) for img in images]


class _WineBackend:
    def __init__(self, work_dir: str) -> None:
        wine_python = os.path.join(work_dir, _WINE_PYTHON_SUBPATH)
        script = os.path.join(work_dir, "vendor", "generateHashes.py")
        dll = os.path.join(work_dir, "vendor", "PhotoDNAx64.dll")

        if not shutil.which("wine64"):
            raise RuntimeError(
                "[PhotoDNA] wine64 was not found. Run setup_photodna() on Linux "
                "or install wine64 manually."
            )
        if not os.path.isfile(wine_python) or not os.path.isfile(script) or not os.path.isfile(dll):
            raise RuntimeError(
                "[PhotoDNA] Linux backend is not set up. Run:\n"
                "  from evohash.hashes.photodna import setup_photodna\n"
                "  setup_photodna()"
            )

        self._wine_python = wine_python
        self._script = script

    def _run(self, image_paths: List[str], timeout: int = 120) -> str:
        result = subprocess.run(
            ["wine64", self._wine_python, self._script, *image_paths],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "WINEDEBUG": "-all"},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"[PhotoDNA] Wine subprocess failed, rc={result.returncode}:\n"
                f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
            )
        return result.stdout

    def compute(self, image: np.ndarray) -> np.ndarray:
        return self.compute_batch([image])[0]

    def compute_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        tmp_paths: List[str] = []
        try:
            for image in images:
                img = to_uint8_rgb(image)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmp_paths.append(f.name)
                Image.fromarray(img).save(tmp_paths[-1])

            stdout = self._run(tmp_paths)
        finally:
            for path in tmp_paths:
                if os.path.isfile(path):
                    os.remove(path)

        lines = [line for line in stdout.splitlines() if line.strip()]
        if len(lines) != len(images):
            raise RuntimeError(f"Expected {len(images)} PhotoDNA lines, got {len(lines)}")
        return [_parse_hash_line(line) for line in lines]


def _shell(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True)


def _apt_install(*packages: str) -> None:
    if not _IS_LINUX:
        return
    missing = [p for p in packages if not shutil.which(p)]
    if not missing:
        return
    _shell("apt-get update -qq")
    _shell(f"apt-get install -y -q {' '.join(missing)}")


def _ensure_vendor_files(work_dir: str) -> None:
    vendor_dir = os.path.join(work_dir, "vendor")
    os.makedirs(vendor_dir, exist_ok=True)

    script_path = os.path.join(vendor_dir, "generateHashes.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(_GENERATE_HASHES_SRC)

    dll_src = os.path.join(work_dir, "PhotoDNAx64.dll")
    dll_dst = os.path.join(vendor_dir, "PhotoDNAx64.dll")
    if os.path.isfile(dll_src):
        if os.path.exists(dll_dst):
            return
        try:
            os.symlink(dll_src, dll_dst)
        except OSError:
            shutil.copy2(dll_src, dll_dst)


def setup_photodna(work_dir: str = _DEFAULT_WORK_DIR, force: bool = False) -> str:
    """Prepare PhotoDNA backend.

    Windows:
        Returns DLL path if it exists, otherwise prints setup guidance.
    Linux/macOS:
        Installs/fetches Wine Python and extracts PhotoDNAx64.dll from FTK ISO.
        Requires shell tools and may require sudo/root for apt-get on Linux.
    """
    os.makedirs(work_dir, exist_ok=True)

    if _IS_WINDOWS:
        dll = _DEFAULT_DLL_PATH or os.path.join(work_dir, "PhotoDNAx64.dll")
        if os.path.isfile(dll):
            print(f"[PhotoDNA] Windows DLL found: {dll}")
            return dll
        print(
            "[PhotoDNA] Windows setup:\n"
            "  1. Extract PhotoDNAx64.dll from the FTK ISO.\n"
            f"  2. Put it into {work_dir}, or set PHOTODNA_DLL, or pass dll_path=.\n"
            "  3. Then create PhotoDNAWrapper()."
        )
        return ""

    wine_python = os.path.join(work_dir, _WINE_PYTHON_SUBPATH)
    dll_path = os.path.join(work_dir, "PhotoDNAx64.dll")

    if not force and os.path.isfile(wine_python) and os.path.isfile(dll_path):
        _ensure_vendor_files(work_dir)
        print(f"[PhotoDNA] Already set up at {work_dir}")
        return wine_python

    orig_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        _apt_install("wine64", "cabextract", "genisoimage", "curl")

        if force or not os.path.isfile(dll_path):
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
                    if "photodna" in fname.lower() and fname.lower().endswith(".dll"):
                        found = os.path.join(root, fname)
                        break
                if found:
                    break
            if not found:
                raise RuntimeError("PhotoDNAx64.dll not found in extracted FTK ISO")
            shutil.copy2(found, dll_path)

            for fname in ("AD_FTK_7.0.0.iso", "Data1.cab"):
                if os.path.isfile(fname):
                    os.remove(fname)
            shutil.rmtree("tmp_cab", ignore_errors=True)

        if force or not os.path.isfile(wine_python):
            _shell(
                "curl -L --progress-bar -o wine_python_39.tar.gz "
                "https://github.com/jankais3r/pyPhotoDNA/releases/download/"
                "wine_python_39/wine_python_39.tar.gz"
            )
            _shell("tar -xf wine_python_39.tar.gz")
            os.remove("wine_python_39.tar.gz")

        _ensure_vendor_files(work_dir)
        print(f"[PhotoDNA] Setup complete: {work_dir}")
        return wine_python
    finally:
        os.chdir(orig_dir)


class PhotoDNAWrapper:
    def __init__(
        self,
        threshold_p: float = 3855.0,
        dll_path: str = _DEFAULT_DLL_PATH,
        work_dir: str = _DEFAULT_WORK_DIR,
    ) -> None:
        self.spec = HashSpec(
            hash_id="photodna",
            threshold_p=float(threshold_p),
            distance_name="hash_l1",
        )
        self._dll_path = dll_path
        self._work_dir = work_dir
        self._backend: Optional[_WindowsBackend | _WineBackend] = None

    @property
    def hash_id(self) -> str:
        return self.spec.hash_id

    @property
    def threshold(self) -> float:
        return self.spec.threshold_p

    def _resolve(self) -> None:
        if self._backend is not None:
            return

        if _IS_WINDOWS:
            dll = (
                self._dll_path
                or os.environ.get("PHOTODNA_DLL", "")
                or os.path.join(self._work_dir, "PhotoDNAx64.dll")
            )
            self._backend = _WindowsBackend(dll)
        else:
            self._backend = _WineBackend(self._work_dir)

    def compute(self, image: np.ndarray) -> np.ndarray:
        self._resolve()
        assert self._backend is not None
        return self._backend.compute(image)

    def compute_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        self._resolve()
        assert self._backend is not None
        return self._backend.compute_batch(images)

    def distance(self, d1: Any, d2: Any) -> float:
        return numeric_l1(d1, d2)
