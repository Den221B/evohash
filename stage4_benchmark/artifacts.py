"""Artifact store for the efficient benchmark stage.

One run produces everything needed for analysis and the diploma figures, so the
attacks never have to be re-run:

```
<output_root>/<run_name>/
  manifest.json                         # config, per-hash budgets/attacks, skips
  metrics.csv                           # one row per (hash, attack, pair)
  summary.csv                           # aggregated per (hash, attack)
  metrics/<hash>/<attack>.json          # per-attack pair rows (resume unit)
  history/<hash>/<attack>.npz           # convergence per pair: (2, N) = [step, best_dist]
  images/<hash>/source/<pair_id>.png    # resized source (once per hash)
  images/<hash>/target/<pair_id>.png    # resized target (once per hash)
  images/<hash>/adv/<attack>/<pair_id>.png   # attacked image
```

``RunStore`` both writes (during a run) and reads (for notebooks / viz).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(image)).save(path)


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def downsample_history(values: Iterable[float], max_points: int) -> np.ndarray:
    """Return a (2, N) array [step_index, best_dist], optionally thinned."""
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((2, 0), dtype=np.float32)
    steps = np.arange(arr.size, dtype=np.float32)
    if max_points and arr.size > max_points:
        idx = np.linspace(0, arr.size - 1, max_points).round().astype(int)
        steps, arr = steps[idx], arr[idx]
    return np.stack([steps, arr], axis=0)


class RunStore:
    """Read/write artifacts for one benchmark run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.root = Path(run_dir)
        self._manifest: dict[str, Any] = {}
        self._metrics = None  # lazy pandas DataFrame
        self._summary = None

    # ---- creation / manifest --------------------------------------------- #

    @classmethod
    def create(cls, manifest: dict[str, Any], *, output_root: str | Path,
               run_name: str | None = None) -> "RunStore":
        run_name = run_name or time.strftime("bench_%Y%m%d_%H%M%S")
        store = cls(Path(output_root) / run_name)
        store.root.mkdir(parents=True, exist_ok=True)
        store._manifest = {
            "run_name": run_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.system(),
            "hashes": {},
            "skipped": {},
            **manifest,
        }
        store._write_manifest()
        return store

    @classmethod
    def open(cls, run_dir: str | Path) -> "RunStore":
        store = cls(run_dir)
        mf = store.root / "manifest.json"
        if mf.exists():
            store._manifest = json.loads(mf.read_text(encoding="utf-8"))
        return store

    def _write_manifest(self) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps(self._manifest, indent=2, default=str), encoding="utf-8"
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    def mark_skipped(self, hash_id: str, reason: str) -> None:
        self._manifest.setdefault("skipped", {})[hash_id] = reason
        self._write_manifest()

    def record_hash(self, hash_id: str, *, budget: int, attacks: list[str], n_pairs: int) -> None:
        self._manifest.setdefault("hashes", {})[hash_id] = {
            "budget": int(budget), "attacks": list(attacks), "n_pairs": int(n_pairs),
        }
        self._write_manifest()

    # ---- paths ----------------------------------------------------------- #

    def _img_dir(self, hash_id: str) -> Path:
        return self.root / "images" / _sanitize(hash_id)

    def _metrics_path(self, hash_id: str, attack: str) -> Path:
        return self.root / "metrics" / _sanitize(hash_id) / f"{_sanitize(attack)}.json"

    def _history_path(self, hash_id: str, attack: str) -> Path:
        return self.root / "history" / _sanitize(hash_id) / f"{_sanitize(attack)}.npz"

    def _adv_path(self, hash_id: str, attack: str, pair_id: str) -> Path:
        return self._img_dir(hash_id) / "adv" / _sanitize(attack) / f"{_sanitize(pair_id)}.png"

    def source_path(self, hash_id: str, pair_id: str) -> Path:
        return self._img_dir(hash_id) / "source" / f"{_sanitize(pair_id)}.png"

    def target_path(self, hash_id: str, pair_id: str) -> Path:
        return self._img_dir(hash_id) / "target" / f"{_sanitize(pair_id)}.png"

    # ---- writing --------------------------------------------------------- #

    def save_pair_images(self, hash_id: str, pairs: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
        """``pairs`` = list of (pair_id, source_img, target_img) already resized."""
        for pair_id, src, tgt in pairs:
            sp, tp = self.source_path(hash_id, pair_id), self.target_path(hash_id, pair_id)
            if not sp.exists():
                save_png(src, sp)
            if not tp.exists():
                save_png(tgt, tp)

    def has_attack(self, hash_id: str, attack: str) -> bool:
        return self._metrics_path(hash_id, attack).exists()

    def save_attack(
        self,
        hash_id: str,
        attack: str,
        attack_type: str,
        rows: list[dict[str, Any]],
        histories: dict[str, np.ndarray],
        adv_images: dict[str, np.ndarray],
    ) -> None:
        for row in rows:
            row.setdefault("hash_id", hash_id)
            row.setdefault("attack", attack)
            row.setdefault("attack_type", attack_type)
        mp = self._metrics_path(hash_id, attack)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

        hp = self._history_path(hash_id, attack)
        hp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hp, **{_sanitize(k): v for k, v in histories.items()})

        for pair_id, img in adv_images.items():
            save_png(img, self._adv_path(hash_id, attack, pair_id))

    def finalize(self) -> None:
        """Aggregate per-attack json into metrics.csv and summary.csv."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for mp in sorted((self.root / "metrics").glob("*/*.json")):
            rows.extend(json.loads(mp.read_text(encoding="utf-8")))
        df = pd.DataFrame(rows)
        df.to_csv(self.root / "metrics.csv", index=False)
        summary = aggregate_summary(df)
        summary.to_csv(self.root / "summary.csv", index=False)
        self._manifest["finalized"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_manifest()
        self._metrics, self._summary = df, summary

    # ---- reading --------------------------------------------------------- #

    @property
    def metrics(self):
        import pandas as pd
        if self._metrics is None:
            path = self.root / "metrics.csv"
            self._metrics = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return self._metrics

    @property
    def summary(self):
        import pandas as pd
        if self._summary is None:
            path = self.root / "summary.csv"
            self._summary = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return self._summary

    def hashes(self) -> list[str]:
        df = self.metrics
        return sorted(df["hash_id"].unique()) if len(df) else []

    def attacks(self, hash_id: str) -> list[str]:
        df = self.metrics
        if not len(df):
            return []
        return sorted(df.loc[df.hash_id == hash_id, "attack"].unique())

    def pairs(self, hash_id: str, attack: str | None = None) -> list[str]:
        df = self.metrics
        if not len(df):
            return []
        sel = df.hash_id == hash_id
        if attack is not None:
            sel &= df.attack == attack
        return list(df.loc[sel, "pair_id"].astype(str).unique())

    def history(self, hash_id: str, attack: str, pair_id: str) -> tuple[np.ndarray, np.ndarray]:
        path = self._history_path(hash_id, attack)
        if not path.exists():
            return np.zeros(0), np.zeros(0)
        with np.load(path) as data:
            key = _sanitize(pair_id)
            if key not in data:
                return np.zeros(0), np.zeros(0)
            arr = data[key]
        return arr[0], arr[1]

    def source_image(self, hash_id: str, pair_id: str) -> np.ndarray:
        return load_png(self.source_path(hash_id, pair_id))

    def target_image(self, hash_id: str, pair_id: str) -> np.ndarray:
        return load_png(self.target_path(hash_id, pair_id))

    def adv_image(self, hash_id: str, attack: str, pair_id: str) -> np.ndarray:
        return load_png(self._adv_path(hash_id, attack, pair_id))

    def images(self, hash_id: str, attack: str, pair_id: str) -> dict[str, np.ndarray]:
        return {
            "source": self.source_image(hash_id, pair_id),
            "target": self.target_image(hash_id, pair_id),
            "adv": self.adv_image(hash_id, attack, pair_id),
        }

    def row(self, hash_id: str, attack: str, pair_id: str) -> dict[str, Any]:
        df = self.metrics
        sel = (df.hash_id == hash_id) & (df.attack == attack) & (df.pair_id.astype(str) == str(pair_id))
        hits = df.loc[sel]
        return hits.iloc[0].to_dict() if len(hits) else {}


def aggregate_summary(df, *, alpha: float = 1.0 / 50.0, asr_floor: float = 0.05):
    """Aggregate per-pair rows into per (hash, attack) summary with the project score."""
    import pandas as pd

    if df is None or not len(df):
        return pd.DataFrame()
    out = []
    for (hash_id, attack), g in df.groupby(["hash_id", "attack"]):
        asr = float(g["success"].mean())
        mean_l2_raw = float(g["pixel_l2_raw"].mean())
        mean_queries = float(g["queries"].mean())
        gap = np.maximum(g["final_hash_l1"].to_numpy() - g["threshold"].to_numpy(), 0.0)
        target_metric = mean_l2_raw + alpha * mean_queries
        score = target_metric / max(asr, asr_floor) + (10_000.0 if asr == 0.0 else 0.0)
        out.append({
            "hash_id": hash_id,
            "attack": attack,
            "attack_type": g["attack_type"].iloc[0] if "attack_type" in g else "",
            "n_pairs": int(len(g)),
            "asr": asr,
            "n_success": int(g["success"].sum()),
            "mean_l2_raw": mean_l2_raw,
            "median_l2_raw": float(g["pixel_l2_raw"].median()),
            "mean_lpips": float(g["lpips"].mean()) if "lpips" in g else float("nan"),
            "mean_queries": mean_queries,
            "median_queries": float(g["queries"].median()),
            "mean_final_hash_l1": float(g["final_hash_l1"].mean()),
            "mean_gap_to_threshold": float(gap.mean()),
            "target_metric": target_metric,
            "score": score,
            "mean_time_sec": float(g["time_sec"].mean()) if "time_sec" in g else float("nan"),
            "budget": int(g["budget"].iloc[0]) if "budget" in g else None,
        })
    cols_sort = ["hash_id", "score"]
    return pd.DataFrame(out).sort_values(cols_sort).reset_index(drop=True)


__all__ = ["RunStore", "save_png", "load_png", "downsample_history", "aggregate_summary"]
