"""Plot OpenEvolve champion score histories.

The exported ``openevolve_checkpoint/champions_*`` folders contain a
record-improvement chain for each hash.  Each next champion has a lower
``metrics.score`` than the previous one, so the champion rank is a stable axis
for showing how the best found program improved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HASH_ORDER = ("phash", "pdq", "photodna", "neuralhash")
HASH_LABELS = {
    "phash": "pHash",
    "pdq": "PDQ",
    "photodna": "PhotoDNA",
    "neuralhash": "NeuralHash",
}
HASH_COLORS = {
    "phash": "#3D5A80",
    "pdq": "#3D8C95",
    "photodna": "#A66A3F",
    "neuralhash": "#6F5A9A",
}
HASH_MARKERS = {
    "phash": "o",
    "pdq": "s",
    "photodna": "^",
    "neuralhash": "D",
}


def load_champion_history(
    checkpoint_root: str | Path = "openevolve_checkpoint",
) -> pd.DataFrame:
    """Load champion JSON sidecars into a tidy dataframe."""
    root = Path(checkpoint_root)
    rows: list[dict[str, Any]] = []
    for hash_dir in sorted(root.glob("champions_*")):
        if not hash_dir.is_dir():
            continue
        hash_id = hash_dir.name.replace("champions_", "")
        for path in sorted(hash_dir.glob("champion_*.json"), key=_rank_from_path):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            provenance = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
            metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
            score = _as_float(metrics.get("score"))
            if score is None:
                continue
            rank = _as_int(provenance.get("rank"), _rank_from_path(path))
            first_seen = provenance.get("checkpoint_first_seen")
            last_seen = provenance.get("checkpoint_last_seen")
            trace_path = provenance.get("trace_path")
            rows.append(
                {
                    "hash_id": hash_id,
                    "hash_label": HASH_LABELS.get(hash_id, hash_id),
                    "champion_rank": rank,
                    "id": data.get("id", path.stem),
                    "kind": provenance.get("kind", ""),
                    "score": score,
                    "fitness": _as_float(provenance.get("fitness")),
                    "asr": _as_float(metrics.get("asr")),
                    "mean_l2": _as_float(metrics.get("mean_l2_raw")),
                    "mean_queries": _as_float(metrics.get("mean_queries")),
                    "checkpoint_first_seen": first_seen,
                    "checkpoint_first_seen_n": _checkpoint_number(first_seen),
                    "checkpoint_last_seen": last_seen,
                    "checkpoint_last_seen_n": _checkpoint_number(last_seen),
                    "trace_path": trace_path,
                    "trace_order": _trace_order(trace_path),
                    "source_path": str(path),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_hash_order"] = df["hash_id"].map({h: i for i, h in enumerate(HASH_ORDER)}).fillna(99)
    df = df.sort_values(["_hash_order", "champion_rank"]).reset_index(drop=True)

    df["score_initial"] = df.groupby("hash_id")["score"].transform("first")
    df["score_best_so_far"] = df.groupby("hash_id")["score"].cummin()
    df["normalized_score"] = df["score"] / df["score_initial"]
    df["best_normalized_score"] = df["score_best_so_far"] / df["score_initial"]
    df["improvement_pct"] = (1.0 - df["score"] / df["score_initial"]) * 100.0
    df["best_improvement_pct"] = (1.0 - df["score_best_so_far"] / df["score_initial"]) * 100.0

    df = df.groupby("hash_id", group_keys=False).apply(_with_chronology_axis)
    return df.drop(columns=["_hash_order"], errors="ignore").reset_index(drop=True)


def summarize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize initial/final score and relative improvement per hash."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for hash_id in HASH_ORDER:
        group = df[df["hash_id"] == hash_id].sort_values("champion_rank")
        if group.empty:
            continue
        first = group.iloc[0]
        best = group.loc[group["score"].idxmin()]
        rows.append(
            {
                "hash_id": hash_id,
                "hash": HASH_LABELS.get(hash_id, hash_id),
                "n_champions": int(len(group)),
                "initial_score": float(first["score"]),
                "best_score": float(best["score"]),
                "score_reduction_pct": float((first["score"] - best["score"]) / first["score"] * 100.0),
                "best_champion_rank": int(best["champion_rank"]),
                "best_asr": float(best["asr"]) if pd.notna(best["asr"]) else np.nan,
                "best_mean_l2": float(best["mean_l2"]) if pd.notna(best["mean_l2"]) else np.nan,
                "best_mean_queries": float(best["mean_queries"]) if pd.notna(best["mean_queries"]) else np.nan,
                "chronology_source": str(best["chronology_source"]),
            }
        )
    return pd.DataFrame(rows)


def chronology_source_table(df: pd.DataFrame) -> pd.DataFrame:
    """Show which chronological signal is available for every hash."""
    rows = []
    for hash_id in HASH_ORDER:
        group = df[df["hash_id"] == hash_id]
        if group.empty:
            continue
        rows.append(
            {
                "hash_id": hash_id,
                "hash": HASH_LABELS.get(hash_id, hash_id),
                "n_champions": int(len(group)),
                "checkpoint_first_seen_present": int(group["checkpoint_first_seen_n"].notna().sum()),
                "trace_order_present": int(group["trace_order"].notna().sum()),
                "chronology_source": str(group["chronology_source"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_by_champion_rank(
    df: pd.DataFrame,
    *,
    normalized: bool = True,
    figsize: tuple[float, float] = (9.2, 5.2),
) -> plt.Figure:
    """Plot score against champion rank for all hashes on one axis."""
    y_col = "best_normalized_score" if normalized else "score_best_so_far"
    ylabel = "Нормированный score (меньше лучше)" if normalized else "Score (меньше лучше)"
    title = (
        "Динамика score в цепочке чемпионов OpenEvolve"
        if normalized
        else "Динамика абсолютного score в цепочке чемпионов OpenEvolve"
    )
    fig, ax = plt.subplots(figsize=figsize)
    for hash_id, group in _iter_hash_groups(df):
        ax.plot(
            group["champion_rank"],
            group[y_col],
            label=HASH_LABELS.get(hash_id, hash_id),
            color=HASH_COLORS.get(hash_id),
            marker=HASH_MARKERS.get(hash_id, "o"),
            linewidth=2.1,
            markersize=5.5,
        )
    ax.set_title(title)
    ax.set_xlabel("Номер улучшения в цепочке чемпионов")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(title="Хэш", ncol=2)
    fig.tight_layout()
    return fig


def plot_by_relative_chronology(
    df: pd.DataFrame,
    *,
    normalized: bool = True,
    figsize: tuple[float, float] = (9.2, 5.2),
) -> plt.Figure:
    """Plot score against the best available per-hash chronological axis."""
    y_col = "best_normalized_score" if normalized else "score_best_so_far"
    ylabel = "Нормированный score (меньше лучше)" if normalized else "Score (меньше лучше)"
    fig, ax = plt.subplots(figsize=figsize)
    for hash_id, group in _iter_hash_groups(df):
        source = str(group["chronology_source"].iloc[0])
        ax.plot(
            group["chronology_progress"],
            group[y_col],
            label=f"{HASH_LABELS.get(hash_id, hash_id)} ({_source_label(source)})",
            color=HASH_COLORS.get(hash_id),
            marker=HASH_MARKERS.get(hash_id, "o"),
            linewidth=2.1,
            markersize=5.5,
        )
    ax.set_title("Динамика score по нормированному ходу запуска OpenEvolve")
    ax.set_xlabel("Нормированный ход запуска внутри хэша")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(title="Хэш и ось", fontsize=9)
    fig.tight_layout()
    return fig


def plot_raw_small_multiples(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (9.5, 6.6),
) -> plt.Figure:
    """Plot raw score per hash in separate axes to avoid scale conflicts."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, hash_id in zip(axes.ravel(), HASH_ORDER):
        group = df[df["hash_id"] == hash_id].sort_values("champion_rank")
        if group.empty:
            ax.axis("off")
            continue
        ax.plot(
            group["champion_rank"],
            group["score_best_so_far"],
            color=HASH_COLORS.get(hash_id),
            marker=HASH_MARKERS.get(hash_id, "o"),
            linewidth=2.0,
            markersize=4.8,
        )
        ax.set_title(HASH_LABELS.get(hash_id, hash_id))
        ax.set_xlabel("Номер улучшения")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.28)
    fig.suptitle("Абсолютный score чемпионов OpenEvolve", y=0.99)
    fig.tight_layout()
    return fig


def plot_chronology_small_multiples(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (9.5, 6.6),
) -> plt.Figure:
    """Plot each hash against its best available chronological source."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, hash_id in zip(axes.ravel(), HASH_ORDER):
        group = df[df["hash_id"] == hash_id].sort_values("chronology_progress")
        if group.empty:
            ax.axis("off")
            continue
        source = str(group["chronology_source"].iloc[0])
        ax.plot(
            group["chronology_progress"],
            group["best_normalized_score"],
            color=HASH_COLORS.get(hash_id),
            marker=HASH_MARKERS.get(hash_id, "o"),
            linewidth=2.0,
            markersize=4.8,
        )
        ax.set_title(f"{HASH_LABELS.get(hash_id, hash_id)}: {_source_label(source)}")
        ax.set_xlabel("Нормированный ход запуска")
        ax.set_ylabel("score / начальный score")
        ax.grid(True, alpha=0.28)
    fig.suptitle("Проверка временной оси для чемпионов OpenEvolve", y=0.99)
    fig.tight_layout()
    return fig


def save_history_artifacts(
    df: pd.DataFrame,
    out_dir: str | Path = "outputs/openevolve_history",
    *,
    dpi: int = 220,
) -> dict[str, Path]:
    """Save tables and experimental figures for the champion histories."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_history(df)
    sources = chronology_source_table(df)

    paths: dict[str, Path] = {}
    paths["history_csv"] = out / "openevolve_champion_history.csv"
    paths["summary_csv"] = out / "openevolve_champion_summary.csv"
    paths["sources_csv"] = out / "openevolve_chronology_sources.csv"
    df.to_csv(paths["history_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    sources.to_csv(paths["sources_csv"], index=False)

    figures = {
        "rank_normalized_png": (
            "openevolve_score_by_champion_rank_normalized.png",
            plot_by_champion_rank(df, normalized=True),
        ),
        "rank_raw_small_multiples_png": (
            "openevolve_score_by_champion_rank_raw_small_multiples.png",
            plot_raw_small_multiples(df),
        ),
        "chronology_normalized_png": (
            "openevolve_score_by_chronology_normalized.png",
            plot_by_relative_chronology(df, normalized=True),
        ),
        "chronology_small_multiples_png": (
            "openevolve_score_by_chronology_small_multiples.png",
            plot_chronology_small_multiples(df),
        ),
    }
    for key, (filename, fig) in figures.items():
        path = out / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths[key] = path
    return paths


def _iter_hash_groups(df: pd.DataFrame):
    for hash_id in HASH_ORDER:
        group = df[df["hash_id"] == hash_id].sort_values("champion_rank")
        if not group.empty:
            yield hash_id, group


def _with_chronology_axis(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("champion_rank").copy()
    non_seed = group[group["champion_rank"] > 0]
    if non_seed["trace_order"].notna().sum() >= max(1, len(non_seed) // 2):
        source = "evaluation_trace"
        values = group["trace_order"].astype("float64")
        min_v = float(non_seed["trace_order"].min())
        max_v = float(non_seed["trace_order"].max())
        values = values.fillna(min_v)
    elif group["checkpoint_first_seen_n"].notna().sum() >= max(2, len(group) // 2):
        source = "checkpoint_first_seen"
        values = group["checkpoint_first_seen_n"].astype("float64")
        min_v = float(values.min())
        max_v = float(values.max())
    else:
        source = "champion_rank"
        values = group["champion_rank"].astype("float64")
        min_v = float(values.min())
        max_v = float(values.max())

    span = max(max_v - min_v, 1.0)
    group["chronology_source"] = source
    group["chronology_value"] = values
    group["chronology_progress"] = ((values - min_v) / span).clip(0.0, 1.0)
    max_rank = max(int(group["champion_rank"].max()), 1)
    group["rank_progress"] = group["champion_rank"] / max_rank
    return group


def _rank_from_path(path: Path) -> int:
    match = re.search(r"champion_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _checkpoint_number(value: Any) -> int | None:
    if value in (None, ""):
        return None
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def _trace_order(value: Any) -> int | None:
    if value in (None, ""):
        return None
    stem = Path(str(value)).stem
    match = re.match(r"0*(\d+)_", stem)
    return int(match.group(1)) if match else None


def _source_label(source: str) -> str:
    return {
        "checkpoint_first_seen": "checkpoint",
        "evaluation_trace": "trace",
        "champion_rank": "rank",
    }.get(source, source)


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "HASH_LABELS",
    "HASH_ORDER",
    "chronology_source_table",
    "load_champion_history",
    "plot_by_champion_rank",
    "plot_by_relative_chronology",
    "plot_chronology_small_multiples",
    "plot_raw_small_multiples",
    "save_history_artifacts",
    "summarize_history",
]
