"""Analysis helpers for push-to-zero ASR experiments.

The experiment runs attacks with a zero match threshold, so an attack stops only
on an exact hash collision or after exhausting the configured budget.  The saved
best observed hash distances can then be reused to compute ASR for arbitrary
thresholds.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import tables as T


DEFAULT_THRESHOLDS = {
    "phash": 12.0,
    "pdq": 92.0,
    "photodna": 3855.0,
    "neuralhash": 17.0,
}

DEFAULT_TAU_GRIDS = {
    "phash": [0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40],
    "pdq": [0, 25, 50, 75, 92, 110, 130, 150],
    "photodna": [0, 500, 1000, 2000, 3000, 3855, 4500, 5500],
    "neuralhash": [0, 4, 8, 12, 17, 20, 24, 32],
}

CONTROL_RATIOS = (0.0, 0.5, 1.0)

ATTACK_STYLES = {
    "ATKScopes": dict(color="#1f77b4", linestyle="-", marker="o"),
    "NES": dict(color="#222222", linestyle="--", marker="s"),
    "Prokos": dict(color="#d55e00", linestyle="-.", marker="^"),
    "SimBA": dict(color="#009e73", linestyle=":", marker="D"),
    "ZO-SignSGD": dict(color="#cc79a7", linestyle=(0, (3, 1, 1, 1)), marker="v"),
    "GigaEvolve": dict(color="#7f7f7f", linestyle="--", marker="P"),
    "OpenEvolve": dict(color="#0072b2", linestyle="-", marker="X"),
}

ENVELOPE_STYLES = {
    "optimized": dict(color="#111111", linestyle="-", marker="o", linewidth=2.0),
    "evolved": dict(color="#666666", linestyle="--", marker="s", linewidth=2.0),
    "best-of-all": dict(color="#0072b2", linestyle="-.", marker="^", linewidth=2.4),
}

DIPLOMA_HASH_ORDER = ("phash", "pdq", "photodna", "neuralhash")

DIPLOMA_ATTACK_STYLES = {
    "ATKScopes": dict(color="#3B6EA8", linestyle="-", marker="o"),
    "NES": dict(color="#C97B2B", linestyle="-", marker="s"),
    "Prokos": dict(color="#4E8F55", linestyle="-", marker="^"),
    "SimBA": dict(color="#A84A4A", linestyle="-", marker="D"),
    "ZO-SignSGD": dict(color="#7A5EA8", linestyle="-", marker="v"),
    "GigaEvolve": dict(color="#7B6156", linestyle="-", marker="P"),
    "OpenEvolve": dict(color="#3D8C95", linestyle="-", marker="X"),
}

DIPLOMA_MILESTONE_LEVELS = (0.25, 0.5, 0.75, 0.9, 1.0)
DIPLOMA_MILESTONE_MARKERS = {
    0.25: "o",
    0.5: "s",
    0.75: "^",
    0.9: "D",
    1.0: "X",
}

DIPLOMA_TRAJECTORY_CMAP = ["#D9C99B", "#8DB6A3", "#4F86A8", "#3D5A80", "#44355B"]


def pretty_attack(name: str, attack_type: str = "") -> str:
    """Return the short attack name used in diploma tables and plots."""
    low = str(name).lower()
    if attack_type == "evolved" or "evolve" in low:
        if "giga" in low:
            return "GigaEvolve"
        if "open" in low:
            return "OpenEvolve"
        return str(name)
    return T.PRETTY.get(str(name), str(name))


def fmt_float(x: Any, nd: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"


def fmt_int(x: Any) -> str:
    if pd.isna(x):
        return "--"
    return f"{int(round(float(x))):,}".replace(",", " ")


def attack_push_summary(
    store: Any,
    *,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
    distance_col: str = "best_hash_l1",
) -> pd.DataFrame:
    """Aggregate exact-collision and default-threshold ASR per attack.

    ``distance_col`` defaults to ``best_hash_l1`` because push-to-zero runs can
    continue after crossing a looser threshold and some attacks may end at a
    worse final point than the best point observed during the run.
    """
    df = store.metrics.copy()
    rows = []
    for (hash_id, attack, attack_type), group in df.groupby(["hash_id", "attack", "attack_type"]):
        dist = group[distance_col].astype(float)
        tau0 = float(thresholds[hash_id])
        rows.append(
            {
                "hash_id": hash_id,
                "hash": T.HASH_PRETTY.get(hash_id, hash_id),
                "attack_id": attack,
                "attack": pretty_attack(attack, attack_type),
                "family": "evolved" if attack_type == "evolved" else "optimized",
                "distance_col": distance_col,
                "asr_0": float((dist <= 0.0).mean()),
                "asr_tau0": float((dist <= tau0).mean()),
                "mean_distance": float(dist.mean()),
                "mean_final_dist": float(group["final_hash_l1"].astype(float).mean()),
                "mean_best_dist": float(group["best_hash_l1"].astype(float).mean()),
                "mean_l2_raw": float(group["pixel_l2_raw"].mean()),
                "mean_lpips": float(group["lpips"].mean()),
                "mean_queries": float(group["queries"].mean()),
                "budget": int(group["budget"].iloc[0]),
                "n_pairs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["hash_id", "asr_0", "asr_tau0", "mean_distance"],
        ascending=[True, False, False, True],
    )


def threshold_curve(
    store: Any,
    *,
    tau_grids: Mapping[str, Sequence[float]] = DEFAULT_TAU_GRIDS,
    distance_col: str = "best_hash_l1",
) -> pd.DataFrame:
    """Return ASR(tau) for each attack from cached best-observed distances."""
    df = store.metrics.copy()
    rows = []
    for (hash_id, attack, attack_type), group in df.groupby(["hash_id", "attack", "attack_type"]):
        dist = group[distance_col].astype(float).to_numpy()
        for tau in tau_grids[hash_id]:
            rows.append(
                {
                    "hash_id": hash_id,
                    "hash": T.HASH_PRETTY.get(hash_id, hash_id),
                    "attack_id": attack,
                    "attack": pretty_attack(attack, attack_type),
                    "family": "evolved" if attack_type == "evolved" else "optimized",
                    "distance_col": distance_col,
                    "tau": float(tau),
                    "tau_ratio": float(tau) / float(DEFAULT_THRESHOLDS[hash_id]),
                    "asr": float((dist <= float(tau)).mean()),
                }
            )
    return pd.DataFrame(rows)


def envelope_curve(
    store: Any,
    *,
    tau_grids: Mapping[str, Sequence[float]] = DEFAULT_TAU_GRIDS,
    distance_col: str = "best_hash_l1",
) -> pd.DataFrame:
    """Return best optimized/evolved/best-of-all ASR(tau) envelopes."""
    df = store.metrics.copy()
    rows = []
    for hash_id, hash_rows in df.groupby("hash_id"):
        groups = {
            "optimized": hash_rows[hash_rows.attack_type == "registry"],
            "evolved": hash_rows[hash_rows.attack_type == "evolved"],
            "best-of-all": hash_rows,
        }
        for family, group in groups.items():
            if not len(group):
                continue
            best_dist = group.groupby("pair_id")[distance_col].min().astype(float).to_numpy()
            for tau in tau_grids[hash_id]:
                rows.append(
                    {
                        "hash_id": hash_id,
                        "hash": T.HASH_PRETTY.get(hash_id, hash_id),
                        "family": family,
                        "distance_col": distance_col,
                        "tau": float(tau),
                        "tau_ratio": float(tau) / float(DEFAULT_THRESHOLDS[hash_id]),
                        "asr": float((best_dist <= float(tau)).mean()),
                    }
                )
    return pd.DataFrame(rows)


def control_points_frame(
    store: Any,
    *,
    ratios: Sequence[float] = CONTROL_RATIOS,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
    distance_col: str = "best_hash_l1",
) -> pd.DataFrame:
    """Compact table for ASR at 0, 0.5*tau0 and tau0.

    For each family and pair, the row with the smallest selected distance is
    used. Its query count is used for the family-level mean query estimate.
    """
    df = store.metrics.copy()
    rows = []
    for hash_id, hash_rows in df.groupby("hash_id"):
        groups = {
            "optimized": hash_rows[hash_rows.attack_type == "registry"],
            "evolved": hash_rows[hash_rows.attack_type == "evolved"],
            "best-of-all": hash_rows,
        }
        tau0 = float(thresholds[hash_id])
        for family, group in groups.items():
            if not len(group):
                continue
            idx = group.groupby("pair_id")[distance_col].idxmin()
            best = group.loc[idx].copy()
            row = {
                "hash_id": hash_id,
                "hash": T.HASH_PRETTY.get(hash_id, hash_id),
                "family": family,
                "distance_col": distance_col,
                "mean_distance": float(best[distance_col].astype(float).mean()),
                "mean_final_dist": float(best["final_hash_l1"].astype(float).mean()),
                "mean_best_dist": float(best["best_hash_l1"].astype(float).mean()),
                "mean_queries": float(best["queries"].astype(float).mean()),
                "n_pairs": int(len(best)),
            }
            for ratio in ratios:
                tau = ratio * tau0
                if ratio == 0:
                    label = "ASR(0)"
                elif ratio == 1:
                    label = "ASR(tau0)"
                else:
                    label = f"ASR({ratio:g}tau0)"
                row[label] = float((best[distance_col].astype(float) <= tau).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def format_push_summary(df: pd.DataFrame) -> pd.DataFrame:
    """String-format the per-attack push summary for notebook display."""
    out = df.copy()
    out["ASR(0)"] = out["asr_0"].map(lambda x: fmt_float(x, 2))
    out["ASR(tau0)"] = out["asr_tau0"].map(lambda x: fmt_float(x, 2))
    out["mean best d"] = out["mean_distance"].map(lambda x: fmt_float(x, 2))
    out["L2"] = out["mean_l2_raw"].map(lambda x: fmt_float(x, 1))
    out["LPIPS"] = out["mean_lpips"].map(lambda x: fmt_float(x, 3))
    out["Queries"] = out["mean_queries"].map(fmt_int)
    out["Budget"] = out["budget"].map(fmt_int)
    return out[
        ["hash", "attack", "family", "ASR(0)", "ASR(tau0)", "mean best d", "L2", "LPIPS", "Queries", "Budget", "n_pairs"]
    ]


def format_control_points(df: pd.DataFrame) -> pd.DataFrame:
    """String-format control points for notebook display."""
    out = df.copy()
    for col in out.columns:
        if col.startswith("ASR("):
            out[col] = out[col].map(lambda x: fmt_float(x, 2))
    out["mean best d"] = out["mean_distance"].map(lambda x: fmt_float(x, 2))
    out["Queries"] = out["mean_queries"].map(fmt_int)
    return out[["hash", "family", *[c for c in out.columns if c.startswith("ASR(")], "mean best d", "Queries", "n_pairs"]]


def format_asr_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Format ASR pivot tables to two decimal places."""
    out = pivot.copy()
    for col in out.columns:
        if isinstance(col, (int, float, np.integer, np.floating)):
            out[col] = out[col].map(lambda x: fmt_float(x, 2))
    return out


def save_or_show(fig, out_path: str | Path | None = None):
    """Save figure if ``out_path`` is provided and always return it."""
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_attack_threshold_curves(
    curve: pd.DataFrame,
    hash_id: str = "phash",
    *,
    out_path: str | Path | None = None,
):
    """Plot detailed ASR(tau) curves for one hash and all attacks."""
    import matplotlib.pyplot as plt

    sub = curve[curve.hash_id == hash_id].sort_values("tau")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for attack, group in sub.groupby("attack", sort=False):
        style = ATTACK_STYLES.get(attack, dict(linestyle="-", marker="o"))
        ax.plot(group["tau"], group["asr"], label=attack, markersize=4, linewidth=1.7, **style)
    tau0 = float(DEFAULT_THRESHOLDS[hash_id])
    ax.axvline(tau0, color="black", linestyle=(0, (2, 2)), linewidth=1.2, label=f"default tau={tau0:g}")
    ax.set_title(f"{T.HASH_PRETTY.get(hash_id, hash_id)}: ASR(tau), push-to-zero, budget x2")
    ax.set_xlabel("collision threshold tau")
    ax.set_ylabel("ASR")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    return save_or_show(fig, out_path)


def plot_envelope_small_multiples(
    envelope: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
):
    """Plot normalized ASR envelopes for all available hashes."""
    import matplotlib.pyplot as plt

    hashes = [h for h in T.HASH_PRETTY if h in set(envelope.hash_id)]
    hashes.extend(sorted(h for h in set(envelope.hash_id) if h not in T.HASH_PRETTY))
    n = len(hashes)
    cols = 2 if n > 1 else 1
    rows = int(ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.6, 3.2 * rows), squeeze=False, sharey=True)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, hash_id in zip(axes.ravel(), hashes):
        ax.set_visible(True)
        sub = envelope[envelope.hash_id == hash_id].sort_values("tau_ratio")
        for family, group in sub.groupby("family", sort=False):
            style = ENVELOPE_STYLES.get(family, dict(linestyle="-", marker="o"))
            ax.plot(group["tau_ratio"], group["asr"], label=family, markersize=4, **style)
        ax.axvline(1.0, color="black", linestyle=(0, (2, 2)), linewidth=1.1)
        ax.set_title(T.HASH_PRETTY.get(hash_id, hash_id))
        ax.set_xlabel("tau / tau0")
        ax.set_ylabel("ASR")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save_or_show(fig, out_path)


def first_crossing(steps: np.ndarray, distances: np.ndarray, tau: float) -> float | None:
    """First history step where distance crosses ``tau``."""
    mask = np.asarray(distances, dtype=float) <= float(tau)
    if not mask.any():
        return None
    return float(np.asarray(steps, dtype=float)[int(np.argmax(mask))])


def convergence_events(
    store: Any,
    hash_id: str,
    attack: str,
    *,
    tau: float | None = None,
    levels: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 1.0),
) -> pd.DataFrame:
    """Return approximate first steps where cumulative ASR reaches each level."""
    tau = float(DEFAULT_THRESHOLDS[hash_id] if tau is None else tau)
    pair_ids = store.pairs(hash_id, attack)
    crossings = []
    for pair_id in pair_ids:
        steps, distances = store.history(hash_id, attack, pair_id)
        if len(steps) == 0:
            continue
        crossing = first_crossing(steps, distances, tau)
        if crossing is not None:
            crossings.append(crossing)
    crossings = sorted(crossings)
    rows = []
    n_pairs = len(pair_ids)
    for level in levels:
        needed = int(ceil(float(level) * n_pairs))
        if needed <= len(crossings):
            rows.append({"level": float(level), "step": float(crossings[needed - 1]), "reached": True})
        else:
            rows.append({"level": float(level), "step": np.nan, "reached": False})
    return pd.DataFrame(rows)


def convergence_threshold_levels(
    store: Any,
    hash_id: str,
    attack: str,
    *,
    levels: Sequence[float] = DIPLOMA_MILESTONE_LEVELS,
    distance_col: str = "best_hash_l1",
) -> pd.DataFrame:
    """Return horizontal distance thresholds that correspond to ASR levels.

    The thresholds are computed from per-pair best distances for one selected
    attack.  For example, the 0.9 row is the smallest observed distance
    threshold at which at least 90% of pairs are counted as successful.
    """
    df = store.metrics
    sel = (df.hash_id == hash_id) & (df.attack == attack)
    rows_df = df.loc[sel].copy()
    if not len(rows_df):
        return pd.DataFrame(columns=["level", "tau", "asr"])

    dist = rows_df[distance_col].astype(float).sort_values().to_numpy()
    n = len(dist)
    rows = []
    for level in levels:
        idx = min(max(int(ceil(float(level) * n)) - 1, 0), n - 1)
        tau = float(dist[idx])
        rows.append(
            {
                "level": float(level),
                "tau": tau,
                "asr": float((dist <= tau).mean()),
                "n_pairs": int(n),
            }
        )
    return pd.DataFrame(rows)


def select_convergence_attack(summary: pd.DataFrame, hash_id: str) -> str:
    """Pick a representative attack for convergence plots."""
    sub = summary[summary.hash_id == hash_id].sort_values(
        ["asr_0", "asr_tau0", "mean_distance"],
        ascending=[False, False, True],
    )
    if not len(sub):
        raise ValueError(f"No attacks found for {hash_id}")
    return str(sub.iloc[0]["attack_id"])


def select_convergence_threshold_attack(
    store: Any,
    hash_id: str,
    *,
    levels: Sequence[float] = DIPLOMA_MILESTONE_LEVELS,
    distance_col: str = "best_hash_l1",
) -> str:
    """Pick an attack whose trajectories make horizontal ASR levels readable."""
    df = store.metrics
    hash_rows = df[df.hash_id == hash_id]
    if not len(hash_rows):
        raise ValueError(f"No attacks found for {hash_id}")

    tau0 = float(DEFAULT_THRESHOLDS[hash_id])
    rows = []
    for attack, group in hash_rows.groupby("attack"):
        dist = group[distance_col].astype(float).sort_values().to_numpy()
        if len(dist) == 0:
            continue
        thresholds = []
        for level in levels:
            idx = min(max(int(ceil(float(level) * len(dist))) - 1, 0), len(dist) - 1)
            thresholds.append(float(dist[idx]))
        rounded = np.round(np.asarray(thresholds, dtype=float), 6)
        unique_levels = int(len(np.unique(rounded)))
        spread_ratio = float((np.nanmax(rounded) - np.nanmin(rounded)) / tau0) if tau0 > 0 else float(np.nanmax(rounded))
        tau0_asr = float((dist <= tau0).mean())
        exact_asr = float((dist <= 0.0).mean())
        mean_distance = float(np.mean(dist))
        rows.append(
            {
                "attack": attack,
                "unique_levels": unique_levels,
                "spread_ratio": spread_ratio,
                "tau0_asr": tau0_asr,
                "exact_asr": exact_asr,
                "mean_distance": mean_distance,
            }
        )

    if not rows:
        raise ValueError(f"No attacks found for {hash_id}")
    ranked = pd.DataFrame(rows).sort_values(
        ["unique_levels", "tau0_asr", "spread_ratio", "exact_asr", "mean_distance"],
        ascending=[False, False, False, True, True],
    )
    return str(ranked.iloc[0]["attack"])


def plot_convergence_histories(
    store: Any,
    hash_id: str,
    attack: str,
    *,
    tau: float | None = None,
    levels: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 1.0),
    max_pairs: int | None = 100,
    out_path: str | Path | None = None,
):
    """Plot per-pair convergence histories with ASR milestone lines."""
    import matplotlib.pyplot as plt

    tau = float(DEFAULT_THRESHOLDS[hash_id] if tau is None else tau)
    pair_ids = store.pairs(hash_id, attack)
    if max_pairs is not None:
        pair_ids = pair_ids[: int(max_pairs)]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    max_y = tau
    for pair_id in pair_ids:
        steps, distances = store.history(hash_id, attack, pair_id)
        if len(steps) == 0:
            continue
        distances = np.asarray(distances, dtype=float)
        max_y = max(max_y, float(np.nanpercentile(distances, 95)))
        ax.plot(steps, distances, color="#4a4a4a", alpha=0.16, linewidth=0.8)

    ax.axhline(tau, color="black", linestyle="--", linewidth=1.4, label=f"default tau={tau:g}")
    events = convergence_events(store, hash_id, attack, tau=tau, levels=levels)
    vstyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, row in events.iterrows():
        if not row["reached"]:
            continue
        level = float(row["level"])
        step = float(row["step"])
        ax.axvline(step, color="#0072b2", linestyle=vstyles[i % len(vstyles)], linewidth=1.2)
        ax.text(
            step,
            max_y * 0.96,
            f"{int(round(level * 100))}%",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#005a8d",
        )

    display_name = pretty_attack(attack, "evolved" if "evolve" in attack.lower() else "registry")
    ax.set_title(f"{T.HASH_PRETTY.get(hash_id, hash_id)} / {display_name}: convergence over pairs")
    ax.set_xlabel("recorded history step")
    ax.set_ylabel("best hash distance")
    ax.set_ylim(bottom=0, top=max_y * 1.05)
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8, frameon=True)
    return save_or_show(fig, out_path)


def plot_convergence_threshold_histories(
    store: Any,
    hash_id: str,
    attack: str,
    *,
    tau: float | None = None,
    levels: Sequence[float] = DIPLOMA_MILESTONE_LEVELS,
    distance_col: str = "best_hash_l1",
    max_pairs: int | None = 100,
    color_by: str = "tau0_crossing_step",
    out_path: str | Path | None = None,
):
    """Plot per-pair trajectories with horizontal ASR threshold levels."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors

    tau0 = float(DEFAULT_THRESHOLDS[hash_id] if tau is None else tau)
    pair_ids = store.pairs(hash_id, attack)
    if max_pairs is not None:
        pair_ids = pair_ids[: int(max_pairs)]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    max_y = tau0
    histories = []
    for pair_id in pair_ids:
        steps, distances = store.history(hash_id, attack, pair_id)
        if len(steps) == 0:
            continue
        distances = np.asarray(distances, dtype=float)
        best_distance = float(np.nanmin(distances))
        finite = distances[np.isfinite(distances)]
        if len(finite):
            max_y = max(max_y, float(np.nanpercentile(finite, 95)))
        histories.append((steps, distances, best_distance))

    if histories:
        if color_by in {"crossing_step", "tau0_crossing_step", "zero_crossing_step"}:
            crossing_tau = 0.0 if color_by == "zero_crossing_step" else tau0
            values = []
            for steps, distances, _best_distance in histories:
                crossing = first_crossing(steps, distances, crossing_tau)
                values.append(float(crossing) if crossing is not None else np.nan)
            finite_values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
            vmax = float(np.nanpercentile(finite_values, 95)) if len(finite_values) else 1.0
            norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1.0))
            cmap = mcolors.LinearSegmentedColormap.from_list("muted_crossing", DIPLOMA_TRAJECTORY_CMAP)
            colorbar_label = "Шаг первого достижения $\\tau=0$" if color_by == "zero_crossing_step" else "Шаг первого достижения $\\tau_0$"
        else:
            values = [best_distance / tau0 if tau0 > 0 else best_distance for *_rest, best_distance in histories]
            finite_values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
            vmax = float(np.nanpercentile(finite_values, 95)) if len(finite_values) else 1.0
            vmax = max(vmax, 1.5)
            norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
            cmap = mcolors.LinearSegmentedColormap.from_list("muted_distance", DIPLOMA_TRAJECTORY_CMAP)
            colorbar_label = "Лучшая дистанция / $\\tau_0$"

        for (steps, distances, _best_distance), value in zip(histories, values):
            if np.isfinite(value):
                color = cmap(norm(value))
            else:
                color = "#bdbdbd"
            ax.plot(steps, distances, color=color, alpha=0.42, linewidth=1.05)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.046)
        cbar.set_label(colorbar_label)

    levels_df = convergence_threshold_levels(
        store,
        hash_id,
        attack,
        levels=levels,
        distance_col=distance_col,
    )
    if len(levels_df):
        max_y = max(max_y, float(levels_df["tau"].max()))
    top_y = max_y * 1.08 if max_y > 0 else 1.0

    df = store.metrics
    sel = (df.hash_id == hash_id) & (df.attack == attack)
    attack_dist = df.loc[sel, distance_col].astype(float)
    tau0_asr = float((attack_dist <= tau0).mean()) if len(attack_dist) else float("nan")
    ax.axhline(
        tau0,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"$\\tau_0$={tau0:g}, ASR={tau0_asr:.2f}",
    )

    level_groups: list[dict[str, Any]] = []
    for _i, row in levels_df.iterrows():
        level = float(row["level"])
        tau_level = float(row["tau"])
        asr_value = float(row["asr"])
        for group in level_groups:
            if np.isclose(group["tau"], tau_level):
                group["levels"].append(level)
                group["asr"] = max(group["asr"], asr_value)
                break
        else:
            level_groups.append({"tau": tau_level, "levels": [level], "asr": asr_value})

    colors = ["#3B6EA8", "#4E8F55", "#8B5E83", "#B66A3C", "#6F6F6F"]
    linestyles = ["-", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    for i, group in enumerate(level_groups):
        tau_level = float(group["tau"])
        asr_value = float(group["asr"])
        level_values = [int(round(float(level) * 100)) for level in group["levels"]]
        if len(level_values) == 1:
            level_label = f"{level_values[0]}%"
        else:
            level_label = f"{min(level_values)}-{max(level_values)}%"
        ax.axhline(
            tau_level,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2.0 if np.isclose(tau_level, 0.0) else 1.5,
            label=f"ASR {level_label}: $\\tau$={tau_level:g} (факт {asr_value:.2f})",
            zorder=4,
        )
        label_y = tau_level + 0.025 * top_y if np.isclose(tau_level, 0.0) else tau_level
        ax.text(
            0.99,
            label_y,
            f"ASR {level_label}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color=colors[i % len(colors)],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        )

    display_name = pretty_attack(attack, "evolved" if "evolve" in attack.lower() else "registry")
    ax.set_title(f"{T.HASH_PRETTY.get(hash_id, hash_id)} / {display_name}: траектории расстояния в хеш-пространстве")
    ax.set_xlabel("Шаг истории атаки")
    ax.set_ylabel("Лучшая дистанция в хеш-пространстве")
    bottom_y = -0.03 * top_y if any(np.isclose(group["tau"], 0.0) for group in level_groups) else 0.0
    ax.set_ylim(bottom=bottom_y, top=top_y)
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    return save_or_show(fig, out_path)


def _ordered_hashes(frame: pd.DataFrame, hash_order: Sequence[str] = DIPLOMA_HASH_ORDER) -> list[str]:
    present = set(frame["hash_id"].dropna().astype(str))
    ordered = [h for h in hash_order if h in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _level_suffix(level: float) -> str:
    return str(int(round(float(level) * 100)))


def _style_for_display_attack(display_name: str) -> dict[str, Any]:
    return DIPLOMA_ATTACK_STYLES.get(
        display_name,
        dict(color="#333333", linestyle="-", marker="o"),
    )


def threshold_quantiles(
    envelope: pd.DataFrame,
    *,
    levels: Sequence[float] = (0.5, 0.9, 0.95, 0.99, 1.0),
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Return first threshold where each family envelope reaches ASR levels."""
    rows = []
    for (hash_id, family), group in envelope.groupby(["hash_id", "family"]):
        tau0 = float(thresholds[str(hash_id)])
        row: dict[str, Any] = {
            "scope": "family-envelope",
            "hash_id": hash_id,
            "hash": T.HASH_PRETTY.get(str(hash_id), str(hash_id)),
            "family": family,
            "tau0": tau0,
        }
        g = group.sort_values("tau_ratio")
        for level in levels:
            suffix = _level_suffix(float(level))
            hit = g[g["asr"].astype(float) >= float(level)]
            if hit.empty:
                row[f"tau_at_{suffix}asr"] = np.nan
                row[f"tau_ratio_at_{suffix}asr"] = np.nan
            else:
                first = hit.iloc[0]
                row[f"tau_at_{suffix}asr"] = float(first["tau"])
                row[f"tau_ratio_at_{suffix}asr"] = float(first["tau_ratio"])
        rows.append(row)
    return pd.DataFrame(rows)


def asr_milestone_table(
    curve: pd.DataFrame,
    *,
    levels: Sequence[float] = DIPLOMA_MILESTONE_LEVELS,
) -> pd.DataFrame:
    """Return per-attack threshold ratios needed to reach ASR milestones."""
    rows = []
    for (hash_id, attack_id), group in curve.groupby(["hash_id", "attack_id"]):
        g = group.sort_values("tau_ratio")
        family = str(g["family"].iloc[0])
        display_attack = str(g["attack"].iloc[0]) if "attack" in g else str(attack_id)
        for level in levels:
            hit = g[g["asr"].astype(float) >= float(level)]
            if hit.empty:
                tau = np.nan
                tau_ratio = np.nan
            else:
                first = hit.iloc[0]
                tau = float(first["tau"])
                tau_ratio = float(first["tau_ratio"])
            rows.append(
                {
                    "hash_id": hash_id,
                    "hash": T.HASH_PRETTY.get(str(hash_id), str(hash_id)),
                    "attack_id": attack_id,
                    "attack": display_attack,
                    "family": family,
                    "level": float(level),
                    "level_label": f"{int(round(float(level) * 100))}%",
                    "tau": tau,
                    "tau_ratio": tau_ratio,
                }
            )
    return pd.DataFrame(rows)


def plot_all_attack_threshold_curves(
    curve: pd.DataFrame,
    *,
    hash_order: Sequence[str] = DIPLOMA_HASH_ORDER,
    out_path: str | Path | None = None,
):
    """Plot normalized ASR curves for every attack with equal visual weight."""
    import matplotlib.pyplot as plt

    hashes = _ordered_hashes(curve, hash_order)
    cols = 2 if len(hashes) > 1 else 1
    rows = int(ceil(len(hashes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), squeeze=False, sharey=True)
    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, hash_id in zip(axes.ravel(), hashes):
        ax.set_visible(True)
        sub = curve[curve["hash_id"] == hash_id].copy()
        for attack_id, group in sub.groupby("attack_id", sort=False):
            g = group.sort_values("tau_ratio")
            display_attack = str(g["attack"].iloc[0]) if "attack" in g else str(attack_id)
            style = _style_for_display_attack(display_attack)
            ax.plot(
                g["tau_ratio"],
                g["asr"],
                label=display_attack,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=5,
                alpha=0.95,
            )
        ax.axvline(1.0, color="black", linestyle=":", linewidth=2.0, label="τ₀")
        ax.set_title(T.HASH_PRETTY.get(str(hash_id), str(hash_id)), fontsize=16)
        ax.set_xlabel("τ / τ₀")
        ax.set_ylabel("ASR")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.9)

    fig.suptitle("Зависимость ASR от выбранного порога коллизии", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_or_show(fig, out_path)


def plot_asr_milestones(
    curve: pd.DataFrame,
    milestones: pd.DataFrame | None = None,
    *,
    hash_order: Sequence[str] = DIPLOMA_HASH_ORDER,
    levels: Sequence[float] = DIPLOMA_MILESTONE_LEVELS,
    out_path: str | Path | None = None,
):
    """Plot threshold-ratio milestones required to reach selected ASR levels."""
    import matplotlib.pyplot as plt

    if milestones is None:
        milestones = asr_milestone_table(curve, levels=levels)

    hashes = _ordered_hashes(curve, hash_order)
    cols = 2 if len(hashes) > 1 else 1
    rows = int(ceil(len(hashes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, hash_id in zip(axes.ravel(), hashes):
        ax.set_visible(True)
        sub_curve = curve[curve["hash_id"] == hash_id]
        sub_ms = milestones[milestones["hash_id"] == hash_id]
        attacks = list(sub_curve["attack_id"].drop_duplicates())
        y_positions = np.arange(len(attacks))
        ax.axvline(1.0, color="black", linestyle=":", linewidth=2.0)

        for y, attack_id in zip(y_positions, attacks):
            attack_curve = sub_curve[sub_curve["attack_id"] == attack_id]
            display_attack = str(attack_curve["attack"].iloc[0]) if "attack" in attack_curve else str(attack_id)
            style = _style_for_display_attack(display_attack)
            attack_ms = sub_ms[sub_ms["attack_id"] == attack_id]
            xs = attack_ms["tau_ratio"].dropna()
            if not xs.empty:
                ax.hlines(y, xs.min(), xs.max(), color=style["color"], linewidth=2.2, alpha=0.85)
            for level in levels:
                row = attack_ms[attack_ms["level"] == float(level)]
                if row.empty or pd.isna(row.iloc[0]["tau_ratio"]):
                    continue
                ax.scatter(
                    float(row.iloc[0]["tau_ratio"]),
                    y,
                    color=style["color"],
                    marker=DIPLOMA_MILESTONE_MARKERS.get(float(level), "o"),
                    s=58,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [
                str(sub_curve[sub_curve["attack_id"] == a]["attack"].iloc[0])
                if "attack" in sub_curve
                else str(a)
                for a in attacks
            ],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_title(T.HASH_PRETTY.get(str(hash_id), str(hash_id)), fontsize=16)
        ax.set_xlabel("τ / τ₀")
        ax.grid(True, axis="x", alpha=0.35)
        xmax = sub_ms["tau_ratio"].dropna().max()
        ax.set_xlim(left=-0.03, right=max(1.05, min(float(xmax) * 1.08, 2.4)) if pd.notna(xmax) else 1.05)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=DIPLOMA_MILESTONE_MARKERS.get(float(level), "o"),
            color="black",
            linestyle="None",
            markersize=8,
            label=f"{int(round(float(level) * 100))}%",
        )
        for level in levels
    ]
    fig.legend(handles=legend_handles, title="Уровень ASR", loc="lower center", ncol=len(levels), frameon=True)
    fig.suptitle("Порог, необходимый для достижения заданного ASR", fontsize=18)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    return save_or_show(fig, out_path)


def _ensure_hash_id(control: pd.DataFrame) -> pd.DataFrame:
    out = control.copy()
    if "hash_id" in out.columns:
        return out
    name_to_id = {name: hash_id for hash_id, name in T.HASH_PRETTY.items()}
    out["hash_id"] = out["hash"].map(name_to_id)
    out["hash_id"] = out["hash_id"].fillna(out["hash"].astype(str).str.lower())
    return out


def diploma_push_tables(control: pd.DataFrame, envelope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact formatted tables for the diploma body and appendix."""
    quant = threshold_quantiles(envelope, levels=(0.9, 1.0))
    control_display = _ensure_hash_id(control)
    quant_cols = quant[
        [
            "hash_id",
            "family",
            "tau0",
            "tau_at_90asr",
            "tau_ratio_at_90asr",
            "tau_at_100asr",
            "tau_ratio_at_100asr",
        ]
    ]
    merged = control_display.merge(quant_cols, on=["hash_id", "family"], how="left")

    hash_rank = {hash_id: i for i, hash_id in enumerate(DIPLOMA_HASH_ORDER)}
    family_rank = {"optimized": 0, "evolved": 1, "best-of-all": 2}
    merged["_hash_rank"] = merged["hash_id"].map(hash_rank).fillna(999)
    merged["_family_rank"] = merged["family"].map(family_rank).fillna(999)
    merged = merged.sort_values(["_hash_rank", "_family_rank", "hash_id", "family"])

    def format_rows(frame: pd.DataFrame, include_family: bool) -> pd.DataFrame:
        out = pd.DataFrame()
        out["hash"] = frame["hash_id"].map(lambda h: T.HASH_PRETTY.get(str(h), str(h)))
        if include_family:
            out["family"] = frame["family"]
        out["ASR(0)"] = frame["ASR(0)"].map(lambda x: fmt_float(x, 2))
        out["ASR(0.5tau0)"] = frame["ASR(0.5tau0)"].map(lambda x: fmt_float(x, 2))
        out["ASR(tau0)"] = frame["ASR(tau0)"].map(lambda x: fmt_float(x, 2))
        out["mean best d"] = frame["mean_best_dist"].map(lambda x: fmt_float(x, 2))
        out["queries"] = frame["mean_queries"].map(fmt_int)
        out["tau90/tau0"] = frame["tau_ratio_at_90asr"].map(lambda x: fmt_float(x, 2))
        out["tau100/tau0"] = frame["tau_ratio_at_100asr"].map(lambda x: fmt_float(x, 2))
        return out

    body = format_rows(merged[merged["family"] == "best-of-all"], include_family=False)
    appendix = format_rows(merged, include_family=True)
    return body, appendix


def save_diploma_push_artifacts(
    out_dir: str | Path,
    *,
    curve: pd.DataFrame,
    envelope: pd.DataFrame,
    control: pd.DataFrame,
) -> dict[str, Path]:
    """Save diploma-ready ASR push figures, milestone tables and TeX tables."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "all_attack_curves": out / "asr_threshold_all_attacks_equal.png",
        "milestone_plot": out / "asr_milestones_threshold_ratios.png",
        "milestone_csv": out / "asr_milestones_by_attack.csv",
        "body_csv": out / "diploma_push_body_table.csv",
        "body_tex": out / "diploma_push_body_table.tex",
        "appendix_csv": out / "diploma_push_appendix_table.csv",
        "appendix_tex": out / "diploma_push_appendix_table.tex",
    }

    milestones = asr_milestone_table(curve)
    milestones.to_csv(paths["milestone_csv"], index=False)
    plot_all_attack_threshold_curves(curve, out_path=paths["all_attack_curves"])
    plot_asr_milestones(curve, milestones=milestones, out_path=paths["milestone_plot"])

    body, appendix = diploma_push_tables(control, envelope)
    body.to_csv(paths["body_csv"], index=False)
    body.to_latex(paths["body_tex"], index=False, escape=False)
    appendix.to_csv(paths["appendix_csv"], index=False)
    appendix.to_latex(paths["appendix_tex"], index=False, escape=False)
    return paths


def save_analysis_tables(
    out_dir: str | Path,
    *,
    summary: pd.DataFrame,
    curve: pd.DataFrame,
    envelope: pd.DataFrame,
    control: pd.DataFrame,
) -> Path:
    """Save analysis CSV files and return output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "push_summary.csv", index=False)
    curve.to_csv(out / "asr_threshold_curve.csv", index=False)
    envelope.to_csv(out / "asr_envelope_curve.csv", index=False)
    control.to_csv(out / "control_points.csv", index=False)
    threshold_quantiles(envelope).to_csv(out / "asr_threshold_quantiles.csv", index=False)
    return out


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_TAU_GRIDS",
    "DIPLOMA_HASH_ORDER",
    "DIPLOMA_MILESTONE_LEVELS",
    "attack_push_summary",
    "threshold_curve",
    "envelope_curve",
    "control_points_frame",
    "threshold_quantiles",
    "asr_milestone_table",
    "format_push_summary",
    "format_control_points",
    "format_asr_pivot",
    "plot_attack_threshold_curves",
    "plot_envelope_small_multiples",
    "plot_all_attack_threshold_curves",
    "plot_asr_milestones",
    "diploma_push_tables",
    "save_diploma_push_artifacts",
    "convergence_events",
    "convergence_threshold_levels",
    "select_convergence_attack",
    "select_convergence_threshold_attack",
    "plot_convergence_histories",
    "plot_convergence_threshold_histories",
    "save_analysis_tables",
]
