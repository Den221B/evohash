"""Render thesis-ready figures from saved experiment artifacts.

This module deliberately does not recompute expensive attacks or hash values.
It reads CSV/NPZ artifacts produced by the benchmark notebooks and redraws the
selected figures in one typography/style for the diploma.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


HASH_ORDER = ("phash", "pdq", "neuralhash", "photodna")
ASR_PANEL_ORDER = ("phash", "pdq", "photodna", "neuralhash")
HASH_LABELS = {
    "phash": "pHash",
    "pdq": "PDQ",
    "neuralhash": "NeuralHash",
    "photodna": "PhotoDNA",
}
DEFAULT_THRESHOLDS = {
    "phash": 12.0,
    "pdq": 92.0,
    "neuralhash": 17.0,
    "photodna": 3855.0,
}

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

HASH_STYLES = {
    "phash": dict(color=OKABE["blue"], marker="o", linestyle="-"),
    "pdq": dict(color=OKABE["orange"], marker="s", linestyle=(0, (4, 2))),
    "neuralhash": dict(color=OKABE["green"], marker="^", linestyle="-."),
    "photodna": dict(color=OKABE["vermillion"], marker="D", linestyle=(0, (1, 1))),
}

FAMILY_STYLES = {
    "optimized": dict(color=OKABE["vermillion"], marker="^", linestyle="-"),
    "evolved": dict(color=OKABE["purple"], marker="D", linestyle=(0, (3, 1, 1, 1))),
    "best-of-all": dict(color=OKABE["black"], marker="o", linestyle=(0, (4, 2))),
}
FAMILY_LABELS = {
    "optimized": "оптимизированные",
    "evolved": "эволюционные",
    "best-of-all": "лучшее",
}

ATTACK_ORDER = (
    "GigaEvolve",
    "OpenEvolve",
    "ATKScopes",
    "NES",
    "Prokos",
    "SimBA",
    "ZO-SignSGD",
)
ATTACK_STYLES = {
    "GigaEvolve": dict(color="#7A5C54", marker="o", linestyle="-"),
    "OpenEvolve": dict(color="#4C9A9A", marker="s", linestyle="-"),
    "ATKScopes": dict(color="#4C78A8", marker="D", linestyle="-"),
    "NES": dict(color="#F28E2B", marker="v", linestyle="-"),
    "Prokos": dict(color="#59A14F", marker="^", linestyle="-"),
    "SimBA": dict(color="#AF7AA1", marker="P", linestyle="-"),
    "ZO-SignSGD": dict(color="#6B6B6B", marker="X", linestyle="-"),
}
MILESTONE_MARKERS = {
    0.25: ("o", "25%"),
    0.50: ("s", "50%"),
    0.75: ("^", "75%"),
    0.90: ("D", "90%"),
    1.00: ("X", "100%"),
}

PANEL_LETTERS = "абвгдежз"

CONVERGENCE_SELECTION = {
    "phash": ("pHash_GigaEvolve", "GigaEvolve"),
    "pdq": ("nes", "NES"),
    "neuralhash": ("NeuralHash_GigaEvolve", "GigaEvolve"),
    "photodna": ("PhotoDNA_GigaEvolve", "GigaEvolve"),
}


def set_diploma_style(base: float = 14.0) -> None:
    """Apply one plot style for all diploma figures."""
    import locale as _locale
    for _loc in ("ru_RU.UTF-8", "Russian_Russia.1251", "ru_RU"):
        try:
            _locale.setlocale(_locale.LC_NUMERIC, _loc)
            break
        except _locale.Error:
            continue
    mpl.rcParams.update(
        {
            "axes.formatter.use_locale": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.2,
            "axes.grid": True,
            "grid.color": "0.86",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def default_paths(project_root: str | Path = ".") -> dict[str, Path]:
    root = Path(project_root)
    asr_root = root / "outputs" / "stage_bench_asr_push" / "push_zero_secret_all_hashes_2x_budget"
    return {
        "root": root,
        "asr_root": asr_root,
        "asr_analysis": asr_root / "analysis_asr",
        "asr_history": asr_root / "history",
        "benign": root / "outputs" / "benign_analysis",
        "openevolve": root / "outputs" / "openevolve_history",
        "out": root / "figures" / "diploma",
    }


def save_figure(fig: plt.Figure, out_dir: str | Path, stem: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": out / f"{stem}.pdf",
        "png": out / f"{stem}.png",
    }
    for path in paths.values():
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def render_all(project_root: str | Path = ".", *, out_dir: str | Path | None = None) -> dict[str, dict[str, Path]]:
    """Render all selected diploma figures from saved artifacts."""
    paths = default_paths(project_root)
    if out_dir is not None:
        paths["out"] = Path(out_dir)

    set_diploma_style()
    rendered: dict[str, dict[str, Path]] = {}
    rendered["openevolve_score"] = save_figure(
        plot_openevolve_score(paths["openevolve"] / "openevolve_champion_history.csv"),
        paths["out"],
        "openevolve_score_progress",
    )
    rendered["asr_threshold"] = save_figure(
        plot_asr_threshold_curves(paths["asr_analysis"] / "asr_threshold_curve.csv"),
        paths["out"],
        "asr_threshold_curves",
    )
    rendered["asr_milestones"] = save_figure(
        plot_asr_milestones(
            paths["asr_analysis"] / "diploma_ready_check" / "asr_milestones_by_attack.csv"
        ),
        paths["out"],
        "asr_required_threshold",
    )
    for hash_id in ASR_PANEL_ORDER:
        rendered[f"asr_convergence_{hash_id}"] = save_figure(
            plot_convergence_trajectory(
                paths["asr_history"],
                paths["asr_analysis"] / "convergence_thresholds",
                hash_id,
            ),
            paths["out"],
            f"asr_convergence_{hash_id}",
        )
    rendered["operating_grid"] = save_figure(
        plot_operating_grid(paths["benign"]),
        paths["out"],
        "defense_operating_grid",
    )
    rendered["consistency_gate"] = save_figure(
        plot_consistency_gate(paths["benign"] / "consistency_gate.csv"),
        paths["out"],
        "defense_consistency_gate",
    )
    rendered["input_purification"] = save_figure(
        plot_input_purification(paths["benign"] / "input_purification.csv"),
        paths["out"],
        "defense_input_purification",
    )
    return rendered


def plot_openevolve_score(history_csv: str | Path) -> plt.Figure:
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    for hash_id in HASH_ORDER:
        group = df[df["hash_id"] == hash_id].sort_values("champion_rank")
        if group.empty:
            continue
        style = HASH_STYLES[hash_id]
        y = group["score_best_so_far"] / group["score_initial"]
        rank = group["champion_rank"].astype(float)
        if len(rank) > 1 and float(rank.max()) > float(rank.min()):
            x = (rank - float(rank.min())) / (float(rank.max()) - float(rank.min()))
        else:
            x = rank * 0.0
        ax.plot(
            x,
            y,
            label=HASH_LABELS[hash_id],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
        )
    ax.set_xlabel("Нормированный номер улучшения")
    ax.set_ylabel("Относительный score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.58, 1.03)
    ax.legend(title="Хеш", ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def plot_asr_threshold_curves(threshold_curve_csv: str | Path) -> plt.Figure:
    df = pd.read_csv(threshold_curve_csv)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True)
    handles_by_label: dict[str, Any] = {}
    tau_handle = None

    for i, (ax, hash_id) in enumerate(zip(axes.ravel(), ASR_PANEL_ORDER)):
        group = df[df["hash_id"] == hash_id].copy()
        for attack in _ordered_attacks(group):
            cur = group[group["attack"] == attack].sort_values("tau_ratio")
            if cur.empty:
                continue
            style = ATTACK_STYLES.get(attack, dict(color="0.3", marker="o", linestyle="-"))
            (line,) = ax.plot(
                cur["tau_ratio"],
                cur["asr"],
                label=attack,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.55,
                markersize=4.0,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
            handles_by_label.setdefault(attack, line)

        tau_handle = ax.axvline(1.0, color="black", linestyle=":", linewidth=1.1, label=r"$\tau_0$")
        ax.set_title(f"{PANEL_LETTERS[i]})", loc="left", fontweight="bold")
        ax.set_title(HASH_LABELS[hash_id], loc="center")
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, color="0.88", linewidth=0.55)

    fig.supxlabel(r"Относительный порог коллизии $\tau/\tau_0$")
    fig.supylabel("ASR")
    legend_handles = [handles_by_label[a] for a in ATTACK_ORDER if a in handles_by_label]
    legend_labels = [a for a in ATTACK_ORDER if a in handles_by_label]
    if tau_handle is not None:
        legend_handles.append(tau_handle)
        legend_labels.append(r"$\tau_0$")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.91])
    return fig


def plot_asr_milestones(milestones_csv: str | Path) -> plt.Figure:
    df = pd.read_csv(milestones_csv)
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.2), sharey=False)
    level_handles: dict[str, Any] = {}

    for i, (ax, hash_id) in enumerate(zip(axes.ravel(), ASR_PANEL_ORDER)):
        group = df[df["hash_id"] == hash_id].copy()
        attacks = _ordered_attacks(group)
        y_positions = {attack: pos for pos, attack in enumerate(attacks)}

        for attack in attacks:
            cur = group[group["attack"] == attack].sort_values("level")
            if cur.empty:
                continue
            y = y_positions[attack]
            style = ATTACK_STYLES.get(attack, dict(color="0.3"))
            ax.hlines(
                y,
                xmin=float(cur["tau_ratio"].min()),
                xmax=float(cur["tau_ratio"].max()),
                color=style["color"],
                linewidth=1.55,
                alpha=0.75,
            )
            for _, row in cur.iterrows():
                level = float(row["level"])
                marker, label = MILESTONE_MARKERS.get(level, ("o", f"{int(level * 100)}%"))
                sc = ax.scatter(
                    float(row["tau_ratio"]),
                    y,
                    marker=marker,
                    s=28,
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
                level_handles.setdefault(label, sc)

        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.1)
        xmax = max(1.05, float(group["tau_ratio"].max()) * 1.08)
        ax.set_xlim(-0.02, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.1f}".replace(".", ",")))
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks)
        ax.invert_yaxis()
        ax.set_title(f"{PANEL_LETTERS[i]})", loc="left", fontweight="bold")
        ax.set_title(HASH_LABELS[hash_id], loc="center")
        ax.grid(axis="x", color="0.88", linewidth=0.55)
        ax.grid(axis="y", visible=False)

    fig.supxlabel(r"Порог коллизии $\tau/\tau_0$")
    ordered_level_labels = [label for _, label in MILESTONE_MARKERS.values()]
    fig.legend(
        [level_handles[label] for label in ordered_level_labels if label in level_handles],
        [label for label in ordered_level_labels if label in level_handles],
        title="Целевой ASR",
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    return fig


def plot_convergence_grid(history_dir: str | Path, levels_dir: str | Path) -> plt.Figure:
    history_dir = Path(history_dir)
    levels_dir = Path(levels_dir)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), sharey=False)
    cmap = mpl.colormaps.get_cmap("viridis")
    for i, (ax, hash_id) in enumerate(zip(axes.ravel(), HASH_ORDER)):
        attack_id, attack_label = CONVERGENCE_SELECTION[hash_id]
        npz_path = history_dir / hash_id / f"{attack_id}.npz"
        levels_path = levels_dir / f"{hash_id}_{attack_id}_convergence_threshold_levels.csv"
        histories = _load_histories(npz_path)
        finals = np.array([h[1][-1] for h in histories if len(h[1])])
        norm = mpl.colors.Normalize(vmin=float(np.nanmin(finals)), vmax=float(np.nanmax(finals)))
        for x, y in histories:
            ax.plot(x, y, color=cmap(norm(y[-1])), alpha=0.26, linewidth=0.7)

        tau0 = DEFAULT_THRESHOLDS[hash_id]
        ax.axhline(tau0, color="black", linestyle="--", linewidth=1.2, label=r"$\tau_0$")
        if levels_path.exists():
            levels = pd.read_csv(levels_path)
            for _, row in levels.iterrows():
                if float(row["level"]) >= 1.0:
                    continue
                ax.axhline(float(row["tau"]), color="0.35", linestyle=":", linewidth=0.75)
                ax.text(
                    0.985,
                    float(row["tau"]),
                    f"{int(round(float(row['level']) * 100))}%",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=8.5,
                    color="0.25",
                )
        ax.set_title(f"{PANEL_LETTERS[i]}) {HASH_LABELS[hash_id]}", loc="left", fontweight="bold")
        ax.text(
            0.98,
            0.94,
            attack_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="0.25",
        )
        ax.set_ylim(bottom=-0.5 if hash_id != "photodna" else -50)
    fig.supxlabel("Шаг истории")
    fig.supylabel("Дистанция в хеш-пространстве")
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 1.0])
    return fig


def plot_convergence_trajectory(history_dir: str | Path, levels_dir: str | Path, hash_id: str) -> plt.Figure:
    history_dir = Path(history_dir)
    levels_dir = Path(levels_dir)
    attack_id, attack_label = CONVERGENCE_SELECTION[hash_id]
    npz_path = history_dir / hash_id / f"{attack_id}.npz"
    levels_path = levels_dir / f"{hash_id}_{attack_id}_convergence_threshold_levels.csv"

    histories = _load_histories(npz_path)
    tau0 = DEFAULT_THRESHOLDS[hash_id]
    crossing_steps = np.array(
        [
            _first_crossing_step(steps, distances, tau0)
            for steps, distances in histories
        ],
        dtype=float,
    )
    finite_crossings = crossing_steps[np.isfinite(crossing_steps)]
    vmax = float(np.nanpercentile(finite_crossings, 95)) if len(finite_crossings) else 1.0
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trajectory_crossing",
        ["#d7c28a", "#9cc6b8", "#6fa6bb", "#5d6f99", "#43335f"],
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    max_y = tau0
    for (steps, distances), crossing in zip(histories, crossing_steps):
        finite_dist = np.asarray(distances, dtype=float)
        finite_dist = finite_dist[np.isfinite(finite_dist)]
        if len(finite_dist):
            max_y = max(max_y, float(np.nanpercentile(finite_dist, 95)))
        color = cmap(norm(crossing)) if np.isfinite(crossing) else "#b8b8b8"
        ax.plot(steps, distances, color=color, alpha=0.44, linewidth=1.0)

    if histories:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
        cbar.set_label(r"Шаг достижения $\tau_0$")

    final_distances = np.array([distances[-1] for _, distances in histories if len(distances)], dtype=float)
    tau0_asr = float((final_distances <= tau0).mean()) if len(final_distances) else float("nan")
    tau0_asr_txt = f"{tau0_asr:.2f}".replace(".", ",")
    ax.axhline(
        tau0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\tau_0$={tau0:g}, ASR={tau0_asr_txt}",
        zorder=4,
    )

    if levels_path.exists():
        levels = pd.read_csv(levels_path)
        if len(levels):
            max_y = max(max_y, float(levels["tau"].max()))
        top_y = max_y * 1.08 if max_y > 0 else 1.0
        for idx, row in levels.iterrows():
            level = float(row["level"])
            tau = float(row["tau"])
            asr_value = float(row["asr"])
            fact_txt = f"{asr_value:.2f}".replace(".", ",")
            color = ["#2F6DA3", "#4F8F55", "#8E6687", "#B06B42", "#6F6F6F"][idx % 5]
            linestyle = ["-", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))][idx % 5]
            ax.axhline(
                tau,
                color=color,
                linestyle=linestyle,
                linewidth=2.0 if np.isclose(tau, 0.0) else 1.35,
                label=rf"ASR {int(round(level * 100))}%: $\tau$={tau:g} (факт {fact_txt})",
                zorder=4,
            )
            label_y = tau + 0.025 * top_y if np.isclose(tau, 0.0) else tau
            ax.text(
                0.99,
                label_y,
                f"ASR {int(round(level * 100))}%",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=9,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.4),
            )
    else:
        top_y = max_y * 1.08 if max_y > 0 else 1.0

    bottom_y = -0.03 * top_y if levels_path.exists() and (pd.read_csv(levels_path)["tau"] == 0).any() else 0.0
    ax.set_ylim(bottom=bottom_y, top=top_y)
    ax.set_xlabel("Шаг атаки")
    ax.set_ylabel("Дистанция в хеш-пространстве")
    ax.grid(True, color="0.84", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9, edgecolor="0.85")
    fig.tight_layout()
    return fig


def plot_operating_grid(benign_dir: str | Path) -> plt.Figure:
    benign_dir = Path(benign_dir)
    frames = []
    for hash_id in HASH_ORDER:
        path = benign_dir / f"{hash_id}_operating_point.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)

    styles = [
        ("benign_recall", "Benign recall", OKABE["green"], "o", "-"),
        ("fpr", "FPR", OKABE["blue"], "s", (0, (4, 2))),
        ("asr_optimized", "ASR opt.", OKABE["vermillion"], "^", "-"),
        ("asr_evolved", "ASR evol.", OKABE["purple"], "D", (0, (3, 1, 1, 1))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharey=True)
    handles = labels = None
    for i, (ax, hash_id) in enumerate(zip(axes.ravel(), HASH_ORDER)):
        group = df[df["hash_id"] == hash_id].sort_values("tau")
        x = group["tau"] / DEFAULT_THRESHOLDS[hash_id]
        for col, label, color, marker, linestyle in styles:
            if col in group:
                ax.plot(
                    x,
                    group[col],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    label=label,
                )
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.1, label=r"$\tau_0$")
        ax.set_title(f"{PANEL_LETTERS[i]})", loc="left", fontweight="bold")
        ax.set_title(HASH_LABELS[hash_id], loc="center")
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="x", visible=False)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    fig.supxlabel(r"Относительный порог $\tau/\tau_0$")
    fig.supylabel("Доля")
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.94])
    return fig


def plot_consistency_gate(gate_csv: str | Path) -> plt.Figure:
    df = pd.read_csv(gate_csv)
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    for hash_id in HASH_ORDER:
        group = df[df["hash"] == hash_id].sort_values("delta")
        if group.empty:
            continue
        style = HASH_STYLES[hash_id]
        ax.plot(
            group["attack_asr_gate"],
            group["benign_recall_gate"],
            label=HASH_LABELS[hash_id],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
    ax.scatter(
        [0],
        [1],
        marker="*",
        s=150,
        color=OKABE["yellow"],
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="Идеал",
    )
    ax.set_xlabel(r"ASR атак после consistency gate ($\downarrow$ лучше)")
    ax.set_ylabel(r"Benign recall ($\uparrow$ лучше)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right", ncol=1, frameon=True, framealpha=0.88, edgecolor="0.85")
    fig.tight_layout()
    return fig


def plot_input_purification(purification_csv: str | Path) -> plt.Figure:
    df = pd.read_csv(purification_csv)
    piv = df.pivot(index="hash", columns="defense", values="attack_asr")
    order = [h for h in HASH_ORDER if h in piv.index]
    cols = (["none"] if "none" in piv.columns else []) + [c for c in piv.columns if c != "none"]
    piv = piv.loc[order, cols]
    defense_labels = {
        "none": "none",
        "blur0.5": "blur 0.5",
        "blur1.0": "blur 1.0",
        "jpeg50": "JPEG 50",
        "jpeg90": "JPEG 90",
    }
    colors = [OKABE["black"], OKABE["blue"], OKABE["skyblue"], OKABE["vermillion"], OKABE["orange"]]
    hatches = ["", "//", "\\\\", "xx", ".."]
    ax = piv.plot(
        kind="bar",
        figsize=(6.8, 4.0),
        width=0.82,
        color=colors[: len(cols)],
        edgecolor="white",
        linewidth=0.4,
    )
    for container, hatch in zip(ax.containers, hatches):
        for bar in container:
            bar.set_hatch(hatch)
    ax.set_ylabel("ASR атак")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels([HASH_LABELS.get(h, h) for h in piv.index], rotation=0)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [defense_labels.get(col, col) for col in cols],
        title="Input purification",
        ncol=len(cols),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
    )
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


def _ordered_attacks(frame: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(frame["attack"].dropna().astype(str)))
    ordered = [attack for attack in ATTACK_ORDER if attack in present]
    ordered.extend(sorted(attack for attack in present if attack not in set(ordered)))
    return ordered


def _first_crossing_step(steps: np.ndarray, distances: np.ndarray, tau: float) -> float:
    steps = np.asarray(steps, dtype=float)
    distances = np.asarray(distances, dtype=float)
    mask = distances <= float(tau)
    if not mask.any():
        return float("nan")
    return float(steps[int(np.argmax(mask))])


def _load_histories(npz_path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    npz = np.load(npz_path, allow_pickle=True)
    histories: list[tuple[np.ndarray, np.ndarray]] = []
    for key in npz.files:
        arr = np.asarray(npz[key], dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] == 0:
            continue
        histories.append((arr[0], arr[1]))
    return histories


__all__ = [
    "default_paths",
    "plot_asr_milestones",
    "plot_asr_threshold_curves",
    "plot_consistency_gate",
    "plot_convergence_grid",
    "plot_convergence_trajectory",
    "plot_input_purification",
    "plot_openevolve_score",
    "plot_operating_grid",
    "render_all",
    "save_figure",
    "set_diploma_style",
]
