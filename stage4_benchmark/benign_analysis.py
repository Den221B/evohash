"""Benign recall / operating-point analysis for the defense section (4.2).

A matcher flags a collision when ``d(H(query), H(ref)) <= tau``. The operating
point is set by three distance populations:

* **genuine**  -- ``d(H(B(z)), H(z))`` for benign edits ``B`` the hash should
  survive (JPEG, resize, blur, brightness). ``benign_recall(tau) = P(<=tau)``.
* **impostor** -- ``d(H(z_i), H(z_j))`` for unrelated images. ``FPR(tau)``.
* **adversarial** -- attack distances (from the push experiment / benchmark).

benign recall lower-bounds how strict ``tau`` may be without losing legitimate
matches; attack ASR wants ``tau`` low; FPR upper-bounds it. The defenses
(threshold, consistency gate, input purification) are judged by whether they
drop ASR while keeping benign recall high.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from evohash.hashes import build_hash_fn
from evohash.hashes.defended import Defense
from evohash.metrics import to_float32
from evohash.optimizer import load_pairs
from evohash.preprocessing import ResizeSpec, apply_resize

DEFAULT_THRESHOLDS = {"phash": 12.0, "pdq": 92.0, "neuralhash": 17.0, "photodna": 3855.0}

# Legitimate edits a perceptual hash is designed to survive (photometric + mild
# geometric). Reported per-edit so a single failing transform is visible.
DEFAULT_BENIGN_EDITS: dict[str, Defense] = {
    "jpeg90": Defense("jpeg", 90),
    "jpeg75": Defense("jpeg", 75),
    "jpeg50": Defense("jpeg", 50),
    "resize0.75": Defense("resize", 0.75),
    "resize0.5": Defense("resize", 0.5),
    "blur0.5": Defense("blur", 0.5),
    "blur1.0": Defense("blur", 1.0),
    "bright1.1": Defense("brightness", 1.1),
    "bright0.9": Defense("brightness", 0.9),
}

# Squeezes for the consistency gate (the hash is ~invariant to these).
DEFAULT_SQUEEZE: list[Defense] = [Defense("jpeg", 85), Defense("median", 3)]


# --------------------------------------------------------------------------- #
# reference images                                                            #
# --------------------------------------------------------------------------- #

def reference_images(
    dataset_zip: str,
    *,
    split: str = "secret",
    n_pairs: int | None = 100,
    resize_size: int | None = 256,
    resample: str = "bilinear",
) -> list[tuple[str, np.ndarray]]:
    """Both source and target images, resized to the benchmark grid -> (id, img)."""
    pairs = load_pairs(dataset_zip, split=split, n_pairs=n_pairs)
    rs = ResizeSpec(size=resize_size, resample=resample)
    out: list[tuple[str, np.ndarray]] = []
    for i, p in enumerate(pairs):
        pid = str(getattr(p, "pair_id", i))
        out.append((f"{pid}__src", apply_resize(to_float32(p.x), rs)))
        out.append((f"{pid}__tgt", apply_resize(to_float32(p.y), rs)))
    return out


# --------------------------------------------------------------------------- #
# distance populations                                                        #
# --------------------------------------------------------------------------- #

def genuine_distances(images, hash_fn, edits: Mapping[str, Defense] = DEFAULT_BENIGN_EDITS):
    """Per (image, benign edit): self-distance d(H(edit(z)), H(z))."""
    import pandas as pd

    rows = []
    for iid, img in images:
        h0 = hash_fn.compute(img)
        for name, edit in edits.items():
            d = float(hash_fn.distance(hash_fn.compute(edit(img)), h0))
            rows.append({"image_id": iid, "edit": name, "dist": d})
    return pd.DataFrame(rows)


def impostor_distances(images, hash_fn, *, n_impostor: int | None = 5000, seed: int = 0):
    """Distances between unrelated images (random unordered pairs)."""
    import pandas as pd

    ids = [iid for iid, _ in images]
    digests = {iid: hash_fn.compute(img) for iid, img in images}
    n = len(ids)
    max_pairs = n * (n - 1) // 2
    n_take = max_pairs if n_impostor is None else min(int(n_impostor), max_pairs)
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    rows = []
    while len(seen) < n_take:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append({"a": ids[a], "b": ids[b],
                     "dist": float(hash_fn.distance(digests[ids[a]], digests[ids[b]]))})
    return pd.DataFrame(rows)


def recall_curve(genuine_df, taus: Sequence[float]):
    import pandas as pd
    d = genuine_df["dist"].to_numpy()
    return pd.DataFrame([{"tau": float(t), "benign_recall": float((d <= float(t)).mean())} for t in taus])


def fpr_curve(impostor_df, taus: Sequence[float]):
    import pandas as pd
    d = impostor_df["dist"].to_numpy()
    return pd.DataFrame([{"tau": float(t), "fpr": float((d <= float(t)).mean())} for t in taus])


def recall_at(genuine_df, tau: float) -> float:
    d = genuine_df["dist"].to_numpy()
    return float((d <= float(tau)).mean())


# --------------------------------------------------------------------------- #
# operating point: benign recall vs attack ASR vs FPR                          #
# --------------------------------------------------------------------------- #

def operating_point(hash_id, genuine_df, impostor_df, asr_envelope_df, taus: Sequence[float]):
    """Merge benign_recall(tau), FPR(tau) with attack ASR(tau) envelope per family.

    ``asr_envelope_df`` is the push ``asr_envelope_curve.csv`` (columns
    hash_id, family, tau, asr). Returns one row per tau with benign_recall, fpr
    and asr per family (optimized / evolved / best-of-all).
    """
    import pandas as pd

    rec = recall_curve(genuine_df, taus).set_index("tau")
    fpr = fpr_curve(impostor_df, taus).set_index("tau")
    out = rec.join(fpr)
    env = asr_envelope_df[asr_envelope_df["hash_id"] == hash_id]
    for fam, g in env.groupby("family"):
        s = g.set_index("tau")["asr"].reindex(taus)
        out[f"asr_{str(fam).replace('-', '_')}"] = s.to_numpy()
    out = out.reset_index().rename(columns={"index": "tau"})
    out.insert(0, "hash_id", hash_id)
    return out


# --------------------------------------------------------------------------- #
# consistency gate (benign vs adversarial self-distance)                       #
# --------------------------------------------------------------------------- #

def _self_distance(image, squeeze, hash_fn) -> float:
    sqs = squeeze if isinstance(squeeze, (list, tuple)) else [squeeze]
    h = hash_fn.compute(image)
    return max(float(hash_fn.distance(h, hash_fn.compute(s(image)))) for s in sqs)


def benign_self_distances(images, hash_fn, squeeze=DEFAULT_SQUEEZE,
                          edits: Mapping[str, Defense] = DEFAULT_BENIGN_EDITS):
    """Self-distance of benign queries (edited references) under squeeze."""
    import pandas as pd
    rows = []
    for iid, img in images:
        for name, edit in edits.items():
            rows.append({"image_id": iid, "edit": name,
                         "self_dist": _self_distance(edit(img), squeeze, hash_fn)})
    return pd.DataFrame(rows)


def attack_self_distances(store, hash_id, squeeze=DEFAULT_SQUEEZE, *, attacks=None):
    """Self-distance of saved adversarial images x_best under squeeze."""
    import pandas as pd
    hf = build_hash_fn(hash_id, thresholds=DEFAULT_THRESHOLDS)
    rows = []
    for attack in (attacks or store.attacks(hash_id)):
        for pid in store.pairs(hash_id, attack):
            adv = store.adv_image(hash_id, attack, pid)
            rows.append({"attack": attack, "pair_id": pid,
                         "self_dist": _self_distance(adv, squeeze, hf)})
    return pd.DataFrame(rows)


def gate_tradeoff(benign_self_df, attack_self_df, attack_dist_df, deltas: Sequence[float],
                  *, tau: float):
    """benign recall and attack pass-rate under the consistency gate per delta.

    ``attack_dist_df`` has the clean attack distance per pair (best/final) so a
    successful attack is one with dist<=tau; the gate additionally requires
    self_dist<=delta. ``benign_self_df`` gives the self-distance of benign
    queries (assumed to match within tau by construction of legitimate edits).
    """
    import pandas as pd
    b = benign_self_df["self_dist"].to_numpy()
    a_self = attack_self_df.set_index(["attack", "pair_id"])["self_dist"]
    a_dist = attack_dist_df.set_index(["attack", "pair_id"])["dist"]
    joined = pd.concat([a_dist.rename("dist"), a_self.rename("self")], axis=1).dropna()
    success = joined["dist"].to_numpy() <= float(tau)
    rows = []
    for d in deltas:
        d = float(d)
        rows.append({
            "delta": d,
            "benign_recall_gate": float((b <= d).mean()),
            "attack_asr_gate": float((success & (joined["self"].to_numpy() <= d)).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# input purification (benign recall + attack ASR under H∘purify)               #
# --------------------------------------------------------------------------- #

def purified_genuine_distances(images, hash_fn, purify: Defense,
                               edits: Mapping[str, Defense] = DEFAULT_BENIGN_EDITS):
    """Genuine distances when both reference and query go through ``purify``."""
    import pandas as pd
    rows = []
    for iid, img in images:
        h0 = hash_fn.compute(purify(img))
        for name, edit in edits.items():
            d = float(hash_fn.distance(hash_fn.compute(purify(edit(img))), h0))
            rows.append({"image_id": iid, "edit": name, "dist": d})
    return pd.DataFrame(rows)


def purified_attack_distances(store, hash_id, purify: Defense, *, attacks=None):
    """Attack distances when reference and adversarial image go through ``purify``."""
    import pandas as pd
    hf = build_hash_fn(hash_id, thresholds=DEFAULT_THRESHOLDS)
    rows = []
    for attack in (attacks or store.attacks(hash_id)):
        for pid in store.pairs(hash_id, attack):
            adv = store.adv_image(hash_id, attack, pid)
            tgt = store.target_image(hash_id, pid)
            d = float(hf.distance(hf.compute(purify(adv)), hf.compute(purify(tgt))))
            rows.append({"attack": attack, "pair_id": pid, "dist": d})
    return pd.DataFrame(rows)


def attack_distances_clean(store, hash_id, *, attacks=None):
    """Clean attack distance per (attack, pair) from the saved metrics."""
    import pandas as pd
    m = store.metrics
    sel = m[m.hash_id == hash_id].copy()
    if attacks is not None:
        sel = sel[sel.attack.isin(list(attacks))]
    return sel.rename(columns={"final_hash_l1": "dist"})[["attack", "pair_id", "dist"]]


# --------------------------------------------------------------------------- #
# figures (publication style: serif/Times, STIX math, grayscale-safe, despined)#
# --------------------------------------------------------------------------- #

HASH_ORDER = ["phash", "pdq", "neuralhash", "photodna"]
HASH_PRETTY = {"phash": "pHash", "pdq": "PDQ", "neuralhash": "NeuralHash", "photodna": "PhotoDNA"}

# Okabe-Ito colorblind-safe palette.
_OKABE = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}


def set_pub_style(base: float = 11) -> None:
    """Apply a thesis/paper-grade matplotlib style (serif/Times, STIX math, despined).

    Idempotent; call before plotting. Saved figures embed fonts (``pdf.fonttype=42``)
    so vector ``.pdf`` exports drop into LaTeX cleanly.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
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
        "lines.linewidth": 1.7,
        "lines.markersize": 4.8,
        "axes.grid": True,
        "grid.color": "0.85",
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
    })


# (column, label, color, marker, linestyle) -- distinct line+marker => B/W-safe.
_OP_STYLE = [
    ("benign_recall", "benign recall", _OKABE["green"], "o", "-"),
    ("fpr", "FPR (импостор)", _OKABE["blue"], "s", (0, (4, 2))),
    ("asr_optimized", "ASR optimized", _OKABE["vermillion"], "^", "-"),
    ("asr_evolved", "ASR evolved", _OKABE["purple"], "D", (0, (3, 1, 1, 1))),
]
_PANEL = "абвгдежз"


def plot_operating_grid(op_all, *, thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
                        use_ratio: bool = True, hashes: Sequence[str] | None = None):
    """2x2 operating-point grid: benign recall / FPR / ASR vs (relative) threshold."""
    import matplotlib.pyplot as plt
    set_pub_style()

    hs = list(hashes) if hashes else [h for h in HASH_ORDER if h in set(op_all.hash_id)]
    nrows = (len(hs) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(7.2, 2.9 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    handles = labels = None
    for i, (ax, h) in enumerate(zip(axes, hs)):
        o = op_all[op_all.hash_id == h].sort_values("tau")
        x = (o.tau / float(thresholds[h])) if use_ratio else o.tau
        for col, lab, c, m, ls in _OP_STYLE:
            if col in o and o[col].notna().any():
                ax.plot(x, o[col], color=c, marker=m, ls=ls, mec="white", mew=0.4, label=lab)
        ax.axvline(1.0 if use_ratio else float(thresholds[h]),
                   color=_OKABE["black"], ls=":", lw=1.1, label=r"$\tau_0$")
        ax.set_title(f"{_PANEL[i]})", loc="left", fontweight="bold")
        ax.set_title(HASH_PRETTY.get(h, h), loc="center")
        ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", visible=False)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for ax in axes[len(hs):]:
        ax.axis("off")
    fig.supxlabel(r"Относительный порог $\tau/\tau_0$" if use_ratio else r"Порог $\tau$")
    fig.supylabel("Доля")
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.93])
    return fig


def plot_gate_tradeoff(gate_all, *, hashes: Sequence[str] | None = None):
    """Single trade-off plot: benign recall vs attack ASR as delta varies (points)."""
    import matplotlib.pyplot as plt
    set_pub_style()

    hs = list(hashes) if hashes else [h for h in HASH_ORDER if h in set(gate_all.hash)]
    styles = {
        "phash": (_OKABE["blue"], "o", "-"),
        "pdq": (_OKABE["orange"], "s", (0, (4, 2))),
        "neuralhash": (_OKABE["green"], "^", "-."),
        "photodna": (_OKABE["vermillion"], "D", (0, (1, 1))),
    }
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for h in hs:
        g = gate_all[gate_all.hash == h].sort_values("delta")
        c, m, ls = styles.get(h, (_OKABE["black"], "o", "-"))
        ax.plot(g.attack_asr_gate, g.benign_recall_gate, color=c, marker=m, ls=ls,
                mec="white", mew=0.4, label=HASH_PRETTY.get(h, h))
    ax.scatter([0], [1], marker="*", s=200, color=_OKABE["yellow"], edgecolor="k",
               linewidth=0.6, zorder=6, label="идеал")
    ax.set_xlabel(r"ASR атак после gate ($\downarrow$ лучше)")
    ax.set_ylabel(r"benign recall ($\uparrow$ лучше)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.85")
    fig.tight_layout()
    return fig


def plot_purification_bars(purification, *, hashes: Sequence[str] | None = None):
    """Grouped bars of attack ASR per (hash, purify mode); benign recall ~1.0."""
    import matplotlib.pyplot as plt
    set_pub_style()

    piv = purification.pivot(index="hash", columns="defense", values="attack_asr")
    order = [h for h in HASH_ORDER if h in piv.index]
    piv = piv.loc[order]
    cols = (["none"] if "none" in piv.columns else []) + [c for c in piv.columns if c != "none"]
    piv = piv[cols]
    bar_colors = [_OKABE["black"], _OKABE["blue"], _OKABE["skyblue"],
                  _OKABE["vermillion"], _OKABE["orange"], _OKABE["green"]]
    hatches = ["", "//", "\\\\", "xx", "..", "++"]
    ax = piv.plot(kind="bar", figsize=(7.2, 3.8), width=0.82,
                  color=bar_colors[:len(cols)], edgecolor="white", linewidth=0.4)
    for cont, hatch in zip(ax.containers, hatches):
        for bar in cont:
            bar.set_hatch(hatch)
    ax.set_ylabel("ASR атак"); ax.set_xlabel(""); ax.set_ylim(0, 1.0)
    ax.set_xticklabels([HASH_PRETTY.get(h, h) for h in piv.index], rotation=0)
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.legend(title="purify", ncol=len(cols), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig = ax.get_figure(); fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_BENIGN_EDITS",
    "HASH_PRETTY",
    "set_pub_style",
    "plot_operating_grid",
    "plot_gate_tradeoff",
    "plot_purification_bars",
    "DEFAULT_SQUEEZE",
    "reference_images",
    "genuine_distances",
    "impostor_distances",
    "recall_curve",
    "fpr_curve",
    "recall_at",
    "operating_point",
    "benign_self_distances",
    "attack_self_distances",
    "gate_tradeoff",
    "purified_genuine_distances",
    "purified_attack_distances",
    "attack_distances_clean",
]
