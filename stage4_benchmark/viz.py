"""Visualization helpers for benchmark artifacts (notebook / diploma figures).

All functions take a :class:`RunStore` and return a matplotlib ``Figure`` so they
can be shown inline or saved with ``fig.savefig(...)``. ``export_triple`` writes a
ready-to-insert PNG for the diploma; ``browse`` is an optional ipywidgets picker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .artifacts import RunStore


def _amplified_diff(source: np.ndarray, adv: np.ndarray, amplify: float) -> np.ndarray:
    return np.clip(np.abs(adv.astype(np.float32) - source.astype(np.float32)) * float(amplify), 0.0, 1.0)


def _metric_caption(store: RunStore, hash_id: str, attack: str, pair_id: str) -> str:
    r = store.row(hash_id, attack, pair_id)
    if not r:
        return ""
    ok = "✓" if r.get("success") else "✗"
    parts = [f"{ok} d={r.get('final_hash_l1'):.0f}/{r.get('threshold'):.0f}",
             f"L2={r.get('pixel_l2_raw'):.1f}", f"q={int(r.get('queries', 0))}"]
    lp = r.get("lpips")
    if lp is not None and np.isfinite(lp):
        parts.append(f"LPIPS={lp:.3f}")
    return "  ".join(parts)


def show_triple(
    store: RunStore,
    hash_id: str,
    attack: str,
    pair_id: str,
    *,
    amplify: float = 10.0,
    show_diff: bool = True,
):
    """Source | target | attacked (+ amplified difference) for one pair."""
    import matplotlib.pyplot as plt

    imgs = store.images(hash_id, attack, pair_id)
    panels = [("source", imgs["source"]), ("target", imgs["target"]), ("attacked", imgs["adv"])]
    if show_diff:
        panels.append((f"|adv−src|×{amplify:g}", _amplified_diff(imgs["source"], imgs["adv"], amplify)))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(np.clip(img, 0, 1)); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle(f"{hash_id} · {attack} · {pair_id}\n{_metric_caption(store, hash_id, attack, pair_id)}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def gallery(
    store: RunStore,
    hash_id: str,
    attack: str,
    pair_ids: Sequence[str],
    *,
    show_target: bool = True,
    amplify: float | None = None,
):
    """Grid: one row per pair = source | [target] | attacked | [diff]."""
    import matplotlib.pyplot as plt

    cols = 2 + int(show_target) + int(amplify is not None)
    n = len(pair_ids)
    fig, axes = plt.subplots(n, cols, figsize=(3.0 * cols, 3.0 * n))
    axes = np.atleast_2d(axes)
    for r, pair_id in enumerate(pair_ids):
        imgs = store.images(hash_id, attack, pair_id)
        seq = [("source", imgs["source"])]
        if show_target:
            seq.append(("target", imgs["target"]))
        seq.append((_metric_caption(store, hash_id, attack, pair_id), imgs["adv"]))
        if amplify is not None:
            seq.append(("diff", _amplified_diff(imgs["source"], imgs["adv"], amplify)))
        for c, (title, img) in enumerate(seq):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1)); ax.axis("off")
            ax.set_title(title, fontsize=8)
        axes[r, 0].set_ylabel(pair_id, fontsize=8)
    fig.suptitle(f"{hash_id} · {attack}", fontsize=11)
    fig.tight_layout()
    return fig


def convergence(
    store: RunStore,
    hash_id: str,
    attack: str,
    pair_ids: Sequence[str] | None = None,
    *,
    logy: bool = False,
):
    """Best hash-distance vs optimization step, one line per pair."""
    import matplotlib.pyplot as plt

    pair_ids = list(pair_ids) if pair_ids is not None else store.pairs(hash_id, attack)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    thr = None
    for pair_id in pair_ids:
        steps, dist = store.history(hash_id, attack, pair_id)
        if len(dist):
            ax.plot(steps, dist, lw=1.0, alpha=0.8, label=pair_id)
        r = store.row(hash_id, attack, pair_id)
        thr = r.get("threshold", thr)
    if thr is not None:
        ax.axhline(float(thr), color="gray", ls=":", label=f"τ={float(thr):g}")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("шаг оптимизации"); ax.set_ylabel("best hash-distance")
    ax.set_title(f"Сходимость · {hash_id} · {attack}")
    if len(pair_ids) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def export_triple(
    store: RunStore,
    hash_id: str,
    attack: str,
    pair_id: str,
    path: str | Path,
    *,
    amplify: float = 10.0,
    show_diff: bool = True,
    dpi: int = 200,
) -> Path:
    """Render and save a source|target|attacked figure for the diploma."""
    fig = show_triple(store, hash_id, attack, pair_id, amplify=amplify, show_diff=show_diff)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def browse(store: RunStore, *, amplify: float = 10.0):
    """Interactive picker (ipywidgets): choose hash/attack/pair, view + export."""
    import ipywidgets as widgets
    from IPython.display import display
    import matplotlib.pyplot as plt

    hashes = store.hashes()
    hash_dd = widgets.Dropdown(options=hashes, description="hash")
    attack_dd = widgets.Dropdown(options=store.attacks(hashes[0]) if hashes else [], description="attack")
    pair_dd = widgets.Dropdown(
        options=store.pairs(hashes[0], attack_dd.value) if hashes else [], description="pair")
    export_name = widgets.Text(value="outputs/diploma_figs/example.png", description="path")
    export_btn = widgets.Button(description="export PNG", button_style="success")
    out = widgets.Output()

    def _refresh_attacks(*_):
        attack_dd.options = store.attacks(hash_dd.value)

    def _refresh_pairs(*_):
        pair_dd.options = store.pairs(hash_dd.value, attack_dd.value)

    def _render(*_):
        with out:
            out.clear_output(wait=True)
            fig = show_triple(store, hash_dd.value, attack_dd.value, pair_dd.value, amplify=amplify)
            plt.show()

    def _export(_):
        p = export_triple(store, hash_dd.value, attack_dd.value, pair_dd.value, export_name.value, amplify=amplify)
        with out:
            print(f"saved -> {p}")

    hash_dd.observe(_refresh_attacks, "value")
    attack_dd.observe(_refresh_pairs, "value")
    for w in (hash_dd, attack_dd, pair_dd):
        w.observe(_render, "value")
    export_btn.on_click(_export)

    display(widgets.HBox([hash_dd, attack_dd, pair_dd]),
            widgets.HBox([export_name, export_btn]), out)
    _render()


__all__ = ["show_triple", "gallery", "convergence", "export_triple", "browse"]
