"""LaTeX table generators for the diploma, fed from benchmark artifacts.

Each function returns a LaTeX string for a ``table`` float (matching the diploma
style: ``\\hline`` + ``|``-columns) with the same ``\\caption``/``\\label`` as the
table it replaces, so the output is a drop-in. ``write_tables`` dumps them to
``.tex`` files for ``\\input``.

Data sources
------------
* per-hash benchmark numbers come from :class:`RunStore` (``store.summary``);
* the original->optimized comparison needs two stores (run the original config
  into one ``run_name`` and the optimized config into another);
* evolved attacks live in the optimized store (``attack_type == "evolved"``);
* threshold / defense tables come from the ``defense_experiment`` DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import RunStore

# Pretty display names for registry attacks.
PRETTY = {
    "atkscopes": "ATKScopes",
    "nes": "NES",
    "prokos": "Prokos",
    "simba": "SimBA",
    "zo_signsgd": "ZO-SignSGD",
    "nes_attack_v0": "NES-v0",
}
_REG_ORDER = ["atkscopes", "nes", "prokos", "simba", "zo_signsgd"]
_EXCLUDED_ATTACKS = {"nes_attack_v0"}
HASH_PRETTY = {"phash": "pHash", "pdq": "PDQ", "photodna": "PhotoDNA", "neuralhash": "NeuralHash"}
HASH_ORDER = tuple(HASH_PRETTY)


# --------------------------------------------------------------------------- #
# formatting helpers                                                          #
# --------------------------------------------------------------------------- #

def _f(x: Any, nd: int = 2, dash: str = "--") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(v):
        return dash
    return f"{v:.{nd}f}"


def _int(x: Any, dash: str = "--") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(v):
        return dash
    return f"{int(round(v)):,}".replace(",", "\\,")


def _pretty_attack(name: str, attack_type: str = "") -> str:
    if attack_type == "evolved" or "evolve" in name.lower():
        low = name.lower()
        if "giga" in low:
            return "GigaEvolve"
        if "open" in low:
            return "OpenEvolve"
        return name.split("_", 1)[-1]
    return PRETTY.get(name, name)


def _summary_frame(store_or_df: Any):
    """Return a normalized benchmark summary DataFrame.

    The current Stage-4 ``RunStore`` uses ``attack``/``attack_type`` columns.
    The older comparison runner used ``candidate``/``candidate_type``.  The
    table generators accept both, so notebooks can keep using existing runs.
    """
    import pandas as pd

    df = store_or_df.summary if hasattr(store_or_df, "summary") else store_or_df
    df = pd.DataFrame(df).copy()
    if df.empty:
        return df

    if "attack" not in df.columns and "candidate" in df.columns:
        df["attack"] = df["candidate"]
    if "attack_type" not in df.columns:
        if "candidate_type" in df.columns:
            df["attack_type"] = df["candidate_type"].replace({"config": "registry"})
        elif "config_variant" in df.columns:
            df["attack_type"] = np.where(df["config_variant"].astype(str).ne(""), "registry", "evolved")
        else:
            df["attack_type"] = ""

    if "status" in df.columns:
        df = df[df["status"].fillna("ok").eq("ok")]
    if "attack" in df.columns:
        df = df[~df["attack"].astype(str).isin(_EXCLUDED_ATTACKS)]
    for col in ("hash_id", "attack", "attack_type"):
        if col not in df.columns:
            df[col] = ""
    return df


def _hashes_from(*frames: Any, hashes: Sequence[str] | None = None) -> list[str]:
    if hashes is not None:
        return list(hashes)
    seen: set[str] = set()
    for frame in frames:
        df = _summary_frame(frame)
        if "hash_id" in df.columns:
            seen.update(str(h) for h in df["hash_id"].dropna().unique())
    ordered = [h for h in HASH_ORDER if h in seen]
    ordered.extend(sorted(h for h in seen if h not in HASH_PRETTY))
    return ordered


def _summary_row_from(df, hash_id: str, attack: str) -> dict[str, Any]:
    hit = df[(df.hash_id == hash_id) & (df.attack == attack)]
    return hit.iloc[0].to_dict() if len(hit) else {}


def _table(body: str, *, col_spec: str, header: str, caption: str, label: str,
           env: str = "tabular", size: str = "", tabcolsep: str | None = None,
           width: str = "\\textwidth") -> str:
    pre = []
    if tabcolsep is not None:
        pre.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    if size:
        pre.append(f"\\{size}")
    begin = (f"\\begin{{{env}}}{{{width}}}{{{col_spec}}}" if env == "tabularx"
             else f"\\begin{{{env}}}{{{col_spec}}}")
    end = f"\\end{{{env}}}"
    return "\n".join([
        "\\begin{table}[H]",
        "\\centering",
        *pre,
        begin,
        "\\hline",
        header + " \\\\",
        "\\hline",
        body.rstrip(),
        "\\hline",
        end,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])


def _summary_row(store: RunStore, hash_id: str, attack: str) -> dict[str, Any]:
    return _summary_row_from(_summary_frame(store), hash_id, attack)


# --------------------------------------------------------------------------- #
# 1. per-hash original -> optimized comparison                                #
# --------------------------------------------------------------------------- #

def latex_baseline_optimized(
    orig_store: RunStore,
    opt_store: RunStore,
    hash_id: str,
    *,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Reproduce ``tab:<hash>_baseline_optimized`` (original->optimized + dScore)."""
    s_o, s_p = _summary_frame(orig_store), _summary_frame(opt_store)
    reg_o = s_o[(s_o.hash_id == hash_id) & (s_o.attack_type == "registry")]
    reg_p = s_p[(s_p.hash_id == hash_id) & (s_p.attack_type == "registry")]
    attacks = set(reg_o["attack"]) | set(reg_p["attack"])
    attacks = [a for a in _REG_ORDER if a in attacks] + sorted(a for a in attacks if a not in _REG_ORDER)

    lines = []
    for a in attacks:
        ro = _summary_row_from(reg_o, hash_id, a)
        rp = _summary_row_from(reg_p, hash_id, a)
        pretty = _pretty_attack(a, "registry")
        so, sp = ro.get("score"), rp.get("score")
        if ro and rp and so and np.isfinite(so) and so > 0 and sp is not None and np.isfinite(sp):
            d = (float(so) - float(sp)) / float(so) * 100.0
            ds = (f"$-{abs(d):.1f}$\\%" if d < 0 else f"{d:.1f}\\%")
        else:
            ds = "--"

        def cells(r):
            if not r:
                return "-- & -- & -- & -- & --"
            return (f"{_f(r.get('asr'), 2)} & {_f(r.get('mean_l2_raw'), 1)} & "
                    f"{_f(r.get('mean_lpips'), 3)} & {_int(r.get('mean_queries'))} & "
                    f"{_f(r.get('score'), 1)}")

        lines.append(f"\\multirow{{2}}{{2.5cm}}{{{pretty}}} & original & {cells(ro)} & -- \\\\")
        lines.append(f"& optimized & {cells(rp)} & {ds} \\\\")
        lines.append("\\hline")

    caption = caption or f"Сравнение обычной и оптимизированной конфигураций для {HASH_PRETTY.get(hash_id, hash_id)}"
    label = label or f"tab:{hash_id}_baseline_optimized"
    header = ("\\textbf{Атака} & \\textbf{Конф.} & \\textbf{ASR} & \\textbf{$L_2$} & "
              "\\textbf{LPIPS} & \\textbf{Queries} & \\textbf{S} & \\textbf{$\\Delta S$}")
    body = "\n".join(lines[:-1])  # drop trailing \hline (added by _table)
    return _table(
        body,
        col_spec="|>{\\raggedright\\arraybackslash}p{2.5cm}|c|c|c|c|c|c|c|",
        header=header, caption=caption, label=label, env="tabular", tabcolsep="2pt",
    )


# --------------------------------------------------------------------------- #
# 2. evolved attacks per hash                                                  #
# --------------------------------------------------------------------------- #

def latex_evolved_results(
    store: RunStore,
    *,
    hashes: Sequence[str] | None = None,
    caption: str = "Результаты эволюционно найденных атак",
    label: str = "tab:evolved_results",
) -> str:
    s = _summary_frame(store)
    ev = s[s.attack_type == "evolved"]
    hashes = _hashes_from(ev, hashes=hashes)
    lines = []
    for h in hashes:
        rows = ev[ev.hash_id == h].sort_values("score")
        for _, r in rows.iterrows():
            lines.append(
                f"{HASH_PRETTY.get(h, h)} & {_pretty_attack(r['attack'], 'evolved')} & "
                f"{_f(r['asr'], 2)} & {_f(r['mean_l2_raw'], 2)} & {_f(r['mean_lpips'], 3)} & "
                f"{_int(r['mean_queries'])} & {_f(r['score'], 2)} \\\\"
            )
        lines.append("\\hline")
    header = ("\\textbf{Хеш} & \\textbf{Кандидат} & \\textbf{ASR} & \\textbf{$L_2$} & "
              "\\textbf{LPIPS} & \\textbf{Queries} & \\textbf{Score}")
    return _table(
        "\n".join(lines[:-1]),
        col_spec="|l|>{\\raggedright\\arraybackslash}p{2.7cm}|c|c|c|c|c|",
        header=header, caption=caption, label=label, env="tabular",
    )


# --------------------------------------------------------------------------- #
# 3. best optimized vs best evolved                                            #
# --------------------------------------------------------------------------- #

def latex_best_optimized_vs_evolved(
    store: RunStore,
    *,
    hashes: Sequence[str] | None = None,
    caption: str = "Сравнение лучших оптимизированных и эволюционно найденных атак",
    label: str = "tab:best_optimized_vs_evolved",
) -> str:
    s = _summary_frame(store)
    hashes = _hashes_from(s, hashes=hashes)
    lines = []
    for h in hashes:
        reg = s[(s.hash_id == h) & (s.attack_type == "registry")].sort_values("score")
        ev = s[(s.hash_id == h) & (s.attack_type == "evolved")].sort_values("score")
        if not len(reg) or not len(ev):
            continue
        ro, eo = reg.iloc[0], ev.iloc[0]
        so, se = float(ro["score"]), float(eo["score"])
        d = (so - se) / so * 100.0 if so > 0 else float("nan")
        ds = (f"$-{abs(d):.1f}$\\%" if d < 0 else f"{d:.1f}\\%") if np.isfinite(d) else "--"
        lines.append(
            f"{HASH_PRETTY.get(h, h)} & {_pretty_attack(ro['attack'], 'registry')}, optimized & "
            f"{_f(so, 2)} & {_pretty_attack(eo['attack'], 'evolved')} & {_f(se, 2)} & {ds} \\\\"
        )
    header = ("\\textbf{Хеш} & \\textbf{Лучшее optimized} & \\textbf{Score opt.} & "
              "\\textbf{Лучшее evolved} & \\textbf{Score evolved} & \\textbf{$\\Delta$score}")
    return _table(
        "\n".join(lines),
        col_spec="|l|X|c|X|c|c|", header=header, caption=caption, label=label,
        env="tabularx",
    )


# --------------------------------------------------------------------------- #
# 4. threshold sweep ASR(tau)                                                  #
# --------------------------------------------------------------------------- #

def latex_threshold_sweep(
    asr_by_attack: Mapping[str, Mapping[float, float]],
    taus: Sequence[float],
    *,
    attack_types: Mapping[str, str] | None = None,
    caption: str = "Зависимость ASR от порога коллизии",
    label: str = "tab:phash_threshold_sweep",
) -> str:
    """``asr_by_attack``: {display_name: {tau: asr}}; columns = taus."""
    attack_types = attack_types or {}
    head_cells = " & ".join(f"$ASR({int(t)})$" for t in taus)
    header = "\\textbf{Атака} & " + head_cells
    lines = []
    for name, curve in asr_by_attack.items():
        vals = " & ".join(_f(curve.get(t, float("nan")), 2) for t in taus)
        lines.append(f"{name} & {vals} \\\\")
    col_spec = "|X|" + "c|" * len(taus)
    return _table("\n".join(lines), col_spec=col_spec, header=header,
                  caption=caption, label=label, env="tabularx")


# --------------------------------------------------------------------------- #
# generic tidy DataFrame -> LaTeX (defense tables, etc.)                        #
# --------------------------------------------------------------------------- #

def dataframe_to_latex(
    df,
    columns: Sequence[str],
    *,
    headers: Sequence[str] | None = None,
    caption: str = "",
    label: str = "",
    col_spec: str | None = None,
    float_fmt: Mapping[str, int] | None = None,
    env: str = "tabularx",
    size: str = "",
) -> str:
    headers = list(headers or columns)
    float_fmt = float_fmt or {}
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if c in float_fmt:
                cells.append(_f(v, float_fmt[c]))
            elif isinstance(v, (int, np.integer)):
                cells.append(_int(v))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + " \\\\")
    if col_spec is None:
        col_spec = "|" + "X|" + "c|" * (len(columns) - 1)
    header = " & ".join(f"\\textbf{{{h}}}" for h in headers)
    return _table("\n".join(lines), col_spec=col_spec, header=header,
                  caption=caption, label=label, env=env, size=size)


def _screen_int(x: Any, dash: str = "--") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(v):
        return dash
    return f"{int(round(v)):,}".replace(",", " ")


def _screen_delta(before: Any, after: Any) -> str:
    try:
        b, a = float(before), float(after)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(b) or not np.isfinite(a) or b <= 0:
        return "--"
    delta = (b - a) / b * 100.0
    return f"{delta:.1f}%"


def _screen_metric_row(row: dict[str, Any], *, score_digits: int = 1) -> dict[str, str]:
    if not row:
        return {"ASR": "--", "L2": "--", "LPIPS": "--", "Queries": "--", "S": "--"}
    return {
        "ASR": _f(row.get("asr"), 2),
        "L2": _f(row.get("mean_l2_raw"), 1),
        "LPIPS": _f(row.get("mean_lpips"), 3),
        "Queries": _screen_int(row.get("mean_queries")),
        "S": _f(row.get("score"), score_digits),
    }


def baseline_optimized_frame(orig_store: RunStore, opt_store: RunStore, hash_id: str):
    """Notebook-friendly version of the original -> optimized diploma table."""
    import pandas as pd

    s_o, s_p = _summary_frame(orig_store), _summary_frame(opt_store)
    reg_o = s_o[(s_o.hash_id == hash_id) & (s_o.attack_type == "registry")]
    reg_p = s_p[(s_p.hash_id == hash_id) & (s_p.attack_type == "registry")]
    attacks = set(reg_o["attack"]) | set(reg_p["attack"])
    attacks = [a for a in _REG_ORDER if a in attacks] + sorted(a for a in attacks if a not in _REG_ORDER)

    rows = []
    for attack in attacks:
        original = _summary_row_from(reg_o, hash_id, attack)
        optimized = _summary_row_from(reg_p, hash_id, attack)
        original_metrics = _screen_metric_row(original)
        optimized_metrics = _screen_metric_row(optimized)
        rows.append({
            "Атака": _pretty_attack(attack, "registry"),
            "Конф.": "original",
            **original_metrics,
            "ΔS": "--",
        })
        rows.append({
            "Атака": "",
            "Конф.": "optimized",
            **optimized_metrics,
            "ΔS": _screen_delta(original.get("score"), optimized.get("score")),
        })
    return pd.DataFrame(rows, columns=["Атака", "Конф.", "ASR", "L2", "LPIPS", "Queries", "S", "ΔS"])


def evolved_results_frame(store: RunStore, *, hashes: Sequence[str] | None = None):
    """Notebook-friendly version of the evolved candidates diploma table."""
    import pandas as pd

    s = _summary_frame(store)
    ev = s[s.attack_type == "evolved"]
    selected = _hashes_from(ev, hashes=hashes)
    rows = []
    for hash_id in selected:
        for _, row in ev[ev.hash_id == hash_id].sort_values("score").iterrows():
            rows.append({
                "Хеш": HASH_PRETTY.get(hash_id, hash_id),
                "Кандидат": _pretty_attack(row["attack"], "evolved"),
                "ASR": _f(row.get("asr"), 2),
                "L2": _f(row.get("mean_l2_raw"), 2),
                "LPIPS": _f(row.get("mean_lpips"), 3),
                "Queries": _screen_int(row.get("mean_queries")),
                "Score": _f(row.get("score"), 2),
            })
    return pd.DataFrame(rows, columns=["Хеш", "Кандидат", "ASR", "L2", "LPIPS", "Queries", "Score"])


def best_optimized_vs_evolved_frame(store: RunStore, *, hashes: Sequence[str] | None = None):
    """Notebook-friendly version of the best optimized vs evolved diploma table."""
    import pandas as pd

    s = _summary_frame(store)
    selected = _hashes_from(s, hashes=hashes)
    rows = []
    for hash_id in selected:
        registry = s[(s.hash_id == hash_id) & (s.attack_type == "registry")].sort_values("score")
        evolved = s[(s.hash_id == hash_id) & (s.attack_type == "evolved")].sort_values("score")
        if not len(registry) or not len(evolved):
            continue
        reg, ev = registry.iloc[0], evolved.iloc[0]
        rows.append({
            "Хеш": HASH_PRETTY.get(hash_id, hash_id),
            "Лучшее optimized": f"{_pretty_attack(reg['attack'], 'registry')}, optimized",
            "Score opt.": _f(reg.get("score"), 2),
            "Лучшее evolved": _pretty_attack(ev["attack"], "evolved"),
            "Score evolved": _f(ev.get("score"), 2),
            "Δscore": _screen_delta(reg.get("score"), ev.get("score")),
        })
    return pd.DataFrame(
        rows,
        columns=["Хеш", "Лучшее optimized", "Score opt.", "Лучшее evolved", "Score evolved", "Δscore"],
    )


def diploma_display_frames(
    orig_store: RunStore,
    opt_store: RunStore,
    *,
    hashes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build human-readable notebook tables with the same structure as the diploma."""
    selected = _hashes_from(orig_store, opt_store, hashes=hashes)
    frames: dict[str, Any] = {
        f"Сравнение обычной и оптимизированной конфигураций для {HASH_PRETTY.get(h, h)}":
            baseline_optimized_frame(orig_store, opt_store, h)
        for h in selected
    }
    frames["Результаты эволюционно найденных атак"] = evolved_results_frame(opt_store, hashes=selected)
    frames["Сравнение лучших оптимизированных и эволюционно найденных атак"] = (
        best_optimized_vs_evolved_frame(opt_store, hashes=selected)
    )
    return frames


def style_diploma_frame(df, *, caption: str | None = None):
    """Apply a compact diploma-like style for Jupyter display."""
    styler = df.style.set_properties(**{
        "border": "1px solid #4a4a4a",
        "padding": "4px 7px",
        "text-align": "center",
        "font-size": "12px",
    }).set_table_styles([
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("border", "1px solid #4a4a4a"),
            ("margin", "8px 0 18px 0"),
        ]},
        {"selector": "th", "props": [
            ("border", "1px solid #4a4a4a"),
            ("padding", "4px 7px"),
            ("text-align", "center"),
            ("font-weight", "bold"),
            ("background-color", "#f2f2f2"),
        ]},
        {"selector": "caption", "props": [
            ("caption-side", "bottom"),
            ("font-weight", "normal"),
            ("padding-top", "6px"),
            ("font-size", "12px"),
        ]},
    ])
    if caption:
        styler = styler.set_caption(caption)
    try:
        return styler.hide(axis="index")
    except TypeError:
        return styler.hide_index()


def latex_diploma_tables(
    orig_store: RunStore,
    opt_store: RunStore,
    *,
    hashes: Sequence[str] | None = None,
) -> dict[str, str]:
    """Build the diploma-ready Stage-4 result tables.

    Returns a mapping suitable for :func:`write_tables`:
    ``baseline_optimized_<hash>``, ``evolved_results`` and
    ``best_optimized_vs_evolved``.
    """
    selected = _hashes_from(orig_store, opt_store, hashes=hashes)
    tables = {
        f"baseline_optimized_{h}": latex_baseline_optimized(orig_store, opt_store, h)
        for h in selected
    }
    tables["evolved_results"] = latex_evolved_results(opt_store, hashes=selected)
    tables["best_optimized_vs_evolved"] = latex_best_optimized_vs_evolved(opt_store, hashes=selected)
    return tables


def write_tables(out_dir: str | Path, **named_latex: str) -> dict[str, Path]:
    """Write ``name=latex`` pairs to ``<out_dir>/<name>.tex``; return paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, tex in named_latex.items():
        p = out / f"{name}.tex"
        p.write_text(tex + "\n", encoding="utf-8")
        paths[name] = p
    return paths


__all__ = [
    "latex_baseline_optimized",
    "latex_evolved_results",
    "latex_best_optimized_vs_evolved",
    "latex_threshold_sweep",
    "latex_diploma_tables",
    "baseline_optimized_frame",
    "evolved_results_frame",
    "best_optimized_vs_evolved_frame",
    "diploma_display_frames",
    "style_diploma_frame",
    "dataframe_to_latex",
    "write_tables",
    "PRETTY",
    "HASH_PRETTY",
]
