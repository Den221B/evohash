"""Extract the N best evolved programs from an OpenEvolve run.

Two complementary, *deduplicated* selections (dedup key = sha256 of normalized
source):

  * **top-N globally** by ``combined_score`` (= ``-score``; higher is better), and
  * the **elite of every occupied MAP-Elites cell** (maximum behavioural diversity).

For ``extract_best_programs`` the source of truth is the latest
``<output_dir>/checkpoints/checkpoint_<N>/`` written by the OpenEvolve
controller. For ``extract_champions`` we scan *all* checkpoints because
OpenEvolve's bounded database can evict older record-holders before the final
checkpoint:

  * ``programs/<id>.json`` — one file per program, ``Program.to_dict()`` with
    ``id`` / ``code`` / ``metrics``.
  * ``metadata.json`` — ``island_feature_maps`` (per-island ``{cell_key: program_id}``),
    ``archive``, ``best_program_id``.

Pure stdlib, no ``openevolve`` import — so it works after the run from any
process (the notebook, the subprocess runner, or a cold shell).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


# Metrics surfaced into best_programs/index.json (others stay in the per-program
# JSON sidecar). Order is cosmetic.
_INDEX_METRIC_KEYS = (
    "score",
    "combined_score",
    "asr",
    "mean_l2_raw",
    "mean_queries",
    "mean_pixel_l2",
    "mean_gap_to_threshold",
    "efficiency",
    "is_valid",
)


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the ``checkpoint_<N>`` dir with the highest N, or None."""
    checkpoints = _all_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def _all_checkpoints(output_dir: str | Path) -> list[Path]:
    """Return checkpoint dirs sorted by checkpoint number, oldest to newest."""
    ckpt_root = Path(output_dir) / "checkpoints"
    if not ckpt_root.is_dir():
        return []
    checkpoints: list[Path] = []
    for child in ckpt_root.glob("checkpoint_*"):
        if not child.is_dir():
            continue
        if _checkpoint_number(child) < 0:
            continue
        checkpoints.append(child)
    return sorted(checkpoints, key=_checkpoint_number)


def extract_best_programs(
    output_dir: str | Path,
    *,
    top_n: int = 5,
    checkpoint: str | Path | None = None,
    dest_dirname: str = "best_programs",
) -> dict[str, Any]:
    """Write the best programs (code + metrics) to ``<output_dir>/best_programs/``.

    Returns the index dict (also written to ``best_programs/index.json``).
    """
    out = Path(output_dir)
    ckpt = Path(checkpoint) if checkpoint is not None else latest_checkpoint(out)
    if ckpt is None or not ckpt.is_dir():
        return {"ok": False, "reason": "no checkpoint found", "output_dir": str(out), "entries": []}

    programs = _load_programs(ckpt)
    if not programs:
        return {"ok": False, "reason": f"no programs in {ckpt}", "checkpoint": str(ckpt), "entries": []}

    dest = out / dest_dirname
    dest.mkdir(parents=True, exist_ok=True)

    seen: dict[str, str] = {}  # code_hash -> stem already written
    entries: list[dict[str, Any]] = []

    # --- 1) Top-N globally by fitness (combined_score desc) ---
    ranked = sorted(programs.values(), key=_fitness, reverse=True)
    rank = 0
    for prog in ranked:
        if rank >= top_n:
            break
        h = _code_hash(prog["code"])
        if h in seen:
            continue
        rank += 1
        stem = f"top_{rank:02d}_{prog['id']}"
        _write_program(dest, stem, prog)
        seen[h] = stem
        entries.append(_index_entry("topN", stem, prog, code_hash=h, rank=rank))

    # --- 2) Elite of every occupied MAP-Elites cell (collapsed across islands) ---
    for cell_key, pid in sorted(_cell_elites(ckpt, programs).items()):
        prog = programs.get(pid)
        if prog is None:
            continue
        h = _code_hash(prog["code"])
        if h in seen:
            # identical code already saved (often also a top-N elite): record the
            # cell membership but don't duplicate the file.
            entries.append(
                _index_entry("cell", seen[h], prog, code_hash=h, cell=cell_key, duplicate_of=seen[h])
            )
            continue
        stem = f"cell_{cell_key}_{prog['id']}"
        _write_program(dest, stem, prog)
        seen[h] = stem
        entries.append(_index_entry("cell", stem, prog, code_hash=h, cell=cell_key))

    index = {
        "ok": True,
        "output_dir": str(out),
        "checkpoint": str(ckpt),
        "dest": str(dest),
        "top_n": int(top_n),
        "n_programs_in_checkpoint": len(programs),
        "n_unique_files": len({e["code_path"] for e in entries}),
        "entries": entries,
    }
    (dest / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def extract_champions(
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    dest_dirname: str = "champions",
    min_delta: float = 1e-9,
) -> dict[str, Any]:
    """Save the *record-improvement chain* of programs from a run.

    The chain is reconstructed from checkpoint snapshots. By default this scans
    all checkpoints, because the final OpenEvolve database is bounded and may
    have evicted older record-holders. If ``checkpoint=...`` is passed, only that
    checkpoint is used.

      1. **Baseline** = the best initial-population program (preloaded
         ``seed_*`` / ``metadata.source=evohash_seed`` programs, plus the
         original generation-0 program if OpenEvolve stores it). This is the
         best starting attack+config, not an evolved child that merely has
         ``iteration_found == 0``.
      2. Then evolved programs in chronological order; a program joins the chain
         only when it **strictly beats** the running-best fitness
         (``> best + min_delta``). A program whose metric did not change (same
         fitness) is a duplicate and is skipped.

    Fitness = ``combined_score`` if present else ``-score`` (seeds carry only
    ``score``; evolved programs carry ``combined_score`` — both reduce to the
    same lower-is-better objective).

    Writes ``<output_dir>/champions/champion_<NN>_<id>.py`` + ``.json`` per
    champion and a single ``aggregate.json`` summarising the chain for
    cross-program / cross-framework comparison. This is checkpoint-level
    reconstruction: a record that appears and is evicted between checkpoints
    cannot be recovered unless a live evaluation trace was recorded.
    """
    out = Path(output_dir)
    checkpoints = [Path(checkpoint)] if checkpoint is not None else _all_checkpoints(out)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.is_dir()]
    if not checkpoints:
        return {"ok": False, "reason": "no checkpoint found", "output_dir": str(out), "champions": []}

    checkpoint_program_map, n_checkpoint_records_loaded = _load_programs_from_checkpoints(checkpoints)
    trace_program_map, n_trace_records_loaded = (
        ({}, 0) if checkpoint is not None else _load_evaluation_trace_programs(out)
    )
    if trace_program_map:
        # Checkpoints remain the source for initial seeds; the trace gives a
        # denser event-level view of evaluated children between checkpoints.
        program_map = {
            pid: prog for pid, prog in checkpoint_program_map.items() if _is_seed(prog)
        }
        program_map.update(trace_program_map)
        checkpoint_source = "evaluation_trace_with_checkpoint_seeds"
        coverage_note = (
            "Evaluation trace reconstruction. Seed baseline comes from checkpoints; "
            "evolved records come from per-evaluation trace files."
        )
    else:
        program_map = checkpoint_program_map
        checkpoint_source = "single_checkpoint" if checkpoint is not None else "all_checkpoints"
        coverage_note = (
            "Checkpoint-level reconstruction. Records that appeared and were evicted "
            "between checkpoints require a live evaluation trace to recover."
        )
    programs = list(program_map.values())
    if not programs:
        return {
            "ok": False,
            "reason": f"no programs in {checkpoints[-1]}",
            "checkpoint": str(checkpoints[-1]),
            "n_checkpoints_scanned": len(checkpoints),
            "champions": [],
        }

    seeds = [p for p in programs if _is_seed(p)]
    evolved = sorted(
        (p for p in programs if not _is_seed(p)),
        key=lambda p: (
            p.get("iteration_found", 0),
            p.get("generation", 0),
            p.get("timestamp", 0.0),
            p.get("checkpoint_first_seen_order", 0),
        ),
    )

    chain: list[tuple[str, dict[str, Any]]] = []
    best = float("-inf")
    if seeds:
        best_seed = max(seeds, key=_fitness)
        chain.append(("seed", best_seed))
        best = _fitness(best_seed)
    for prog in evolved:
        fit = _fitness(prog)
        if fit > best + min_delta:  # strictly beats the record (equal metric = duplicate -> skip)
            chain.append(("evolved", prog))
            best = fit

    dest = out / dest_dirname
    dest.mkdir(parents=True, exist_ok=True)
    _clear_previous_champion_export(dest)

    champions: list[dict[str, Any]] = []
    prev_fit: float | None = None
    for rank, (kind, prog) in enumerate(chain):
        stem = f"champion_{rank:02d}_{_safe_stem(prog['id'])}"
        (dest / f"{stem}.py").write_text(prog["code"], encoding="utf-8")
        fit = _fitness(prog)
        rec: dict[str, Any] = {
            "rank": rank,
            "kind": kind,
            "id": prog["id"],
            "iteration_found": int(prog.get("iteration_found", 0) or 0),
            "generation": int(prog.get("generation", 0) or 0),
            "parent_id": prog.get("parent_id"),
            "checkpoint_first_seen": prog.get("checkpoint_first_seen"),
            "checkpoint_last_seen": prog.get("checkpoint_last_seen"),
            "code_path": f"{stem}.py",
            "metrics_path": f"{stem}.json",
            "fitness": fit,
            "delta_fitness": None if prev_fit is None else (fit - prev_fit),
        }
        if prog.get("trace_path"):
            rec["trace_path"] = prog.get("trace_path")
        metrics = prog.get("metrics") or {}
        for key in _INDEX_METRIC_KEYS:
            if key in metrics:
                try:
                    rec[key] = float(metrics[key])
                except (TypeError, ValueError):
                    continue
        (dest / f"{stem}.json").write_text(
            json.dumps(
                {
                    "id": prog["id"],
                    "provenance": {
                        "rank": rank,
                        "kind": kind,
                        "iteration_found": rec["iteration_found"],
                        "generation": rec["generation"],
                        "parent_id": rec["parent_id"],
                        "fitness": fit,
                        "checkpoint_first_seen": rec["checkpoint_first_seen"],
                        "checkpoint_last_seen": rec["checkpoint_last_seen"],
                        "trace_path": rec.get("trace_path"),
                    },
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        champions.append(rec)
        prev_fit = fit

    aggregate = {
        "ok": True,
        "framework": "openevolve",
        "output_dir": str(out),
        "checkpoint": str(checkpoints[-1]),
        "checkpoint_source": checkpoint_source,
        "n_checkpoints_scanned": len(checkpoints),
        "checkpoints_scanned": [str(ckpt) for ckpt in checkpoints],
        "n_checkpoint_program_records_loaded": n_checkpoint_records_loaded,
        "n_trace_program_records_loaded": n_trace_records_loaded,
        "n_program_records_loaded": n_checkpoint_records_loaded + n_trace_records_loaded,
        "n_programs_total": len(programs),
        "n_seeds": len(seeds),
        "n_champions": len(champions),
        "coverage_note": coverage_note,
        "baseline": champions[0] if champions else None,
        "best": champions[-1] if champions else None,
        "total_improvement_fitness": (
            champions[-1]["fitness"] - champions[0]["fitness"] if champions else 0.0
        ),
        "champions": champions,
    }
    (dest / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load_programs(ckpt: Path) -> dict[str, dict[str, Any]]:
    progs: dict[str, dict[str, Any]] = {}
    pdir = ckpt / "programs"
    if not pdir.is_dir():
        return progs
    for pf in pdir.glob("*.json"):
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        code = data.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        pid = str(data.get("id") or pf.stem)
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        # Extra provenance fields (used by extract_champions; ignored by
        # extract_best_programs).
        progs[pid] = {
            "id": pid,
            "code": code,
            "metrics": metrics,
            "iteration_found": _as_int(data.get("iteration_found"), 0),
            "generation": _as_int(data.get("generation"), 0),
            "timestamp": _as_float(data.get("timestamp"), 0.0),
            "parent_id": data.get("parent_id"),
            "metadata": data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        }
    return progs


def _load_programs_from_checkpoints(checkpoints: list[Path]) -> tuple[dict[str, dict[str, Any]], int]:
    """Merge program snapshots from many checkpoints by program id.

    Program JSON files can appear in several checkpoint dumps. We keep the first
    copy for chronology/provenance and update only the last-seen checkpoint.
    Each checkpoint's ``best_program.py`` is also included as a safety net: if a
    best program was no longer present in ``programs/`` when a later checkpoint
    was written, the checkpoint-level best snapshot can still recover its code.
    """
    merged: dict[str, dict[str, Any]] = {}
    records_loaded = 0
    for order, ckpt in enumerate(checkpoints):
        for prog in _load_programs(ckpt).values():
            records_loaded += 1
            _remember_checkpoint_program(merged, prog, ckpt, order)
        best = _load_checkpoint_best_program(ckpt)
        if best is not None:
            records_loaded += 1
            _remember_checkpoint_program(merged, best, ckpt, order)
    return merged, records_loaded


def _remember_checkpoint_program(
    merged: dict[str, dict[str, Any]],
    prog: dict[str, Any],
    ckpt: Path,
    order: int,
) -> None:
    pid = str(prog.get("id") or "")
    if not pid:
        return
    existing = merged.get(pid)
    if existing is not None:
        existing["checkpoint_last_seen"] = ckpt.name
        existing["checkpoint_last_seen_order"] = order
        return

    item = dict(prog)
    item["checkpoint_first_seen"] = ckpt.name
    item["checkpoint_first_seen_order"] = order
    item["checkpoint_last_seen"] = ckpt.name
    item["checkpoint_last_seen_order"] = order
    merged[pid] = item


def _load_checkpoint_best_program(ckpt: Path) -> dict[str, Any] | None:
    info_path = ckpt / "best_program_info.json"
    code_path = ckpt / "best_program.py"
    if not info_path.is_file() or not code_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        code = code_path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return None
    if not code.strip():
        return None

    metrics = info.get("metrics") if isinstance(info.get("metrics"), dict) else {}
    ckpt_n = _checkpoint_number(ckpt)
    pid = str(info.get("id") or f"checkpoint_best_{ckpt_n}")
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    metadata = {**metadata, "source": metadata.get("source") or "openevolve_checkpoint_best"}
    return {
        "id": pid,
        "code": code,
        "metrics": metrics,
        "iteration_found": _as_int(
            info.get("iteration_found", info.get("current_iteration")),
            ckpt_n if ckpt_n >= 0 else 0,
        ),
        "generation": _as_int(info.get("generation"), 0),
        "timestamp": _as_float(info.get("timestamp"), 0.0),
        "parent_id": info.get("parent_id"),
        "metadata": metadata,
    }


def _load_evaluation_trace_programs(output_dir: Path) -> tuple[dict[str, dict[str, Any]], int]:
    trace_root = output_dir / "evaluation_trace"
    if not trace_root.is_dir():
        return {}, 0

    programs: dict[str, dict[str, Any]] = {}
    records_loaded = 0
    for order, path in enumerate(sorted(trace_root.glob("*.json"))):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        code = data.get("code")
        metrics = data.get("metrics")
        if not isinstance(code, str) or not code.strip() or not isinstance(metrics, dict):
            continue
        records_loaded += 1
        pid = str(data.get("id") or path.stem)
        programs[pid] = {
            "id": pid,
            "code": code,
            "metrics": metrics,
            # The trace records evaluation-completion order, not OpenEvolve's
            # internal iteration id. Keep iteration_found at 0 and sort traced
            # children by timestamp/trace_order below.
            "iteration_found": _as_int(data.get("iteration_found"), 0),
            "generation": _as_int(data.get("generation"), 1),
            "timestamp": _as_float(data.get("timestamp"), 0.0),
            "parent_id": data.get("parent_id"),
            "metadata": {"source": "openevolve_evaluation_trace"},
            "trace_order": order,
            "trace_path": str(path),
        }
    return programs, records_loaded


def _cell_elites(ckpt: Path, programs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Collapse per-island feature maps into one elite per cell (fittest wins)."""
    meta_path = ckpt / "metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    best_per_cell: dict[str, str] = {}
    for island_map in meta.get("island_feature_maps") or []:
        if not isinstance(island_map, dict):
            continue
        for cell_key, pid in island_map.items():
            pid = str(pid)
            current = best_per_cell.get(cell_key)
            if current is None or _fitness(programs.get(pid, {})) > _fitness(programs.get(current, {})):
                best_per_cell[cell_key] = pid
    return best_per_cell


def _fitness(prog: dict[str, Any]) -> float:
    """Higher = better. Mirrors openevolve get_fitness_score for our metrics.

    Seeds carry only ``score`` (scored via validate_program); evolved programs
    carry ``combined_score`` (= -score, injected in runtime.evaluate). Both
    reduce to the same lower-is-better objective.
    """
    metrics = prog.get("metrics") or {}
    for key, sign in (("combined_score", 1.0), ("score", -1.0)):
        if key in metrics:
            try:
                return sign * float(metrics[key])
            except (TypeError, ValueError):
                continue
    return float("-inf")


def _is_seed(prog: dict[str, Any]) -> bool:
    """A program present before mutation, not just any program at iteration 0.

    OpenEvolve checkpoints can contain generated children with
    ``iteration_found == 0``. Treating every such program as a seed corrupts the
    record chain baseline: a strong early child can be mislabeled as the best
    seed. Reliable initial-population signals are our preloaded seed metadata/id,
    or OpenEvolve's original generation-0 program with no parent.
    """
    metadata = prog.get("metadata") or {}
    if metadata.get("source") == "evohash_seed":
        return True
    if str(prog.get("id", "")).startswith("seed_"):
        return True
    return (
        int(prog.get("iteration_found", 0) or 0) == 0
        and int(prog.get("generation", 0) or 0) == 0
        and prog.get("parent_id") in (None, "")
    )


def _clear_previous_champion_export(dest: Path) -> None:
    """Remove stale files from a previous champion export in the same folder."""
    for pattern in ("champion_*.py", "champion_*.json", "aggregate.json"):
        for path in dest.glob(pattern):
            if path.is_file():
                path.unlink()


def _checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return -1


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return (stem[:80] or "program")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _code_hash(code: str) -> str:
    norm = "\n".join(line.rstrip() for line in code.strip().splitlines())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _write_program(dest: Path, stem: str, prog: dict[str, Any]) -> None:
    (dest / f"{stem}.py").write_text(prog["code"], encoding="utf-8")
    (dest / f"{stem}.json").write_text(
        json.dumps({"id": prog["id"], "metrics": prog.get("metrics", {})}, indent=2),
        encoding="utf-8",
    )


def _index_entry(
    kind: str,
    stem: str,
    prog: dict[str, Any],
    *,
    code_hash: str,
    rank: int | None = None,
    cell: str | None = None,
    duplicate_of: str | None = None,
) -> dict[str, Any]:
    metrics = prog.get("metrics") or {}
    entry: dict[str, Any] = {
        "kind": kind,
        "id": prog["id"],
        "code_path": f"{stem}.py",
        "metrics_path": f"{stem}.json",
        "code_hash": code_hash,
    }
    if rank is not None:
        entry["rank"] = rank
    if cell is not None:
        entry["cell"] = cell
    if duplicate_of is not None:
        entry["duplicate_of"] = duplicate_of
    for key in _INDEX_METRIC_KEYS:
        if key in metrics:
            try:
                entry[key] = float(metrics[key])
            except (TypeError, ValueError):
                continue
    return entry
