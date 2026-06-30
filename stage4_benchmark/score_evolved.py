"""Easy scoring of evolved attack programs.

Thin convenience wrapper around the production single-program scorer
(:func:`evohash.stage3_common.validate.validate_program`). The returned score is
*identical* to the objective the Stage 3 evolution ranked by
(``metrics["score"]`` = ``ProgramEvalSummary.score``), so numbers line up with
Stage 3 / Stage 4 reporting.

Typical use (from the repo root or a notebook)::

    from evohash.stage4_benchmark.score_evolved import (
        score_evolved_program, score_evolved_programs,
    )

    # one program -> ProgramEvalSummary (has .score and .metrics)
    s = score_evolved_program("pHash_GigaEvolve.py")
    print(s.score, s.metrics["asr"], s.metrics["mean_l2_raw"])

    # every program for a hash (or all hashes) -> pandas DataFrame
    df = score_evolved_programs("phash")          # both pHash_* programs
    df = score_evolved_programs()                 # everything in evolved_programs/

The ``hash_id`` is inferred from the filename prefix (text before the first
``_``, case-insensitive): ``pHash_GigaEvolve.py`` -> ``phash``,
``PDQ_OpenEvolve.py`` -> ``pdq``. Pass ``hash_id=`` explicitly to override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evohash.stage3_common.config import EvaluationConfig
from evohash.stage3_common.validate import ProgramEvalSummary, validate_program

# .../evohash (the package dir, which contains evolved_programs/)
_PKG_DIR = Path(__file__).resolve().parents[1]
# repo root (parent of the package) holds dataset_v1_public.zip
_REPO_ROOT = _PKG_DIR.parent

DEFAULT_EVOLVED_DIR = _PKG_DIR / "evolved_programs"
DEFAULT_DATASET = _REPO_ROOT / "dataset_v1_public.zip"

# Longest names first is not required (we match on the prefix token), but keep
# the canonical set so a stray filename fails loudly instead of silently.
KNOWN_HASHES = ("phash", "pdq", "photodna", "neuralhash")


def infer_hash_id(program_path: str | Path) -> str:
    """Infer the hash id from an evolved program filename (``pHash_X`` -> ``phash``)."""

    prefix = Path(program_path).stem.split("_", 1)[0].lower()
    if prefix not in KNOWN_HASHES:
        raise ValueError(
            f"Cannot infer hash_id from {Path(program_path).name!r} "
            f"(prefix {prefix!r} not in {KNOWN_HASHES}). Pass hash_id=... explicitly."
        )
    return prefix


def _resolve_program(program_path: str | Path) -> Path:
    """Accept an absolute path, a cwd-relative path, or a bare filename."""

    path = Path(program_path)
    if path.is_file():
        return path.resolve()
    candidate = DEFAULT_EVOLVED_DIR / path.name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Evolved program not found: {program_path!r} "
        f"(also tried {candidate})."
    )


def discover_evolved_programs(
    hash_id: str | None = None,
    *,
    evolved_dir: str | Path = DEFAULT_EVOLVED_DIR,
) -> list[Path]:
    """List evolved program files, optionally filtered by inferred hash id."""

    root = Path(evolved_dir)
    files = [
        path
        for path in sorted(root.glob("*.py"))
        if path.is_file() and not path.name.startswith("__")
    ]
    if hash_id is not None:
        want = hash_id.lower()
        files = [path for path in files if infer_hash_id(path) == want]
    return files


def score_evolved_program(
    program_path: str | Path,
    *,
    hash_id: str | None = None,
    max_pairs: int | None = 30,
    budget: int = 10_000,
    dataset_zip: str | Path | None = None,
    show_progress: bool = False,
    **config_kwargs: Any,
) -> ProgramEvalSummary:
    """Score one evolved program through the production evaluator.

    Returns a :class:`ProgramEvalSummary`; read ``.score`` (lower is better) and
    ``.metrics`` (``asr``, ``mean_l2_raw``, ``mean_queries``, ``is_valid``, ...).
    Any extra keyword is forwarded to :class:`EvaluationConfig` (e.g. ``seed``,
    ``resize_size``, ``parallel_backend``, ``compute_lpips`` is not a field here).
    """

    resolved = _resolve_program(program_path)
    cfg = EvaluationConfig(
        hash_id=hash_id or infer_hash_id(resolved),
        dataset_zip=str(dataset_zip or DEFAULT_DATASET),
        max_pairs=max_pairs,
        budget=int(budget),
        show_progress=bool(show_progress),
        **config_kwargs,
    )
    return validate_program(resolved, cfg)


def score_evolved_programs(
    hash_id: str | None = None,
    *,
    evolved_dir: str | Path = DEFAULT_EVOLVED_DIR,
    as_dataframe: bool = True,
    **score_kwargs: Any,
):
    """Score every evolved program (optionally for one hash) into a table.

    With ``as_dataframe=True`` (default) returns a pandas DataFrame sorted by
    score (best first); otherwise a list of plain dicts. Extra keywords are
    passed to :func:`score_evolved_program`.
    """

    rows: list[dict[str, Any]] = []
    for path in discover_evolved_programs(hash_id, evolved_dir=evolved_dir):
        summary = score_evolved_program(path, **score_kwargs)
        metrics = summary.metrics
        rows.append(
            {
                "program": path.name,
                "hash_id": infer_hash_id(path),
                "score": summary.score,
                "asr": metrics.get("asr"),
                "mean_l2_raw": metrics.get("mean_l2_raw"),
                "mean_queries": metrics.get("mean_queries"),
                "mean_final_hash_l1": metrics.get("mean_final_hash_l1"),
                "is_valid": metrics.get("is_valid"),
                "n_errors": len(summary.errors),
                "path": str(path),
            }
        )

    if not as_dataframe:
        return rows

    import pandas as pd

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["hash_id", "score"]).reset_index(drop=True)
    return df


__all__ = [
    "DEFAULT_EVOLVED_DIR",
    "DEFAULT_DATASET",
    "KNOWN_HASHES",
    "infer_hash_id",
    "discover_evolved_programs",
    "score_evolved_program",
    "score_evolved_programs",
]
