"""Local per-hash research notes for evolution prompts."""

from __future__ import annotations

from pathlib import Path


NOTES_DIR = Path(__file__).resolve().parent / "problem_notes"


def load_problem_notes(hash_id: str) -> str:
    path = NOTES_DIR / f"{hash_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()
