"""Export top EvoHash/GigaEvolve programs from Redis JSON keys."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save top programs from a GigaEvolve Redis run.")
    parser.add_argument("--prefix", required=True, help="Redis key prefix, usually problem.name.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--metric", default="score")
    parser.add_argument("--minimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--state", default="done", choices=["queued", "running", "done", "discarded", "any"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/gigaevolve_exports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    programs = fetch_programs(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        prefix=args.prefix,
    )
    ranked = rank_programs(
        programs,
        metric=args.metric,
        minimize=args.minimize,
        state_filter=None if args.state == "any" else args.state,
        top_n=args.top_n,
    )
    if not ranked:
        print(f"No programs found for prefix={args.prefix!r}, metric={args.metric!r}.", file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir) / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    save_ranked_programs(ranked, output_dir, metric=args.metric)
    print(json.dumps(summary(ranked, metric=args.metric, output_dir=output_dir), ensure_ascii=False, indent=2))


def fetch_programs(*, host: str, port: int, db: int, prefix: str) -> list[dict[str, Any]]:
    try:
        import redis
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("The redis package is required for export.") from exc

    client = redis.Redis(host=host, port=port, db=db, decode_responses=True, socket_connect_timeout=3)
    client.ping()
    programs: list[dict[str, Any]] = []
    for key in client.scan_iter(match=f"{prefix}:program:*", count=1000):
        raw = client.get(key)
        if not raw:
            continue
        try:
            program = json.loads(raw)
        except json.JSONDecodeError:
            continue
        program["_redis_key"] = key
        programs.append(program)
    return programs


def rank_programs(
    programs: list[dict[str, Any]],
    *,
    metric: str,
    minimize: bool,
    state_filter: str | None,
    top_n: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for program in programs:
        if state_filter and _state(program) != state_filter:
            continue
        metrics = program.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(metric)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value_f):
            continue
        ranked.append(program)
    ranked.sort(key=lambda program: float(program["metrics"][metric]), reverse=not minimize)
    return ranked[:top_n]


def save_ranked_programs(programs: list[dict[str, Any]], output_dir: Path, *, metric: str) -> None:
    entries: list[dict[str, Any]] = []
    for index, program in enumerate(programs, 1):
        metric_value = float(program["metrics"][metric])
        program_id = str(program.get("id", "unknown"))
        stem = f"rank{index:03d}_{_safe_metric(metric_value)}_{program_id[:10]}"
        code_path = output_dir / f"{stem}.py"
        metrics_path = output_dir / f"{stem}.metrics.json"
        code_path.write_text(str(program.get("code", "")), encoding="utf-8")
        metrics_path.write_text(
            json.dumps(program.get("metrics", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        entries.append(_program_entry(program, index=index, metric=metric, code_path=code_path))

    best = programs[0]
    (output_dir / "best_program.py").write_text(str(best.get("code", "")), encoding="utf-8")
    (output_dir / "best_program.metrics.json").write_text(
        json.dumps(best.get("metrics", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "top_programs.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summary(programs: list[dict[str, Any]], *, metric: str, output_dir: Path) -> dict[str, Any]:
    best = programs[0]
    return {
        "ok": True,
        "output_dir": str(output_dir),
        "n_exported": len(programs),
        "best_id": best.get("id"),
        "best_generation": _generation(best),
        "best_state": _state(best),
        "metric": metric,
        "best_metric": best.get("metrics", {}).get(metric),
        "best_metrics": best.get("metrics", {}),
        "best_program": str(output_dir / "best_program.py"),
    }


def _program_entry(program: dict[str, Any], *, index: int, metric: str, code_path: Path) -> dict[str, Any]:
    lineage = program.get("lineage") if isinstance(program.get("lineage"), dict) else {}
    return {
        "rank": index,
        "id": program.get("id"),
        "state": _state(program),
        "generation": _generation(program),
        "metric": metric,
        "metric_value": program.get("metrics", {}).get(metric),
        "metrics": program.get("metrics", {}),
        "parents": lineage.get("parents", []),
        "mutation": lineage.get("mutation"),
        "created_at": program.get("created_at"),
        "redis_key": program.get("_redis_key"),
        "code_path": str(code_path),
    }


def _state(program: dict[str, Any]) -> str:
    state = program.get("state", "")
    if isinstance(state, dict):
        return str(state.get("value", ""))
    return str(state)


def _generation(program: dict[str, Any]) -> int | None:
    value = program.get("generation")
    if value is None and isinstance(program.get("lineage"), dict):
        value = program["lineage"].get("generation")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_metric(value: float) -> str:
    return f"{value:.6g}".replace("+", "").replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        raise
