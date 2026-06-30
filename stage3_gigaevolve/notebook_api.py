"""Small notebook-first API for EvoHash GigaEvolve experiments."""

from __future__ import annotations

import ast
from datetime import datetime
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

from evohash.stage3_gigaevolve.env_check import check_environment
from evohash.stage3_gigaevolve.export_redis import (
    fetch_programs,
    rank_programs,
    save_ranked_programs,
    summary as export_summary,
)
from evohash.stage3_gigaevolve.llm_check import check_llm
from evohash.stage3_gigaevolve.models import DEFAULT_OPENROUTER_MODEL, openrouter_aliases, resolve_openrouter_model
from evohash.stage3_gigaevolve.snapshot_redis import save_snapshot

_BACKGROUND_PROCS: dict[str, subprocess.Popen] = {}
_LOG_TAILERS: dict[str, threading.Event] = {}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Resolve notebook-facing relative paths from the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def package_parent() -> Path:
    return project_root().parent


def model_table(*, as_dataframe: bool = True):
    """Return known OpenRouter aliases with model ids and model-page links."""
    rows = [
        {
            "alias": alias,
            "model": model,
            "default": alias == DEFAULT_OPENROUTER_MODEL,
            "url": f"https://openrouter.ai/{model}",
        }
        for alias, model in openrouter_aliases().items()
    ]
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


def preflight(
    *,
    dataset: str | Path,
    llm_source: str = "openrouter",
    model: str = DEFAULT_OPENROUTER_MODEL,
    local_base_url: str = "http://localhost:8000/v1",
    api_key: str = "None",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    check_redis: bool = False,
    check_llm_connection: bool = True,
    diagnostic_llm: bool = False,
) -> dict[str, Any]:
    """Check environment and, optionally, LLM/Redis connectivity."""
    env = check_environment(
        dataset=dataset,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        check_redis=check_redis,
    )
    result: dict[str, Any] = {"ok": env["ok"], "environment": env}
    if check_llm_connection:
        llm = check_llm(
            source=llm_source,
            model=model,
            local_base_url=local_base_url,
            dotenv=project_root() / ".env",
            diagnostic=diagnostic_llm,
        )
        result["llm"] = llm
        result["ok"] = bool(result["ok"] and llm["ok"])
    return result


def preflight_table(preflight_result: dict[str, Any], *, as_dataframe: bool = True):
    """Return a compact check table for notebook display."""
    env = preflight_result.get("environment") or {}
    llm = preflight_result.get("llm") or {}
    redis_status = env.get("redis") or {}
    imports = env.get("imports") or {}
    optional_imports = env.get("optional_imports") or {}
    missing_imports = env.get("missing_imports") or []
    missing_optional = env.get("missing_optional_imports") or []

    rows = [
        {
            "check": "project",
            "ok": bool(env.get("gigaevolve_core_exists")),
            "details": str(env.get("gigaevolve_core", "")),
            "recommendation": "" if env.get("gigaevolve_core_exists") else "Check third_party/gigaevolve_core.",
        },
        {
            "check": "dataset",
            "ok": bool(env.get("dataset_exists")),
            "details": str(env.get("dataset", "")),
            "recommendation": "" if env.get("dataset_exists") else "Set DATASET to an existing zip.",
        },
        {
            "check": "imports",
            "ok": not missing_imports,
            "details": ", ".join(name for name, ok in imports.items() if ok),
            "recommendation": "" if not missing_imports else "Install: " + ", ".join(missing_imports),
        },
        {
            "check": "optional",
            "ok": not missing_optional,
            "details": ", ".join(name for name, ok in optional_imports.items() if ok),
            "recommendation": "" if not missing_optional else "Optional install: " + ", ".join(missing_optional),
        },
    ]
    if redis_status:
        rows.append(
            {
                "check": "redis",
                "ok": bool(redis_status.get("ok")),
                "details": f"{redis_status.get('host')}:{redis_status.get('port')}/{redis_status.get('db')}",
                "recommendation": "" if redis_status.get("ok") else "Start Redis or set REDIS_HOST/REDIS_PORT.",
            }
        )
    if llm:
        rows.append(
            {
                "check": "llm",
                "ok": bool(llm.get("ok")),
                "details": f"{llm.get('source')} | {llm.get('model')} | key={bool(llm.get('api_key_seen'))}",
                "recommendation": "" if llm.get("ok") else str(llm.get("error", ""))[:300],
            }
        )
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


def check_redis(*, host: str = "localhost", port: int = 6379, db: int = 0) -> dict[str, Any]:
    try:
        import redis

        client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3)
        return {"ok": bool(client.ping()), "host": host, "port": port, "db": db}
    except Exception as exc:
        return {
            "ok": False,
            "host": host,
            "port": port,
            "db": db,
            "error": f"{type(exc).__name__}: {exc}",
            "recommendations": ["Start Redis: docker compose up -d redis, or point REDIS_HOST/REDIS_PORT to an existing Redis."],
        }


def redis_lock_status(*, prefix: str, host: str = "localhost", port: int = 6379, db: int = 0) -> dict[str, Any]:
    """Return the GigaEvolve instance-lock state for a Redis prefix."""
    key = f"{prefix}:__instance_lock__"
    try:
        import redis

        client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3)
        value = client.get(key)
        return {
            "ok": True,
            "locked": value is not None,
            "key": key,
            "value": value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value,
        }
    except Exception as exc:
        return {"ok": False, "locked": None, "key": key, "error": f"{type(exc).__name__}: {exc}"}


def redis_prefix_status(*, prefix: str, host: str = "localhost", port: int = 6379, db: int = 0) -> dict[str, Any]:
    """Return GigaEvolve Redis key/status counts for one prefix."""
    try:
        import redis

        client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3)
        statuses = ("queued", "running", "done", "discarded", "failed")
        status_counts = {status: int(client.scard(f"{prefix}:status:{status}")) for status in statuses}
        key_count = sum(1 for _ in client.scan_iter(match=f"{prefix}:*", count=1000))
        return {
            "ok": True,
            "prefix": prefix,
            "db": db,
            "has_data": bool(key_count),
            "key_count": key_count,
            "status_counts": status_counts,
        }
    except Exception as exc:
        return {"ok": False, "prefix": prefix, "db": db, "error": f"{type(exc).__name__}: {exc}"}


def clear_redis_instance_lock(*, prefix: str, host: str = "localhost", port: int = 6379, db: int = 0) -> dict[str, Any]:
    """Delete a stale GigaEvolve instance lock after you have checked no run is alive."""
    key = f"{prefix}:__instance_lock__"
    try:
        import redis

        client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3)
        value = client.get(key)
        deleted = client.delete(key)
        return {
            "ok": True,
            "key": key,
            "deleted": int(deleted),
            "previous_value": value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value,
        }
    except Exception as exc:
        return {"ok": False, "key": key, "deleted": 0, "error": f"{type(exc).__name__}: {exc}"}


def clear_redis_prefix(*, prefix: str, host: str = "localhost", port: int = 6379, db: int = 0) -> dict[str, Any]:
    """Delete all Redis keys for one GigaEvolve prefix."""
    try:
        import redis

        client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3)
        keys = list(client.scan_iter(match=f"{prefix}:*", count=1000))
        deleted = int(client.delete(*keys)) if keys else 0
        return {"ok": True, "prefix": prefix, "db": db, "matched": len(keys), "deleted": deleted}
    except Exception as exc:
        return {"ok": False, "prefix": prefix, "db": db, "deleted": 0, "error": f"{type(exc).__name__}: {exc}"}


def start_redis() -> dict[str, Any]:
    """Start the repo Redis service with docker compose."""
    completed = subprocess.run(
        ["docker", "compose", "up", "-d", "redis"],
        cwd=str(project_root()),
        text=True,
        capture_output=True,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "recommendations": [] if completed.returncode == 0 else ["Check Docker Desktop/daemon and docker-compose.yml."],
    }


def benchmark_backends(
    *,
    hash_id: str = "phash",
    dataset: str | Path = "../dataset_v1_public.zip",
    max_pairs: int = 10,
    budget: int = 300,
    backends: Iterable[str] = ("sequential", "thread", "process"),
    workers: int | None = None,
    worker_cap: int = 30,
    problem_output_dir: str | Path = "outputs/gigaevolve_problems",
    as_dataframe: bool = True,
):
    """Run the real evaluator backend benchmark and return a compact comparison table."""
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for backend in backends:
        cmd = [
            sys.executable,
            "-m",
            "evohash.stage3_gigaevolve.bench",
            "--hash",
            hash_id,
            "--dataset",
            str(dataset),
            "--max-pairs",
            str(max_pairs),
            "--budget",
            str(budget),
            "--backend",
            backend,
            "--worker-cap",
            str(worker_cap),
            "--problem-output-dir",
            str(problem_output_dir),
        ]
        if workers is not None:
            cmd += ["--workers", str(workers)]
        completed = subprocess.run(
            cmd,
            cwd=str(project_root()),
            env=_subprocess_env(),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            errors.append(
                {
                    "requested_backend": backend,
                    "ok": False,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-2000:],
                    "stdout": completed.stdout[-2000:],
                }
            )
            continue
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as exc:
            errors.append({"requested_backend": backend, "ok": False, "error": f"{type(exc).__name__}: {exc}", "stdout": completed.stdout[-2000:]})
            continue
        rows.append(_benchmark_row(payload))

    result = _maybe_dataframe(rows, as_dataframe=as_dataframe)
    if errors and hasattr(result, "attrs"):
        result.attrs["errors"] = errors
    elif errors:
        return {"ok": False, "rows": rows, "errors": errors}
    return result


def export_best_program(
    *,
    prefix: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    metric: str = "score",
    minimize: bool = True,
    top_n: int = 10,
    output_dir: str | Path = "outputs/gigaevolve_exports",
) -> dict[str, Any]:
    programs = fetch_programs(host=redis_host, port=redis_port, db=redis_db, prefix=prefix)
    ranked = rank_programs(programs, metric=metric, minimize=minimize, state_filter="done", top_n=top_n)
    if not ranked:
        return {"ok": False, "error": f"No done programs with metric {metric!r} for prefix {prefix!r}."}
    out = project_path(output_dir) / prefix
    out.mkdir(parents=True, exist_ok=True)
    save_ranked_programs(ranked, out, metric=metric)
    return export_summary(ranked, metric=metric, output_dir=out)


def save_run_snapshot(
    *,
    prefix: str,
    file: str | Path,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
) -> dict[str, Any]:
    return save_snapshot(file=project_path(file), host=redis_host, port=redis_port, db=redis_db, prefix=prefix)


def read_metrics_jsonl(path: str | Path, *, as_dataframe: bool = True):
    """Read a metrics.jsonl produced by run_gigaevolve into rows/table."""
    rows: list[dict[str, Any]] = []
    metrics_path = Path(path)
    if not metrics_path.exists():
        return _maybe_dataframe(rows, as_dataframe=as_dataframe)
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        row = {
            "event_index": event.get("event_index"),
            "time": event.get("time"),
            "program_id": event.get("program_id"),
            "best_score_so_far": event.get("best_score_so_far"),
        }
        metrics = event.get("metrics") or {}
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


def list_gigaevolve_runs(
    log_roots: Iterable[str | Path] | None = None,
    *,
    as_dataframe: bool = True,
) -> Any:
    """List saved GigaEvolve run directories with manifest status."""
    roots = list(log_roots or [project_path("outputs/gigaevolve_run_logs")])
    rows: list[dict[str, Any]] = []
    for root in roots:
        root_path = project_path(root)
        if not root_path.exists():
            continue
        for run_path in root_path.iterdir():
            if not run_path.is_dir():
                continue
            manifest = run_status(run_path)
            params = manifest.get("params") if isinstance(manifest.get("params"), dict) else {}
            rows.append(
                {
                    "run_dir": str(run_path.resolve()),
                    "name": run_path.name,
                    "mtime": datetime.fromtimestamp(run_path.stat().st_mtime).isoformat(timespec="seconds"),
                    "status": manifest.get("status"),
                    "pid": manifest.get("pid"),
                    "returncode": manifest.get("returncode"),
                    "run_name": manifest.get("run_name"),
                    "hash": params.get("hash_id"),
                    "generations": params.get("generations"),
                    "max_pairs": params.get("max_pairs"),
                    "log_path": str((run_path / "run.log").resolve()),
                    "metrics_path": str((run_path / "metrics.jsonl").resolve()),
                }
            )
    rows.sort(key=lambda row: str(row.get("mtime") or ""), reverse=True)
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


def find_gigaevolve_run_by_pid(
    pid: int,
    log_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Find a saved run directory by the PID stored in manifest.json."""
    target_pid = int(pid)
    for row in list_gigaevolve_runs(log_roots, as_dataframe=False):
        try:
            row_pid = int(row.get("pid"))
        except (TypeError, ValueError):
            continue
        if row_pid == target_pid:
            return {"ok": True, **row}
    return {"ok": False, "pid": target_pid, "error": "No manifest with this pid was found."}


def latest_gigaevolve_run(log_roots: Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Return the newest saved GigaEvolve run directory."""
    rows = list_gigaevolve_runs(log_roots, as_dataframe=False)
    if not rows:
        return {"ok": False, "error": "No run directories found."}
    return {"ok": True, **rows[0]}


def build_gigaevolve_command(
    *,
    hash_id: str,
    dataset: str | Path,
    run_name: str,
    split: str = "public",
    max_pairs: int = 10,
    budget: int = 300,
    generations: int = 3,
    eval_backend: str = "auto",
    eval_workers: int | None = None,
    worker_cap: int = 30,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    llm_source: str = "openrouter",
    model: str = DEFAULT_OPENROUTER_MODEL,
    local_base_url: str = "http://localhost:8000/v1",
    max_mutations_per_generation: int = 4,
    max_elites_per_generation: int = 4,
    max_concurrent_dags: int = 3,
    dag_prefetch_factor: int = 1,
    stage_timeout: float = 1200,
    dag_timeout: float = 3600,
    llm_temperature: float | None = 0.6,
    llm_max_retries: int = 8,
    llm_max_tokens: int | None = 81920,
    llm_request_timeout: float | None = 600,
    resume: bool = False,
    problem_output_dir: str | Path = "outputs/gigaevolve_problems",
    seed_config: str | Path = "results/attack_configs_optimized.json",
    seed_attacks: str = "all",
    require_proxy: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Build the exact subprocess command used by the notebook launchers."""
    return _build_gigaevolve_cmd(
        hash_id=hash_id,
        dataset=dataset,
        run_name=run_name,
        split=split,
        max_pairs=max_pairs,
        budget=budget,
        generations=generations,
        eval_backend=eval_backend,
        eval_workers=eval_workers,
        worker_cap=worker_cap,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        llm_source=llm_source,
        model=model,
        local_base_url=local_base_url,
        max_mutations_per_generation=max_mutations_per_generation,
        max_elites_per_generation=max_elites_per_generation,
        max_concurrent_dags=max_concurrent_dags,
        dag_prefetch_factor=dag_prefetch_factor,
        stage_timeout=stage_timeout,
        dag_timeout=dag_timeout,
        llm_temperature=llm_temperature,
        llm_max_retries=llm_max_retries,
        llm_max_tokens=llm_max_tokens,
        llm_request_timeout=llm_request_timeout,
        resume=resume,
        problem_output_dir=problem_output_dir,
        seed_config=seed_config,
        seed_attacks=seed_attacks,
        require_proxy=require_proxy,
        dry_run=dry_run,
    )


def start_gigaevolve_background(
    *,
    hash_id: str,
    dataset: str | Path,
    run_name: str,
    split: str = "public",
    max_pairs: int = 10,
    budget: int = 300,
    generations: int = 3,
    eval_backend: str = "auto",
    eval_workers: int | None = None,
    worker_cap: int = 30,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    llm_source: str = "openrouter",
    model: str = DEFAULT_OPENROUTER_MODEL,
    local_base_url: str = "http://localhost:8000/v1",
    max_mutations_per_generation: int = 4,
    max_elites_per_generation: int = 4,
    max_concurrent_dags: int = 3,
    dag_prefetch_factor: int = 1,
    stage_timeout: float = 1200,
    dag_timeout: float = 3600,
    llm_temperature: float | None = 0.6,
    llm_max_retries: int = 8,
    llm_max_tokens: int | None = 81920,
    llm_request_timeout: float | None = 600,
    resume: bool = False,
    clear_existing_prefix: bool = False,
    problem_output_dir: str | Path = "outputs/gigaevolve_problems",
    seed_config: str | Path = "results/attack_configs_optimized.json",
    seed_attacks: str = "all",
    require_proxy: bool = False,
    dry_run: bool = False,
    tracking: bool = False,
    log_dir: str | Path = "outputs/gigaevolve_run_logs",
) -> dict[str, Any]:
    """Start GigaEvolve in the background with all console output redirected to run.log."""
    cmd = _build_gigaevolve_cmd(
        tracking=tracking,
        hash_id=hash_id,
        dataset=dataset,
        run_name=run_name,
        split=split,
        max_pairs=max_pairs,
        budget=budget,
        generations=generations,
        eval_backend=eval_backend,
        eval_workers=eval_workers,
        worker_cap=worker_cap,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        llm_source=llm_source,
        model=model,
        local_base_url=local_base_url,
        max_mutations_per_generation=max_mutations_per_generation,
        max_elites_per_generation=max_elites_per_generation,
        max_concurrent_dags=max_concurrent_dags,
        dag_prefetch_factor=dag_prefetch_factor,
        stage_timeout=stage_timeout,
        dag_timeout=dag_timeout,
        llm_temperature=llm_temperature,
        llm_max_retries=llm_max_retries,
        llm_max_tokens=llm_max_tokens,
        llm_request_timeout=llm_request_timeout,
        resume=resume,
        problem_output_dir=problem_output_dir,
        seed_config=seed_config,
        seed_attacks=seed_attacks,
        require_proxy=require_proxy,
        dry_run=dry_run,
    )
    if clear_existing_prefix and not resume and not dry_run:
        clear_redis_prefix(prefix=run_name, host=redis_host, port=redis_port, db=redis_db)
    run_log_dir = _run_log_dir(project_path(log_dir), run_name)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / "run.log"
    metrics_path = run_log_dir / "metrics.jsonl"
    manifest_path = run_log_dir / "manifest.json"
    command_path = run_log_dir / "command.txt"

    command_path.write_text(" ".join(str(part) for part in cmd) + "\n", encoding="utf-8")
    manifest = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "pid": None,
        "returncode": None,
        "project_root": str(project_root()),
        "python": sys.executable,
        "cmd": cmd,
        "paths": {
            "run_dir": str(run_log_dir),
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "manifest_path": str(manifest_path),
            "command_path": str(command_path),
        },
        "params": {
            "hash_id": hash_id,
            "dataset": str(dataset),
            "split": split,
            "max_pairs": max_pairs,
            "budget": budget,
            "generations": generations,
            "eval_backend": eval_backend,
            "eval_workers": eval_workers,
            "worker_cap": worker_cap,
            "max_concurrent_dags": max_concurrent_dags,
            "dag_prefetch_factor": dag_prefetch_factor,
            "redis_host": redis_host,
            "redis_port": redis_port,
            "redis_db": redis_db,
            "llm_source": llm_source,
            "model": model,
            "local_base_url": local_base_url,
            "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens,
            "llm_request_timeout": llm_request_timeout,
            "llm_max_retries": llm_max_retries,
            "resume": resume,
            "clear_existing_prefix": clear_existing_prefix,
            "seed_config": str(seed_config),
            "seed_attacks": seed_attacks,
        },
        "git": _git_info(),
    }
    _write_manifest(manifest_path, manifest)

    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root()),
        env=_subprocess_env(),
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    manifest["pid"] = proc.pid
    _write_manifest(manifest_path, manifest)
    _BACKGROUND_PROCS[str(run_log_dir.resolve())] = proc

    waiter = threading.Thread(
        target=_wait_and_finalize_background_run,
        args=(proc, log_file, manifest_path),
        daemon=True,
    )
    waiter.start()
    return {
        "ok": True,
        "pid": proc.pid,
        "run_dir": str(run_log_dir),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "command_path": str(command_path),
    }


def run_gigaevolve_live_panel(
    *,
    hash_id: str,
    dataset: str | Path,
    run_name: str,
    split: str = "public",
    max_pairs: int = 10,
    budget: int = 300,
    generations: int = 3,
    eval_backend: str = "auto",
    eval_workers: int | None = None,
    worker_cap: int = 30,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    llm_source: str = "openrouter",
    model: str = DEFAULT_OPENROUTER_MODEL,
    local_base_url: str = "http://localhost:8000/v1",
    max_mutations_per_generation: int = 4,
    max_elites_per_generation: int = 4,
    max_concurrent_dags: int = 3,
    dag_prefetch_factor: int = 1,
    stage_timeout: float = 1200,
    dag_timeout: float = 3600,
    llm_temperature: float | None = 0.6,
    llm_max_retries: int = 8,
    llm_max_tokens: int | None = 81920,
    llm_request_timeout: float | None = 600,
    resume: bool = False,
    clear_existing_prefix: bool = False,
    problem_output_dir: str | Path = "outputs/gigaevolve_problems",
    seed_config: str | Path = "results/attack_configs_optimized.json",
    seed_attacks: str = "all",
    require_proxy: bool = False,
    dry_run: bool = False,
    log_dir: str | Path = "outputs/gigaevolve_run_logs",
    tail_lines: int = 500,
    height: str = "520px",
    width: str = "100%",
    title: str = "GigaEvolve live log",
    refresh_sec: float = 0.2,
) -> dict[str, Any]:
    """Run GigaEvolve and stream the child process stdout into one notebook panel."""
    cmd = _build_gigaevolve_cmd(
        hash_id=hash_id,
        dataset=dataset,
        run_name=run_name,
        split=split,
        max_pairs=max_pairs,
        budget=budget,
        generations=generations,
        eval_backend=eval_backend,
        eval_workers=eval_workers,
        worker_cap=worker_cap,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        llm_source=llm_source,
        model=model,
        local_base_url=local_base_url,
        max_mutations_per_generation=max_mutations_per_generation,
        max_elites_per_generation=max_elites_per_generation,
        max_concurrent_dags=max_concurrent_dags,
        dag_prefetch_factor=dag_prefetch_factor,
        stage_timeout=stage_timeout,
        dag_timeout=dag_timeout,
        llm_temperature=llm_temperature,
        llm_max_retries=llm_max_retries,
        llm_max_tokens=llm_max_tokens,
        llm_request_timeout=llm_request_timeout,
        resume=resume,
        problem_output_dir=problem_output_dir,
        seed_config=seed_config,
        seed_attacks=seed_attacks,
        require_proxy=require_proxy,
        dry_run=dry_run,
    )
    if clear_existing_prefix and not resume and not dry_run:
        clear_redis_prefix(prefix=run_name, host=redis_host, port=redis_port, db=redis_db)

    run_log_dir = _run_log_dir(project_path(log_dir), run_name)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / "run.log"
    metrics_path = run_log_dir / "metrics.jsonl"
    manifest_path = run_log_dir / "manifest.json"
    command_path = run_log_dir / "command.txt"
    command_path.write_text(" ".join(str(part) for part in cmd) + "\n", encoding="utf-8")

    manifest = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "pid": None,
        "returncode": None,
        "project_root": str(project_root()),
        "python": sys.executable,
        "cmd": cmd,
        "paths": {
            "run_dir": str(run_log_dir),
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "manifest_path": str(manifest_path),
            "command_path": str(command_path),
        },
        "params": {
            "hash_id": hash_id,
            "dataset": str(dataset),
            "split": split,
            "max_pairs": max_pairs,
            "budget": budget,
            "generations": generations,
            "eval_backend": eval_backend,
            "eval_workers": eval_workers,
            "worker_cap": worker_cap,
            "max_concurrent_dags": max_concurrent_dags,
            "dag_prefetch_factor": dag_prefetch_factor,
            "redis_host": redis_host,
            "redis_port": redis_port,
            "redis_db": redis_db,
            "llm_source": llm_source,
            "model": model,
            "local_base_url": local_base_url,
            "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens,
            "llm_request_timeout": llm_request_timeout,
            "llm_max_retries": llm_max_retries,
            "resume": resume,
            "clear_existing_prefix": clear_existing_prefix,
            "seed_config": str(seed_config),
            "seed_attacks": seed_attacks,
        },
        "git": _git_info(),
    }
    _write_manifest(manifest_path, manifest)

    lines: list[str] = []
    metric_events = 0
    generations_done = 0
    best_score: float | None = None
    latest_metrics: dict[str, Any] | None = None
    stopped_by_user = False
    proc: subprocess.Popen[str] | None = None

    panel = _LiveLogPanel(
        title=title,
        run_path=run_log_dir,
        height=height,
        width=width,
    )

    def current_status(*, suffix: str = "") -> dict[str, Any]:
        status = {
            "status": "running" if proc is None or proc.poll() is None else ("finished" if proc.returncode == 0 else "failed"),
            "pid": None if proc is None else proc.pid,
            "returncode": None if proc is None else proc.poll(),
        }
        detail = f"gen={generations_done}/{generations} evals={metric_events}"
        if best_score is not None:
            detail += f" best={best_score:.4g}"
        if suffix:
            detail += f" | {suffix}"
        status["detail"] = detail
        return status

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root()),
            env=_subprocess_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert proc.stdout is not None
        manifest["pid"] = proc.pid
        _write_manifest(manifest_path, manifest)
        panel.update(lines, current_status())

        next_refresh = 0.0
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file, metrics_path.open(
            "w", encoding="utf-8", buffering=1
        ) as metrics_file:
            for line in proc.stdout:
                log_file.write(line)
                clean = line.rstrip()
                lines.append(clean)
                if len(lines) > int(tail_lines):
                    del lines[: len(lines) - int(tail_lines)]

                generation_event = _parse_generation_progress(clean)
                if generation_event is not None:
                    generations_done = max(generations_done, generation_event)

                event = _parse_fetch_metrics(clean)
                if event is not None:
                    metric_events += 1
                    latest_metrics = event["metrics"]
                    best_score = _best_score(best_score, latest_metrics.get("score"))
                    event["event_index"] = metric_events
                    event["best_score_so_far"] = best_score
                    metrics_file.write(json.dumps(event, ensure_ascii=False) + "\n")

                now = time.monotonic()
                if now >= next_refresh or event is not None or generation_event is not None:
                    panel.update(lines, current_status())
                    next_refresh = now + max(0.05, float(refresh_sec))

        returncode = proc.wait()
    except KeyboardInterrupt:
        stopped_by_user = True
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait(timeout=10)
        else:
            returncode = None if proc is None else proc.returncode
        panel.update(lines, current_status(suffix="stopped by user"))
    finally:
        if proc is not None:
            returncode = proc.poll()
            if returncode is None:
                returncode = proc.wait()
        else:
            returncode = None
        status = "stopped" if stopped_by_user else ("finished" if returncode == 0 else "failed")
        manifest["status"] = status
        manifest["returncode"] = returncode
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_manifest(manifest_path, manifest)
        panel.update(lines, current_status(suffix=status))

    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "run_dir": str(run_log_dir),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "command_path": str(command_path),
        "metric_events": metric_events,
        "generations_done": generations_done,
        "total_generations": generations,
        "best_score": best_score,
        "latest_metrics": latest_metrics,
        "stopped_by_user": stopped_by_user,
        "tail": lines[-80:],
    }


def monitor_gigaevolve_run(
    run_dir: str | Path,
    *,
    poll_sec: float = 2.0,
    show_progress: bool = True,
    progress_mode: str = "generations",
    total_generations: int | None = None,
    stop_when_finished: bool = True,
    max_idle_sec: float | None = None,
) -> dict[str, Any]:
    """Parse a background run.log into metrics.jsonl and show a quiet tqdm."""
    run_path = Path(run_dir)
    log_path = run_path / "run.log"
    metrics_path = run_path / "metrics.jsonl"
    manifest_path = run_path / "manifest.json"
    manifest = run_status(run_path)
    if total_generations is None:
        total_generations = _manifest_total_generations(manifest)
    progress_total = total_generations if progress_mode == "generations" else None
    progress_desc = f"{run_path.name} generations" if progress_mode == "generations" else f"{run_path.name} evaluations"
    progress_unit = "gen" if progress_mode == "generations" else "eval"
    progress = _progress_bar(total=progress_total, enabled=show_progress, desc=progress_desc, unit=progress_unit)
    seen_program_ids: set[str] = set()
    event_index = 0
    generations_done = 0
    best_score: float | None = None
    latest_metrics: dict[str, Any] | None = None
    last_change = time.monotonic()
    position = 0

    if metrics_path.exists():
        for row in read_metrics_jsonl(metrics_path, as_dataframe=False):
            pid = str(row.get("program_id") or "")
            if pid:
                seen_program_ids.add(pid)
            event_index = max(event_index, int(row.get("event_index") or 0))
            best_score = _best_score(best_score, row.get("score"))

    try:
        while True:
            changed = False
            if log_path.exists():
                with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
                    log_file.seek(position)
                    for line in log_file:
                        clean = line.rstrip()
                        generation_event = _parse_generation_progress(clean)
                        if generation_event is not None:
                            generations_done = max(generations_done, generation_event)
                            changed = True
                            if progress is not None and progress_mode == "generations":
                                if total_generations is not None and progress.total != total_generations:
                                    progress.total = total_generations
                                if generations_done > progress.n:
                                    progress.update(generations_done - progress.n)
                                postfix: dict[str, Any] = {"gen": generations_done}
                                if total_generations:
                                    postfix["total"] = total_generations
                                if best_score is not None:
                                    postfix["best"] = best_score
                                progress.set_postfix({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in postfix.items()})
                            continue

                        event = _parse_fetch_metrics(clean)
                        if event is None:
                            continue
                        program_id = str(event["program_id"])
                        if program_id in seen_program_ids:
                            continue
                        seen_program_ids.add(program_id)
                        event_index += 1
                        latest_metrics = event["metrics"]
                        best_score = _best_score(best_score, latest_metrics.get("score"))
                        event["event_index"] = event_index
                        event["best_score_so_far"] = best_score
                        with metrics_path.open("a", encoding="utf-8") as metrics_file:
                            metrics_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                        changed = True
                        if progress is not None and progress_mode == "evaluations":
                            progress.update(1)
                            score = _float_or_none(latest_metrics.get("score"))
                            postfix = {"score": score, "best": best_score}
                            progress.set_postfix({k: f"{v:.4g}" for k, v in postfix.items() if v is not None})
                    position = log_file.tell()
            if changed:
                last_change = time.monotonic()

            status = run_status(run_path)
            if stop_when_finished and status.get("status") in {"finished", "failed"}:
                break
            if max_idle_sec is not None and time.monotonic() - last_change >= max_idle_sec:
                break
            time.sleep(max(0.2, float(poll_sec)))
    finally:
        if progress is not None:
            progress.close()

    return {
        "ok": True,
        "run_dir": str(run_path),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "metric_events": event_index,
        "generations_done": generations_done,
        "total_generations": total_generations,
        "best_score": best_score,
        "latest_metrics": latest_metrics,
        "status": run_status(run_path),
    }


def tail_run_log(run_dir: str | Path, *, n: int = 80) -> list[str]:
    """Return the last n lines of a background run log."""
    log_path = Path(run_dir) / "run.log"
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-int(n):]


def watch_run_log(run_dir: str | Path, *, n: int = 80, poll_sec: float = 2.0, stop_when_finished: bool = False) -> None:
    """Notebook live log viewer for a second notebook."""
    try:
        from IPython.display import clear_output
    except Exception:
        clear_output = None

    while True:
        if clear_output is not None:
            clear_output(wait=True)
        print("\n".join(tail_run_log(run_dir, n=n)))
        status = run_status(run_dir)
        print(f"\nstatus={status.get('status')} pid={status.get('pid')} returncode={status.get('returncode')}")
        if stop_when_finished and status.get("status") in {"finished", "failed"}:
            break
        time.sleep(max(0.2, float(poll_sec)))


def watch_run_log_panel(
    run_dir: str | Path,
    *,
    n: int = 120,
    poll_sec: float = 2.0,
    stop_when_finished: bool = False,
    height: str = "560px",
    width: str = "100%",
    title: str = "run.log tail",
    auto_refresh: bool = True,
    follow: bool = True,
) -> dict[str, Any]:
    """Notebook log viewer that updates one stable output area."""
    run_path = Path(run_dir)
    try:
        return _watch_run_log_widget_panel(
            run_path,
            n=n,
            poll_sec=poll_sec,
            stop_when_finished=stop_when_finished,
            height=height,
            width=width,
            title=title,
            auto_refresh=auto_refresh,
            follow=follow,
        )
    except ImportError:
        return _watch_run_log_html_panel(
            run_path,
            n=n,
            poll_sec=poll_sec,
            stop_when_finished=stop_when_finished,
            height=height,
            width=width,
            title=title,
            auto_refresh=auto_refresh,
        )


def watch_run_log_stream_panel(
    run_dir: str | Path,
    *,
    n: int = 120,
    poll_sec: float = 1.0,
    stop_when_finished: bool = False,
    height: str = "560px",
    width: str = "100%",
    title: str = "run.log stream",
    filter_text: str = "",
) -> dict[str, Any]:
    """Start a non-blocking, append-only notebook tailer for run.log."""
    run_path = Path(run_dir)
    log_path = run_path / "run.log"
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        return watch_run_log_panel(
            run_path,
            n=n,
            poll_sec=poll_sec,
            stop_when_finished=stop_when_finished,
            height=height,
            width=width,
            title=title,
        )

    stop_key = str(run_path.resolve())
    previous = _LOG_TAILERS.get(stop_key)
    if previous is not None:
        previous.set()

    stop_event = threading.Event()
    _LOG_TAILERS[stop_key] = stop_event

    header = widgets.HTML()
    pause_toggle = widgets.ToggleButton(
        value=False,
        description="Pause",
        tooltip="Pause appending new log lines",
        icon="pause",
    )
    clear_button = widgets.Button(
        description="Clear",
        tooltip="Clear this viewer output only",
        icon="trash",
    )
    stop_button = widgets.Button(
        description="Stop viewer",
        tooltip="Stop this tailer thread; it does not stop GigaEvolve",
        icon="stop",
    )
    filter_input = widgets.Text(
        value=filter_text,
        placeholder="filter new lines",
        description="Filter",
        layout=widgets.Layout(width="300px"),
    )
    output = widgets.Output(
        layout=widgets.Layout(
            width=width,
            height=height,
            overflow_y="auto",
            border="1px solid #d0d7de",
        )
    )
    controls = widgets.HBox([pause_toggle, clear_button, stop_button, filter_input])
    display(widgets.VBox([header, controls, output]))

    def set_header(*, suffix: str = "") -> None:
        status = run_status(run_path)
        header.value = _log_panel_header_html(
            title=title,
            run_path=run_path,
            status=status,
            line_count=0,
            suffix=suffix,
        )

    def clear_output(_=None) -> None:
        output.clear_output(wait=True)

    def stop_viewer(_=None) -> None:
        stop_event.set()
        set_header(suffix="viewer stopped")

    clear_button.on_click(clear_output)
    stop_button.on_click(stop_viewer)

    def append_line(line: str) -> None:
        needle = filter_input.value.strip()
        if needle and needle.lower() not in line.lower():
            return
        output.append_stdout(line)

    def tail_loop() -> None:
        position = 0
        try:
            if log_path.exists():
                initial = tail_run_log(run_path, n=n)
                if initial:
                    output.append_stdout("\n".join(initial) + "\n")
                position = log_path.stat().st_size
            set_header(suffix="streaming")
            while not stop_event.is_set():
                status = run_status(run_path)
                if not pause_toggle.value and log_path.exists():
                    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
                        log_file.seek(position)
                        for line in log_file:
                            append_line(line)
                        position = log_file.tell()
                suffix = "paused" if pause_toggle.value else "streaming"
                set_header(suffix=suffix)
                if stop_when_finished and status.get("status") in {"finished", "failed"}:
                    break
                time.sleep(max(0.2, float(poll_sec)))
        except Exception as exc:
            output.append_stderr(f"\n[viewer error] {type(exc).__name__}: {exc}\n")
            set_header(suffix="viewer error")
        finally:
            stop_event.set()
            if _LOG_TAILERS.get(stop_key) is stop_event:
                _LOG_TAILERS.pop(stop_key, None)
            set_header(suffix="viewer stopped")

    thread = threading.Thread(target=tail_loop, name=f"log-tail-{run_path.name}", daemon=True)
    thread.start()
    return {
        "ok": True,
        "run_dir": str(run_path),
        "log_path": str(log_path),
        "thread": thread.name,
        "mode": "append-only",
    }


def run_status(run_dir: str | Path) -> dict[str, Any]:
    """Return manifest status and live Popen status when available in this kernel."""
    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"status": "unknown", "manifest_error": "invalid json"}
    proc = _BACKGROUND_PROCS.get(str(run_path.resolve()))
    if proc is not None:
        returncode = proc.poll()
        manifest["returncode"] = returncode
        if returncode is None:
            manifest["status"] = "running"
        elif returncode == 0:
            manifest["status"] = "finished"
        else:
            manifest["status"] = "failed"
    return manifest or {"status": "unknown", "run_dir": str(run_path)}


def stop_gigaevolve_background(run_dir: str | Path, *, timeout: float = 10.0) -> dict[str, Any]:
    """Terminate a background run started in this kernel."""
    run_path = Path(run_dir)
    proc = _BACKGROUND_PROCS.get(str(run_path.resolve()))
    if proc is None:
        return {"ok": False, "error": "This kernel does not own the background process.", "status": run_status(run_path)}
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
    return {"ok": True, "status": run_status(run_path)}


def run_gigaevolve(
    *,
    hash_id: str,
    dataset: str | Path,
    run_name: str,
    split: str = "public",
    max_pairs: int = 10,
    budget: int = 300,
    generations: int = 3,
    eval_backend: str = "auto",
    eval_workers: int | None = None,
    worker_cap: int = 30,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    llm_source: str = "openrouter",
    model: str = DEFAULT_OPENROUTER_MODEL,
    local_base_url: str = "http://localhost:8000/v1",
    max_mutations_per_generation: int = 4,
    max_elites_per_generation: int = 4,
    max_concurrent_dags: int = 3,
    dag_prefetch_factor: int = 1,
    stage_timeout: float = 1200,
    dag_timeout: float = 3600,
    llm_temperature: float | None = 0.6,
    llm_max_retries: int = 8,
    llm_max_tokens: int | None = 81920,
    llm_request_timeout: float | None = 600,
    resume: bool = False,
    clear_existing_prefix: bool = False,
    problem_output_dir: str | Path = "outputs/gigaevolve_problems",
    seed_config: str | Path = "results/attack_configs_optimized.json",
    seed_attacks: str = "all",
    require_proxy: bool = False,
    dry_run: bool = False,
    tracking: bool = False,
    stream: bool = True,
    log_dir: str | Path = "outputs/gigaevolve_run_logs",
    show_progress: bool = True,
    echo_log: bool = False,
) -> dict[str, Any]:
    """Launch GigaEvolve. This uses a subprocess because Hydra owns process config."""
    cmd = _build_gigaevolve_cmd(
        tracking=tracking,
        hash_id=hash_id,
        dataset=dataset,
        run_name=run_name,
        split=split,
        max_pairs=max_pairs,
        budget=budget,
        generations=generations,
        eval_backend=eval_backend,
        eval_workers=eval_workers,
        worker_cap=worker_cap,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        llm_source=llm_source,
        model=model,
        local_base_url=local_base_url,
        max_mutations_per_generation=max_mutations_per_generation,
        max_elites_per_generation=max_elites_per_generation,
        max_concurrent_dags=max_concurrent_dags,
        dag_prefetch_factor=dag_prefetch_factor,
        stage_timeout=stage_timeout,
        dag_timeout=dag_timeout,
        llm_temperature=llm_temperature,
        llm_max_retries=llm_max_retries,
        llm_max_tokens=llm_max_tokens,
        llm_request_timeout=llm_request_timeout,
        resume=resume,
        problem_output_dir=problem_output_dir,
        seed_config=seed_config,
        seed_attacks=seed_attacks,
        require_proxy=require_proxy,
        dry_run=dry_run,
    )
    if clear_existing_prefix and not resume and not dry_run:
        clear_redis_prefix(prefix=run_name, host=redis_host, port=redis_port, db=redis_db)

    if stream:
        run_log_dir = _run_log_dir(project_path(log_dir), run_name)
        run_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_log_dir / "run.log"
        metrics_path = run_log_dir / "metrics.jsonl"
        progress = _progress_bar(
            total=None,
            enabled=show_progress and not dry_run,
            desc=f"{run_name} evaluations",
        )
        metric_events = 0
        best_score: float | None = None
        latest_metrics: dict[str, Any] | None = None
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root()),
            env=_subprocess_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        with log_path.open("w", encoding="utf-8") as log_file, metrics_path.open("w", encoding="utf-8") as metrics_file:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                if echo_log:
                    print(line, end="")
                clean = line.rstrip()
                lines.append(clean)
                event = _parse_fetch_metrics(clean)
                if event is not None:
                    metric_events += 1
                    latest_metrics = event["metrics"]
                    score = _float_or_none(latest_metrics.get("score"))
                    if score is not None and (best_score is None or score < best_score):
                        best_score = score
                    event["event_index"] = metric_events
                    event["best_score_so_far"] = best_score
                    metrics_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    metrics_file.flush()
                    if progress is not None:
                        progress.update(1)
                        postfix = {"score": score, "best": best_score}
                        progress.set_postfix({k: f"{v:.4g}" for k, v in postfix.items() if v is not None})
        if progress is not None:
            progress.close()
        return {
            "ok": proc.wait() == 0,
            "returncode": proc.returncode,
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "metric_events": metric_events,
            "best_score": best_score,
            "latest_metrics": latest_metrics,
            "tail": lines[-80:],
        }

    completed = subprocess.run(cmd, cwd=str(project_root()), env=_subprocess_env(), text=True, capture_output=True)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _benchmark_row(payload: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "requested_backend",
        "eval_backend",
        "wall_sec",
        "score",
        "target_metric",
        "asr",
        "mean_l2_raw",
        "mean_queries",
        "mean_gap_to_threshold",
        "n_pairs",
        "is_valid",
    ]
    return {key: payload.get(key) for key in keep}


def _build_gigaevolve_cmd(
    *,
    hash_id: str,
    dataset: str | Path,
    run_name: str,
    split: str,
    max_pairs: int,
    budget: int,
    generations: int,
    eval_backend: str,
    eval_workers: int | None,
    worker_cap: int,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    llm_source: str,
    model: str,
    local_base_url: str,
    max_mutations_per_generation: int,
    max_elites_per_generation: int,
    max_concurrent_dags: int,
    dag_prefetch_factor: int,
    stage_timeout: float,
    dag_timeout: float,
    llm_temperature: float | None,
    llm_max_retries: int,
    llm_max_tokens: int | None,
    llm_request_timeout: float | None,
    resume: bool,
    problem_output_dir: str | Path,
    seed_config: str | Path,
    seed_attacks: str,
    require_proxy: bool,
    dry_run: bool,
    tracking: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "evohash.stage3_gigaevolve.run",
        "--hash",
        hash_id,
        "--dataset",
        str(dataset),
        "--split",
        split,
        "--max-pairs",
        str(max_pairs),
        "--budget",
        str(budget),
        "--generations",
        str(generations),
        "--problem-name",
        run_name,
        "--problem-output-dir",
        str(problem_output_dir),
        "--seed-config",
        str(seed_config),
        "--seed-attacks",
        seed_attacks,
        "--eval-backend",
        eval_backend,
        "--worker-cap",
        str(worker_cap),
        "--redis-host",
        redis_host,
        "--redis-port",
        str(redis_port),
        "--redis-db",
        str(redis_db),
        "--max-mutations-per-generation",
        str(max_mutations_per_generation),
        "--max-elites-per-generation",
        str(max_elites_per_generation),
        "--max-concurrent-dags",
        str(max_concurrent_dags),
        "--dag-prefetch-factor",
        str(dag_prefetch_factor),
        "--stage-timeout",
        str(stage_timeout),
        "--dag-timeout",
        str(dag_timeout),
        "--llm-max-retries",
        str(llm_max_retries),
        "--llm",
        llm_source,
        "--model",
        model,
    ]
    if llm_temperature is not None:
        cmd += ["--temperature", str(llm_temperature)]
    if llm_max_tokens is not None:
        cmd += ["--max-tokens", str(llm_max_tokens)]
    if llm_request_timeout is not None:
        cmd += ["--request-timeout", str(llm_request_timeout)]
    if resume:
        cmd += ["--resume"]
    if eval_workers is not None:
        cmd += ["--eval-workers", str(eval_workers)]
    if llm_source == "local":
        cmd += ["--local-base-url", local_base_url]
    if require_proxy:
        cmd += ["--require-proxy"]
    if tracking:
        cmd += ["--tracking"]
    if dry_run:
        cmd += ["--dry-run"]
    return cmd


def _wait_and_finalize_background_run(proc: subprocess.Popen, log_file, manifest_path: Path) -> None:
    returncode = proc.wait()
    try:
        log_file.flush()
        log_file.close()
    except Exception:
        pass
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest["returncode"] = returncode
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = "finished" if returncode == 0 else "failed"
    _write_manifest(manifest_path, manifest)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_info() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={project_root()}", *args],
            cwd=str(project_root()),
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": run_git("status", "--short"),
    }


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(package_parent()), str(project_root())]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("WANDB_CONSOLE", "off")
    return env


def _maybe_dataframe(rows: list[dict[str, Any]], *, as_dataframe: bool):
    if not as_dataframe:
        return rows
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows


_FETCH_METRICS_RE = re.compile(r"\[FetchMetrics\]\s+(?P<program_id>\S+)\s+metrics=(?P<metrics>\{.*\})")


def _parse_fetch_metrics(line: str) -> dict[str, Any] | None:
    match = _FETCH_METRICS_RE.search(line)
    if not match:
        return None
    try:
        raw_metrics = ast.literal_eval(match.group("metrics"))
    except Exception:
        return None
    metrics = {key: _coerce_metric(value) for key, value in raw_metrics.items()}
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "program_id": match.group("program_id"),
        "metrics": metrics,
    }


def _coerce_metric(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _best_score(current: float | None, candidate: Any) -> float | None:
    score = _float_or_none(candidate)
    if score is None:
        return current
    if current is None or score < current:
        return score
    return current


_GENERATION_DONE_RE = re.compile(r"\[EvolutionEngine\]\s+gen=(?P<gen>\d+)\s+done\b")
_SELECTING_ELITES_RE = re.compile(r"selecting elites \(gen=(?P<gen>\d+),")
_STOP_MAX_GENERATIONS_RE = re.compile(r"Stop: max_generations=(?P<max_generations>\d+)")


def _parse_generation_progress(line: str) -> int | None:
    """Return completed generation count parsed from GigaEvolve log lines."""
    match = _GENERATION_DONE_RE.search(line) or _SELECTING_ELITES_RE.search(line)
    if match:
        return int(match.group("gen")) + 1
    stop_match = _STOP_MAX_GENERATIONS_RE.search(line)
    if stop_match:
        return int(stop_match.group("max_generations"))
    return None


def _manifest_total_generations(manifest: dict[str, Any]) -> int | None:
    params = manifest.get("params") if isinstance(manifest, dict) else None
    if not isinstance(params, dict):
        return None
    value = params.get("generations")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _progress_bar(*, total: int | None, enabled: bool, desc: str, unit: str = "eval"):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, unit=unit)
    except Exception:
        return None


class _LiveLogPanel:
    """Small notebook output panel that prefers ipywidgets and falls back to HTML."""

    def __init__(self, *, title: str, run_path: Path, height: str, width: str) -> None:
        self.title = title
        self.run_path = run_path
        self.height = height
        self.width = width
        self.header: Any | None = None
        self.textarea: Any | None = None
        self.handle: Any | None = None
        self.html_cls: Any | None = None

        try:
            import ipywidgets as widgets
            from IPython.display import display

            self.header = widgets.HTML()
            self.textarea = widgets.Textarea(
                value="",
                disabled=True,
                layout=widgets.Layout(width=width, height=height),
            )
            display(widgets.VBox([self.header, self.textarea]))
            return
        except Exception:
            pass

        try:
            from IPython.display import HTML, display

            self.html_cls = HTML
            self.handle = display(HTML(""), display_id=True)
        except Exception:
            self.handle = None

    def update(self, lines: list[str], status: dict[str, Any]) -> None:
        suffix = str(status.get("detail") or "")
        if self.textarea is not None and self.header is not None:
            self.textarea.value = "\n".join(lines)
            self.header.value = _log_panel_header_html(
                title=self.title,
                run_path=self.run_path,
                status=status,
                line_count=len(lines),
                suffix=suffix,
            )
            return

        if self.handle is not None and self.html_cls is not None:
            self.handle.update(
                self.html_cls(
                    _log_panel_html(
                        title=self.title,
                        run_path=self.run_path,
                        status=status,
                        lines=lines,
                        height=self.height,
                        width=self.width,
                        suffix=suffix,
                    )
                )
            )
            return

        if lines:
            print(lines[-1], flush=True)


def _watch_run_log_widget_panel(
    run_path: Path,
    *,
    n: int,
    poll_sec: float,
    stop_when_finished: bool,
    height: str,
    width: str,
    title: str,
    auto_refresh: bool,
    follow: bool,
) -> dict[str, Any]:
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:
        raise ImportError from exc

    header = widgets.HTML()
    refresh_toggle = widgets.ToggleButton(
        value=auto_refresh,
        description="Auto",
        tooltip="Keep refreshing the log panel",
        icon="refresh",
    )
    follow_toggle = widgets.ToggleButton(
        value=follow,
        description="Follow",
        tooltip="Jump to the newest log lines on each refresh",
        icon="angle-double-down",
    )
    refresh_button = widgets.Button(
        description="Refresh",
        tooltip="Refresh once",
        icon="search",
    )
    lines_input = widgets.BoundedIntText(
        value=int(n),
        min=20,
        max=5000,
        step=20,
        description="Lines",
        layout=widgets.Layout(width="160px"),
    )
    filter_input = widgets.Text(
        value="",
        placeholder="filter text",
        description="Filter",
        layout=widgets.Layout(width="260px"),
    )
    textarea = widgets.Textarea(
        value="",
        disabled=False,
        layout=widgets.Layout(width=width, height=height),
    )
    controls = widgets.HBox([refresh_toggle, follow_toggle, refresh_button, lines_input, filter_input])
    display(widgets.VBox([header, controls, textarea]))

    stopped_by_user = False
    refresh_requested = True
    last_rendered = ""

    def request_refresh(_=None) -> None:
        nonlocal refresh_requested
        refresh_requested = True

    refresh_button.on_click(request_refresh)

    def render(*, suffix: str = "", force_content: bool = False) -> bool:
        nonlocal last_rendered
        status = run_status(run_path)
        line_count = int(lines_input.value)
        lines = tail_run_log(run_path, n=line_count)
        needle = filter_input.value.strip()
        if needle:
            lines = [line for line in lines if needle.lower() in line.lower()]
        text = "\n".join(lines)
        should_update_text = force_content or follow_toggle.value or not last_rendered
        if should_update_text and text != last_rendered:
            textarea.value = text
            last_rendered = text
            if follow_toggle.value:
                textarea.focus()
        header.value = _log_panel_header_html(
            title=title,
            run_path=run_path,
            status=status,
            line_count=len(lines),
            suffix=suffix
            or (
                "auto status; content frozen until Follow/Refresh"
                if refresh_toggle.value and not follow_toggle.value
                else ("auto refresh on" if refresh_toggle.value else "paused")
            ),
        )
        return bool(status.get("status") in {"finished", "failed"})

    try:
        while True:
            if refresh_toggle.value or refresh_requested:
                force_content = refresh_requested
                refresh_requested = False
                is_finished = render(force_content=force_content)
            else:
                status = run_status(run_path)
                header.value = _log_panel_header_html(
                    title=title,
                    run_path=run_path,
                    status=status,
                    line_count=len(last_rendered.splitlines()) if last_rendered else 0,
                    suffix="paused",
                )
                is_finished = bool(status.get("status") in {"finished", "failed"})
            if stop_when_finished and is_finished:
                break
            time.sleep(max(0.2, float(poll_sec)))
    except KeyboardInterrupt:
        stopped_by_user = True
        render(suffix="viewer stopped by user")
    return {"ok": True, "run_dir": str(run_path), "status": run_status(run_path), "stopped_by_user": stopped_by_user}


def _watch_run_log_html_panel(
    run_path: Path,
    *,
    n: int,
    poll_sec: float,
    stop_when_finished: bool,
    height: str,
    width: str,
    title: str,
    auto_refresh: bool,
) -> dict[str, Any]:
    from IPython.display import HTML, display

    handle = display(HTML(""), display_id=True)
    stopped_by_user = False
    if not auto_refresh:
        lines = tail_run_log(run_path, n=n)
        status = run_status(run_path)
        handle.update(
            HTML(
                _log_panel_html(
                    title=title,
                    run_path=run_path,
                    status=status,
                    lines=lines,
                    height=height,
                    width=width,
                    suffix="static snapshot; rerun cell to refresh",
                )
            )
        )
        return {"ok": True, "run_dir": str(run_path), "status": status, "stopped_by_user": stopped_by_user}
    try:
        while True:
            status = run_status(run_path)
            lines = tail_run_log(run_path, n=n)
            handle.update(HTML(_log_panel_html(title=title, run_path=run_path, status=status, lines=lines, height=height, width=width)))
            if stop_when_finished and status.get("status") in {"finished", "failed"}:
                break
            time.sleep(max(0.2, float(poll_sec)))
    except KeyboardInterrupt:
        stopped_by_user = True
        status = run_status(run_path)
        lines = tail_run_log(run_path, n=n)
        handle.update(
            HTML(
                _log_panel_html(
                    title=title,
                    run_path=run_path,
                    status=status,
                    lines=lines,
                    height=height,
                    width=width,
                    suffix="stopped by user",
                )
            )
        )
    return {"ok": True, "run_dir": str(run_path), "status": run_status(run_path), "stopped_by_user": stopped_by_user}


def _log_panel_header_html(
    *,
    title: str,
    run_path: Path,
    status: dict[str, Any],
    line_count: int,
    suffix: str = "",
) -> str:
    parts = [
        f"<b>{html.escape(title)}</b>",
        f"run={html.escape(run_path.name)}",
        f"status={html.escape(str(status.get('status', 'unknown')))}",
        f"pid={html.escape(str(status.get('pid', '')))}",
        f"returncode={html.escape(str(status.get('returncode', '')))}",
        f"lines={line_count}",
    ]
    if suffix:
        parts.append(html.escape(suffix))
    return "<span style='font-family:system-ui,sans-serif;font-size:13px'>" + " | ".join(parts) + "</span>"


def _log_panel_html(
    *,
    title: str,
    run_path: Path,
    status: dict[str, Any],
    lines: list[str],
    height: str,
    width: str,
    suffix: str = "",
) -> str:
    body = html.escape("\n".join(lines))
    header = _log_panel_header_html(title=title, run_path=run_path, status=status, line_count=len(lines), suffix=suffix)
    return f"""
<div style="width:{html.escape(width)}">
  <div style="margin:0 0 6px 0">{header}</div>
  <pre style="height:{html.escape(height)}; overflow:auto; white-space:pre-wrap; margin:0; padding:10px; border:1px solid #d0d7de; border-radius:6px; background:#0b1020; color:#d6deff; font-size:12px; line-height:1.35">{body}</pre>
</div>
"""


def _run_log_dir(base_dir: str | Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_name)
    return Path(base_dir) / f"{safe_name}_{stamp}"
