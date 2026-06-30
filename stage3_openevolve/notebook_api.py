"""Notebook-first launcher for OpenEvolve runs over EvoHash problems.

Two ways to run, mirroring ``stage3_gigaevolve``:

* **Background subprocess (recommended, used by the notebook).**  OpenEvolve calls
  ``asyncio.run()`` and uses process-based parallel evaluation, which fight with a
  Jupyter kernel.  ``start_openevolve_background`` spawns ``python -m
  evohash.stage3_openevolve.run`` as a separate process and ``live_tail`` shows a
  non-drifting log view.  This is the stable single entrypoint.

* **In-process (secondary).**  ``run_openevolve`` calls ``openevolve.run_evolution``
  directly.  Fine for scripts/CI, but blocks and can clash with a running event loop.

The parity helpers (``model_table``, ``preflight``/``preflight_table``,
``preview_problem``, ``read_checkpoint_metrics``, ``run_status`` …) match the
GigaEvolve notebook API so the two notebooks read the same.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evohash.stage3_common.config import EvaluationConfig, LLMConfig
from evohash.stage3_common.llm import OPENROUTER_MODELS, normalize_llm_config

from .export import extract_best_programs, extract_champions, latest_checkpoint
from .generator import (
    OpenEvolveProblemConfig,
    generate_openevolve_problem,
    _system_prompt,
)
from .run import _load_dotenv, build_configs_from_spec
from .runtime import ENV_TRACE_DIR, export_eval_env
from .seeding import prepopulate_seed_database

# run_name -> handle dict (kept so the notebook can re-query a launched run)
_BACKGROUND_RUNS: dict[str, dict[str, Any]] = {}
_BACKGROUND_FILES: dict[str, Any] = {}

# Metric columns surfaced in tables (subset of what the evaluator returns).
_METRIC_COLS = (
    "score",
    "combined_score",
    "asr",
    "mean_l2_raw",
    "mean_queries",
    "mean_pixel_l2",
    "is_valid",
)


# ---------------------------------------------------------------------------
# Path helpers (mirror stage3_gigaevolve.notebook_api)
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """The ``evohash`` package directory (parent of stage3_openevolve)."""
    return Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Resolve a notebook-facing relative path from the package root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate


def package_parent() -> Path:
    """Directory that should be on PYTHONPATH so ``import evohash`` works."""
    return project_root().parent


# ---------------------------------------------------------------------------
# Models + preflight
# ---------------------------------------------------------------------------

def model_table(*, as_dataframe: bool = True):
    """Known OpenRouter aliases -> model ids (same registry GigaEvolve uses)."""
    rows = [
        {"alias": alias, "model": model, "url": f"https://openrouter.ai/{model}"}
        for alias, model in OPENROUTER_MODELS.items()
    ]
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


def preflight(
    *,
    dataset: str | Path | None = None,
    seed_config: str | Path | None = None,
    llm: LLMConfig | dict[str, Any] | None = None,
    dotenv: str | Path | None = None,
    check_llm_connection: bool = False,
    require_openevolve: bool = True,
) -> dict[str, Any]:
    """Check openevolve + optional deps + dataset/seed-config presence.

    When ``check_llm_connection`` and ``llm`` are given, also pings the LLM
    (see :func:`check_llm`) to catch a dead key / wrong endpoint / empty-content
    (reasoning) model BEFORE launching a run.
    """
    info: dict[str, Any] = {"ok": True, "checks": {}}

    oe: dict[str, Any] = {"available": False}
    try:
        import openevolve  # type: ignore

        oe["available"] = True
        oe["version"] = getattr(openevolve, "__version__", "unknown")
        oe["path"] = getattr(openevolve, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        oe["error"] = f"{type(exc).__name__}: {exc}"
    info["checks"]["openevolve"] = oe
    if require_openevolve and not oe["available"]:
        info["ok"] = False

    for lib in ("numpy", "yaml", "pandas", "ipywidgets", "torch", "lpips"):
        chk: dict[str, Any] = {"available": False, "optional": lib in {"torch", "lpips", "pandas", "ipywidgets"}}
        try:
            module = __import__(lib)
            chk["available"] = True
            chk["version"] = getattr(module, "__version__", "")
        except Exception as exc:  # noqa: BLE001
            chk["error"] = f"{type(exc).__name__}: {exc}"
            if not chk["optional"]:
                info["ok"] = False
        info["checks"][lib] = chk

    if dataset is not None:
        p = Path(dataset)
        info["checks"]["dataset"] = {"path": str(p), "exists": p.exists()}
        if not p.exists():
            info["ok"] = False
    if seed_config is not None:
        p = Path(seed_config)
        info["checks"]["seed_config"] = {"path": str(p), "exists": p.exists()}
        if not p.exists():
            info["ok"] = False

    if check_llm_connection and llm is not None:
        llm_chk = check_llm(llm=llm, dotenv=dotenv)
        info["checks"]["llm"] = llm_chk
        # Empty content / bad key / HTTP error blocks the run; a reasoning-model
        # warning does NOT (it responded — just flagged as risky).
        if not llm_chk.get("available"):
            info["ok"] = False

    return info


def check_llm(
    *,
    llm: LLMConfig | dict[str, Any],
    dotenv: str | Path | None = None,
    prompt: str = "Reply with exactly one word: pong",
    max_tokens: int = 640,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Ping the configured LLM and verify it returns non-empty content.

    Catches the common pre-launch failures: missing/invalid key, wrong endpoint,
    and — crucially — models that return **empty content** (which makes
    OpenEvolve's client raise ``'NoneType' object is not subscriptable``).
    Reasoning models (response carries a ``reasoning`` field) are flagged as a
    *warning*, not a hard fail: they answer short prompts but can return empty
    content under long evolution prompts. ``available`` stays True if the model
    actually returned content, so a working local reasoning LLM is not blocked.
    """
    import json as _json
    import urllib.error
    import urllib.request

    if dotenv is not None:
        _load_dotenv(Path(dotenv))

    if isinstance(llm, dict):
        from .run import _build_llm

        llm_obj = _build_llm(llm)
    else:
        llm_obj = llm

    result: dict[str, Any] = {
        "available": False,
        "source": getattr(llm_obj, "source", "?"),
        "model": getattr(llm_obj, "model", "?"),
    }

    try:
        cfg = normalize_llm_config(llm_obj)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["details"] = "config/key error"
        result["recommendation"] = "Set OPENROUTER_API_KEY in .env, or pass api_key / fix base_url."
        return result
    result["model"] = cfg.model

    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if cfg.source == "openrouter":
        headers.setdefault("HTTP-Referer", "https://localhost/evohash")
        headers.setdefault("X-Title", "EvoHash preflight")

    url = cfg.base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    req = urllib.request.Request(
        url, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result["error"] = f"HTTP {exc.code}: {body[:300]}"
        result["details"] = f"HTTP {exc.code}"
        result["recommendation"] = "Check API key, model id, and endpoint."
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["details"] = "request failed"
        result["recommendation"] = "Check base_url / network / that the local server is running."
        return result

    try:
        data = _json.loads(raw)
        choices = data.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        content = msg.get("content")
        finish = choices[0].get("finish_reason") if choices else None
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"unparseable response: {str(exc)[:160]}; raw={raw[:160]}"
        result["details"] = "bad response shape"
        result["recommendation"] = "Endpoint is not OpenAI-compatible?"
        return result

    reasoning = bool(msg.get("reasoning") or msg.get("reasoning_details"))
    has_content = isinstance(content, str) and content.strip() != ""
    result["finish_reason"] = finish
    result["reasoning_model"] = reasoning

    if not has_content:
        result["available"] = False
        result["details"] = "empty content" + (" (reasoning model)" if reasoning else "")
        result["recommendation"] = (
            "Empty content -> OpenEvolve's client will fail ('NoneType not subscriptable'). "
            "Use a non-reasoning instruct model or a local LLM."
        )
        return result

    result["available"] = True
    preview = content.strip().replace("\n", " ")[:50]
    if reasoning:
        result["warning"] = "reasoning model"
        result["details"] = f"ok, but REASONING model | '{preview}'"
        result["recommendation"] = (
            "Responded on a short prompt, but reasoning models can return empty content "
            "under long evolution prompts and break OpenEvolve. Prefer a non-reasoning "
            "model or a local LLM if the run fails early."
        )
    else:
        result["details"] = f"ok | '{preview}'"
    return result


def preflight_table(preflight_result: dict[str, Any], *, as_dataframe: bool = True):
    """Compact check table for notebook display."""
    rows = []
    for name, chk in (preflight_result.get("checks") or {}).items():
        ok = chk.get("available", chk.get("exists"))
        details = chk.get("details") or chk.get("version") or chk.get("path") or chk.get("error") or ""
        if chk.get("recommendation") and (not ok or chk.get("warning")):
            details = f"{details} -- {chk['recommendation']}"
        row = {"check": name, "ok": bool(ok), "details": str(details)}
        if chk.get("warning"):
            row["warning"] = chk["warning"]
        if chk.get("optional"):
            row["optional"] = True
        rows.append(row)
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------

def preview_problem(spec: dict[str, Any], *, default_output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate the problem package from ``spec`` WITHOUT running evolution.

    Returns the rendered system prompt (same notes GigaEvolve uses), the resolved
    seed pool, and the generated config.yaml — analogous to GigaEvolve's dry_run.

    Loads ``spec['dotenv']`` first so an OpenRouter key in ``.env`` is visible to
    ``normalize_llm_config`` — this runs in-kernel, unlike the subprocess runner
    which loads ``.env`` on its own.
    """
    _load_dotenv(Path(spec.get("dotenv") or (project_root() / ".env")))
    out = str(default_output_dir) if default_output_dir else str(project_path("outputs/openevolve_problems"))
    eval_cfg, llm_cfg, problem_cfg = build_configs_from_spec(spec, default_output_dir=out)
    problem_dir = generate_openevolve_problem(config=problem_cfg, llm=llm_cfg, evaluation=eval_cfg)

    bookkeeping = {}
    bk_path = problem_dir / "evohash_problem.json"
    if bk_path.is_file():
        try:
            bookkeeping = json.loads(bk_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bookkeeping = {}

    return {
        "problem_dir": str(problem_dir),
        "hash_id": eval_cfg.hash_id,
        "llm": {"source": llm_cfg.source, "model": llm_cfg.model},
        "feature_dimensions": list(problem_cfg.feature_dimensions),
        "iterations": int(problem_cfg.iterations),
        "system_prompt": _system_prompt(eval_cfg.hash_id),
        "config_yaml": (problem_dir / "config.yaml").read_text(encoding="utf-8"),
        "seed_attacks": bookkeeping.get("seed_attacks"),
        "primary_attack": bookkeeping.get("primary_attack"),
        "db_seed_attacks": bookkeeping.get("db_seed_attacks"),
    }


# ---------------------------------------------------------------------------
# Background launch + monitoring (subprocess; the stable entrypoint)
# ---------------------------------------------------------------------------

def start_openevolve_background(
    spec: dict[str, Any],
    *,
    spec_path: str | Path | None = None,
    python_exe: str | None = None,
) -> dict[str, Any]:
    """Launch ``run.py`` as a subprocess and return a handle dict.

    The handle carries the live ``proc`` plus ``output_dir`` / ``runner_log`` /
    ``run_name`` for monitoring.  A ``manifest.json`` is written so the run is
    discoverable later via :func:`list_openevolve_runs`.
    """
    spec = dict(spec)
    hash_id = (spec.get("evaluation") or {}).get("hash_id", "run")
    run_name = spec.get("run_name") or f"oe_{hash_id}_{int(time.time())}"
    spec["run_name"] = run_name

    output_dir = Path(
        spec.get("output_dir") or (project_path("outputs/openevolve_runs") / run_name)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec["output_dir"] = str(output_dir)
    runner_log = output_dir / "runner.log"
    subprocess_log = output_dir / "subprocess.log"

    spec_file = Path(spec_path) if spec_path else (output_dir / "spec.json")
    spec_file.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Force UTF-8 so OpenEvolve's unicode log lines (e.g. ✅) don't crash logging
    # on a Windows console defaulting to cp1251/cp1252.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    pythonpath_parts = [str(package_parent()), str(project_root())]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    cmd = [
        python_exe or sys.executable,
        "-m",
        "evohash.stage3_openevolve.run",
        "--spec",
        str(spec_file),
    ]
    # Do not use stdout=PIPE here. If the notebook does not continuously drain
    # the pipe, Windows can block the child process once the pipe buffer fills.
    # The runner itself writes structured logs to runner.log; this file captures
    # raw console output from openevolve/subprocess logging.
    subprocess_file = subprocess_log.open("a", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess_file,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(project_root()),
    )

    handle: dict[str, Any] = {
        "run_name": run_name,
        "pid": proc.pid,
        "proc": proc,
        "output_dir": str(output_dir),
        "runner_log": str(runner_log),
        "subprocess_log": str(subprocess_log),
        "spec_path": str(spec_file),
        "cmd": cmd,
        "started_at": time.time(),
    }
    _BACKGROUND_RUNS[run_name] = handle
    _BACKGROUND_FILES[run_name] = subprocess_file
    manifest = {k: v for k, v in handle.items() if k != "proc"}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    _start_manifest_waiter(run_name, proc, output_dir / "manifest.json")
    return handle


def live_tail(
    handle: dict[str, Any],
    *,
    max_lines: int = 500,
    height: str = "420px",
    poll: float = 0.5,
    follow: bool = True,
) -> dict[str, Any]:
    """Non-drifting live view of ``runner.log``.

    Uses an ``ipywidgets.Textarea`` whose *value* is replaced each refresh (fixed
    height, ring-buffered to ``max_lines``) so the output never accumulates or
    scrolls the notebook — the pattern from the prior project.  Falls back to
    printing only new lines if ipywidgets is unavailable.  Blocks until the
    process exits when ``follow=True``.
    """
    runner_log = Path(handle.get("runner_log", ""))
    proc = handle.get("proc")

    area = None
    try:
        import ipywidgets as widgets
        from IPython.display import display

        area = widgets.Textarea(
            value="",
            layout=widgets.Layout(width="100%", height=height),
            disabled=True,
        )
        display(area)
    except Exception:  # noqa: BLE001 - headless / no ipywidgets
        area = None

    printed = 0

    def alive() -> bool:
        return proc is not None and proc.poll() is None

    while True:
        lines: list[str] = []
        if runner_log.exists():
            try:
                lines = runner_log.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            if area is not None:
                area.value = "\n".join(lines[-max_lines:])
            elif len(lines) > printed:
                for line in lines[printed:]:
                    print(line, flush=True)
                printed = len(lines)
        if not follow or not alive():
            # one final refresh after exit
            if runner_log.exists() and area is not None:
                area.value = "\n".join(
                    runner_log.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
                )
            break
        time.sleep(poll)

    rc = proc.poll() if proc is not None else None
    return {"returncode": rc, "running": rc is None, "runner_log": str(runner_log)}


def read_checkpoint_metrics(output_dir: str | Path, *, as_dataframe: bool = True):
    """Per-checkpoint best metrics, oldest→newest (parallel to read_metrics_jsonl)."""
    root = Path(output_dir) / "checkpoints"
    rows: list[dict[str, Any]] = []

    def ckpt_num(d: Path) -> int:
        try:
            return int(d.name.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    if root.is_dir():
        for ckpt in sorted((d for d in root.glob("checkpoint_*") if d.is_dir()), key=ckpt_num):
            info = ckpt / "best_program_info.json"
            if not info.is_file():
                continue
            try:
                data = json.loads(info.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            metrics = data.get("metrics") or {}
            row: dict[str, Any] = {
                "iteration": data.get("current_iteration", ckpt_num(ckpt)),
                "program_id": data.get("id"),
            }
            for key in _METRIC_COLS:
                if key in metrics:
                    try:
                        row[key] = float(metrics[key])
                    except (TypeError, ValueError):
                        continue
            rows.append(row)
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


def run_status(handle: dict[str, Any]) -> dict[str, Any]:
    """Quick status of a launched run (running?, checkpoints, best index)."""
    proc = handle.get("proc")
    out = Path(handle.get("output_dir", ""))
    rc = proc.poll() if proc is not None else None
    ck_root = out / "checkpoints"
    n_ckpt = len([d for d in ck_root.glob("checkpoint_*") if d.is_dir()]) if ck_root.is_dir() else 0
    best_index = out / "best_programs" / "index.json"
    return {
        "run_name": handle.get("run_name"),
        "pid": handle.get("pid"),
        "running": rc is None,
        "returncode": rc,
        "output_dir": str(out),
        "checkpoints": n_ckpt,
        "best_programs_index": str(best_index) if best_index.exists() else None,
    }


def list_openevolve_runs(root: str | Path | None = None, *, as_dataframe: bool = True):
    """List runs under ``outputs/openevolve_runs`` (via their manifest.json)."""
    base = Path(root) if root else project_path("outputs/openevolve_runs")
    rows: list[dict[str, Any]] = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            manifest = d / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            ck_root = d / "checkpoints"
            rows.append(
                {
                    "run_name": data.get("run_name", d.name),
                    "output_dir": str(d),
                    "pid": data.get("pid"),
                    "checkpoints": len([x for x in ck_root.glob("checkpoint_*") if x.is_dir()])
                    if ck_root.is_dir()
                    else 0,
                    "started_at": data.get("started_at"),
                }
            )
    return _maybe_dataframe(rows, as_dataframe=as_dataframe)


def stop_openevolve_background(handle: dict[str, Any], *, timeout: float = 10.0) -> dict[str, Any]:
    """Best-effort stop of a launched subprocess."""
    proc = handle.get("proc")
    if proc is None:
        return {"ok": False, "reason": "handle has no process"}
    if proc.poll() is not None:
        return {"ok": True, "already_exited": True, "returncode": proc.returncode}
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
    _close_background_file(handle.get("run_name"))
    return {"ok": True, "returncode": proc.poll()}


# ===========================================================================
# In-process API (secondary).  Kept for scripts/CI; the notebook uses the
# subprocess path above.
# ===========================================================================

@dataclass
class OpenEvolveRunResult:
    problem_dir: Path
    best_program: Path | None
    best_metrics: dict[str, float]
    output_dir: Path
    framework_result: Any  # native openevolve return value (opaque)
    seed_report: dict[str, Any] = field(default_factory=dict)
    best_programs_index: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": "openevolve",
            "problem_dir": str(self.problem_dir),
            "best_program": str(self.best_program) if self.best_program else None,
            "best_metrics": dict(self.best_metrics),
            "output_dir": str(self.output_dir),
            "seed_report": dict(self.seed_report),
        }


def run_openevolve(
    *,
    evaluation: EvaluationConfig,
    llm: LLMConfig,
    problem: OpenEvolveProblemConfig,
    extra_env: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
    top_n: int = 5,
) -> OpenEvolveRunResult:
    """Generate the problem package and run openevolve in-process (blocking)."""
    try:
        import openevolve  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openevolve is not installed. Run `pip install openevolve` "
            "or call preflight() to verify your environment."
        ) from exc

    llm_cfg = normalize_llm_config(llm)
    problem_dir = generate_openevolve_problem(config=problem, llm=llm_cfg, evaluation=evaluation)
    seed_report = prepopulate_seed_database(
        problem_dir=problem_dir, config=problem, evaluation=evaluation
    )

    out_root = Path(output_dir) if output_dir is not None else problem_dir / "openevolve_run"
    out_root.mkdir(parents=True, exist_ok=True)

    env_updates = export_eval_env(evaluation)
    env_updates[ENV_TRACE_DIR] = str(out_root / "evaluation_trace")
    if extra_env:
        env_updates.update(extra_env)
    os.environ.update(env_updates)

    framework_result = _invoke_openevolve(
        openevolve,
        initial_program=str(problem_dir / "initial_program.py"),
        evaluator=str(problem_dir / "evaluator.py"),
        config=str(problem_dir / "config.yaml"),
        iterations=int(problem.iterations),
        output_dir=str(out_root),
    )

    best_program, best_metrics = _extract_best(framework_result, out_root)
    index = extract_best_programs(out_root, top_n=top_n)
    return OpenEvolveRunResult(
        problem_dir=problem_dir,
        best_program=best_program,
        best_metrics=best_metrics,
        output_dir=out_root,
        framework_result=framework_result,
        seed_report=seed_report,
        best_programs_index=index,
    )


def _invoke_openevolve(
    openevolve_module: Any,
    *,
    initial_program: str,
    evaluator: str,
    config: str,
    iterations: int,
    output_dir: str,
) -> Any:
    """Call whichever openevolve entry point the installed version provides."""
    if hasattr(openevolve_module, "run_evolution"):
        # >= 0.2 functional API. NOTE: the kwarg is ``config`` (path or Config
        # object), NOT ``config_path``; and cleanup=False keeps output_dir +
        # checkpoints on disk for extraction.
        return openevolve_module.run_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            config=config,
            iterations=iterations,
            output_dir=output_dir,
            cleanup=False,
        )
    if hasattr(openevolve_module, "OpenEvolve"):
        evolve = openevolve_module.OpenEvolve(
            initial_program_path=initial_program,
            evaluation_file=evaluator,
            config_path=config,
            output_dir=output_dir,
        )
        run_fn = getattr(evolve, "run", None)
        if run_fn is None:
            raise RuntimeError("OpenEvolve instance has no run() method")
        result = run_fn(iterations=iterations)
        if hasattr(result, "__await__"):
            import asyncio

            return asyncio.run(result)
        return result
    raise RuntimeError(
        "Installed openevolve has neither run_evolution() nor OpenEvolve class; "
        "update the wrapper in stage3_openevolve/notebook_api.py to match."
    )


def _extract_best(framework_result: Any, out_dir: Path) -> tuple[Path | None, dict[str, float]]:
    """Best single program from the EvolutionResult, else from the latest checkpoint."""
    best_metrics: dict[str, float] = {}
    metrics_attr = getattr(framework_result, "metrics", None) or getattr(
        framework_result, "best_metrics", None
    )
    if isinstance(metrics_attr, dict):
        for key, value in metrics_attr.items():
            try:
                best_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    # best_code -> write it next to the run for convenience
    best_path: Path | None = None
    best_code = getattr(framework_result, "best_code", None)
    if isinstance(best_code, str) and best_code.strip():
        best_path = out_dir / "best_code.py"
        try:
            best_path.write_text(best_code, encoding="utf-8")
        except OSError:
            best_path = None

    if best_path is None:
        ckpt = latest_checkpoint(out_dir)
        if ckpt is not None:
            candidate = ckpt / "best_program.py"
            if candidate.exists():
                best_path = candidate

    return best_path, best_metrics


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _maybe_dataframe(rows: list[dict[str, Any]], *, as_dataframe: bool):
    if as_dataframe:
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except Exception:  # noqa: BLE001 - pandas optional
            pass
    return rows


def _start_manifest_waiter(run_name: str, proc: subprocess.Popen, manifest_path: Path) -> None:
    def wait_and_update() -> None:
        returncode = proc.wait()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - best effort status update
            manifest = {}
        manifest["returncode"] = returncode
        manifest["finished_at"] = time.time()
        manifest["status"] = "finished" if returncode == 0 else "failed"
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        finally:
            _close_background_file(run_name)

    threading.Thread(target=wait_and_update, name=f"openevolve-wait-{run_name}", daemon=True).start()


def _close_background_file(run_name: Any) -> None:
    if not run_name:
        return
    file_obj = _BACKGROUND_FILES.pop(str(run_name), None)
    if file_obj is not None:
        try:
            file_obj.close()
        except Exception:
            pass
