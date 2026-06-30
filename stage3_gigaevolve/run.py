"""CLI for running EvoHash through the external GigaEvolve framework."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from evohash.stage3_gigaevolve.generator import GigaEvolveProblemConfig, generate_problem_package
from evohash.stage3_gigaevolve.models import DEFAULT_OPENROUTER_MODEL, resolve_openrouter_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and run a GigaEvolve EvoHash problem.")
    parser.add_argument("--hash", dest="hash_id", default="phash", choices=["phash", "pdq", "neuralhash", "photodna"])
    parser.add_argument("--dataset", default="../dataset_v1_public.zip")
    parser.add_argument("--split", default="public")
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--eval-backend", default="auto", choices=["auto", "sequential", "thread", "process"])
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--worker-cap", type=int, default=30)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--max-mutations-per-generation", type=int, default=None)
    parser.add_argument("--max-elites-per-generation", type=int, default=None)
    parser.add_argument("--max-concurrent-dags", type=int, default=3)
    parser.add_argument("--dag-prefetch-factor", type=int, default=1)
    parser.add_argument("--stage-timeout", type=float, default=None)
    parser.add_argument("--dag-timeout", type=float, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm", default="openrouter", help="Hydra llm config name, e.g. openrouter or single.")
    parser.add_argument("--model", default="", help="Model id or OpenRouter alias.")
    parser.add_argument("--local-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--proxy-port", type=int, default=8100)
    parser.add_argument("--no-proxy", action="store_true", help="Call OpenRouter directly instead of proxying.")
    parser.add_argument("--require-proxy", action="store_true", help="Fail instead of falling back to direct OpenRouter.")
    parser.add_argument("--dotenv", default=".env", help="Optional .env file loaded before launch.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=None)
    parser.add_argument("--gigaevo-root", default=None, help="Defaults to this repo's third_party/gigaevolve_core.")
    parser.add_argument("--config-dir", default=None, help="Hydra config dir. Defaults to <gigaevo-core>/config.")
    parser.add_argument("--problem-output-dir", default="outputs/gigaevolve_problems")
    parser.add_argument("--problem-name", default=None)
    parser.add_argument("--seed-config", default="results/attack_configs_optimized.json")
    parser.add_argument("--seed-attacks", default="all")
    parser.add_argument(
        "--tracking",
        action="store_true",
        help=(
            "Enable GigaEvolve per-generation CSV/plot export. OFF by default: "
            "that export does a full Redis get_all (MGET of every program) every "
            "generation, which on large runs times out / blocks Redis. Leave off "
            "and export once at the end if needed."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _load_dotenv(Path(args.dotenv))
    problem_dir = generate_problem_package(
        GigaEvolveProblemConfig(
            hash_id=args.hash_id,
            output_dir=args.problem_output_dir,
            problem_name=args.problem_name,
            seed_config_path=args.seed_config,
            seed_attacks=tuple(item.strip() for item in args.seed_attacks.split(",") if item.strip()),
        )
    ).resolve()

    project_root = Path(__file__).resolve().parents[1]
    gigaevo_core = (
        Path(args.gigaevo_root).resolve()
        if args.gigaevo_root
        else (project_root / "third_party" / "gigaevolve_core").resolve()
    )
    config_dir = Path(args.config_dir).resolve() if args.config_dir else (gigaevo_core / "config").resolve()
    dataset = Path(args.dataset).resolve()
    llm_model = _resolve_model(args.llm, args.model)
    openrouter_base_url = args.openrouter_base_url
    proxy_proc: subprocess.Popen | None = None
    proxy_log: object | None = None
    if args.llm == "openrouter" and not args.no_proxy:
        try:
            proxy_proc, proxy_log = _start_openrouter_proxy(args.proxy_port, project_root)
            openrouter_base_url = f"http://localhost:{args.proxy_port}/api/v1"
        except Exception as exc:
            if args.require_proxy:
                raise
            print(f"WARNING: OpenRouter proxy is unavailable ({exc}); using direct OpenRouter.", file=sys.stderr)

    env = dict(os.environ)
    pydantic_warning_filter = "ignore:Pydantic serializer warnings:UserWarning:pydantic.main"
    if env.get("PYTHONWARNINGS"):
        env["PYTHONWARNINGS"] = f"{env['PYTHONWARNINGS']},{pydantic_warning_filter}"
    else:
        env["PYTHONWARNINGS"] = pydantic_warning_filter

    project_parent = project_root.parent
    pythonpath_parts = [str(project_parent), str(project_root), str(gigaevo_core)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if args.llm == "openrouter" and env.get("OPENROUTER_API_KEY") and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["OPENROUTER_API_KEY"]

    env.update(
        {
            "EVOHASH_HASH_ID": args.hash_id,
            "EVOHASH_DATASET_ZIP": str(dataset),
            "EVOHASH_SPLIT": args.split,
            "EVOHASH_MAX_PAIRS": str(args.max_pairs),
            "EVOHASH_BUDGET": str(args.budget),
            "EVOHASH_RESIZE_SIZE": str(args.resize_size),
            "EVOHASH_PARALLEL_BACKEND": args.eval_backend,
            "EVOHASH_PARALLEL_WORKER_CAP": str(args.worker_cap),
            "EVOHASH_ATTACK_CONFIG": str(Path(args.seed_config).resolve()),
        }
    )
    if args.eval_workers is not None:
        env["EVOHASH_PARALLEL_WORKERS"] = str(args.eval_workers)

    cmd = [
        sys.executable,
        str(gigaevo_core / "run.py"),
        "--config-dir",
        str(config_dir),
        f"problem.name={problem_dir.name}",
        f"problem.dir={problem_dir}",
        f"llm={args.llm}",
        f"max_generations={args.generations}",
        f"redis.db={args.redis_db}",
        f"redis.host={args.redis_host}",
        f"redis.port={args.redis_port}",
        "redis_max_connections=512",
        "redis_connection_pool_timeout=120.0",
        "redis_health_check_interval=30",
        "redis_max_retries=10",
        "redis_retry_delay=1.0",
        "pipeline=with_context",
        f"logging.level={args.log_level}",
        f"max_concurrent_dags={args.max_concurrent_dags}",
        f"+runner_config.prefetch_factor={args.dag_prefetch_factor}",
        f"prompts.dir={problem_dir / 'prompts'}",
        f"redis.resume={str(bool(args.resume)).lower()}",
    ]
    # Per-generation CSV/plot export is opt-in: it runs a full Redis get_all
    # (MGET of every program) each generation, which on large runs exceeds the
    # client read timeout and stalls Redis for concurrent workers. Default off.
    if args.tracking:
        cmd += [
            "tracking.enabled=true",
            "tracking.minimize=true",
            "tracking.fitness_col=metric_score",
        ]
    if args.max_mutations_per_generation is not None:
        cmd.append(f"max_mutations_per_generation={args.max_mutations_per_generation}")
    if args.max_elites_per_generation is not None:
        cmd.append(f"max_elites_per_generation={args.max_elites_per_generation}")
    if args.stage_timeout is not None:
        cmd.append(f"stage_timeout={args.stage_timeout}")
    if args.dag_timeout is not None:
        cmd.append(f"dag_timeout={args.dag_timeout}")
    if args.temperature is not None:
        cmd.append(f"temperature={args.temperature}")
    if args.max_tokens is not None:
        cmd.append(f"max_tokens={args.max_tokens}")
    if args.request_timeout is not None:
        cmd.append(f"request_timeout={args.request_timeout}")
    if args.llm_max_retries is not None:
        cmd.append(f"llm_max_retries={args.llm_max_retries}")
    if llm_model:
        cmd.append(f"model_name={llm_model}")
    if args.llm == "local":
        cmd.append(f"llm_base_url={args.local_base_url}")
    if args.llm == "openrouter":
        cmd.append(f"openrouter_base_url={openrouter_base_url}")

    print("Generated problem:", problem_dir)
    print("LLM:", args.llm, llm_model or "(config default)")
    if proxy_proc is not None:
        print("OpenRouter proxy:", openrouter_base_url)
    print("Command:")
    print(" ".join(str(part) for part in cmd))
    if args.dry_run:
        _stop_proxy(proxy_proc, proxy_log)
        return

    try:
        raise SystemExit(subprocess.run(cmd, cwd=str(gigaevo_core), env=env).returncode)
    finally:
        _stop_proxy(proxy_proc, proxy_log)


def _resolve_model(llm: str, model: str) -> str:
    model = model.strip()
    if llm == "openrouter":
        return resolve_openrouter_model(model or DEFAULT_OPENROUTER_MODEL)
    if llm == "local":
        if not model:
            raise SystemExit("--model is required when --llm local.")
        return model
    return model


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and _is_unset(os.environ.get(key)):
            os.environ[key] = value


def _is_unset(value: str | None) -> bool:
    return value is None or not value.strip() or value.strip().lower() == "none"


def _start_openrouter_proxy(port: int, project_root: Path) -> tuple[subprocess.Popen, object]:
    log_path = project_root / "outputs" / "logs" / "openrouter_proxy.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(project_root / "stage3_gigaevolve" / "openrouter_proxy.py"), "--port", str(port), "-v"],
        stdout=subprocess.DEVNULL,
        stderr=log_file,
        cwd=str(project_root),
        env=os.environ.copy(),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    url = f"http://localhost:{port}/"
    for _ in range(50):
        if proc.poll() is not None:
            log_file.flush()
            log_file.close()
            raise RuntimeError(f"OpenRouter proxy failed to start. See {log_path}")
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            return proc, log_file
        except Exception:
            time.sleep(0.1)
    proc.terminate()
    log_file.close()
    raise RuntimeError(f"OpenRouter proxy did not become ready on {url}. See {log_path}")


def _stop_proxy(proc: subprocess.Popen | None, log_file: object | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
