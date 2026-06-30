"""Environment preflight for notebook-driven GigaEvolve runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REQUIRED_IMPORTS = {
    "redis": "redis",
    "loguru": "loguru",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "langchain_openai": "langchain-openai",
}

OPTIONAL_IMPORTS = {
    "aiohttp": "aiohttp",  # OpenRouter proxy
    "tqdm": "tqdm",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local dependencies and files for GigaEvolve runs.")
    parser.add_argument("--dataset", default="../dataset_v1_public.zip")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--check-redis", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checks = check_environment(
        dataset=args.dataset,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        check_redis=args.check_redis,
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    raise SystemExit(0 if checks["ok"] else 1)


def check_environment(
    *,
    dataset: str | Path = "../dataset_v1_public.zip",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    check_redis: bool = False,
) -> dict:
    """Return notebook-friendly environment status and install recommendations."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(dataset).resolve()
    imports = {name: _has_import(name) for name in REQUIRED_IMPORTS}
    optional_imports = {name: _has_import(name) for name in OPTIONAL_IMPORTS}
    missing = [name for name, ok in imports.items() if not ok]
    missing_optional = [name for name, ok in optional_imports.items() if not ok]

    recommendations: list[str] = []
    if missing:
        packages = [REQUIRED_IMPORTS[name] for name in missing]
        recommendations.append("Install required packages: python -m pip install " + " ".join(packages))
    if missing_optional:
        packages = [OPTIONAL_IMPORTS[name] for name in missing_optional]
        recommendations.append("Optional helpers: python -m pip install " + " ".join(packages))
    if "hydra" in missing:
        recommendations.append("Important: install hydra-core, not the old package named hydra.")
    if not dataset_path.exists():
        recommendations.append(f"Dataset not found: {dataset_path}. Set DATASET to the correct zip path.")

    checks = {
        "ok": False,
        "project_root": str(project_root),
        "gigaevolve_core": str(project_root / "third_party" / "gigaevolve_core"),
        "gigaevolve_core_exists": (project_root / "third_party" / "gigaevolve_core" / "run.py").exists(),
        "dataset": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "imports": imports,
        "optional_imports": optional_imports,
        "missing_imports": missing,
        "missing_optional_imports": missing_optional,
        "recommendations": recommendations,
    }
    if check_redis:
        checks["redis"] = _check_redis(redis_host, redis_port, redis_db)
        if not checks["redis"].get("ok", False):
            recommendations.append(
                "Start Redis, for example: docker compose up -d redis "
                f"(or set REDIS_HOST/REDIS_PORT; current {redis_host}:{redis_port}/{redis_db})."
            )
    checks["ok"] = (
        checks["gigaevolve_core_exists"]
        and checks["dataset_exists"]
        and not missing
        and (not check_redis or checks["redis"].get("ok", False))
    )
    return checks


def _has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_redis(host: str, port: int, db: int) -> dict:
    try:
        import redis

        client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3)
        return {"ok": bool(client.ping()), "host": host, "port": port, "db": db}
    except Exception as exc:
        return {"ok": False, "host": host, "port": port, "db": db, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        raise
