"""Benchmark generated GigaEvolve problem evaluation backends."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from evohash.stage3_gigaevolve.generator import GigaEvolveProblemConfig, generate_problem_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one generated initial program.")
    parser.add_argument("--hash", dest="hash_id", default="phash")
    parser.add_argument("--dataset", default="../dataset_v1_public.zip")
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--backend", default="thread", choices=["sequential", "thread", "process", "auto"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--worker-cap", type=int, default=30)
    parser.add_argument("--seed-program", default="atkscopes_seed.py")
    parser.add_argument("--problem-output-dir", default="outputs/gigaevolve_bench")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    problem_dir = generate_problem_package(
        GigaEvolveProblemConfig(
            hash_id=args.hash_id,
            output_dir=args.problem_output_dir,
            problem_name=f"evohash_{args.hash_id}_bench",
        )
    ).resolve()
    seed_path = problem_dir / "initial_programs" / args.seed_program
    if not seed_path.exists():
        raise FileNotFoundError(seed_path)

    old_env = dict(os.environ)
    try:
        os.environ.update(
            {
                "EVOHASH_HASH_ID": args.hash_id,
                "EVOHASH_DATASET_ZIP": str(Path(args.dataset).resolve()),
                "EVOHASH_MAX_PAIRS": str(args.max_pairs),
                "EVOHASH_BUDGET": str(args.budget),
                "EVOHASH_ATTACK_CONFIG": str((project_root / "results" / "attack_configs_optimized.json").resolve()),
                "EVOHASH_PARALLEL_BACKEND": args.backend,
                "EVOHASH_PARALLEL_WORKER_CAP": str(args.worker_cap),
                "GIGAEVO_PROGRAM_CODE_PATH": str(seed_path.resolve()),
            }
        )
        if args.workers is not None:
            os.environ["EVOHASH_PARALLEL_WORKERS"] = str(args.workers)
        else:
            os.environ.pop("EVOHASH_PARALLEL_WORKERS", None)

        context_ns: dict = {}
        exec((problem_dir / "context.py").read_text(encoding="utf-8"), context_ns)
        validate_ns: dict = {}
        exec((problem_dir / "validate.py").read_text(encoding="utf-8"), validate_ns)
        program_ns: dict = {}
        exec(seed_path.read_text(encoding="utf-8"), program_ns)

        context = context_ns["build_context"]()
        t0 = time.perf_counter()
        payload = program_ns["entrypoint"](context)
        metrics = validate_ns["validate"](context, payload)
        wall_sec = time.perf_counter() - t0
        print(
            json.dumps(
                {
                    "problem_dir": str(problem_dir),
                    "seed_program": seed_path.name,
                    "requested_backend": args.backend,
                    "eval_backend": payload.get("eval_backend"),
                    "wall_sec": wall_sec,
                    **metrics,
                },
                sort_keys=True,
            )
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)


if __name__ == "__main__":
    main()
