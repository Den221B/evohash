"""Generate a GigaEvolve problem package backed by EvoHash evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evohash.optimizer import clean_public_params, load_attack_config
from evohash.stage3_common.attack_source import build_inlined_attack_source
from evohash.stage3_common.prompt_templates import write_gigaevolve_templates
from evohash.stage3_common.prompts import build_evolution_prompt
from evohash.stage3_common.programs import resolve_seed_attacks


@dataclass(frozen=True)
class GigaEvolveProblemConfig:
    hash_id: str
    output_dir: str | Path = "outputs/gigaevolve_problems"
    problem_name: str | None = None
    seed_config_path: str | Path = "results/attack_configs_optimized.json"
    seed_attacks: tuple[str, ...] = ("all",)
    inline_seed_code: bool = True
    """Use the actual attack module source as seeds instead of registry wrappers."""


def generate_problem_package(config: GigaEvolveProblemConfig) -> Path:
    cfg = load_attack_config(config.seed_config_path)
    name = config.problem_name or f"evohash_{config.hash_id}"
    problem_dir = Path(config.output_dir) / name
    initial_dir = problem_dir / "initial_programs"
    initial_dir.mkdir(parents=True, exist_ok=True)

    attacks = resolve_seed_attacks(
        hash_id=config.hash_id,
        attack_ids=config.seed_attacks,
        config=cfg,
        seed_config_path=config.seed_config_path,
    )
    for attack_id in attacks:
        params = clean_public_params(dict(cfg.get("configs", {}).get(config.hash_id, {}).get(attack_id, {})))
        (initial_dir / f"{attack_id}_seed.py").write_text(
            _seed_program_code(
                attack_id=attack_id,
                params=params,
                inline_seed_code=config.inline_seed_code,
            ),
            encoding="utf-8",
        )

    (problem_dir / "context.py").write_text(_context_code(), encoding="utf-8")
    (problem_dir / "validate.py").write_text(_validate_code(), encoding="utf-8")
    (problem_dir / "metrics.yaml").write_text(_metrics_yaml(), encoding="utf-8")
    write_gigaevolve_templates(problem_dir / "prompts")
    (problem_dir / "task_description.txt").write_text(
        _task_description(config.hash_id, attacks, cfg),
        encoding="utf-8",
    )
    (problem_dir / "evohash_problem.json").write_text(
        json.dumps(
            {
                "hash_id": config.hash_id,
                "seed_config_path": str(config.seed_config_path),
                "seed_attacks": list(attacks),
                "inline_seed_code": bool(config.inline_seed_code),
                "prompts_dir": str(problem_dir / "prompts"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return problem_dir


def _context_code() -> str:
    return '''"""GigaEvolve runtime context for EvoHash."""

from evohash.stage3_gigaevolve.runtime import build_context_from_env


def build_context() -> dict:
    return build_context_from_env()
'''


def _validate_code() -> str:
    return '''"""GigaEvolve validator for EvoHash candidates."""

from typing import Any

from evohash.stage3_gigaevolve.runtime import validate_payload_with_feedback


def validate(context: dict, data: Any):
    return validate_payload_with_feedback(data, context)
'''


def _seed_program_code(
    *,
    attack_id: str,
    params: dict[str, Any],
    inline_seed_code: bool = True,
) -> str:
    if inline_seed_code:
        source = build_inlined_attack_source(
            attack_id=attack_id,
            params=params,
            use_evolve_block=False,
            include_compat_run=False,
        )
        return source.rstrip() + '''


from evohash.stage3_gigaevolve.runtime import evaluate_entrypoint_attack


def entrypoint(context: dict) -> dict:
    return evaluate_entrypoint_attack(
        run_attack,
        context,
        params=DEFAULT_PARAMS,
        attack_id=ATTACK_ID,
    )
'''

    params_literal = repr(params)
    return f'''"""Seed wrapper for EvoHash attack {attack_id}."""

from __future__ import annotations

from typing import Any

from evohash.attacks import build_attack_registry
from evohash.attacks.base import AttackRawResult
from evohash.stage3_gigaevolve.runtime import evaluate_entrypoint_attack


ATTACK_ID = {attack_id!r}
PARAMS = {params_literal}
DEFAULT_PARAMS = dict(PARAMS)


def run_attack(
    x_source,
    oracle,
    params: dict[str, Any] | None = None,
    budget: int | None = 10000,
    seed: int = 0,
) -> AttackRawResult:
    attack = build_attack_registry().get(ATTACK_ID)
    cfg = {{**PARAMS, **(params or {{}})}}
    return attack.run_attack(
        x_source=x_source,
        oracle=oracle,
        params=cfg,
        budget=budget,
        seed=seed,
    )


def entrypoint(context: dict) -> dict:
    return evaluate_entrypoint_attack(
        run_attack,
        context,
        params=PARAMS,
        attack_id=ATTACK_ID,
    )
'''


def _metrics_yaml() -> str:
    return """specs:
  score:
    description: "Stage2 optimizer objective: (mean_l2_raw + alpha * mean_queries) / max(asr, asr_floor) + failure_gap_penalty * mean_gap_to_threshold. Lower is better."
    decimals: 6
    is_primary: true
    higher_is_better: false
    lower_bound: 0.0
    upper_bound: 100000000.0
    include_in_prompts: true
    significant_change: 1.0
    sentinel_value: 1000000000.0
  is_valid:
    description: "1.0 if all image-pair evaluations ran without errors, else below 1.0."
    decimals: 0
    is_primary: false
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
    include_in_prompts: true
    significant_change: 1.0
    sentinel_value: 0.0
  target_metric:
    description: "mean_l2_raw + alpha * mean_queries. Lower is better."
    decimals: 6
    is_primary: false
    higher_is_better: false
    lower_bound: 0.0
    upper_bound: 100000000.0
    include_in_prompts: true
    significant_change: 1.0
    sentinel_value: 1000000000.0
  asr:
    description: "Attack success rate over evaluated image pairs."
    decimals: 4
    is_primary: false
    higher_is_better: true
    lower_bound: 0.0
    upper_bound: 1.0
    include_in_prompts: true
    significant_change: 0.01
    sentinel_value: 0.0
  mean_l2_raw:
    description: "Mean raw L2 distortion. Lower is better."
    decimals: 4
    is_primary: false
    higher_is_better: false
    lower_bound: 0.0
    upper_bound: 1000000.0
    include_in_prompts: true
    significant_change: 0.5
    sentinel_value: 1000000000.0
  mean_queries:
    description: "Mean number of hash queries per image."
    decimals: 1
    is_primary: false
    higher_is_better: false
    lower_bound: 0.0
    upper_bound: 100000.0
    include_in_prompts: true
    significant_change: 10.0
    sentinel_value: 100000.0
  mean_final_hash_l1:
    description: "Mean final hash distance to the target. Lower is better."
    decimals: 3
    is_primary: false
    higher_is_better: false
    lower_bound: 0.0
    upper_bound: 1000000.0
    include_in_prompts: true
    significant_change: 1.0
    sentinel_value: 1000000000.0
  mean_gap_to_threshold:
    description: "Mean positive distance above collision threshold. Lower is better."
    decimals: 3
    is_primary: false
    higher_is_better: false
    lower_bound: 0.0
    upper_bound: 1000000.0
    include_in_prompts: true
    significant_change: 1.0
    sentinel_value: 1000000000.0
"""


def _task_description(hash_id: str, attacks: tuple[str, ...], cfg: dict[str, Any]) -> str:
    configs = cfg.get("configs", {}).get(hash_id, {})
    prompt = build_evolution_prompt(
        hash_id,
        contract_lines=[
            "Programs must define entrypoint(context).",
            "Keep the helper pattern: define run_attack(...) and let entrypoint call evaluate_entrypoint_attack(run_attack, context, ...).",
            "Mutate the visible attack implementation, including helper functions/classes when useful.",
            "The primary metric is score and lower is better.",
            "Do not fabricate rows or metrics; validate.py summarizes evaluator rows and invalid outputs receive a bad score.",
        ],
    )
    lines = [
        prompt.strip(),
        "",
        "Allowed seed attacks and optimized configs:",
    ]
    for attack_id in attacks:
        params = clean_public_params(dict(configs.get(attack_id, {})))
        lines.append(f"- {attack_id}: {json.dumps(params, sort_keys=True)}")
    return "\n".join(lines).strip() + "\n"
