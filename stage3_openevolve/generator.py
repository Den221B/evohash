"""Generate an OpenEvolve problem package backed by EvoHash evaluation.

Layout (mirrors what ``openevolve-run.py`` expects)::

    <output_dir>/<problem_name>/
        initial_program.py        ← seed wrapper with EVOLVE-BLOCK markers
        evaluator.py              ← thin shim that calls stage3_openevolve.runtime
        config.yaml               ← LLM + iteration + prompt settings
        task_description.txt      ← problem notes (shared with GigaEvolve)
        evohash_problem.json      ← bookkeeping for the notebook launcher
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evohash.optimizer import clean_public_params, load_attack_config
from evohash.stage3_common.attack_source import build_inlined_attack_source
from evohash.stage3_common.config import EvaluationConfig, LLMConfig
from evohash.stage3_common.prompt_templates import write_openevolve_templates
from evohash.stage3_common.prompts import build_evolution_prompt
from evohash.stage3_common.programs import resolve_seed_attacks


@dataclass(frozen=True)
class OpenEvolveProblemConfig:
    """Inputs for the OpenEvolve problem generator.

    The shape mirrors ``GigaEvolveProblemConfig`` so notebooks can switch
    backends with minimal code changes.  Every field maps to a key in the
    generated ``config.yaml``; defaults match OpenEvolve's own defaults so
    behaviour is reproducible without surprises.

    For a fair benchmark vs GigaEvolve set ``cascade_evaluation=False`` —
    GigaEvolve does full evaluation for every candidate, OpenEvolve by
    default short-circuits on a cheap first stage.
    """

    hash_id: str
    output_dir: str | Path = "outputs/openevolve_problems"
    problem_name: str | None = None
    seed_config_path: str | Path = "results/attack_configs_optimized.json"
    seed_attacks: tuple[str, ...] = ("all",)

    # === Run length ===
    iterations: int = 50
    checkpoint_interval: int = 10
    random_seed: int = 42

    # === Population structure (MAP-Elites + islands) ===
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5
    feature_dimensions: tuple[str, ...] = ("asr", "mean_l2_raw", "mean_queries")
    feature_bins: int | dict[str, int] = field(
        default_factory=lambda: {"asr": 4, "mean_l2_raw": 8, "mean_queries": 6}
    )
    """Bins per MAP-Elites axis. An int applies to every dimension; a dict gives
    per-dimension resolution (e.g. {"asr": 4, "mean_l2_raw": 8, "mean_queries": 6})
    — useful because asr has few meaningful levels while l2/queries are continuous.
    Dimensions absent from the dict fall back to OpenEvolve's default (10)."""
    diversity_reference_size: int = 20

    # === Migration ===
    migration_interval: int = 50
    migration_rate: float = 0.1

    # === Parent selection ratios (must sum to 1.0) ===
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7

    # === Prompt context ===
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    use_custom_prompt_templates: bool = True
    prompt_template_dirname: str = "prompt_templates"
    suggest_simplification_after_chars: int | None = 30_000

    # === Evaluator ===
    cascade_evaluation: bool = False
    """Disabled by default for fair comparison with GigaEvolve, which always
    runs full evaluation.  Set True for a faster (but unequal) run."""

    cascade_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9)
    parallel_evaluations: int = 4
    evaluator_timeout_s: int = 300
    evaluator_max_retries: int = 3

    # === Seed sharing ===
    use_evolve_block: bool = True
    """Wrap the mutable region in ``EVOLVE-BLOCK-*`` markers (recommended)."""

    inline_seed_code: bool = True
    """Use the actual attack module source as the seed program instead of a
    registry wrapper. This lets the LLM see and mutate the real attack logic."""

    max_code_length: int = 30_000
    """OpenEvolve rejects children above this many characters. Inline attacks,
    especially ATKScopes, need a higher limit than the upstream default."""

    prepopulate_database: bool = True
    """OpenEvolve takes a single ``initial_program.py``.  When True, all *other*
    resolved seed attacks are evaluated (via ``validate_program``) and inserted
    into a saved ``ProgramDatabase`` *before* evolution starts — the same diverse
    pool GigaEvolve loads from ``initial_programs/`` on day 0.  Crucially this
    keeps the per-mutation ``system_message`` clean: seeds live in the population,
    not in every prompt.  Requires ``openevolve`` to be installed at run time."""

    seed_db_dirname: str = "seeded_db"
    """Sub-directory of the problem package where the pre-populated database is
    saved; wired into ``database.db_path`` so the controller auto-loads it."""

    extra_config: dict[str, Any] = field(default_factory=dict)
    """Free-form YAML fragments merged into the generated config.yaml."""

    def selection_ratios_sum(self) -> float:
        return float(self.elite_selection_ratio + self.exploration_ratio + self.exploitation_ratio)


def generate_openevolve_problem(
    config: OpenEvolveProblemConfig,
    llm: LLMConfig,
    evaluation: EvaluationConfig | None = None,
) -> Path:
    """Materialize an OpenEvolve problem package on disk.

    ``evaluation`` is optional; if provided it is recorded as bookkeeping so
    the notebook launcher can rebuild the env vars later.
    """
    cfg = load_attack_config(config.seed_config_path)
    name = config.problem_name or f"evohash_{config.hash_id}"
    problem_dir = Path(config.output_dir) / name
    problem_dir.mkdir(parents=True, exist_ok=True)

    attacks = resolve_seed_attacks(
        hash_id=config.hash_id,
        attack_ids=config.seed_attacks,
        config=cfg,
        seed_config_path=config.seed_config_path,
    )
    if not attacks:
        raise ValueError(
            f"No seed attacks resolved for hash_id={config.hash_id!r}; "
            "check seed_config_path."
        )

    primary_attack = attacks[0]
    primary_params = clean_public_params(
        dict(cfg.get("configs", {}).get(config.hash_id, {}).get(primary_attack, {}))
    )

    # The "extra" seeds are everything besides primary_attack.  Instead of
    # stuffing them into every prompt, we write them to ``seeds/`` so the
    # launcher can evaluate them and pre-populate the ProgramDatabase — they
    # then live in the population (like GigaEvolve's initial_programs/) and the
    # system_message stays clean across all mutations.
    extra_seed_ids: list[str] = list(attacks[1:])
    seed_db_path = (problem_dir / config.seed_db_dirname).resolve()
    if config.prepopulate_database and extra_seed_ids:
        seeds_dir = problem_dir / "seeds"
        seeds_dir.mkdir(exist_ok=True)
        for attack_id in extra_seed_ids:
            params = clean_public_params(
                dict(cfg.get("configs", {}).get(config.hash_id, {}).get(attack_id, {}))
            )
            (seeds_dir / f"{attack_id}.py").write_text(
                _initial_program_code(
                    attack_id=attack_id,
                    params=params,
                    use_evolve_block=False,  # seeds are scored as-is, not mutated here
                    inline_seed_code=config.inline_seed_code,
                ),
                encoding="utf-8",
            )

    (problem_dir / "initial_program.py").write_text(
        _initial_program_code(
            attack_id=primary_attack,
            params=primary_params,
            use_evolve_block=config.use_evolve_block,
            inline_seed_code=config.inline_seed_code,
        ),
        encoding="utf-8",
    )
    if config.use_custom_prompt_templates:
        write_openevolve_templates(problem_dir / config.prompt_template_dirname)
    (problem_dir / "evaluator.py").write_text(_evaluator_code(), encoding="utf-8")
    (problem_dir / "config.yaml").write_text(
        _config_yaml(
            config=config,
            llm=llm,
            db_path=str(seed_db_path) if (config.prepopulate_database and extra_seed_ids) else None,
        ),
        encoding="utf-8",
    )
    (problem_dir / "task_description.txt").write_text(
        _task_description(config.hash_id, attacks, cfg), encoding="utf-8"
    )

    bookkeeping = {
        "framework": "openevolve",
        "hash_id": config.hash_id,
        "seed_config_path": str(config.seed_config_path),
        "seed_attacks": list(attacks),
        "primary_attack": primary_attack,
        "inline_seed_code": bool(config.inline_seed_code),
        "use_custom_prompt_templates": bool(config.use_custom_prompt_templates),
        "prepopulate_database": bool(config.prepopulate_database and extra_seed_ids),
        "seed_db_path": str(seed_db_path) if (config.prepopulate_database and extra_seed_ids) else None,
        "db_seed_attacks": extra_seed_ids if config.prepopulate_database else [],
        "iterations": int(config.iterations),
    }
    if evaluation is not None:
        bookkeeping["evaluation"] = {
            "hash_id": evaluation.hash_id,
            "dataset_zip": str(evaluation.dataset_zip),
            "split": evaluation.split,
            "max_pairs": evaluation.max_pairs,
            "budget": int(evaluation.budget),
            "resize_size": evaluation.resize_size,
        }
    (problem_dir / "evohash_problem.json").write_text(
        json.dumps(bookkeeping, indent=2, sort_keys=True), encoding="utf-8"
    )
    return problem_dir


# ---------------------------------------------------------------------------
# File-content builders
# ---------------------------------------------------------------------------

def _initial_program_code(
    *,
    attack_id: str,
    params: dict[str, Any],
    use_evolve_block: bool,
    inline_seed_code: bool = True,
) -> str:
    """Build an OpenEvolve seed file.

    By default this inlines the actual attack implementation so the LLM sees
    mutable search logic, helper functions, and state classes. The registry
    wrapper path remains available as a fallback.
    """
    if inline_seed_code:
        return build_inlined_attack_source(
            attack_id=attack_id,
            params=params,
            use_evolve_block=use_evolve_block,
            include_compat_run=False,
        )

    params_literal = repr(params)
    block_start = "# EVOLVE-BLOCK-START\n" if use_evolve_block else ""
    block_end = "# EVOLVE-BLOCK-END\n" if use_evolve_block else ""

    return f'''"""OpenEvolve seed program for EvoHash attack {attack_id}.

Mutate ONLY the run_attack body.  Keep the function signature and the import
of ``AttackRawResult`` unchanged so the evaluator can score the result.
"""

from __future__ import annotations

from typing import Any

from evohash.attacks import build_attack_registry
from evohash.attacks.base import AttackRawResult


ATTACK_ID = {attack_id!r}
PARAMS = {params_literal}
DEFAULT_PARAMS = dict(PARAMS)


{block_start}def run_attack(
    x_source,
    oracle,
    params: dict[str, Any] | None = None,
    budget: int | None = 10000,
    seed: int = 0,
) -> AttackRawResult:
    """Initial implementation: delegate to the optimized registry attack."""
    attack = build_attack_registry().get(ATTACK_ID)
    cfg = {{**PARAMS, **(params or {{}})}}
    return attack.run_attack(
        x_source=x_source,
        oracle=oracle,
        params=cfg,
        budget=budget,
        seed=seed,
    )
{block_end}'''


def _evaluator_code() -> str:
    """Build the thin evaluator.py that OpenEvolve loads with importlib.

    All the heavy lifting (dataset, oracle, scoring) lives in stage3_common.
    """
    return '''"""OpenEvolve evaluator for EvoHash. Delegates to stage3_openevolve.runtime."""

from evohash.stage3_openevolve.runtime import evaluate as _evaluate


def evaluate(program_path):
    """Required entry point for OpenEvolve."""
    return _evaluate(program_path)
'''


def _config_yaml(
    *,
    config: OpenEvolveProblemConfig,
    llm: LLMConfig,
    db_path: str | None = None,
) -> str:
    """Build a config.yaml that exposes every knob the user can tune.

    All scalars that live on ``OpenEvolveProblemConfig`` map to one YAML key
    so the file is self-documenting.  ``config.extra_config`` is appended
    verbatim at the bottom — that's the escape hatch for keys we don't
    surface explicitly yet.

    ``db_path`` (when given) points at the pre-populated seed database; the
    OpenEvolve controller auto-loads it on startup so the seed pool is in the
    population from iteration 0.
    """
    api_base, api_key_hint = _llm_endpoint(llm)
    system_message = _system_prompt(config.hash_id)

    ratios_sum = config.selection_ratios_sum()
    ratios_comment = ""
    if abs(ratios_sum - 1.0) > 1e-6:
        ratios_comment = (
            f"  # WARNING: elite+exploration+exploitation = {ratios_sum:.3f} (expected 1.0)"
        )

    lines: list[str] = []
    lines.append("# Auto-generated by evohash.stage3_openevolve.generator")
    lines.append(f"max_iterations: {int(config.iterations)}")
    lines.append(f"checkpoint_interval: {int(config.checkpoint_interval)}")
    lines.append(f"random_seed: {int(config.random_seed)}")
    lines.append(f"max_code_length: {int(config.max_code_length)}")
    lines.append("")
    lines.append("llm:")
    lines.append(f"  api_base: {_yaml_scalar(api_base)}")
    # OpenEvolve's LLMConfig has NO ``model`` field; it uses ``primary_model``
    # (a back-compat shim whose __post_init__ builds the ``models`` list).
    # Emitting ``model:`` would be silently dropped by dacite, leaving
    # ``models=[]`` → run_evolution raises "No LLM models configured".
    lines.append(f"  primary_model: {_yaml_scalar(llm.model)}")
    lines.append(f"  temperature: {float(llm.temperature)}")
    lines.append(f"  max_tokens: {int(llm.max_tokens)}")
    lines.append(f"  timeout: {int(llm.timeout_s)}")
    if api_key_hint:
        # ``${VAR}`` is resolved from the environment at config-load time
        # (openevolve.config._resolve_env_var); the secret never lands in YAML.
        lines.append(f"  api_key: {_yaml_scalar('${' + api_key_hint + '}')}")
    lines.append("")
    lines.append("database:")
    if db_path:
        # Pre-populated seed pool; controller auto-loads it on startup.
        # ``_yaml_scalar`` (json.dumps) escapes Windows backslashes so the path
        # survives YAML double-quoted parsing.
        lines.append(f"  db_path: {_yaml_scalar(db_path)}")
    lines.append(f"  population_size: {int(config.population_size)}")
    lines.append(f"  archive_size: {int(config.archive_size)}")
    lines.append(f"  num_islands: {int(config.num_islands)}")
    lines.append(f"  migration_interval: {int(config.migration_interval)}")
    lines.append(f"  migration_rate: {float(config.migration_rate)}")
    lines.append(f"  elite_selection_ratio: {float(config.elite_selection_ratio)}")
    lines.append(f"  exploration_ratio: {float(config.exploration_ratio)}")
    lines.append(f"  exploitation_ratio: {float(config.exploitation_ratio)}")
    if ratios_comment:
        lines.append(ratios_comment)
    lines.append("  feature_dimensions:")
    for dim in config.feature_dimensions:
        lines.append(f'    - "{dim}"')
    if isinstance(config.feature_bins, dict):
        lines.append("  feature_bins:")
        for dim, bins in config.feature_bins.items():
            lines.append(f"    {dim}: {int(bins)}")
    else:
        lines.append(f"  feature_bins: {int(config.feature_bins)}")
    lines.append(f"  diversity_reference_size: {int(config.diversity_reference_size)}")
    lines.append("")
    lines.append("evaluator:")
    lines.append(f"  timeout: {int(config.evaluator_timeout_s)}")
    lines.append(f"  max_retries: {int(config.evaluator_max_retries)}")
    lines.append(f"  cascade_evaluation: {str(bool(config.cascade_evaluation)).lower()}")
    lines.append("  cascade_thresholds:")
    for threshold in config.cascade_thresholds:
        lines.append(f"    - {float(threshold)}")
    lines.append(f"  parallel_evaluations: {int(config.parallel_evaluations)}")
    lines.append("")
    lines.append("prompt:")
    if config.use_custom_prompt_templates:
        lines.append(f"  template_dir: {_yaml_scalar(config.prompt_template_dirname)}")
    lines.append("  system_message: |")
    for line in system_message.splitlines() or [""]:
        lines.append(f"    {line}")
    lines.append("  include_artifacts: true")
    lines.append("  max_artifact_bytes: 8192")
    if config.suggest_simplification_after_chars is None:
        lines.append("  suggest_simplification_after_chars: null")
    else:
        lines.append(
            f"  suggest_simplification_after_chars: {int(config.suggest_simplification_after_chars)}"
        )
    lines.append(f"  num_top_programs: {int(config.num_top_programs)}")
    lines.append(f"  num_diverse_programs: {int(config.num_diverse_programs)}")

    if config.extra_config:
        import yaml  # local import: yaml is an openevolve dep, may be absent otherwise

        lines.append("")
        lines.append("# --- extra_config (verbatim, overrides anything above) ---")
        lines.append(yaml.safe_dump(config.extra_config, sort_keys=True).rstrip())

    return "\n".join(lines) + "\n"


def _yaml_scalar(value: str) -> str:
    """Emit a string as a YAML-safe double-quoted scalar.

    ``json.dumps`` produces a valid YAML scalar (YAML is a JSON superset) and,
    crucially, escapes backslashes — so Windows paths like
    ``C:\\Users\\...\\seeded_db`` don't trip YAML's double-quote escape rules
    (``\\U`` etc.).  Leaves ``${VAR}`` references intact for env resolution.
    """
    return json.dumps(str(value))


def _llm_endpoint(llm: LLMConfig) -> tuple[str, str | None]:
    if llm.source == "openrouter":
        return ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    if llm.source == "local":
        return (llm.base_url or "http://localhost:8000/v1", None)
    return (llm.base_url or "https://api.openai.com/v1", "OPENAI_API_KEY")


def _system_prompt(hash_id: str) -> str:
    """Same prompt content as GigaEvolve — keeps the two frameworks comparable.

    Intentionally clean: the diverse seed pool is pre-loaded into the
    ProgramDatabase (see ``prepopulate_database``), not injected into every
    mutation prompt.  OpenEvolve already surfaces top/diverse programs from the
    population as in-context examples, so the LLM still sees the seeds.
    """
    return build_evolution_prompt(
        hash_id,
        contract_lines=[
            "Mutate the visible attack implementation while preserving run_attack(x_source, oracle, params=None, budget=10000, seed=0).",
            "Return evohash.attacks.base.AttackRawResult; the evaluator recomputes all metrics.",
            "Use oracle.query(x) for hash distance and oracle.threshold for the success threshold.",
            "Helper functions and state classes may be mutated when they support run_attack.",
            "Keep EVOLVE-BLOCK markers and the AttackRawResult import intact when present.",
        ],
    )


def _task_description(hash_id: str, attacks: tuple[str, ...], cfg: dict[str, Any]) -> str:
    """Write the shared prompt plus seed metadata for notebook inspection."""
    configs = cfg.get("configs", {}).get(hash_id, {})
    lines = [
        _system_prompt(hash_id).strip(),
        "",
        "Allowed seed attacks and optimized configs:",
    ]
    for attack_id in attacks:
        params = clean_public_params(dict(configs.get(attack_id, {})))
        lines.append(f"- {attack_id}: {json.dumps(params, sort_keys=True)}")
    return "\n".join(lines).strip() + "\n"
