"""Render prompt previews for the stage3 evolution adapters.

The files produced here are inspection artifacts, not training/evaluation data.
They use the real adapter prompt builders and inlined seed code, while marking
runtime-selected values such as parent metrics/history as preview-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evohash.optimizer import clean_public_params, load_attack_config
from evohash.stage3_common.config import LLMConfig
from evohash.stage3_common.prompt_templates import (
    GIGAEVOLVE_MUTATION_SYSTEM_TEMPLATE,
    GIGAEVOLVE_MUTATION_USER_TEMPLATE,
    OPENEVOLVE_DIFF_USER_TEMPLATE,
)
from evohash.stage3_common.programs import resolve_seed_attacks
from evohash.stage3_gigaevolve.generator import (
    _metrics_yaml as _giga_metrics_yaml,
    _seed_program_code as _giga_seed_program_code,
    _task_description as _giga_task_description,
)
from evohash.stage3_openevolve.generator import (
    OpenEvolveProblemConfig,
    _config_yaml as _openevolve_config_yaml,
    _initial_program_code as _openevolve_initial_program_code,
    _system_prompt as _openevolve_system_prompt,
)


DEFAULT_HASH_IDS = ("phash", "pdq", "neuralhash", "photodna")


@dataclass(frozen=True)
class PromptPreviewResult:
    hash_id: str
    attack_id: str
    files: list[Path]


def write_prompt_previews(
    *,
    hash_ids: tuple[str, ...] = DEFAULT_HASH_IDS,
    output_dir: str | Path = "outputs/prompt_previews",
    seed_config_path: str | Path = "results/attack_configs_optimized.json",
    seed_attacks: tuple[str, ...] = ("all",),
) -> list[PromptPreviewResult]:
    """Write prompt preview files for OpenEvolve and GigaEvolve."""
    cfg = load_attack_config(seed_config_path)
    out_root = _project_path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[PromptPreviewResult] = []
    summary_lines = [
        "# EvoHash Prompt Previews",
        "",
        "These files show what the model prompt looks like for a first-mutation preview.",
        "Runtime metrics, history, and artifact contents are selected during a live run,",
        "so preview-only values are explicitly marked.",
        "",
    ]

    for hash_id in hash_ids:
        attacks = resolve_seed_attacks(
            hash_id=hash_id,
            attack_ids=seed_attacks,
            config=cfg,
            seed_config_path=seed_config_path,
        )
        if not attacks:
            summary_lines.append(f"- {hash_id}: skipped, no seed attacks resolved.")
            continue

        attack_id = attacks[0]
        params = clean_public_params(
            dict(cfg.get("configs", {}).get(hash_id, {}).get(attack_id, {}))
        )
        hash_dir = out_root / hash_id
        hash_dir.mkdir(parents=True, exist_ok=True)

        open_file = hash_dir / "openevolve_first_mutation_prompt.txt"
        giga_file = hash_dir / "gigaevolve_first_mutation_prompt.txt"
        manifest_file = hash_dir / "manifest.json"

        open_file.write_text(
            render_openevolve_prompt_preview(hash_id, attack_id, params),
            encoding="utf-8",
        )
        giga_file.write_text(
            render_gigaevolve_prompt_preview(hash_id, attacks, cfg, attack_id, params),
            encoding="utf-8",
        )
        manifest = {
            "hash_id": hash_id,
            "primary_seed_attack": attack_id,
            "seed_attacks": list(attacks),
            "files": [open_file.name, giga_file.name],
            "prompt_layers": {
                "system": [
                    "shared black-box objective, threat model, technique families, mutation discipline",
                    "hash-specific context from stage3_common/problem_notes",
                    "framework adapter contract",
                ],
                "user": [
                    "parent metrics and evaluation feedback artifact",
                    "visible inlined parent attack source code",
                    "framework-specific mutation/output task",
                ],
            },
            "important_note": (
                "The parent code shown to the model is the inlined attack module "
                "source with optimized DEFAULT_PARAMS, so the LLM sees the actual "
                "search logic rather than a build_attack_registry() wrapper."
            ),
        }
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        results.append(
            PromptPreviewResult(
                hash_id=hash_id,
                attack_id=attack_id,
                files=[open_file, giga_file, manifest_file],
            )
        )
        summary_lines.append(
            f"- {hash_id}: primary seed `{attack_id}` -> "
            f"`{hash_id}/{open_file.name}`, `{hash_id}/{giga_file.name}`"
        )

    (out_root / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return results


def render_openevolve_prompt_preview(
    hash_id: str,
    attack_id: str,
    params: dict[str, Any],
) -> str:
    """Render an OpenEvolve first-mutation prompt with real parent code."""
    parent_code = _openevolve_initial_program_code(
        attack_id=attack_id,
        params=params,
        use_evolve_block=True,
        inline_seed_code=True,
    )
    system_prompt = _openevolve_system_prompt(hash_id)
    config_yaml = _openevolve_config_yaml(
        config=OpenEvolveProblemConfig(hash_id=hash_id),
        llm=LLMConfig(source="local", model="preview-model"),
    )
    _ = config_yaml  # rendered by generate_openevolve_problem; not part of LLM messages

    try:
        from openevolve.config import PromptConfig
        from openevolve.prompt.sampler import PromptSampler

        prompt_config = PromptConfig(
            system_message=system_prompt,
            num_top_programs=3,
            num_diverse_programs=2,
            include_artifacts=True,
            max_artifact_bytes=8192,
            suggest_simplification_after_chars=30_000,
            use_template_stochasticity=False,
        )
        sampler = PromptSampler(prompt_config)
        sampler.template_manager.add_template("diff_user", OPENEVOLVE_DIFF_USER_TEMPLATE)
        prompt = sampler.build_prompt(
            current_program=parent_code,
            parent_program=parent_code,
            program_metrics=_preview_openevolve_metrics(),
            previous_programs=[],
            top_programs=[],
            inspirations=[],
            language="python",
            evolution_round=1,
            diff_based_evolution=True,
            program_artifacts={"evohash_feedback.txt": _preview_feedback_text()},
            feature_dimensions=["asr", "mean_l2_raw", "mean_queries"],
        )
        system = prompt["system"]
        user = prompt["user"]
    except Exception as exc:  # pragma: no cover - preview fallback
        system = system_prompt
        user = (
            "OpenEvolve PromptSampler could not be imported. "
            f"Fallback preview only. Error: {type(exc).__name__}: {exc}\n\n"
            "# Current Program\n```python\n"
            + parent_code
            + "\n```"
        )

    return _join_prompt_file(
        title="OpenEvolve first-mutation prompt preview",
        hash_id=hash_id,
        attack_id=attack_id,
        notes=[
            "OpenEvolve iteration.py/process_parallel.py pass current_program=parent.code.",
            "The project diff_user template includes that code under '## Current Program'.",
            "Metrics/artifacts below are preview placeholders; live values come from parent.metrics and database artifacts.",
            "For seed parents, the visible code is now the inlined attack implementation from attacks/*.py.",
        ],
        extra_sections={
            "PREVIEW COMPOSITION CHECK": _composition_check(system, user, parent_code),
            "LLM SYSTEM MESSAGE": system,
            "LLM USER MESSAGE": user,
        },
    )


def render_gigaevolve_prompt_preview(
    hash_id: str,
    attacks: tuple[str, ...],
    cfg: dict[str, Any],
    attack_id: str,
    params: dict[str, Any],
) -> str:
    """Render a GigaEvolve mutation prompt with real parent code."""
    parent_code = _giga_seed_program_code(
        attack_id=attack_id,
        params=params,
        inline_seed_code=True,
    )
    task_description = _giga_task_description(hash_id, attacks, cfg)
    system_template = GIGAEVOLVE_MUTATION_SYSTEM_TEMPLATE
    user_template = GIGAEVOLVE_MUTATION_USER_TEMPLATE
    metrics_description = _giga_metrics_description()
    system = system_template.format(
        task_description=task_description,
        metrics_description=metrics_description,
    )
    parent_context = _preview_giga_mutation_context()
    parent_blocks = f"""=== Parent 1 ===
```python
{parent_code}
```

{parent_context}
"""
    user = user_template.format(count=1, parent_blocks=parent_blocks)

    return _join_prompt_file(
        title="GigaEvolve first-mutation prompt preview",
        hash_id=hash_id,
        attack_id=attack_id,
        notes=[
            "GigaEvolve MutationAgent._build_parent_blocks inserts p.code before formatted mutation context.",
            "The project mutation/user.txt template includes {parent_blocks} under '# Parent Candidates'.",
            "Metrics/insights/lineage/artifacts below are preview placeholders; live values are attached by MutationContextStage.",
            "For seed parents, the visible code is now the inlined attack implementation from attacks/*.py plus entrypoint().",
        ],
        extra_sections={
            "PREVIEW COMPOSITION CHECK": _composition_check(system, user, parent_code),
            "LLM SYSTEM MESSAGE": system,
            "LLM USER MESSAGE": user,
        },
    )


def _join_prompt_file(
    *,
    title: str,
    hash_id: str,
    attack_id: str,
    notes: list[str],
    extra_sections: dict[str, str],
) -> str:
    parts = [
        f"# {title}",
        "",
        f"hash_id: {hash_id}",
        f"primary_seed_attack: {attack_id}",
        "",
        "## Preview Metadata (not sent to model)",
    ]
    parts.extend(f"- {note}" for note in notes)
    for name, content in extra_sections.items():
        parts.extend(["", f"=== {name} ===", content.rstrip(), f"=== END {name} ==="])
    return "\n".join(parts).rstrip() + "\n"


def _excerpt_config(config_yaml: str) -> str:
    lines = config_yaml.splitlines()
    keep: list[str] = []
    in_prompt = False
    for line in lines:
        if line.startswith("prompt:"):
            in_prompt = True
        if in_prompt:
            keep.append(line)
    return "\n".join(keep)


def _composition_check(system: str, user: str, parent_code: str) -> str:
    checks = {
        "common_objective": "Operational objective, in priority order:" in system,
        "threat_model": "Threat model:" in system,
        "hash_context": "Hash-specific context:" in system,
        "framework_contract": "Framework adapter contract:" in system or "OUTPUT CONTRACT" in system,
        "inlined_parent_code": "Inlined EvoHash seed attack:" in user,
        "parent_code_exactly_present": parent_code.strip() in user,
        "no_registry_wrapper": "build_attack_registry" not in user,
    }
    return "\n".join(f"- {name}: {'ok' if ok else 'MISSING'}" for name, ok in checks.items())


def _preview_openevolve_metrics() -> dict[str, float]:
    return {
        "score": 1234.0,
        "combined_score": -1234.0,
        "asr": 0.0,
        "mean_l2_raw": 50.0,
        "mean_queries": 300.0,
        "mean_gap_to_threshold": 4.0,
        "is_valid": 1.0,
    }


def _preview_feedback_text() -> str:
    return (
        "PREVIEW ONLY: in a live run this artifact is produced by validate_program().\n"
        "It contains success/failure, progress-to-threshold, query budget use, "
        "runtime errors, plateau/convergence hints, and next mutation focus."
    )


def _preview_giga_mutation_context() -> str:
    return """## Mutation Context (preview)
Metrics:
- score : 1234.000000 (preview placeholder; lower is better)
- asr : 0.0000 (preview placeholder)
- mean_l2_raw : 50.0000 (preview placeholder)
- mean_queries : 300.0 (preview placeholder)

Formatted artifact:
PREVIEW ONLY: in a live run validate_payload_with_feedback() returns this text as
the validator artifact, and MutationContextStage attaches it to the parent."""


def _giga_metrics_description() -> str:
    try:
        import yaml
        from gigaevo.programs.metrics.context import MetricsContext
        from gigaevo.programs.metrics.formatter import MetricsFormatter

        data = yaml.safe_load(_giga_metrics_yaml())
        specs = data.get("specs", data)
        return MetricsFormatter(MetricsContext.from_dict(specs=specs)).format_metrics_description()
    except Exception:
        return _giga_metrics_yaml()


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def _parse_hash_ids(raw: str) -> tuple[str, ...]:
    raw = raw.strip()
    if raw.lower() == "all":
        return DEFAULT_HASH_IDS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render EvoHash evolution prompt previews.")
    parser.add_argument("--hash", dest="hash_ids", default="all", help="'all' or comma-separated hash ids")
    parser.add_argument("--output-dir", default="outputs/prompt_previews")
    parser.add_argument("--seed-config", default="results/attack_configs_optimized.json")
    parser.add_argument("--seed-attacks", default="all", help="'all' or comma-separated attack ids")
    args = parser.parse_args(argv)

    seed_attacks = tuple(item.strip() for item in args.seed_attacks.split(",") if item.strip())
    results = write_prompt_previews(
        hash_ids=_parse_hash_ids(args.hash_ids),
        output_dir=args.output_dir,
        seed_config_path=args.seed_config,
        seed_attacks=seed_attacks or ("all",),
    )
    for result in results:
        print(
            f"{result.hash_id}: {result.attack_id} -> "
            + ", ".join(str(path) for path in result.files)
        )


if __name__ == "__main__":
    main()
