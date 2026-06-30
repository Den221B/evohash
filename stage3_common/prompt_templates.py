"""Project-owned prompt templates for stage3 evolution adapters."""

from __future__ import annotations

from pathlib import Path


OPENEVOLVE_DIFF_USER_TEMPLATE = """# Parent Candidate

## Parent Metrics
{metrics}

## Evaluation Feedback
{artifacts}

## Evolution Context
- Fitness: {fitness_score}
- MAP-Elites dimensions: {feature_dimensions}
- Current MAP-Elites coordinates, if assigned: {feature_coords}
- Focus areas:
{improvement_areas}

## Related Programs
{evolution_history}

## Current Program
```{language}
{current_program}
```

# Mutation Task
Improve this attack while preserving the evaluator contract. Use the parent
metrics, feedback artifact, and visible source code to decide what to change.
Prefer one narrow, diagnosable mechanism change unless the feedback points to a
structural bottleneck.

Return SEARCH/REPLACE blocks only, using this exact format:

<<<<<<< SEARCH
exact code from the current program
=======
replacement code
>>>>>>> REPLACE

Rules:
- Each SEARCH block must match the current program exactly.
- You may change helper functions/classes as well as run_attack.
- Keep run_attack(x_source, oracle, params=None, budget=10000, seed=0).
- Keep deterministic seed handling and respect the query budget.
"""


GIGAEVOLVE_MUTATION_SYSTEM_TEMPLATE = """{task_description}

MUTATION ROLE
You mutate executable Python attack programs inside an evolutionary optimizer.
Use the shared objective order from the task description: threshold success
first, then raw L2, then query cost.

AVAILABLE METRICS
{metrics_description}

OUTPUT CONTRACT
The framework enforces structured fields:
- archetype: short strategy label.
- justification: 2-3 sentences tied to metrics, feedback, or parent code.
- insights_used: copied strings from available feedback when useful; may be empty on cold start.
- changes: concise reusable descriptions of the important edits.
- code: complete valid Python source only.

Mutation discipline:
- Preserve entrypoint(context) and run_attack(...).
- Mutate visible attack logic, helpers, and state classes when useful.
- Use parent metrics, validator artifacts, insights, and lineage when present.
- Fix invalid code, timeouts, or budget leaks before optimizing quality.
- Do not fabricate metrics, rows, images, hashes, or evaluation results.
- Prefer one clear mechanism change per mutation unless feedback demands a rewrite.
"""


GIGAEVOLVE_MUTATION_USER_TEMPLATE = """# Parent Candidates

{parent_blocks}

# Mutation Task
Create one improved complete Python program. Use the parent code and mutation
context above. Preserve the EvoHash adapter contract and keep the result
directly executable.

Focus order:
1. If the parent is invalid, timed out, or exceeded budget, repair that first.
2. Otherwise improve threshold success progress.
3. Then reduce raw L2 distortion.
4. Then reduce query cost.
"""


def write_openevolve_templates(template_dir: str | Path) -> None:
    root = Path(template_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "diff_user.txt").write_text(OPENEVOLVE_DIFF_USER_TEMPLATE, encoding="utf-8")


def write_gigaevolve_templates(template_dir: str | Path) -> None:
    root = Path(template_dir) / "mutation"
    root.mkdir(parents=True, exist_ok=True)
    (root / "system.txt").write_text(GIGAEVOLVE_MUTATION_SYSTEM_TEMPLATE, encoding="utf-8")
    (root / "user.txt").write_text(GIGAEVOLVE_MUTATION_USER_TEMPLATE, encoding="utf-8")
