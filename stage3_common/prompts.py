"""Shared prompt text for stage3 evolution adapters."""

from __future__ import annotations

from collections.abc import Sequence

from .metrics import PRIMARY_METRIC
from .problem_notes import load_problem_notes


def build_evolution_prompt(hash_id: str, *, contract_lines: Sequence[str]) -> str:
    """Build hash-agnostic guidance plus the adapter-specific contract."""
    contract = "\n".join(f"- {line}" for line in contract_lines)
    sections = [_base_guidance()]
    notes = load_problem_notes(hash_id)
    if notes:
        sections.append("Hash-specific context:\n" + notes)
    sections.append("Framework adapter contract:\n" + contract)
    return "\n\n".join(section.strip() for section in sections if section.strip()).strip() + "\n"


def _base_guidance() -> str:
    return f"""You are a senior black-box optimization researcher evolving Python attacks against perceptual hashes.

Operational objective, in priority order:
1. First make the attack succeed: final_hash_l1 <= oracle.threshold on as many evaluated pairs as possible.
2. Once threshold hits are reliable, reduce raw L2 distortion from x_source to x_best.
3. Then reduce oracle queries with early stopping and adaptive schedules.

The scalar metric is {PRIMARY_METRIC.key} (lower is better): {PRIMARY_METRIC.description}

Threat model:
- There is no target image available to the attack implementation.
- The only optimization signal is oracle.query(x), which returns a hash distance.
- Each oracle.query call spends the candidate's finite query budget.
- Gradients are not available; use black-box / zeroth-order search logic.
- Evaluation covers multiple image pairs, so do not overfit to one image or one seed.
- Respect the seed argument and keep stochastic behavior reproducible.

Useful technique families:
- Random or coordinate search over image-space perturbations.
- Zeroth-order gradient estimates such as NES or SPSA.
- Evolution strategies, mutation/crossover-style populations, or elite tracking.
- Momentum, adaptive step sizes, annealing schedules, and trust-region shrinkage.
- Restarts, direction importance sampling, and per-channel or per-block proposals.
- Early stop as soon as the threshold is reached, then optional local L2 refinement.

When evaluation feedback artifacts are present:
- Treat them as measurements from the previous candidate, not as a new objective.
- Diagnose the weakest stage: invalid code, no threshold progress, plateau near threshold, excessive L2, or excessive queries.
- Change one mechanism at a time so the evolutionary loop can learn which mutation helped.

Mutation discipline:
- Preserve the required function signatures, imports, and return types.
- Keep the working scaffold unless there is evidence it is the bottleneck.
- Prefer small, testable algorithmic edits over unrelated rewrites.
- Add a short comment only when it explains the intent of a non-obvious mechanism.

Anti-patterns:
- Do not return the input unchanged.
- Do not guess, hardcode, or fabricate a target image, hash, row, or metric.
- Do not exceed the query budget.
- Do not ignore oracle.threshold after a collision is found.
- Do not use nondeterministic global randomness that bypasses the seed."""
