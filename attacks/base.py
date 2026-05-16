"""Minimal functional attack contract.

Attack modules should expose:

    run_attack(x_source, oracle, params=None, budget=10_000, seed=0) -> AttackRawResult

Optionally they may also expose the student-repo compatible:

    run(context, **kwargs) -> dict

No attack classes are required. The registry below stores functions, not attack
implementations/classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np


@dataclass
class AttackRawResult:
    """Raw result returned by an attack.

    This is intentionally not the final evaluation result. The evaluator must
    recompute final_hash_l1, success, pixel_l2, etc. from x_best.
    """

    x_best: np.ndarray
    history: list[float] = field(default_factory=list)
    queries: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


RunAttackFn = Callable[..., AttackRawResult]


@dataclass(frozen=True)
class RegisteredAttack:
    attack_id: str
    run_attack: RunAttackFn
    default_params: dict[str, Any] = field(default_factory=dict)


class AttackRegistry:
    """Tiny registry for attack functions."""

    def __init__(self) -> None:
        self._reg: dict[str, RegisteredAttack] = {}

    def register(
        self,
        attack_id: str,
        run_attack: RunAttackFn,
        default_params: Optional[dict[str, Any]] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        if attack_id in self._reg and not overwrite:
            raise KeyError(f"Attack {attack_id!r} already registered")
        self._reg[attack_id] = RegisteredAttack(
            attack_id=attack_id,
            run_attack=run_attack,
            default_params=dict(default_params or {}),
        )

    def get(self, attack_id: str) -> RegisteredAttack:
        if attack_id not in self._reg:
            raise KeyError(f"Unknown attack {attack_id!r}. Available: {self.list_ids()}")
        return self._reg[attack_id]

    def list_ids(self) -> list[str]:
        return sorted(self._reg)

    def items(self) -> Iterable[tuple[str, RegisteredAttack]]:
        return self._reg.items()


def resolve_max_iters(
    params: dict[str, Any],
    *,
    budget: int | Any | None,
    queries_per_iter: float,
    default_unlimited: int = 100_000,
) -> int:
    """Resolve the technical iteration cap from budget unless explicit."""
    if "max_iters" in params:
        return int(params["max_iters"])
    if "n_iter" in params:
        return int(params["n_iter"])

    max_queries = getattr(budget, "max_queries", budget)
    if max_queries is None:
        return int(default_unlimited)

    qpi = max(float(queries_per_iter), 1.0)
    return max(1, int(math.ceil(float(max_queries) / qpi)) + 2)
