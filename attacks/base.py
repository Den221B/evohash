"""Base types for attack implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle


@dataclass(frozen=True)
class AttackSpec:
    attack_id: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    x_best: np.ndarray          # best adversarial candidate found
    best_score: float           # hash distance at x_best
    queries_used: int
    runtime_ms: int
    stopped_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Attack(Protocol):
    spec: AttackSpec

    def run(
        self,
        x0: np.ndarray,
        oracle: HashOracle,
        budget: BudgetSpec,
    ) -> AttackResult:
        """Run the attack.

        Args:
            x0:     Source image, float32 [0,1] HxWxC.
            oracle: Black-box query interface.
            budget: Budget constraints (already embedded in oracle, passed
                    here for convenience / seed access).

        Returns:
            AttackResult with the best adversarial candidate found.
        """
        ...


class AttackRegistry:
    def __init__(self) -> None:
        self._reg: Dict[str, Attack] = {}

    def register(self, attack: Attack, *, overwrite: bool = False) -> None:
        aid = attack.spec.attack_id
        if aid in self._reg and not overwrite:
            raise ValueError(
                f"Attack '{aid}' already registered. Use overwrite=True to replace."
            )
        self._reg[aid] = attack

    def get(self, attack_id: str) -> Attack:
        if attack_id not in self._reg:
            raise KeyError(
                f"Unknown attack_id '{attack_id}'. "
                f"Registered: {self.list_ids()}"
            )
        return self._reg[attack_id]

    def list_ids(self) -> List[str]:
        return sorted(self._reg.keys())

    def __contains__(self, attack_id: str) -> bool:
        return attack_id in self._reg
