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
    x_best: np.ndarray          # лучший adversarial кандидат
    best_score: float           # hash distance (меньше = лучше)
    queries_used: int
    runtime_ms: int
    stopped_reason: Optional[str] = None   # "success" | "budget" | "no_init" | "error"
    history: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def runtime_s(self) -> float:
        return self.runtime_ms / 1000.0

    @property
    def l2(self) -> float:
        return float(self.extra.get("l2", 0.0))

    @property
    def queries_per_sec(self) -> float:
        if self.runtime_s > 0:
            return self.queries_used / self.runtime_s
        return 0.0


@runtime_checkable
class Attack(Protocol):
    spec: AttackSpec

    def run(
        self,
        x0: np.ndarray,
        oracle: HashOracle,
        budget: BudgetSpec,
    ) -> AttackResult:
        """Запустить атаку.

        Args:
            x0:     Исходное изображение, float32 [0,1] HxWxC.
            oracle: Black-box интерфейс. oracle.query(x) → дистанция до target hash.
            budget: Бюджет (используется для seed и передачи в sub-атаки).

        Returns:
            AttackResult с лучшим найденным кандидатом.
        """
        ...


class AttackRegistry:
    def __init__(self) -> None:
        self._reg: Dict[str, Any] = {}

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