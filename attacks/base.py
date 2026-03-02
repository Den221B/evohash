from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class AttackSpec:
    """Metadata about an attack algorithm."""
    attack_id: str
    name: str
    description: str = ""


@dataclass
class AttackResult:
    """Result of running an attack on a single (image, hash) instance."""
    success: bool
    best_dist: float
    queries_used: int

    # Best candidate found (optional, depending on attack implementation)
    best_x: Optional[Any] = None

    # Per-iteration history; attack may store whatever it wants here
    history: Dict[str, Any] = field(default_factory=dict)

    # Additional debug info / config dump / timings, etc.
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Attack(Protocol):
    """Attack interface."""
    spec: AttackSpec

    def run(self, x0, oracle, budget) -> AttackResult:
        ...


class AttackRegistry:
    """Simple registry of attack instances."""
    def __init__(self) -> None:
        self._reg: Dict[str, Attack] = {}

    def register(self, attack: Attack) -> None:
        if not hasattr(attack, "spec") or not getattr(attack, "spec", None):
            raise TypeError("Attack must have a .spec field (AttackSpec).")
        aid = attack.spec.attack_id
        if aid in self._reg:
            raise KeyError(f"Attack '{aid}' already registered.")
        self._reg[aid] = attack

    def get(self, attack_id: str) -> Attack:
        if attack_id not in self._reg:
            raise KeyError(f"Unknown attack '{attack_id}'. Available: {list(self._reg.keys())}")
        return self._reg[attack_id]

    def list_ids(self) -> list[str]:
        return list(self._reg.keys())

    def items(self):
        return self._reg.items()