"""Attack implementations and registry."""
from .base import AttackSpec, AttackResult, Attack, AttackRegistry
from .random_search import RandomSearchAttack

__all__ = [
    "AttackSpec",
    "AttackResult",
    "Attack",
    "AttackRegistry",
    "RandomSearchAttack",
]


def build_default_registry() -> AttackRegistry:
    """Build a registry with the baseline attacks available out of the box."""
    reg = AttackRegistry()
    reg.register(RandomSearchAttack(sigma=0.05, iters=200))
    return reg
