"""Attack registry for the functional attack contract."""
from __future__ import annotations

from evohash.attacks.base import AttackRawResult, AttackRegistry, RegisteredAttack


def build_attack_registry() -> AttackRegistry:
    reg = AttackRegistry()

    from evohash.attacks import nes, prokos, nes_attack_v0, simba, zo_signsgd, atkscopes

    for mod in [nes, prokos, nes_attack_v0, simba, zo_signsgd, atkscopes]:
        reg.register(
            getattr(mod, "ATTACK_ID"),
            mod.run_attack,
            getattr(mod, "DEFAULT_PARAMS", {}),
            overwrite=True,
        )

    return reg


__all__ = [
    "AttackRawResult",
    "AttackRegistry",
    "RegisteredAttack",
    "build_attack_registry",
]
