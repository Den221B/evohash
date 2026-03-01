"""Attack implementations and registry.

Available attacks
-----------------
Baseline:
    RandomSearchAttack   — random Gaussian perturbations (sanity check)

Score-based (continuous hash distance oracle):
    NESAttack            — Natural Evolution Strategies with antithetic sampling
    ProkosAttack         — NES + momentum + grayscale (USENIX Sec 2023)
    SimBaAttack          — Simple Black-box Attack (pixel or DCT basis)
    ZOSignSGDAttack      — Zeroth-Order Sign SGD

Multiresolution:
    AtkScopesAttack      — DCT-domain coordinate descent (USENIX Sec 2025)

Decision-based (binary collision feedback only):
    HSJAAttack           — HopSkipJumpAttack adapted for hash collision

Hybrid (score-based stage 1 + HSJA stage 2):
    build_nes_hsja()
    build_simba_hsja()
    build_zo_hsja()
"""
from .base import AttackSpec, AttackResult, Attack, AttackRegistry
from .nes import NESAttack, ProkosAttack
from .simba import SimBaAttack
from .zo_sign_sgd import ZOSignSGDAttack
from .atkscopes import AtkScopesAttack
from .hsja import HSJAAttack
from .hybrid import build_nes_hsja, build_simba_hsja, build_zo_hsja

# keep random search available for sanity checks / notebooks
try:
    from .random_search import RandomSearchAttack
    _has_random = True
except ImportError:
    _has_random = False

__all__ = [
    # types
    "AttackSpec",
    "AttackResult",
    "Attack",
    "AttackRegistry",
    # score-based
    "NESAttack",
    "ProkosAttack",
    "SimBaAttack",
    "ZOSignSGDAttack",
    # multiresolution
    "AtkScopesAttack",
    # decision-based
    "HSJAAttack",
    # hybrid builders
    "build_nes_hsja",
    "build_simba_hsja",
    "build_zo_hsja",
    # registry factory
    "build_default_registry",
]

if _has_random:
    __all__.append("RandomSearchAttack")


def build_default_registry() -> AttackRegistry:
    """Build a registry with all baseline attacks available out of the box.

    Sensible defaults are chosen; hyperparameters can be tuned per-experiment.
    """
    reg = AttackRegistry()

    if _has_random:
        reg.register(RandomSearchAttack(sigma=0.05, iters=200))

    reg.register(NESAttack(
        sigma=1.0, lr=0.5, n_samples=10, max_iters=3000, grayscale=False
    ))
    reg.register(ProkosAttack(
        sigma=1.0, lr=0.5, n_samples=10, max_iters=3000, rho=0.5
    ))
    reg.register(SimBaAttack(
        epsilon=0.05, max_iters=3000, basis="dct"
    ))
    reg.register(ZOSignSGDAttack(
        mu=0.005, lr=0.003, n_samples=20, max_iters=3000
    ))
    # Atkscopes registered 3×: one per hash family scale
    reg.register(AtkScopesAttack(scale="global", lr=1.0, a=1.0, max_iters=2000),
                 overwrite=False)   # attack_id = "atkscopes" (global by default)
    reg.register(HSJAAttack(
        max_iters=40, init_queries=100, grad_queries=100, step_size=0.02
    ))
    reg.register(build_nes_hsja(stage1_query_fraction=0.6))
    reg.register(build_simba_hsja(stage1_query_fraction=0.6))
    reg.register(build_zo_hsja(stage1_query_fraction=0.6))

    return reg
