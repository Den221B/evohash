# """Attack implementations and registry.

# Available attacks
# -----------------
# Baseline:
#     RandomSearchAttack   — random Gaussian perturbations (sanity check)

# Score-based (continuous hash distance oracle):
#     NESAttack            — Natural Evolution Strategies with antithetic sampling
#     ProkosAttack         — NES + momentum + grayscale (USENIX Sec 2023)
#     SimBaAttack          — Simple Black-box Attack (pixel or DCT basis)
#     ZOSignSGDAttack      — Zeroth-Order Sign SGD

# Multiresolution:
#     AtkScopesAttack      — DCT-domain coordinate descent (USENIX Sec 2025)

# Decision-based (binary collision feedback only):
#     HSJAAttack           — HopSkipJumpAttack adapted for hash collision

# Hybrid (score-based stage 1 + HSJA stage 2):
#     build_nes_hsja()
#     build_simba_hsja()
#     build_zo_hsja()
# """
# from .base import AttackSpec, AttackResult, Attack, AttackRegistry
# from .nes import NESAttack, ProkosAttack
# from .simba import SimBaAttack
# from .zo_sign_sgd import ZOSignSGDAttack
# from .atkscopes import AtkScopesAttack
# from .hsja import HSJAAttack
# from .hybrid import build_nes_hsja, build_simba_hsja, build_zo_hsja

# # keep random search available for sanity checks / notebooks
# try:
#     from .random_search import RandomSearchAttack
#     _has_random = True
# except ImportError:
#     _has_random = False

# __all__ = [
#     # types
#     "AttackSpec",
#     "AttackResult",
#     "Attack",
#     "AttackRegistry",
#     # score-based
#     "NESAttack",
#     "ProkosAttack",
#     "SimBaAttack",
#     "ZOSignSGDAttack",
#     # multiresolution
#     "AtkScopesAttack",
#     # decision-based
#     "HSJAAttack",
#     # hybrid builders
#     "build_nes_hsja",
#     "build_simba_hsja",
#     "build_zo_hsja",
#     # registry factory
#     "build_default_registry",
# ]

# if _has_random:
#     __all__.append("RandomSearchAttack")


# def build_default_registry() -> AttackRegistry:
#     """Build a registry with all baseline attacks available out of the box.

#     Sensible defaults are chosen; hyperparameters can be tuned per-experiment.
#     """
#     reg = AttackRegistry()

#     if _has_random:
#         reg.register(RandomSearchAttack(sigma=0.05, iters=200))

#     reg.register(NESAttack(
#         sigma=1.0, lr=0.5, n_samples=10, max_iters=3000, grayscale=False
#     ))
#     reg.register(ProkosAttack(
#         sigma=1.0, lr=0.5, n_samples=10, max_iters=3000, rho=0.5
#     ))
#     reg.register(SimBaAttack(
#         epsilon=0.05, max_iters=3000, basis="dct"
#     ))
#     reg.register(ZOSignSGDAttack(
#         mu=0.005, lr=0.003, n_samples=20, max_iters=3000
#     ))
#     # Atkscopes registered 3×: one per hash family scale
#     reg.register(AtkScopesAttack(scale="global", lr=1.0, a=1.0, max_iters=2000),
#                  overwrite=False)   # attack_id = "atkscopes" (global by default)
#     reg.register(HSJAAttack(
#         max_iters=40, init_queries=100, grad_queries=100, step_size=0.02
#     ))
#     reg.register(build_nes_hsja(stage1_query_fraction=0.6))
#     reg.register(build_simba_hsja(stage1_query_fraction=0.6))
#     reg.register(build_zo_hsja(stage1_query_fraction=0.6))

#     return reg
# attacks/__init__.py
"""Attack implementations and registry.

We aim for:
- reproducibility (budget.seed -> deterministic RNG inside attacks)
- comparability (same budget constraints: max_queries / max_time)
- consistent outputs (AttackResult: success, best_dist, queries_used, best_x, history, extra)
- time-aware history (elapsed_ms per iter + runtime_ms total)
"""

from __future__ import annotations

from .base import AttackSpec, AttackResult, Attack, AttackRegistry

from .nes import NESAttack, ProkosAttack
from .simba import SimBAAttack
from .zo_sign_sgd import ZOSignSGDAttack
from .atkscopes import AtkScopesAttack
from .hsja import HSJAAttack
from .hybrid import HybridAttack

def _with_id(attack: Attack, attack_id: str, name: str | None = None) -> Attack:
    """Return the same attack instance but with a new spec id (and optional name)."""
    spec = getattr(attack, "spec", None)
    if spec is None:
        raise TypeError("Attack has no .spec")
    attack.spec = AttackSpec(
        attack_id=attack_id,
        name=name or spec.name,
        description=spec.description,
    )
    return attack


__all__ = [
    # types
    "AttackSpec",
    "AttackResult",
    "Attack",
    "AttackRegistry",
    # attacks
    "NESAttack",
    "ProkosAttack",
    "SimBAAttack",
    "ZOSignSGDAttack",
    "AtkScopesAttack",
    "HSJAAttack",
    "HybridAttack",
    # registry factory
    "build_default_registry",
]

def build_default_registry() -> AttackRegistry:
    """Build a registry with baseline attacks.

    Defaults are chosen to be reasonable for images in [0,1].
    Tune per-hash if needed (hash distance quantization differs a lot).
    """
    reg = AttackRegistry()

    # ----------------------------
    # NES / Prokos
    # ----------------------------
    reg.register(_with_id(NESAttack(
        sigma=0.05,
        lr=0.1,
        n_samples=20,
        antithetic=True,
        grayscale=False,
        normalize_grad=True,
    ), "nes"))

    reg.register(_with_id(ProkosAttack(
        sigma=0.05,
        lr=0.02,
        n_samples=20,
        rho=0.5,
        double_sample=False,
        grayscale=False,
        normalize_grad=True,
    ), "prokos"))

    # ----------------------------
    # SimBA (two variants)
    # ----------------------------
    reg.register(_with_id(SimBAAttack(
        epsilon=0.2,
        basis="dct",
        freq_dims=28,
        order="strided",
        stride=7,
        grayscale=False,
        normalize_direction=True,
    ), "simba_dct", "SimBA-DCT"))

    reg.register(_with_id(SimBAAttack(
        epsilon=0.2,
        basis="pixel",
        freq_dims=0,     # unused
        order="rand",
        stride=1,
        grayscale=False,
        normalize_direction=True,
    ), "simba_pixel", "SimBA-Pixel"))

    # ----------------------------
    # ZO-signSGD (two estimators)
    # ----------------------------
    reg.register(_with_id(ZOSignSGDAttack(
        mu=0.10,
        lr=0.01,
        n_samples=20,
        estimator="forward",
        momentum=0.0,
        grayscale=False,
    ), "zo_sign_sgd_fwd", "ZO-signSGD (forward)"))

    reg.register(_with_id(ZOSignSGDAttack(
        mu=0.10,
        lr=0.01,
        n_samples=20,
        estimator="central",  # 2x queries per direction, but often more stable
        momentum=0.0,
        grayscale=False,
    ), "zo_sign_sgd_central", "ZO-signSGD (central)"))

    # ----------------------------
    # ATKSCOPES (three scales; MUST have unique IDs)
    # ----------------------------
    reg.register(_with_id(AtkScopesAttack(
        scale="global",
        lr=0.05,
        a=0.10,
        patch_size=None,
        beta1=0.9, beta2=0.999, eps=1e-8,
    ), "atkscopes_global", "ATKSCOPES (global)"))

    reg.register(_with_id(AtkScopesAttack(
        scale="mid",
        lr=0.05,
        a=0.10,
        patch_size=None,  # defaults to H//4
        beta1=0.9, beta2=0.999, eps=1e-8,
    ), "atkscopes_mid", "ATKSCOPES (mid)"))

    reg.register(_with_id(AtkScopesAttack(
        scale="pixel",
        lr=0.05,
        a=0.10,
        patch_size=None,
        beta1=0.9, beta2=0.999, eps=1e-8,
    ), "atkscopes_pixel", "ATKSCOPES (pixel)"))

    # ----------------------------
    # HSJA
    # ----------------------------
    reg.register(_with_id(HSJAAttack(
        init_max_tries=200,
        init_sigmas=(0.05, 0.1, 0.2, 0.4, 0.8),
        bin_search_steps=12,
        grad_queries=40,
        probe_delta_ratio=0.01,
        step_ratio=0.15,
        grayscale=False,
    ), "hsja"))

    # ----------------------------
    # Hybrid example: SimBA-DCT -> HSJA
    # (time split handled inside HybridAttack via stage1_time_frac)
    # ----------------------------
    reg.register(_with_id(HybridAttack(
        stage1=SimBAAttack(
            epsilon=0.2,
            basis="dct",
            freq_dims=28,
            order="strided",
            stride=7,
            grayscale=False,
            normalize_direction=True,
        ),
        stage2=HSJAAttack(
            init_max_tries=200,
            init_sigmas=(0.05, 0.1, 0.2, 0.4, 0.8),
            bin_search_steps=12,
            grad_queries=40,
            probe_delta_ratio=0.01,
            step_ratio=0.15,
            grayscale=False,
        ),
        stage1_time_frac=0.35,
        stage1_max_queries=None,
    ), "hybrid_simba_hsja", "Hybrid (SimBA→HSJA)"))

    return reg