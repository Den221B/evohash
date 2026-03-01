"""Hybrid two-stage attacks: score-based phase + HSJA decision-based refinement.

Stage 1 (score-based): Run NES/SimBa/ZO-Sign-SGD using continuous hash
    distance signal.  Stop early once within `switch_threshold` of the
    collision threshold (or after `stage1_queries` queries).

Stage 2 (HSJA): Starting from the best point found in stage 1, refine
    using binary decision feedback only.  This boundary-walking phase
    minimises L2 distortion while maintaining the collision.

Hybrids evaluated in:
    Madden et al., "Assessing the adversarial security of perceptual
    hashing algorithms", arXiv 2406.00918, 2024.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from .base import AttackResult, AttackSpec
from .nes import NESAttack, ProkosAttack
from .simba import SimBaAttack
from .zo_sign_sgd import ZOSignSGDAttack
from .hsja import HSJAAttack


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _wrap_hybrid(stage1_id: str, stage1_instance, stage2_instance: HSJAAttack):
    """Factory that produces a Hybrid attack dataclass combining two stages."""

    @dataclass
    class _HybridAttack:
        stage1: type = field(default_factory=lambda: stage1_instance)
        stage2: HSJAAttack = field(default_factory=lambda: stage2_instance)
        stage1_query_fraction: float = 0.5   # fraction of budget for stage 1
        spec: AttackSpec = field(init=False)

        def __post_init__(self) -> None:
            self.spec = AttackSpec(
                attack_id=f"{stage1_id}+hsja",
                params=dict(
                    stage1=stage1_id,
                    stage1_query_fraction=self.stage1_query_fraction,
                ),
            )

        def run(
            self,
            x0: np.ndarray,
            oracle: HashOracle,
            budget: BudgetSpec,
        ) -> AttackResult:
            t0 = time.monotonic()
            history: List[float] = []
            threshold = getattr(oracle, "threshold", 0.0)

            # --- Stage 1 ---
            max_q = getattr(budget, "max_queries", int(1e9))
            stage1_q = int(max_q * self.stage1_query_fraction)

            # temporarily monkey-patch budget for stage 1
            stage1_budget = _BudgetProxy(budget, stage1_q)
            r1 = self.stage1.run(x0, oracle, stage1_budget)
            history.extend(r1.history)

            x_mid = r1.x_best
            best_score = r1.best_score

            if best_score <= threshold:
                # stage 1 already succeeded
                elapsed = int((time.monotonic() - t0) * 1000)
                return AttackResult(
                    x_best=x_mid,
                    best_score=best_score,
                    queries_used=oracle.queries_used,
                    runtime_ms=elapsed,
                    stopped_reason="success_stage1",
                    history=history,
                    extra={"l2": _l2(x0, x_mid), "stage": 1},
                )

            # --- Stage 2: HSJA starting from x_mid ---
            r2 = self.stage2.run(x_mid, oracle, budget)
            history.extend(r2.history)

            best_x = r2.x_best if r2.best_score < best_score else x_mid
            best_score = min(r2.best_score, best_score)

            reason = "success" if best_score <= threshold else "budget"
            elapsed = int((time.monotonic() - t0) * 1000)
            return AttackResult(
                x_best=best_x,
                best_score=best_score,
                queries_used=oracle.queries_used,
                runtime_ms=elapsed,
                stopped_reason=reason,
                history=history,
                extra={"l2": _l2(x0, best_x), "stage": 2},
            )

    return _HybridAttack()


class _BudgetProxy:
    """Thin wrapper that forwards attribute access but caps max_queries."""
    def __init__(self, budget, cap: int):
        self._budget = budget
        self.max_queries = cap

    def __getattr__(self, name):
        return getattr(self._budget, name)


# ---------------------------------------------------------------------------
# Public hybrid constructors
# ---------------------------------------------------------------------------

def build_nes_hsja(
    nes_kwargs: Optional[dict] = None,
    hsja_kwargs: Optional[dict] = None,
    stage1_query_fraction: float = 0.5,
):
    """NES (score-based) → HSJA (decision-based) hybrid."""
    nes = NESAttack(**(nes_kwargs or {}))
    hsja = HSJAAttack(**(hsja_kwargs or {}))
    hybrid = _wrap_hybrid("nes", nes, hsja)
    hybrid.stage1_query_fraction = stage1_query_fraction
    return hybrid


def build_simba_hsja(
    simba_kwargs: Optional[dict] = None,
    hsja_kwargs: Optional[dict] = None,
    stage1_query_fraction: float = 0.5,
):
    """SimBa (score-based) → HSJA (decision-based) hybrid."""
    simba = SimBaAttack(**(simba_kwargs or {}))
    hsja = HSJAAttack(**(hsja_kwargs or {}))
    hybrid = _wrap_hybrid("simba", simba, hsja)
    hybrid.stage1_query_fraction = stage1_query_fraction
    return hybrid


def build_zo_hsja(
    zo_kwargs: Optional[dict] = None,
    hsja_kwargs: Optional[dict] = None,
    stage1_query_fraction: float = 0.5,
):
    """ZO-Sign-SGD (score-based) → HSJA (decision-based) hybrid."""
    zo = ZOSignSGDAttack(**(zo_kwargs or {}))
    hsja = HSJAAttack(**(hsja_kwargs or {}))
    hybrid = _wrap_hybrid("zo_sign_sgd", zo, hsja)
    hybrid.stage1_query_fraction = stage1_query_fraction
    return hybrid
