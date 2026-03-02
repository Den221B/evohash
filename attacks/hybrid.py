"""Hybrid two-stage attacks: score-based phase + HSJA decision-based refinement.

Stage 1 (score-based): NES / SimBa / ZO — continuous hash distance.
    Использует первые stage1_time_fraction времени.
    Останавливается при успехе или по времени.

Stage 2 (HSJA): decision-based refinement.
    Стартует из x_mid (лучшая точка stage 1).
    y_init передаётся в HSJA для гарантированного начала если stage 1 не нашёл коллизию.

Key fixes vs original:
  - stage split теперь time-based (_BudgetProxy по времени), не query-based.
    Оригинал делил max_queries, что бессмысленно при time-only бюджете.
  - y_init передаётся в HSJA — без него HSJA почти всегда fails на random парах.
  - L2: RMS sqrt(mean(...)), согласованно с utils.l2_img.
  - Убран _l2 с sqrt(sum(...)), везде используется RMS.
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
    """RMS L2 — согласованно с utils.l2_img."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


# ---------------------------------------------------------------------------
# Hybrid factory
# ---------------------------------------------------------------------------

def _make_hybrid(stage1_id: str, stage1_attack, stage2_attack: HSJAAttack):

    @dataclass
    class _HybridAttack:
        stage1: object = field(default_factory=lambda: stage1_attack)
        stage2: HSJAAttack = field(default_factory=lambda: stage2_attack)
        stage1_time_fraction: float = 0.5
        y_init: Optional[np.ndarray] = None   # устанавливается из evaluator
        spec: AttackSpec = field(init=False)

        def __post_init__(self) -> None:
            self.spec = AttackSpec(
                attack_id=f"{stage1_id}+hsja",
                params=dict(stage1=stage1_id, stage1_time_fraction=self.stage1_time_fraction),
            )

        def run(self, x0: np.ndarray, oracle: HashOracle, budget: BudgetSpec) -> AttackResult:
            t0 = time.monotonic()
            history: List[float] = []
            thr = oracle.threshold

            # Stage 1: ограничиваем время через временную подмену oracle.budget.
            # oracle.budget_ok() сравнивает elapsed с budget.max_time_s.
            # started_at — время создания oracle. Если подставим stage1_time,
            # stage 1 остановится когда oracle проживёт stage1_time секунд.
            total_time = budget.max_time_s
            if total_time is not None:
                stage1_budget = BudgetSpec(
                    max_time_s=total_time * self.stage1_time_fraction,
                    max_queries=budget.max_queries,
                    seed=budget.seed,
                )
                orig_budget = oracle.budget
                oracle.budget = stage1_budget
                r1 = self.stage1.run(x0, oracle, stage1_budget)
                oracle.budget = orig_budget  # восстанавливаем для stage 2
            else:
                r1 = self.stage1.run(x0, oracle, budget)
            history.extend(r1.history)

            x_mid = r1.x_best
            best_score = r1.best_score

            if best_score <= thr:
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

            # Stage 2: HSJA стартует от оригинального x0.
            # Stage 1 не нашёл коллизию (иначе вернули бы выше),
            # поэтому берём y_init как гарантированную начальную коллизию.
            self.stage2.y_init = self.y_init
            r2 = self.stage2.run(x0, oracle, budget)  # x0 — оригинальный source
            history.extend(r2.history)

            best_x = r2.x_best if r2.best_score < best_score else x_mid
            best_score = min(r2.best_score, best_score)

            reason = "success" if best_score <= thr else "budget"
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


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------

def build_nes_hsja(
    nes_kwargs: Optional[dict] = None,
    hsja_kwargs: Optional[dict] = None,
    stage1_time_fraction: float = 0.6,
):
    """NES → HSJA."""
    nes  = NESAttack(**(nes_kwargs or {}))
    hsja = HSJAAttack(**(hsja_kwargs or {}))
    h = _make_hybrid("nes", nes, hsja)
    h.stage1_time_fraction = stage1_time_fraction
    return h


def build_simba_hsja(
    simba_kwargs: Optional[dict] = None,
    hsja_kwargs: Optional[dict] = None,
    stage1_time_fraction: float = 0.6,
):
    """SimBa → HSJA."""
    simba = SimBaAttack(**(simba_kwargs or {}))
    hsja  = HSJAAttack(**(hsja_kwargs or {}))
    h = _make_hybrid("simba", simba, hsja)
    h.stage1_time_fraction = stage1_time_fraction
    return h


def build_zo_hsja(
    zo_kwargs: Optional[dict] = None,
    hsja_kwargs: Optional[dict] = None,
    stage1_time_fraction: float = 0.6,
):
    """ZO-Sign-SGD → HSJA."""
    zo   = ZOSignSGDAttack(**(zo_kwargs or {}))
    hsja = HSJAAttack(**(hsja_kwargs or {}))
    h = _make_hybrid("zo_sign_sgd", zo, hsja)
    h.stage1_time_fraction = stage1_time_fraction
    return h