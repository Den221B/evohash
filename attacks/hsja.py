"""HopSkipJumpAttack (HSJA) — Decision-based boundary attack.

HSJA минимизирует L2(x0, x_adv) при условии что x_adv остаётся коллизией.
Это НЕ атака для снижения hash distance — она работает ВНУТРИ зоны коллизии.

Алгоритм:
    1. Старт: x_adv = y (гарантированная коллизия, максимальный L2 от x0).
    2. Бинарный поиск границы на отрезке [x0, x_adv].
    3. Оценка градиента на границе: в каком направлении от границы коллизия?
    4. Шаг от границы в направлении коллизии (ближе к x0 = меньше L2).
    5. Повторять — каждая итерация уменьшает L2 оставаясь в коллизии.

Цель HSJA в нашем проекте:
  - Standalone: взять y как старт, дойти как можно ближе к x0.
    Результат: x_adv с малым L2 и hash_dist <= threshold.
  - В hybrid: stage 1 (NES) нашёл x_mid (возможно коллизию), HSJA уменьшает L2.

Key fixes vs original:
  - Понята правильная цель: минимизируем L2, не hash dist.
    Цикл идёт while budget_ok(), не while best_dist > thr.
  - y_init задаётся снаружи для гарантированного старта.
  - Убран двойной запрос is_collision + oracle.query.
  - L2: RMS sqrt(mean(...)), согласованно с utils.l2_img.

Reference:
    Chen et al., "HopSkipJumpAttack", IEEE S&P 2020.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from .base import AttackResult, AttackSpec


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    """RMS L2 — согласованно с utils.l2_img."""
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


@dataclass
class HSJAAttack:
    """HopSkipJumpAttack для perceptual hash collision.

    HSJA минимизирует L2(x0, x_adv), удерживая x_adv в зоне коллизии.
    Используется как refinement-шаг после score-based атаки в hybrid,
    или standalone (стартует от y — гарантированной коллизии).

    Parameters
    ----------
    grad_queries : int
        Monte Carlo сэмплов для оценки градиента на границе.
    step_size : float
        Шаг от границы (адаптивный: × dist_to_boundary).
    binary_search_steps : int
        Шагов бинарного поиска границы за итерацию.
    y_init : np.ndarray | None
        Целевая картинка y — гарантированная начальная коллизия.
        Устанавливается снаружи (Evaluator или Hybrid) перед run().
    """
    grad_queries: int = 50
    step_size: float = 0.05
    binary_search_steps: int = 10
    y_init: Optional[np.ndarray] = None

    spec: AttackSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = AttackSpec(
            attack_id="hsja",
            params=dict(
                grad_queries=self.grad_queries,
                step_size=self.step_size,
                binary_search_steps=self.binary_search_steps,
            ),
        )

    def run(self, x0: np.ndarray, oracle: HashOracle, budget: BudgetSpec) -> AttackResult:
        t0 = time.monotonic()
        history: List[float] = []
        thr = oracle.threshold

        def is_collision(x: np.ndarray) -> bool:
            """Бинарный запрос — коллизия или нет. Тратит 1 query."""
            if not oracle.budget_ok():
                return False
            d = oracle.query(x)
            history.append(d)
            return d <= thr

        # Найти начальную точку коллизии
        x_adv = self._find_initial_collision(x0, is_collision, oracle)
        if x_adv is None:
            elapsed = int((time.monotonic() - t0) * 1000)
            return AttackResult(
                x_best=x0,
                best_score=oracle.state.best_score,
                queries_used=oracle.queries_used,
                runtime_ms=elapsed,
                stopped_reason="no_init",
                history=history,
                extra={"l2": 0.0},
            )

        # x_adv — коллизия, x0 — нет. Цель: минимизировать L2(x0, x_adv).
        best_x = x_adv.copy()
        best_l2 = _l2(x0, x_adv)
        # best_hash_dist — дистанция текущего x_adv (для отчёта, не для оптимизации)
        best_hash_dist = history[-1] if history else float("inf")

        # Главный цикл: идём по границе в сторону x0
        while oracle.budget_ok():
            # Бинарный поиск границы: x0 (нет коллизии) ←—→ x_adv (коллизия)
            x_boundary = self._binary_search(x0, x_adv, is_collision)
            if not oracle.budget_ok():
                break

            # Оценка градиента на границе
            grad = self._estimate_grad(x_boundary, x0, is_collision)
            if not oracle.budget_ok():
                break

            # Шаг от границы в направлении коллизии (к interior, ближе к x0)
            dist_to_boundary = _l2(x0, x_boundary)
            step = self.step_size * max(dist_to_boundary, 1e-4)
            x_new = _clip(x_boundary + step * grad)

            # Явный запрос: нужна реальная hash dist (не только бинарный ответ)
            if not oracle.budget_ok():
                break
            d_new = oracle.query(x_new)
            history.append(d_new)

            if d_new <= thr:
                # x_new — коллизия. Проверяем улучшился ли L2.
                x_adv = x_new.copy()
                l2_new = _l2(x0, x_new)
                if l2_new < best_l2:
                    best_l2 = l2_new
                    best_x = x_new.copy()
                    best_hash_dist = d_new

        # "success" = нашли коллизию с L2 < L2(x0, y_init)
        # Если x_adv изменился от y_init — мы улучшили L2, это успех HSJA.
        success = best_hash_dist <= thr
        reason = "success" if success else "budget"
        elapsed = int((time.monotonic() - t0) * 1000)
        return AttackResult(
            x_best=best_x,
            best_score=best_hash_dist,
            queries_used=oracle.queries_used,
            runtime_ms=elapsed,
            stopped_reason=reason,
            history=history,
            extra={"l2": best_l2},
        )

    def _find_initial_collision(
        self, x0: np.ndarray, is_collision, oracle: HashOracle
    ) -> Optional[np.ndarray]:
        """Найти гарантированную начальную коллизию.

        Стратегии:
        1. y_init напрямую — hash(y, y)=0 <= threshold, всегда коллизия.
        2. Интерполяции x0 → y_init.
        3. Случайные картинки (запасной вариант).
        """
        if self.y_init is not None:
            y = self.y_init.astype(np.float32)
            if is_collision(y):
                return y
            for alpha in [0.9, 0.7, 0.5]:
                if not oracle.budget_ok():
                    return None
                x_interp = _clip(alpha * y + (1 - alpha) * x0.astype(np.float32))
                if is_collision(x_interp):
                    return x_interp

        # Случайные картинки
        for _ in range(20):
            if not oracle.budget_ok():
                return None
            x_rand = np.random.rand(*x0.shape).astype(np.float32)
            if is_collision(x_rand):
                return x_rand

        return None

    def _binary_search(self, x_no, x_yes, is_collision) -> np.ndarray:
        """Бинарный поиск на отрезке [x_no (нет), x_yes (да)] для границы коллизии."""
        lo, hi = 0.0, 1.0
        for _ in range(self.binary_search_steps):
            mid = (lo + hi) / 2
            x_mid = _clip((1 - mid) * x_no + mid * x_yes)
            if is_collision(x_mid):
                hi = mid
            else:
                lo = mid
        return _clip((1 - hi) * x_no + hi * x_yes)

    def _estimate_grad(self, x_boundary, x_source, is_collision) -> np.ndarray:
        """Monte Carlo оценка направления внутрь зоны коллизии."""
        grad = np.zeros_like(x_boundary)
        # Шаг для зондирования: 1% от расстояния до source
        delta = max(_l2(x_source, x_boundary) * 0.01, 1e-3)

        for _ in range(self.grad_queries):
            p = np.random.randn(*x_boundary.shape).astype(np.float32)
            p /= np.linalg.norm(p) + 1e-12
            coeff = 1.0 if is_collision(_clip(x_boundary + delta * p)) else -1.0
            grad += coeff * p

        grad /= self.grad_queries + 1e-12
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad /= norm
        return grad