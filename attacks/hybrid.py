# attacks/hybrid.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict, Optional

from .base import AttackSpec, AttackResult


def _get_threshold(oracle):
    return getattr(oracle, "threshold_p", getattr(oracle, "threshold", None))


def _budget_to_seconds(budget) -> Optional[float]:
    """
    Try to interpret budget time limit in seconds, supporting either max_time_s or max_time_ms.
    Returns None if not present.
    """
    if hasattr(budget, "max_time_s") and getattr(budget, "max_time_s") is not None:
        return float(getattr(budget, "max_time_s"))
    if hasattr(budget, "max_time_ms") and getattr(budget, "max_time_ms") is not None:
        return float(getattr(budget, "max_time_ms")) / 1000.0
    return None


def _make_stage_budget_like(budget, max_time_s: Optional[float] = None, max_queries: Optional[int] = None):
    """
    Make a budget-like object with same seed and compatible time fields.
    We use SimpleNamespace so oracle.budget_ok can access attributes.
    """
    seed = getattr(budget, "seed", 0)
    b = SimpleNamespace()
    b.seed = seed

    # queries
    if max_queries is None:
        b.max_queries = getattr(budget, "max_queries", None)
    else:
        b.max_queries = int(max_queries)

    # time: keep BOTH fields if original had them, for compatibility
    # oracle likely uses one of them.
    if hasattr(budget, "max_time_s"):
        b.max_time_s = max_time_s
    if hasattr(budget, "max_time_ms"):
        b.max_time_ms = None if max_time_s is None else int(max_time_s * 1000)

    # If original had neither, still set max_time_s for our oracle implementation
    if not hasattr(b, "max_time_s") and not hasattr(b, "max_time_ms"):
        b.max_time_s = max_time_s

    return b


@dataclass
class HybridAttack:
    """
    Two-stage hybrid:
      - stage1 runs with a fraction of the overall time (or same time if no time budget)
      - stage2 runs with the remaining time budget (best-effort)

    History keeps both sub-histories:
      history["stage1"] = ...
      history["stage2"] = ...
    """

    stage1: Any
    stage2: Any
    stage1_time_frac: float = 0.35  # fraction of time budget for stage1
    stage1_max_queries: Optional[int] = None  # optional override

    spec: AttackSpec = AttackSpec(
        attack_id="hybrid",
        name="Hybrid",
        description="Two-stage hybrid attack (stage1 then stage2/HSJA); deterministic via budget seed; logs time.",
    )

    def run(self, x0, oracle, budget) -> AttackResult:
        thr = _get_threshold(oracle)
        if thr is None:
            return AttackResult(
                success=False,
                best_dist=float(getattr(oracle.state, "best_dist", float("inf"))),
                queries_used=int(getattr(oracle, "queries_used", 0)),
                best_x=None,
                history={},
                extra={"error": "oracle has no threshold_p/threshold"},
            )

        t0 = perf_counter()
        orig_budget = getattr(oracle, "budget", None)

        history: Dict[str, Any] = {}
        extra: Dict[str, Any] = {
            "stage1_id": getattr(getattr(self.stage1, "spec", None), "attack_id", type(self.stage1).__name__),
            "stage2_id": getattr(getattr(self.stage2, "spec", None), "attack_id", type(self.stage2).__name__),
            "stage1_time_frac": float(self.stage1_time_frac),
        }

        # Compute time split if possible
        total_time_s = _budget_to_seconds(budget)
        stage1_time_s = None
        if total_time_s is not None:
            stage1_time_s = max(0.0, float(total_time_s) * float(self.stage1_time_frac))

        # Stage1 budget (optional)
        stage1_budget = _make_stage_budget_like(
            budget,
            max_time_s=stage1_time_s,
            max_queries=self.stage1_max_queries,
        )

        # Run stage1 with temporary budget
        try:
            oracle.budget = stage1_budget
            r1 = self.stage1.run(x0, oracle, stage1_budget)
        finally:
            oracle.budget = orig_budget

        history["stage1"] = getattr(r1, "history", {})
        extra["stage1_extra"] = getattr(r1, "extra", {})
        best_x = getattr(r1, "best_x", None)
        best_dist = float(getattr(r1, "best_dist", float("inf")))
        success = bool(getattr(r1, "success", False))

        # If stage1 already succeeded and we don't need refinement, still run stage2 only if budget allows
        # Stage2 budget: remaining time (best effort)
        if total_time_s is not None and stage1_time_s is not None:
            stage2_time_s = max(0.0, float(total_time_s) - float(stage1_time_s))
        else:
            stage2_time_s = total_time_s  # may be None

        stage2_budget = _make_stage_budget_like(
            budget,
            max_time_s=stage2_time_s,
            max_queries=getattr(budget, "max_queries", None),
        )

        # Run stage2 using remaining budget
        r2 = self.stage2.run(x0, oracle, stage2_budget)
        history["stage2"] = getattr(r2, "history", {})
        extra["stage2_extra"] = getattr(r2, "extra", {})

        # Select best between stage1 and stage2 based on distortion if available (HSJA gives best_rmse),
        # otherwise fall back to best_dist then keep stage2 if it improved dist.
        r2_best_x = getattr(r2, "best_x", None)
        r2_best_dist = float(getattr(r2, "best_dist", float("inf")))
        r2_success = bool(getattr(r2, "success", False))

        # Prefer any successful result over non-success
        if (not success) and r2_success:
            best_x, best_dist, success = r2_best_x, r2_best_dist, True
        elif success and r2_success:
            # both succeeded: choose the one with smaller RMSE if provided
            r1_rmse = getattr(getattr(r1, "extra", {}), "get", lambda k, d=None: d)("best_rmse", None)
            r2_rmse = getattr(getattr(r2, "extra", {}), "get", lambda k, d=None: d)("best_rmse", None)

            if r1_rmse is not None and r2_rmse is not None:
                if float(r2_rmse) < float(r1_rmse):
                    best_x, best_dist = r2_best_x, r2_best_dist
            else:
                # fallback: smaller dist
                if r2_best_dist < best_dist:
                    best_x, best_dist = r2_best_x, r2_best_dist
        else:
            # neither succeeded (rare if HSJA found init); choose smaller dist
            if r2_best_dist < best_dist:
                best_x, best_dist = r2_best_x, r2_best_dist
            success = success or r2_success

        total_ms = (perf_counter() - t0) * 1000.0

        return AttackResult(
            success=bool(success),
            best_dist=float(best_dist),
            queries_used=int(getattr(oracle, "queries_used", 0)),
            best_x=best_x,
            history=history,
            extra={**extra, "runtime_ms": float(total_ms)},
        )