"""RandomSearch — simple noise-based baseline for pipeline testing.

Not a serious attack; used only to verify the pipeline end-to-end.
"""
from __future__ import annotations

import time

import numpy as np

from evohash.oracle import BudgetSpec, HashOracle
from evohash.utils import to_float32
from .base import AttackResult, AttackSpec


class RandomSearchAttack:
    """Randomly perturb x0 with Gaussian noise, keep the best candidate.

    Params:
        sigma  : std of Gaussian noise (in float [0,1] pixel space)
        iters  : maximum number of perturbation attempts
    """

    def __init__(self, sigma: float = 0.05, iters: int = 200) -> None:
        self.spec = AttackSpec(
            attack_id="random_search",
            params={"sigma": sigma, "iters": iters},
        )

    def run(
        self,
        x0: np.ndarray,
        oracle: HashOracle,
        budget: BudgetSpec,
    ) -> AttackResult:
        t0 = int(time.time() * 1000)
        rng = np.random.default_rng(budget.seed)
        x0 = to_float32(x0)
        sigma = float(self.spec.params["sigma"])
        iters = int(self.spec.params["iters"])

        for _ in range(iters):
            if not oracle.budget_ok():
                break
            noise = rng.normal(0.0, sigma, size=x0.shape).astype(np.float32)
            candidate = oracle.project(x0 + noise)
            oracle.score(candidate)

        runtime = int(time.time() * 1000) - t0
        x_best = oracle.state.best_x if oracle.state.best_x is not None else x0

        return AttackResult(
            x_best=x_best,
            best_score=float(oracle.state.best_score),
            queries_used=int(oracle.state.queries_used),
            runtime_ms=runtime,
            stopped_reason=oracle.state.stopped_reason,
        )
