"""Benchmark helpers for comparing optimized configs and evolved programs."""

from .compare import (
    BenchmarkResult,
    BenchmarkSpec,
    discover_evolved_programs,
    run_benchmark,
    summary_dataframe,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSpec",
    "discover_evolved_programs",
    "run_benchmark",
    "summary_dataframe",
]

