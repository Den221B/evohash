"""EvoHash: LLM-guided discovery of perceptual-hash collision attacks."""
from evohash.hashes import HashRegistry, HashSpec, build_default_registry as build_hash_registry
from evohash.attacks import AttackRegistry, build_attack_registry
from evohash.oracle import BudgetSpec, ConstraintSpec, HashOracle
from evohash.dataset import Dataset, PairSample
from evohash.evaluator import EvalRow, Evaluator, run_eval_on_ds
from evohash.metrics import asr_over_l2, aggregate, build_comparison_table, print_comparison_table
from evohash.preprocessing import ResizeSpec, resize_rgb01

__version__ = "0.1.0"

__all__ = [
    # hashes
    "HashRegistry", "HashSpec", "build_hash_registry",
    # attacks
    "AttackRegistry", "build_attack_registry",
    # oracle
    "BudgetSpec", "ConstraintSpec", "HashOracle",
    # dataset
    "Dataset", "PairSample",
    # evaluator
    "EvalRow", "Evaluator", "run_eval_on_ds",
    # metrics
    "asr_over_l2", "aggregate", "build_comparison_table", "print_comparison_table",
    # preprocessing
    "ResizeSpec", "resize_rgb01",
]
