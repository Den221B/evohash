"""Configuration objects for the GigaEvolve evaluation backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LLMSource = Literal["local", "openrouter", "none"]
ParallelBackend = Literal["auto", "sequential", "thread", "process"]


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible LLM endpoint settings.

    ``source="local"`` expects an OpenAI-compatible server such as vLLM,
    llama.cpp server, Ollama's /v1 API, or a compatible proxy.
    """

    source: LLMSource = "local"
    model: str = ""
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout_s: float = 120.0

    @classmethod
    def none(cls) -> "LLMConfig":
        return cls(source="none")


@dataclass(frozen=True)
class EvaluationConfig:
    """Shared evaluation settings for evolved programs."""

    hash_id: str
    dataset_zip: str | Path
    split: str = "public"
    max_pairs: int | None = 5
    budget: int = 10_000
    seed: int = 0
    resize_size: int | None = 256
    resize_resample: str = "bilinear"
    parallel_backend: ParallelBackend = "auto"
    workers: int | None = None
    worker_cap: int = 30
    photodna_worker_cap: int | None = None
    process_start_method: str | None = None
    show_progress: bool = False
    alpha: float = 1.0 / 50.0
    asr_floor: float = 0.05
    failure_gap_penalty: float = 0.0
    attack_config_path: str | Path = "results/attack_configs_optimized.json"

    def normalized(self) -> "EvaluationConfig":
        return EvaluationConfig(
            hash_id=str(self.hash_id),
            dataset_zip=str(Path(self.dataset_zip)),
            split=str(self.split),
            max_pairs=self.max_pairs,
            budget=int(self.budget),
            seed=int(self.seed),
            resize_size=self.resize_size,
            resize_resample=str(self.resize_resample),
            parallel_backend=self.parallel_backend,
            workers=self.workers,
            worker_cap=int(self.worker_cap),
            photodna_worker_cap=self.photodna_worker_cap,
            process_start_method=self.process_start_method,
            show_progress=bool(self.show_progress),
            alpha=float(self.alpha),
            asr_floor=float(self.asr_floor),
            failure_gap_penalty=float(self.failure_gap_penalty),
            attack_config_path=str(Path(self.attack_config_path)),
        )
