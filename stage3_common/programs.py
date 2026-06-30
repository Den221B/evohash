"""Seed program and candidate file helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from evohash.attacks import build_attack_registry
from evohash.optimizer import clean_public_params, load_attack_config


def ensure_seed_programs(
    *,
    seed_dir: str | Path,
    hash_id: str,
    attack_ids: tuple[str, ...],
    seed_config_path: str | Path | None,
) -> list[Path]:
    """Create import-based seed wrappers for the current attack registry."""
    out = Path(seed_dir)
    out.mkdir(parents=True, exist_ok=True)
    registry = build_attack_registry()
    config: dict[str, Any] = {}
    if seed_config_path is not None and Path(seed_config_path).exists():
        config = load_attack_config(seed_config_path)
    resolved_attack_ids = resolve_seed_attacks(
        hash_id=hash_id,
        attack_ids=attack_ids,
        config=config,
        seed_config_path=seed_config_path,
    )

    paths: list[Path] = []
    for attack_id in resolved_attack_ids:
        registry.get(attack_id)
        params = _params_for_seed(config, hash_id, attack_id)
        path = out / f"{attack_id}_seed.py"
        path.write_text(_seed_code(attack_id=attack_id, params=params), encoding="utf-8")
        paths.append(path)
    return paths


def resolve_seed_attacks(
    *,
    hash_id: str,
    attack_ids: tuple[str, ...],
    config: dict[str, Any] | None = None,
    seed_config_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Resolve user-facing seed attack selectors to concrete registry IDs.

    The special selector ``all`` means: use the configured attack order from
    ``allowed_attacks``, but only keep attacks with a config for this hash.
    That keeps the initial program pool tied to the optimized JSON rather than
    accidentally evaluating every registered attack with default params.
    """
    cfg = config or {}
    requested = tuple(item.strip() for item in attack_ids if item and item.strip())
    if not requested:
        requested = ("all",)

    configured_for_hash = cfg.get("configs", {}).get(hash_id, {})
    allowed = cfg.get("allowed_attacks") or list(configured_for_hash.keys())

    resolved: list[str] = []
    for attack_id in requested:
        if attack_id.lower() == "all":
            if not cfg:
                source = f" from {seed_config_path}" if seed_config_path else ""
                raise ValueError(f"seed_attacks='all' requires a readable seed config{source}")
            resolved.extend(str(item) for item in allowed if item in configured_for_hash)
            continue
        resolved.append(attack_id)

    deduped: list[str] = []
    seen: set[str] = set()
    for attack_id in resolved:
        if attack_id not in seen:
            deduped.append(attack_id)
            seen.add(attack_id)
    return tuple(deduped)


def copy_candidate(*, code: str, candidate_dir: str | Path, prefix: str) -> Path:
    """Write a candidate program to a stable content-addressed file."""
    out = Path(candidate_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]
    path = out / f"{prefix}_{digest}.py"
    path.write_text(code.rstrip() + "\n", encoding="utf-8")
    return path


def read_code(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _params_for_seed(config: dict[str, Any], hash_id: str, attack_id: str) -> dict[str, Any]:
    try:
        params = config["configs"][hash_id][attack_id]
    except Exception:
        return {}
    return clean_public_params(dict(params))


def _seed_code(*, attack_id: str, params: dict[str, Any]) -> str:
    params_literal = repr(params)
    return f'''"""Seed wrapper for the existing {attack_id} attack."""

from __future__ import annotations

from typing import Any

from evohash.attacks import build_attack_registry
from evohash.attacks.base import AttackRawResult


PARAMS = {params_literal}
ATTACK_ID = {attack_id!r}
DEFAULT_PARAMS = dict(PARAMS)


def run_attack(
    x_source,
    oracle,
    params: dict[str, Any] | None = None,
    budget: int | None = 10000,
    seed: int = 0,
) -> AttackRawResult:
    attack = build_attack_registry().get({attack_id!r})
    cfg = {{**PARAMS, **(params or {{}})}}
    return attack.run_attack(
        x_source=x_source,
        oracle=oracle,
        params=cfg,
        budget=budget,
        seed=seed,
    )

'''.strip()
