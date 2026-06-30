"""Defense experiment on a single hash stand (default: pHash).

Cheap, non-adaptive evaluation of an input-transformation defense:

1. Run each attack **once** against the clean hash and cache the adversarial
   image ``x_best`` (this is the Stage-1/Stage-4 run; nothing extra to compute).
2. For a grid of defenses, recompute hashes on the cached ``x_best`` and check
   whether the collision survives purification -> ASR(defense).
3. For the same grid, measure the legitimate-match cost: do benignly edited
   copies of clean images still match under the defended hash -> recall(defense).
4. From the cached final hash distances, build ASR(threshold) curves for free.

No re-running of Optuna/evolution is required. (If you *do* want the adaptive
version, pass a ``DefendedHash`` as ``hash_fn`` to :func:`collect_records` and
the same attack is run inside the defended oracle.)

All analysis functions return tidy ``pandas`` frames / numpy arrays for plotting
in a notebook.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from evohash.attacks import build_attack_registry
from evohash.evaluator import evaluate_attack
from evohash.hashes import build_hash_fn
from evohash.hashes.defended import Defense, make_defended, with_threshold
from evohash.metrics import to_float32
from evohash.preprocessing import ResizeSpec, apply_resize

# Re-export so a notebook can do everything from this one module.
from evohash.optimizer import (  # noqa: F401
    attack_params_from_config,
    load_attack_config,
    load_pairs,
)

_PKG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZED_CONFIG = _PKG_DIR / "results" / "attack_configs_optimized.json"

DEFAULT_THRESHOLDS = {"phash": 12.0, "pdq": 92.0, "neuralhash": 17.0, "photodna": 3855.0}

# Base attacks for the diploma protocol (nes_attack_v0 is intentionally excluded).
DIPLOMA_BASE_ATTACKS = ("nes", "simba", "zo_signsgd", "prokos", "atkscopes")

# A reasonable default defense sweep for pHash (JPEG purification by quality).
DEFAULT_DEFENSES: list[Defense] = [
    Defense("none"),
    Defense("jpeg", 90),
    Defense("jpeg", 75),
    Defense("jpeg", 50),
    Defense("jpeg", 30),
    Defense("blur", 0.5),
    Defense("blur", 1.0),
    Defense("blur", 1.5),
]

# Benign edits a legitimate near-duplicate might undergo (measure recall cost).
DEFAULT_BENIGN_EDITS: list[Defense] = [
    Defense("jpeg", 75),
    Defense("resize", 0.8),
    Defense("blur", 0.5),
    Defense("brightness", 1.1),
]


# --------------------------------------------------------------------------- #
# Loading attacks                                                             #
# --------------------------------------------------------------------------- #

def load_evolved_module(program_path: str | Path):
    """Import an evolved program file (defines ``run_attack``) as a module."""
    path = Path(program_path)
    mod_name = f"evolved_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass inside the module can resolve its
    # __module__ via sys.modules (otherwise: 'NoneType' has no attribute __dict__).
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def load_optimized_attack(
    attack_id: str,
    hash_id: str = "phash",
    *,
    config_path: str | Path = DEFAULT_OPTIMIZED_CONFIG,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(registered_attack, params)`` for an optimized registry attack.

    Use with :func:`collect_records` as ``attack=<obj>, params=<dict>``.
    """
    cfg = load_attack_config(str(config_path))
    attack = build_attack_registry().get(attack_id)
    params = attack_params_from_config(cfg, hash_id, attack_id)
    return attack, params


def build_attack_specs(
    *,
    evolved_programs: Sequence[str | Path] = (),
    optimized_attacks: Sequence[str] = (),
    hash_id: str = "phash",
    config_path: str | Path = DEFAULT_OPTIMIZED_CONFIG,
) -> list[tuple[str, Any, dict[str, Any]]]:
    """Build a ``[(name, attack, params), ...]`` list for :func:`collect_many`.

    Evolved programs are named ``<stem> (evolved)``; optimized registry attacks
    ``<attack_id> (opt)``.
    """
    specs: list[tuple[str, Any, dict[str, Any]]] = []
    for attack_id in optimized_attacks:
        attack, params = load_optimized_attack(attack_id, hash_id, config_path=config_path)
        specs.append((f"{attack_id} (opt)", attack, params))
    for path in evolved_programs:
        module = load_evolved_module(path)
        specs.append((f"{Path(path).stem} (evolved)", module, {}))
    return specs


# --------------------------------------------------------------------------- #
# Records: run each attack once, cache x_best + resized source/target          #
# --------------------------------------------------------------------------- #

@dataclass
class AttackRecord:
    pair_id: str
    x_best: np.ndarray            # resized adversarial image, float32 [0,1]
    x_src: np.ndarray             # resized source image
    x_tgt: np.ndarray             # resized target image
    threshold: float
    clean_final: float            # final hash distance under the clean hash
    clean_success: bool
    pixel_l2_raw: float
    queries: int
    extra: dict[str, Any] = field(default_factory=dict)


def collect_records(
    *,
    attack: Any,
    pairs: Sequence[Any],
    hash_fn: Any | None = None,
    hash_id: str = "phash",
    params: dict[str, Any] | None = None,
    budget: int = 10_000,
    seed: int = 0,
    resize_size: int | None = 256,
    resize_resample: str = "bilinear",
    thresholds: dict[str, float] | None = None,
    early_stop: bool = True,
    workers: int | None = None,
    progress: bool = True,
) -> list[AttackRecord]:
    """Run ``attack`` over ``pairs`` once and cache adversarial images.

    ``attack`` is anything the evaluator accepts: a registered attack, an evolved
    module (with ``run_attack``), or a plain ``run_attack`` callable. Pass
    ``hash_fn=`` to override the clean hash (e.g. a ``DefendedHash`` for the
    adaptive variant); otherwise it is built from ``hash_id``.

    ``early_stop=True`` (default) keeps the realistic attack behaviour: it stops
    on the first collision, so ``x_best`` is a low-distortion successful image
    (the right input for the defense-transfer experiment). ``early_stop=False``
    sets the run threshold to ``0`` (exact match) so the attack keeps minimizing
    the hash distance until it either reaches ``0`` or exhausts the budget -- use
    this for meaningful ASR(threshold) curves below the default threshold. (The
    attack still stops at an exact match, since the distance cannot go lower.)

    Per-pair seeds are independent, so results are identical regardless of
    ``workers`` (thread pool over pairs; pHash/PDQ release the GIL well).
    """
    hf_clean = hash_fn if hash_fn is not None else build_hash_fn(
        hash_id, thresholds=thresholds or DEFAULT_THRESHOLDS
    )
    real_tau = float(hf_clean.spec.threshold_p)
    # For no-early-stop, the attack queries a view whose threshold is 0, so it only
    # stops on an exact match (distance cannot go below 0) and otherwise minimizes
    # to the budget. The digest/distance are unchanged, so recomputed metrics stay
    # correct (we record the real threshold below).
    hf_run = hf_clean if early_stop else with_threshold(hf_clean, 0.0)
    resize_spec = ResizeSpec(size=resize_size, resample=resize_resample)
    n = len(pairs)

    def _run_one(index: int, sample: Any) -> AttackRecord:
        row = evaluate_attack(
            attack=attack,
            hash_fn=hf_run,
            x_source=sample.x,
            x_target=sample.y,
            params=params,
            budget=budget,
            seed=int(seed) + index,
            pair_id=getattr(sample, "pair_id", str(index)),
            resize_size=resize_size,
            resize_resample=resize_resample,
        )
        final = float(row.final_hash_l1)
        return AttackRecord(
            pair_id=str(row.pair_id),
            x_best=to_float32(row.x_best),
            x_src=apply_resize(to_float32(sample.x), resize_spec),
            x_tgt=apply_resize(to_float32(sample.y), resize_spec),
            threshold=real_tau,
            clean_final=final,
            clean_success=bool(final <= real_tau),
            pixel_l2_raw=float(row.pixel_l2_raw),
            queries=int(row.queries),
            extra={"early_stop": bool(early_stop)},
        )

    n_workers = _resolve_workers(workers, n)
    records: list[AttackRecord] = [None] * n  # type: ignore[list-item]
    if n_workers <= 1:
        for i, sample in enumerate(pairs):
            records[i] = _run_one(i, sample)
            if progress:
                r = records[i]
                print(f"  [{i + 1}/{n}] {r.pair_id}: final={r.clean_final:.1f} "
                      f"success={r.clean_success} q={r.queries}", flush=True)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_one, i, s): i for i, s in enumerate(pairs)}
            for fut in as_completed(futures):
                i = futures[fut]
                records[i] = fut.result()
                done += 1
                if progress:
                    r = records[i]
                    print(f"  [{done}/{n}] {r.pair_id}: final={r.clean_final:.1f} "
                          f"success={r.clean_success} q={r.queries}", flush=True)
    return records


def _resolve_workers(workers: int | None, n_jobs: int) -> int:
    if workers is not None:
        return max(1, int(workers))
    return max(1, min(8, (os.cpu_count() or 4), n_jobs))


def collect_many(
    specs: Sequence[tuple[str, Any, dict[str, Any]]],
    pairs: Sequence[Any],
    *,
    hash_id: str = "phash",
    budget: int = 10_000,
    seed: int = 0,
    resize_size: int | None = 256,
    progress: bool = True,
    **kwargs: Any,
) -> dict[str, list[AttackRecord]]:
    """Run several ``(name, attack, params)`` specs once each; cache per name."""
    out: dict[str, list[AttackRecord]] = {}
    for name, attack, params in specs:
        if progress:
            print(f"== {name} ==", flush=True)
        out[name] = collect_records(
            attack=attack,
            params=params,
            pairs=pairs,
            hash_id=hash_id,
            budget=budget,
            seed=seed,
            resize_size=resize_size,
            progress=progress,
            **kwargs,
        )
    return out


def clean_summary(records_by_name: dict[str, list[AttackRecord]]):
    """Clean-hash sanity table (ASR / L2 / queries per attack, no defense)."""
    rows = []
    for name, recs in records_by_name.items():
        rows.append(
            {
                "attack": name,
                "clean_asr": float(np.mean([r.clean_success for r in recs])),
                "mean_l2_raw": float(np.mean([r.pixel_l2_raw for r in recs])),
                "mean_queries": float(np.mean([r.queries for r in recs])),
                "n_pairs": int(len(recs)),
            }
        )
    import pandas as pd

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Defense transfer: does the collision survive purification?                  #
# --------------------------------------------------------------------------- #

def defended_final_distances(
    records: Sequence[AttackRecord],
    defense: Defense,
    inner_hash_fn: Any,
) -> np.ndarray:
    """Recompute, per record, the hash distance of x_best under a defended hash."""
    hd = make_defended(inner_hash_fn, defense)
    out = []
    for r in records:
        target_digest = hd.compute(r.x_tgt)
        out.append(float(hd.distance(hd.compute(r.x_best), target_digest)))
    return np.asarray(out, dtype=np.float64)


def defense_sweep(
    records: Sequence[AttackRecord],
    defenses: Sequence[Defense] = DEFAULT_DEFENSES,
    *,
    inner_hash_fn: Any | None = None,
    hash_id: str = "phash",
    thresholds: dict[str, float] | None = None,
):
    """ASR of the cached attack under each defense (at the hash's threshold)."""
    inner = inner_hash_fn or build_hash_fn(hash_id, thresholds=thresholds or DEFAULT_THRESHOLDS)
    tau = float(inner.spec.threshold_p)
    rows = []
    for d in defenses:
        dist = defended_final_distances(records, d, inner)
        success = dist <= tau
        rows.append(
            {
                "defense": d.label,
                "kind": d.kind,
                "strength": d.strength,
                "asr": float(success.mean()),
                "n_success": int(success.sum()),
                "n_pairs": int(len(records)),
                "mean_final_dist": float(dist.mean()),
                "median_final_dist": float(np.median(dist)),
                "threshold": tau,
            }
        )
    import pandas as pd

    return pd.DataFrame(rows)


def defense_sweep_multi(
    records_by_name: dict[str, list[AttackRecord]],
    defenses: Sequence[Defense] = DEFAULT_DEFENSES,
    *,
    inner_hash_fn: Any | None = None,
    hash_id: str = "phash",
    thresholds: dict[str, float] | None = None,
):
    """ASR under each defense for every attack -> one tidy frame (column ``attack``)."""
    inner = inner_hash_fn or build_hash_fn(hash_id, thresholds=thresholds or DEFAULT_THRESHOLDS)
    import pandas as pd

    frames = []
    for name, recs in records_by_name.items():
        df = defense_sweep(recs, defenses, inner_hash_fn=inner)
        df.insert(0, "attack", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def asr_at_thresholds(distances: np.ndarray, taus: Iterable[float]) -> np.ndarray:
    """ASR as a function of threshold for a vector of final distances."""
    dist = np.asarray(distances, dtype=np.float64)
    return np.array([float((dist <= float(t)).mean()) for t in taus], dtype=np.float64)


def asr_threshold_table(
    records: Sequence[AttackRecord],
    taus: Sequence[float],
    *,
    defense: Defense | None = None,
    inner_hash_fn: Any | None = None,
    hash_id: str = "phash",
    thresholds: dict[str, float] | None = None,
):
    """ASR(threshold) curve under the clean hash (default) or a given defense."""
    if defense is None or defense.kind == "none":
        dist = np.array([r.clean_final for r in records], dtype=np.float64)
    else:
        inner = inner_hash_fn or build_hash_fn(hash_id, thresholds=thresholds or DEFAULT_THRESHOLDS)
        dist = defended_final_distances(records, defense, inner)
    import pandas as pd

    return pd.DataFrame({"threshold": list(taus), "asr": asr_at_thresholds(dist, taus)})


# --------------------------------------------------------------------------- #
# Benign cost: do legitimate near-duplicates still match under the defense?    #
# --------------------------------------------------------------------------- #

def source_images(records: Sequence[AttackRecord]) -> list[np.ndarray]:
    """Resized clean source images, usable as the legitimate database."""
    return [r.x_src for r in records]


def benign_recall(
    images: Sequence[np.ndarray],
    defenses: Sequence[Defense] = DEFAULT_DEFENSES,
    benign_edits: Sequence[Defense] = DEFAULT_BENIGN_EDITS,
    *,
    inner_hash_fn: Any | None = None,
    hash_id: str = "phash",
    thresholds: dict[str, float] | None = None,
    threshold: float | None = None,
):
    """Recall (true-positive rate) of benign near-duplicates under each defense.

    For each defense and benign edit, the fraction of clean images whose edited
    copy still hashes within ``threshold`` of the enrolled (defended) hash.
    """
    inner = inner_hash_fn or build_hash_fn(hash_id, thresholds=thresholds or DEFAULT_THRESHOLDS)
    tau = float(threshold if threshold is not None else inner.spec.threshold_p)
    rows = []
    for d in defenses:
        hd = make_defended(inner, d)
        enrolled = [hd.compute(z) for z in images]
        for edit in benign_edits:
            matches = []
            for z, ref in zip(images, enrolled):
                edited = edit(z)
                matches.append(hd.distance(hd.compute(edited), ref) <= tau)
            rows.append(
                {
                    "defense": d.label,
                    "kind": d.kind,
                    "strength": d.strength,
                    "benign_edit": edit.label,
                    "recall": float(np.mean(matches)),
                    "n_images": int(len(images)),
                    "threshold": tau,
                }
            )
    import pandas as pd

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Consistency-gate defense (feature squeezing) -- the hash H is NOT modified.  #
# A query is accepted only if it ALSO matches AND is self-consistent:          #
#   d(H(x), H(squeeze(x))) <= delta.                                           #
# Clean images are squeeze-invariant (self-distance ~0), so enrolled hashes    #
# and legitimate matching are untouched; adversarial images that rely on       #
# fragile near-threshold bits get a large self-distance and are rejected.      #
# --------------------------------------------------------------------------- #

# Squeezes pHash is (near-)invariant to: benign self-distance stays ~0, while
# adversarial fragile bits collapse. Aggregated by max over the list.
DEFAULT_SQUEEZE: list[Defense] = [Defense("jpeg", 85), Defense("median", 3)]


def self_distance(image: np.ndarray, squeeze: Any, inner_hash_fn: Any) -> float:
    """Max hash distance between an image and its squeezed version(s)."""
    squeezes = squeeze if isinstance(squeeze, (list, tuple)) else [squeeze]
    h = inner_hash_fn.compute(image)
    return max(
        float(inner_hash_fn.distance(h, inner_hash_fn.compute(s(image))))
        for s in squeezes
    )


def _resolve_inner(inner_hash_fn, hash_id, thresholds):
    return inner_hash_fn or build_hash_fn(hash_id, thresholds=thresholds or DEFAULT_THRESHOLDS)


def _benign_match_self(images, benign_edits, squeeze, inner, tau):
    """Per benign legit query (edited enrolled image): (matched<=tau, self_distance)."""
    matched, self_d = [], []
    for z in images:
        hz = inner.compute(z)
        for edit in benign_edits:
            q = edit(z)
            matched.append(float(inner.distance(inner.compute(q), hz)) <= tau)
            self_d.append(self_distance(q, squeeze, inner))
    return np.asarray(matched, dtype=bool), np.asarray(self_d, dtype=np.float64)


def consistency_distributions(
    records_by_name: dict[str, list[AttackRecord]],
    images: Sequence[np.ndarray],
    *,
    squeeze: Any = DEFAULT_SQUEEZE,
    benign_edits: Sequence[Defense] = DEFAULT_BENIGN_EDITS,
    inner_hash_fn: Any | None = None,
    hash_id: str = "phash",
    thresholds: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Self-distance arrays for benign legit queries and each attack's x_best.

    Returns ``{"benign": <array>, "<attack name>": <array>, ...}`` -- plot as
    histograms to show that adversarial images separate from benign ones.
    """
    inner = _resolve_inner(inner_hash_fn, hash_id, thresholds)
    out: dict[str, np.ndarray] = {}
    _, benign_self = _benign_match_self(images, benign_edits, squeeze, inner, float(inner.spec.threshold_p))
    out["benign"] = benign_self
    for name, recs in records_by_name.items():
        out[name] = np.asarray([self_distance(r.x_best, squeeze, inner) for r in recs], dtype=np.float64)
    return out


def consistency_detector_sweep(
    records_by_name: dict[str, list[AttackRecord]],
    images: Sequence[np.ndarray],
    deltas: Sequence[float],
    *,
    squeeze: Any = DEFAULT_SQUEEZE,
    benign_edits: Sequence[Defense] = DEFAULT_BENIGN_EDITS,
    inner_hash_fn: Any | None = None,
    hash_id: str = "phash",
    thresholds: dict[str, float] | None = None,
):
    """Sweep the gate margin ``delta``: defended ASR per attack + benign recall.

    Defended ASR = fraction of pairs that were a clean success AND pass the gate
    (``self_distance(x_best) <= delta``). Benign recall = fraction of benign legit
    queries that still match AND pass the gate. The hash is never modified.
    """
    inner = _resolve_inner(inner_hash_fn, hash_id, thresholds)
    tau = float(inner.spec.threshold_p)
    benign_matched, benign_self = _benign_match_self(images, benign_edits, squeeze, inner, tau)

    adv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, recs in records_by_name.items():
        adv_self = np.asarray([self_distance(r.x_best, squeeze, inner) for r in recs], dtype=np.float64)
        adv_ok = np.asarray([bool(r.clean_success) for r in recs], dtype=bool)
        adv[name] = (adv_ok, adv_self)

    rows = []
    for d in deltas:
        d = float(d)
        benign_recall = float(np.mean(benign_matched & (benign_self <= d)))
        for name, (adv_ok, adv_self) in adv.items():
            rows.append(
                {
                    "attack": name,
                    "delta": d,
                    "defended_asr": float(np.mean(adv_ok & (adv_self <= d))),
                    "benign_recall": benign_recall,
                }
            )
    import pandas as pd

    return pd.DataFrame(rows)


def security_cost_frame(sweep_df, recall_df):
    """Join defense ASR (security) with mean benign recall (cost) per defense.

    Returns one row per defense: ``asr`` (lower is safer) and ``mean_recall``
    (higher keeps legitimate matching), for an ASR-vs-recall trade-off plot.
    """
    import pandas as pd

    cost = recall_df.groupby("defense", as_index=False)["recall"].mean()
    cost = cost.rename(columns={"recall": "mean_recall"})
    out = pd.merge(sweep_df[["defense", "kind", "strength", "asr"]], cost, on="defense", how="left")
    return out.sort_values(["kind", "strength"]).reset_index(drop=True)


__all__ = [
    "AttackRecord",
    "Defense",
    "DEFAULT_DEFENSES",
    "DEFAULT_BENIGN_EDITS",
    "DIPLOMA_BASE_ATTACKS",
    "DEFAULT_OPTIMIZED_CONFIG",
    "load_pairs",
    "load_evolved_module",
    "load_optimized_attack",
    "build_attack_specs",
    "collect_records",
    "collect_many",
    "clean_summary",
    "defended_final_distances",
    "defense_sweep",
    "defense_sweep_multi",
    "asr_at_thresholds",
    "asr_threshold_table",
    "source_images",
    "benign_recall",
    "security_cost_frame",
    "DEFAULT_SQUEEZE",
    "self_distance",
    "consistency_distributions",
    "consistency_detector_sweep",
]
