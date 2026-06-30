"""
find_ASR.py
===========
Движок поиска атакующих конфигураций для perceptual hashing.

Используется из ноутбуков как чёрный ящик:
    from find_ASR import SearchConfig, Searcher, plot_all, print_final_report

Внутри:
  - Random search по пространству гиперпараметров
  - Local search (окрестность лучших конфигов) — на малом разрешении
  - Validation + iterative local search на 256×256 до SR=100%
  - SimBA с агрессивным epsilon (basis=dct, grayscale-fix встроен в конфиг)
  - Метрика quality_score = pixel_l2 + queries / 50
  - Финальный репорт: copy-paste конфиги
"""

from __future__ import annotations

import json
import random
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter
from scipy.stats import spearmanr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted")

# ── Lazy evohash imports (не ломаем если вызывают вне репо) ──────────────────
try:
    from evohash.attacks import build_attack_registry
    from evohash.attacks.base import AttackRawResult
    from evohash.dataset import Dataset, PairSample
    from evohash.evaluator import evaluate_attack
    from evohash.metrics import to_float32
    from evohash.oracle import BudgetSpec, HashOracle
    from evohash.preprocessing import resize_rgb01
    _EVOHASH_OK = True
except ImportError:
    _EVOHASH_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchConfig:
    """Все параметры одного прогона поиска."""
    hash_id:            str             # "phash" | "pdq" | "photodna" | "neuralhash"
    results_dir:        Path            # куда писать (уникальная для каждого хэша)

    # Dataset
    resize_small:       int   = 64      # размер для быстрого поиска
    resize_full:        int   = 256     # финальная валидация
    resize_resample:    str   = "bilinear"
    n_pairs:            int   = 5
    pair_start:         int   = 0
    dataset_split:      str   = "public"
    dataset_zip:        Optional[Path] = None

    # Search budget
    n_trials_per_attack:  int = 30      # random search
    n_local_small:        int = 20      # local search на малом разрешении
    n_local_256:          int = 30      # local search на 256 (если SR<100%)
    local_256_rounds:     int = 3       # сколько раундов пробуем на 256
    local_radius:         float = 0.5   # log-окрестность ×exp(±radius)
    query_budget_small:   int = 4000    # бюджет на малом разрешении
    query_budget_full:    int = 20_000  # бюджет на 256

    # Scoring (только для SR=100% конфигов)
    # quality_score = pixel_l2 + queries / query_divisor  (меньше = лучше)
    query_divisor:        float = 50.0

    # ── Parallel backend (reuses evohash.parallel_eval.run_eval_jobs) ──────────
    eval_backend:   str = "sequential"   # "sequential" | "thread" | "process" | "auto"
    eval_workers:   Optional[int] = None # None → pool default (cores)

    # ── Two-phase search @ resize_full (опц.; для neuralhash) ─────────────────
    # Фаза-1: скрин всех атак на малом бюджете (screen_budget) → ранжируем по
    # success_rate, затем скорости сходимости (mean_final_dist). Фаза-2: топ-K
    # кандидатов на полном бюджете (query_budget_full) + local refine.
    two_phase:       bool = False
    screen_budget:   int = 2000          # бюджет запросов на фазе-1 (скрин)
    screen_top_k:    int = 3             # сколько кандидатов на атаку → фаза-2

    # Known-good configs — оцениваются первыми, до random search.
    # Dict[attack_id -> List[params_dict]]
    seed_configs: Dict[str, List[dict]] = field(default_factory=dict)

    # Samplers (устанавливаются из ноутбука)
    attack_samplers: Dict[str, Callable] = field(default_factory=dict)

    # Categorical params для отдельного анализа
    categorical_params: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SAMPLING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sample_log(lo: float, hi: float, rng: np.random.Generator) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def sample_choice(choices: list, rng: np.random.Generator):
    return choices[int(rng.integers(len(choices)))]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMBA-GRAY (встроен в движок)
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_pair(seed: int) -> "PairSample":
    def synth(s):
        rng = np.random.default_rng(s)
        big = 256
        img  = Image.new("RGB", (big, big),
                         color=tuple(rng.integers(80, 200, 3).tolist()))
        draw = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(8, 18))):
            kind = rng.choice(["ellipse", "rectangle"])
            x0, y0 = rng.integers(0, big - 40, 2)
            x1 = min(big - 1, x0 + int(rng.integers(30, 120)))
            y1 = min(big - 1, y0 + int(rng.integers(30, 120)))
            color = tuple(rng.integers(0, 256, 3).tolist())
            if kind == "ellipse":
                draw.ellipse([x0, y0, x1, y1], fill=color)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=color)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        return np.asarray(img, dtype=np.uint8)
    return PairSample(
        pair_id=f"synthetic_{seed}",
        x=synth(2 * seed + 1),
        y=synth(2 * seed + 2),
        meta={},
    )


def load_pairs(cfg: SearchConfig, resize_size: int) -> Tuple[List, List]:
    """Загружает пары из zip или синтетических данных."""
    samples = []
    if cfg.dataset_zip is not None and cfg.dataset_zip.exists():
        try:
            ds = Dataset(
                str(cfg.dataset_zip),
                split=cfg.dataset_split,
                max_pairs=cfg.pair_start + cfg.n_pairs,
            )
            samples = list(ds)[cfg.pair_start: cfg.pair_start + cfg.n_pairs]
            print(f"✅ Loaded {len(samples)} pairs from {cfg.dataset_zip.name}")
        except Exception as e:
            print(f"⚠️  zip failed: {e} — using synthetic")

    if not samples:
        samples = [make_synthetic_pair(i) for i in range(cfg.n_pairs)]
        print(f"✅ Generated {len(samples)} synthetic pairs")

    pairs = [
        (
            resize_rgb01(s.x, resize_size, resample=cfg.resize_resample),
            resize_rgb01(s.y, resize_size, resample=cfg.resize_resample),
        )
        for s in samples
    ]
    return pairs, samples


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRIAL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one_pair(attack, hash_fn, x, y, params, budget, seed) -> Optional[dict]:
    try:
        row = evaluate_attack(
            attack=attack, hash_fn=hash_fn,
            x_source=x, x_target=y,
            params=deepcopy(params),
            budget=budget, seed=seed,
        )
        return {
            "initial_dist": row.initial_hash_l1,
            "final_dist":   row.final_hash_l1,
            "success":      row.success,
            "rel_impr":     row.relative_improvement,
            "queries":      row.queries,
            "pixel_l2":     row.pixel_l2_raw,
            "time_sec":     row.time_sec,
            "history":      row.history[::max(1, len(row.history) // 100)],
        }
    except Exception as e:
        return {"error": str(e)}


def _aggregate(per_pair: list) -> dict:
    valid = [p for p in per_pair if "error" not in p]
    if not valid:
        return {}
    return {
        "mean_final_dist": float(np.mean([p["final_dist"] for p in valid])),
        "min_final_dist":  float(np.min ([p["final_dist"] for p in valid])),
        "mean_rel_impr":   float(np.mean([p["rel_impr"]   for p in valid])),
        "success_rate":    float(np.mean([p["success"]     for p in valid])),
        "mean_queries":    float(np.mean([p["queries"]     for p in valid])),
        "mean_pixel_l2":   float(np.mean([p["pixel_l2"]   for p in valid])),
        "mean_time_sec":   float(np.mean([p["time_sec"]   for p in valid])),
        "histories":       [p["history"] for p in valid],
    }


def quality_score(result: dict, query_divisor: float = 50.0) -> float:
    """Меньше = лучше. Только для SR=100% конфигов."""
    return result["mean_pixel_l2"] + result["mean_queries"] / query_divisor


# ── Parallel evaluation of a batch of trials (reuses run_eval_jobs) ───────────

def _resolve_backend(cfg: SearchConfig) -> str:
    b = (cfg.eval_backend or "sequential").lower()
    if b == "auto":
        import platform
        return "process" if platform.system() != "Windows" else "thread"
    return b


def _agg_rows(ok_rows: list) -> dict:
    """Aggregate parallel run_eval_jobs rows for ONE config across pairs."""
    return {
        "mean_final_dist": float(np.mean([r["final_hash_l1"]       for r in ok_rows])),
        "min_final_dist":  float(np.min ([r["final_hash_l1"]       for r in ok_rows])),
        "mean_rel_impr":   float(np.mean([r["relative_improvement"] for r in ok_rows])),
        "success_rate":    float(np.mean([float(bool(r["success"])) for r in ok_rows])),
        "mean_queries":    float(np.mean([r["queries"]             for r in ok_rows])),
        "mean_pixel_l2":   float(np.mean([r["pixel_l2_raw"]        for r in ok_rows])),
        "mean_time_sec":   float(np.mean([r.get("time_sec", 0.0)   for r in ok_rows])),
    }


def _mk_record(tr: dict, agg: dict) -> dict:
    rec = {
        "attack_id": tr["attack_id"],
        "trial_idx": tr["trial_idx"],
        "params":    tr["params"],
        "timestamp": datetime.now().isoformat(),
        **agg,
    }
    if tr.get("is_seed"):
        rec["is_seed"] = True
    return rec


def _has_real_history(val) -> bool:
    """True if val already contains at least one non-empty list of floats."""
    if not isinstance(val, list) or not val:
        return False
    return any(isinstance(h, list) and len(h) > 0 for h in val)


def _collect_histories(
    df: pd.DataFrame,
    hash_fn,
    pairs: List,
    all_attacks: dict,
    budget: int,
    *,
    top_k: int = 5,
    force: bool = False,
    hist_budget: Optional[int] = None,
) -> pd.DataFrame:
    """Collect convergence histories for the top-K configs per attack.

    Runs sequentially — used only for plots, not for search.
    Updates the ``histories`` column in a copy of ``df`` for the selected rows.

    **Skip rows that already have real histories** (from the parallel search run)
    unless ``force=True``.  This avoids re-running expensive evaluations when the
    data is already saved in the jsonl files.

    ``hist_budget``: query budget for history collection.  Defaults to ``budget``,
    but for convergence plots you usually only need a fraction (e.g. 500–1000).
    """
    df = df.copy()
    if "histories" not in df.columns:
        df["histories"] = [[] for _ in range(len(df))]

    run_budget = hist_budget if hist_budget is not None else budget
    n_skipped = n_rerun = 0

    for atk_id in sorted(df["attack_id"].unique()):
        sub = df[df["attack_id"] == atk_id].sort_values(
            ["success_rate", "mean_rel_impr"], ascending=False
        ).head(top_k)
        attack = all_attacks.get(atk_id)
        if attack is None:
            continue
        for idx in sub.index:
            existing = df.at[idx, "histories"]
            if not force and _has_real_history(existing):
                n_skipped += 1
                continue           # already have data — don't re-run
            params = df.at[idx, "params"]
            if not isinstance(params, dict):
                try:
                    params = json.loads(params)
                except Exception:
                    continue
            per_pair = [
                _run_one_pair(attack, hash_fn, x, y, params, run_budget, pi)
                for pi, (x, y) in enumerate(pairs)
            ]
            df.at[idx, "histories"] = [
                p.get("history", p.get("histories", []))
                for p in per_pair if "error" not in p
            ]
            n_rerun += 1

    print(f"_collect_histories: {n_skipped} skipped (already had data), "
          f"{n_rerun} re-run at budget={run_budget}")
    return df


def _write_records(out_file, recs) -> None:
    """Append+flush records to jsonl (incremental persistence for resume)."""
    if out_file is None or not recs:
        return
    with open(out_file, "a") as fout:
        for r in recs:
            fout.write(json.dumps(r) + "\n")
        fout.flush()


def _eval_trials(
    cfg: SearchConfig,
    hash_fn,
    trials: List[dict],
    pairs: List,
    budget: int,
    all_attacks: dict,
    *,
    keep_history: bool = False,
    desc: str = "",
    out_file: Optional[Path] = None,
    chunk_size: Optional[int] = None,
) -> List[dict]:
    """Evaluate a batch of trials over all pairs and aggregate per trial.

    Each trial: {"attack_id", "trial_idx", "params", "seed_base"(opt), "is_seed"(opt)}.
    Returns records with the SAME aggregate keys the sequential path produced, so
    plots/downstream are unaffected. Parallel via parallel_eval unless
    cfg.eval_backend == "sequential".

    If ``out_file`` is given, records are appended+flushed **incrementally** (per
    trial in sequential mode, per chunk in parallel mode) so an interrupted run
    resumes near where it stopped. The parallel path reuses ONE worker pool across
    chunks (no re-spawn / no model reload per chunk).
    """
    backend = _resolve_backend(cfg)
    n_pairs = len(pairs)
    records: List[dict] = []
    if not trials:
        return records

    if backend == "sequential":
        for tr in tqdm(trials, desc=desc or f"eval [{cfg.hash_id}]"):
            attack = all_attacks.get(tr["attack_id"])
            if attack is None:
                continue
            base = int(tr.get("seed_base", 0))
            per_pair = [
                _run_one_pair(attack, hash_fn, x, y, tr["params"], budget, base * 1000 + pi)
                for pi, (x, y) in enumerate(pairs)
            ]
            agg = _aggregate(per_pair)
            if not agg:
                continue
            if not keep_history:
                agg.pop("histories", None)
            rec = _mk_record(tr, agg)
            records.append(rec)
            _write_records(out_file, [rec])   # incremental
        return records

    # ── Parallel path: one persistent pool, chunked + incremental writes ──
    from evohash.parallel_eval import EvalJobRunner
    if chunk_size is None:
        # Each chunk still has chunk_size * n_pairs jobs >> workers, so the pool
        # stays saturated; small enough that an interrupt loses ≤1 chunk.
        chunk_size = max(int(cfg.eval_workers or 8), 10)
    context = {
        "config":          {"thresholds": {cfg.hash_id: float(hash_fn.threshold)}},
        "budget":          int(budget),
        "resize_size":     None,           # pairs already resized by load_pairs
        "resize_resample": "bilinear",
        "alpha":           1.0 / 50.0,
        "compute_lpips":   False,
        "keep_history":    bool(keep_history),
        "backend":         backend,
    }
    chunks = [trials[i:i + chunk_size] for i in range(0, len(trials), chunk_size)]
    with EvalJobRunner(context=context, backend=backend,
                       max_workers=cfg.eval_workers, show_progress=False) as runner:
        for chunk in tqdm(chunks, desc=desc or f"eval [{cfg.hash_id}]"):
            jobs = []
            for ti, tr in enumerate(chunk):
                base = int(tr.get("seed_base", 0))
                for pi, (x, y) in enumerate(pairs):
                    jobs.append({
                        "job_index":  ti * n_pairs + pi,
                        "hash_id":    cfg.hash_id,
                        "attack_id":  tr["attack_id"],
                        "pair_index": pi,
                        "pair_id":    str(pi),
                        "x_source":   np.asarray(x),
                        "x_target":   np.asarray(y),
                        "params":     dict(tr["params"]),
                        "seed":       base * 1000 + pi,
                    })
            rows = runner.run(jobs)
            by_trial: dict = {}
            for r in rows:
                by_trial.setdefault(int(r["job_index"]) // n_pairs, []).append(r)
            chunk_recs = []
            for ti, tr in enumerate(chunk):
                ok = [r for r in by_trial.get(ti, []) if r.get("status") == "ok"]
                if not ok:
                    continue
                agg = _agg_rows(ok)
                if keep_history:
                    agg["histories"] = [r.get("history", []) for r in ok]
                chunk_recs.append(_mk_record(tr, agg))
            records.extend(chunk_recs)
            _write_records(out_file, chunk_recs)   # incremental per chunk
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RANDOM SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def run_random_search(
    cfg: SearchConfig,
    hash_fn,
    pairs: List,
    atk_reg,
    *,
    extra_attacks: Optional[dict] = None,   # {"simba_gray": simba_gray_obj}
    rng_seed: int = 7,
) -> pd.DataFrame:
    """
    Random search по пространству гиперпараметров.
    Пишет results в cfg.results_dir/trials.jsonl (с resume).
    Возвращает DataFrame всех результатов.
    """
    out_file = cfg.results_dir / "trials.jsonl"
    done_keys: set = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r["attack_id"], r["trial_idx"]))
        print(f"Resume: {len(done_keys)} trials already done")

    all_attacks = {aid: atk_reg.get(aid) for aid in atk_reg.list_ids()}
    if extra_attacks:
        all_attacks.update(extra_attacks)

    # Build trials: seed configs first (deterministic ids), then random samples.
    rng = np.random.default_rng(rng_seed)
    trials: List[dict] = []
    for atk_id, seed_list in cfg.seed_configs.items():
        if atk_id not in all_attacks:
            continue
        for si, sp in enumerate(seed_list):
            trials.append({"attack_id": atk_id, "trial_idx": f"seed_{si}",
                           "params": dict(sp), "seed_base": si, "is_seed": True})
    for atk_id, sampler in cfg.attack_samplers.items():
        if atk_id not in all_attacks:
            continue
        for trial_idx in range(cfg.n_trials_per_attack):
            trials.append({"attack_id": atk_id, "trial_idx": trial_idx,
                           "params": sampler(rng), "seed_base": int(trial_idx) + 1})

    pending = [t for t in trials if (t["attack_id"], t["trial_idx"]) not in done_keys]
    print(f"Total trials: {len(trials)} | pending: {len(pending)} | "
          f"backend={_resolve_backend(cfg)}")

    records = _eval_trials(cfg, hash_fn, pending, pairs, cfg.query_budget_small,
                           all_attacks, keep_history=True, out_file=out_file,
                           desc=f"Random search [{cfg.hash_id}]")
    print(f"✅ {len(records)} new trials → {out_file}")
    return _load_jsonl(out_file)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LOCAL SEARCH (малое разрешение)
# ═══════════════════════════════════════════════════════════════════════════════

_PARAM_BOUNDS: dict[str, tuple] = {
    # (min, max) — enforced after log-perturbation.
    # beta1/beta2 must stay strictly in (0, 1) for Adam/SGD to be valid.
    "beta1": (0.01, 0.999),
    "beta2": (0.01, 0.9999),
    "eps":   (1e-12, 1e-4),
}


def _perturb(params: dict, radius: float, rng: np.random.Generator) -> dict:
    new = {}
    for k, v in params.items():
        # bool MUST be checked before int: isinstance(True, int) is True in Python,
        # so without this booleans (antithetic, grayscale_noise, eval_updated_point)
        # get corrupted into 1/2 by the int branch below.
        if isinstance(v, bool):
            new[k] = bool(v)
        elif isinstance(v, float) and v > 0:
            perturbed = float(np.exp(rng.uniform(np.log(v) - radius,
                                                  np.log(v) + radius)))
            if k in _PARAM_BOUNDS:
                lo, hi = _PARAM_BOUNDS[k]
                perturbed = max(lo, min(hi, perturbed))
            new[k] = perturbed
        elif isinstance(v, int) and v > 0 and k != "max_iters":
            choices = sorted({max(1, int(v * 0.5)), v, min(int(v * 2), 512)})
            new[k] = int(rng.choice(choices))
        else:
            new[k] = v
    return new


def run_local_search(
    cfg: SearchConfig,
    hash_fn,
    pairs: List,
    atk_reg,
    df_random: pd.DataFrame,
    *,
    extra_attacks: Optional[dict] = None,
    rng_seed: int = 42,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Уточняет параметры в окрестности лучших конфигов из random search.
    suffix — для разных фаз (напр. "", "_256").
    """
    out_file = cfg.results_dir / f"local_search{suffix}.jsonl"
    done: set = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                done.add(json.loads(line)["trial_idx"])

    all_attacks = {aid: atk_reg.get(aid) for aid in atk_reg.list_ids()}
    if extra_attacks:
        all_attacks.update(extra_attacks)

    rng    = np.random.default_rng(rng_seed)
    n_jobs = cfg.n_local_small if not suffix else cfg.n_local_256

    trials: List[dict] = []
    for atk_id in sorted(cfg.attack_samplers.keys()):
        sub = df_random[df_random["attack_id"] == atk_id].copy()
        if sub.empty:
            continue
        best = sub.sort_values(
            ["success_rate", "mean_rel_impr"], ascending=False
        ).iloc[0]
        best_params = best["params"] if isinstance(best["params"], dict) \
            else json.loads(best["params"])
        for i in range(n_jobs):
            trials.append({
                "attack_id": atk_id,
                "trial_idx": f"local{suffix}_{atk_id}_{i}",
                "params":    _perturb(best_params, cfg.local_radius, rng),
                "seed_base": i,
            })

    budget  = cfg.query_budget_small if not suffix else cfg.query_budget_full
    pending = [t for t in trials if t["trial_idx"] not in done]

    records = _eval_trials(cfg, hash_fn, pending, pairs, budget, all_attacks,
                           keep_history=True, out_file=out_file,
                           desc=f"Local search{suffix} [{cfg.hash_id}]")
    return _load_jsonl(out_file)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VALIDATION + ITERATIVE LOCAL SEARCH @ 256
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation_and_refine(
    cfg: SearchConfig,
    hash_fn,
    pairs_256: List,
    atk_reg,
    df_small: pd.DataFrame,      # лучшие конфиги с малого разрешения
    *,
    extra_attacks: Optional[dict] = None,
) -> pd.DataFrame:
    """
    1. Берёт топ-1 конфиг (по success_rate, rel_impr) для каждой атаки.
    2. Запускает на 256×256.
    3. Если SR < 100% — итеративно ищет окрестности на 256 до cfg.local_256_rounds раундов.
    4. Возвращает DataFrame всех результатов на 256.
    """
    all_attacks = {aid: atk_reg.get(aid) for aid in atk_reg.list_ids()}
    if extra_attacks:
        all_attacks.update(extra_attacks)

    out_file = cfg.results_dir / "validation_256.jsonl"
    done: set = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                done.add(json.loads(line)["trial_idx"])

    rng = np.random.default_rng(99)

    def _eval_config(atk_id, trial_id, params):
        if trial_id in done:
            return None
        attack = all_attacks.get(atk_id)
        if attack is None:
            return None
        per_pair = [
            _run_one_pair(attack, hash_fn, x, y,
                          params, cfg.query_budget_full, pi)
            for pi, (x, y) in enumerate(pairs_256)
        ]
        agg = _aggregate(per_pair)
        if not agg:
            return None
        return {
            "attack_id": atk_id,
            "trial_idx": trial_id,
            "params":    params,
            "timestamp": datetime.now().isoformat(),
            **agg,
        }

    with open(out_file, "a") as fout:
        for atk_id in tqdm(sorted(cfg.attack_samplers.keys()),
                           desc=f"Validation 256 [{cfg.hash_id}]"):
            sub = df_small[df_small["attack_id"] == atk_id].copy()
            if sub.empty:
                continue

            best_row = sub.sort_values(
                ["success_rate", "mean_rel_impr"], ascending=False
            ).iloc[0]
            best_params = best_row["params"] if isinstance(best_row["params"], dict) \
                else json.loads(best_row["params"])

            # Шаг 1: валидация лучшего конфига
            rec = _eval_config(atk_id, f"val256_{atk_id}_0", best_params)
            if rec:
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                done.add(rec["trial_idx"])
                sr = rec["success_rate"]
            else:
                # уже было
                existing = [json.loads(l) for l in open(out_file)
                            if json.loads(l)["attack_id"] == atk_id]
                sr = max((r["success_rate"] for r in existing), default=0.0)

            # Шаг 2: если SR < 100% → итеративный local search на 256
            if sr < 1.0:
                print(f"  {atk_id}: SR={sr:.0%} at 256 → refining "
                      f"({cfg.local_256_rounds} rounds × {cfg.n_local_256} trials)")
                for round_idx in range(cfg.local_256_rounds):
                    # Берём лучший из уже записанных на 256
                    existing = [json.loads(l) for l in open(out_file)
                                if json.loads(l)["attack_id"] == atk_id]
                    if not existing:
                        break
                    cur_best = max(existing,
                                   key=lambda r: (r["success_rate"],
                                                  r.get("mean_rel_impr", 0)))
                    cur_sr   = cur_best["success_rate"]
                    if cur_sr >= 1.0:
                        print(f"    round {round_idx}: SR=100% ✅")
                        break

                    cur_params = cur_best["params"] if isinstance(cur_best["params"], dict) \
                        else json.loads(cur_best["params"])

                    for i in range(cfg.n_local_256):
                        tid = f"val256_{atk_id}_r{round_idx}_{i}"
                        perturbed = _perturb(cur_params, cfg.local_radius, rng)
                        rec = _eval_config(atk_id, tid, perturbed)
                        if rec:
                            fout.write(json.dumps(rec) + "\n"); fout.flush()
                            done.add(rec["trial_idx"])

                    # Обновляем SR
                    existing = [json.loads(l) for l in open(out_file)
                                if json.loads(l)["attack_id"] == atk_id]
                    best_sr  = max(r["success_rate"] for r in existing)
                    print(f"    round {round_idx}: best SR={best_sr:.0%}")
                    if best_sr >= 1.0:
                        break

    return _load_jsonl(out_file)


# ═══════════════════════════════════════════════════════════════════════════════
# 8b. TWO-PHASE SEARCH @ resize_full (screen low-budget → refine top-K full-budget)
# ═══════════════════════════════════════════════════════════════════════════════

def run_two_phase_search(
    cfg: SearchConfig,
    hash_fn,
    pairs: List,
    atk_reg,
    *,
    extra_attacks: Optional[dict] = None,
    rng_seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Successive-halving search, all at one resolution (cfg.resize_full).

    Phase 1 (screen): every attack × n_trials at the SMALL ``screen_budget`` —
        cheap look at ASR + convergence speed (mean_final_dist).
    Promote: top ``screen_top_k`` per attack (SR desc, then lower final_dist).
    Phase 2 (refine): promoted candidates at the FULL ``query_budget_full`` with
        ``local_256_rounds`` rounds of local search around the current best.

    Returns ``(df_screen, df_refine)``. Use df_refine for the final report,
    df_screen for factor-importance plots.
    """
    all_attacks = {aid: atk_reg.get(aid) for aid in atk_reg.list_ids()}
    if extra_attacks:
        all_attacks.update(extra_attacks)

    # ── Phase 1: screen ──
    screen_file = cfg.results_dir / "screen.jsonl"
    done: set = set()
    if screen_file.exists():
        with open(screen_file) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["attack_id"], r["trial_idx"]))

    rng = np.random.default_rng(rng_seed)
    trials: List[dict] = []
    for atk_id, seed_list in cfg.seed_configs.items():
        if atk_id not in all_attacks:
            continue
        for si, sp in enumerate(seed_list):
            trials.append({"attack_id": atk_id, "trial_idx": f"seed_{si}",
                           "params": dict(sp), "seed_base": si, "is_seed": True})
    for atk_id, sampler in cfg.attack_samplers.items():
        if atk_id not in all_attacks:
            continue
        for ti in range(cfg.n_trials_per_attack):
            trials.append({"attack_id": atk_id, "trial_idx": ti,
                           "params": sampler(rng), "seed_base": ti + 1})
    pending = [t for t in trials if (t["attack_id"], t["trial_idx"]) not in done]
    print(f"[Phase 1 · screen @{cfg.resize_full} · budget={cfg.screen_budget}] "
          f"trials={len(trials)} pending={len(pending)} backend={_resolve_backend(cfg)}")
    recs = _eval_trials(cfg, hash_fn, pending, pairs, cfg.screen_budget,
                        all_attacks, keep_history=True, out_file=screen_file,
                        desc=f"Screen [{cfg.hash_id}]")
    df_screen = _load_jsonl(screen_file)

    # ── Promote top-K per attack ──
    promoted: List[dict] = []
    print("\n[Promote] top candidates per attack (SR desc, then convergence=lower final_dist):")
    for atk_id in sorted(cfg.attack_samplers.keys()):
        sub = df_screen[df_screen["attack_id"] == atk_id]
        if sub.empty:
            continue
        top = sub.sort_values(["success_rate", "mean_final_dist"],
                              ascending=[False, True]).head(cfg.screen_top_k)
        for rank, (_, row) in enumerate(top.iterrows()):
            params = row["params"] if isinstance(row["params"], dict) else json.loads(row["params"])
            promoted.append({"attack_id": atk_id,
                             "trial_idx": f"promote_{atk_id}_{rank}",
                             "params": params, "seed_base": rank})
            print(f"  {atk_id:<16} #{rank}: SR={row['success_rate']:.0%}  "
                  f"dist={row['mean_final_dist']:.1f}  q={row['mean_queries']:.0f}")

    # ── Phase 2: refine ──
    refine_file = cfg.results_dir / "refine.jsonl"
    done2: set = set()
    if refine_file.exists():
        with open(refine_file) as f:
            for line in f:
                done2.add(json.loads(line)["trial_idx"])
    print(f"\n[Phase 2 · refine @{cfg.resize_full} · budget={cfg.query_budget_full}] "
          f"{len(promoted)} candidates × up to {cfg.local_256_rounds} rounds")

    cur = [t for t in promoted if t["trial_idx"] not in done2]
    recs0 = _eval_trials(cfg, hash_fn, cur, pairs, cfg.query_budget_full,
                         all_attacks, keep_history=True, out_file=refine_file,
                         desc=f"Refine r0 [{cfg.hash_id}]")
    done2.update(r["trial_idx"] for r in recs0)

    rng2 = np.random.default_rng(99)
    for rnd in range(cfg.local_256_rounds):
        df_ref = _load_jsonl(refine_file)
        batch: List[dict] = []
        for atk_id in sorted(cfg.attack_samplers.keys()):
            sub = df_ref[df_ref["attack_id"] == atk_id]
            if sub.empty:
                continue
            best = sub.sort_values(["success_rate", "mean_final_dist"],
                                   ascending=[False, True]).iloc[0]
            if best["success_rate"] >= 1.0:
                continue   # already perfect → don't waste budget
            bp = best["params"] if isinstance(best["params"], dict) else json.loads(best["params"])
            for i in range(cfg.n_local_256):
                tid = f"refine_{atk_id}_r{rnd}_{i}"
                if tid in done2:
                    continue
                batch.append({"attack_id": atk_id, "trial_idx": tid,
                              "params": _perturb(bp, cfg.local_radius, rng2),
                              "seed_base": i})
        if not batch:
            print(f"  round {rnd}: nothing to refine (all SR=100% or empty) — stop")
            break
        recs = _eval_trials(cfg, hash_fn, batch, pairs, cfg.query_budget_full,
                            all_attacks, keep_history=True, out_file=refine_file,
                            desc=f"Refine r{rnd} [{cfg.hash_id}]")
        done2.update(r["trial_idx"] for r in recs)
        df_ref = _load_jsonl(refine_file)
        srs = {a: float(df_ref[df_ref["attack_id"] == a]["success_rate"].max())
               for a in sorted(df_ref["attack_id"].unique())}
        print(f"  round {rnd}: best SR/attack = "
              + ", ".join(f"{a}:{v:.0%}" for a, v in srs.items()))

    return df_screen, _load_jsonl(refine_file)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SELECT BEST CONFIGS FOR 256
# ═══════════════════════════════════════════════════════════════════════════════

def select_best_256(
    df_256: pd.DataFrame,
    cfg: SearchConfig,
    *,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Выбирает лучшие конфиги для 256:
    1. Сортировка: success_rate desc
    2. Среди SR=100% → quality_score = pixel_l2 + queries/query_divisor asc
    3. Если нет SR=100% → best by rel_impr

    Возвращает top_k строк per attack.
    """
    df = df_256.copy()
    df["quality_score"] = df.apply(
        lambda r: r["mean_pixel_l2"] + r["mean_queries"] / cfg.query_divisor
        if r["success_rate"] >= 1.0 else np.inf,
        axis=1,
    )
    result = []
    for atk_id in sorted(df["attack_id"].unique()):
        sub = df[df["attack_id"] == atk_id].copy()
        sr100 = sub[sub["success_rate"] >= 1.0]
        if not sr100.empty:
            top = sr100.sort_values("quality_score").head(top_k)
        else:
            top = sub.sort_values(
                ["success_rate", "mean_rel_impr"], ascending=False
            ).head(top_k)
        result.append(top)
    return pd.concat(result, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _interp100(h):
    if len(h) < 2:
        return None
    return np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(h)), h)


def plot_hash_distance_curves(df: pd.DataFrame, cfg: SearchConfig, hash_fn):
    attacks = sorted(df["attack_id"].unique())
    n = len(attacks)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).ravel()
    n_trials = cfg.n_trials_per_attack
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(n_trials, 1)))

    for ax_idx, atk_id in enumerate(attacks):
        ax  = axes[ax_idx]
        sub = df[df["attack_id"] == atk_id].sort_values("mean_rel_impr")
        n_with_hist = 0
        for i, (_, row) in enumerate(sub.iterrows()):
            hists = row.get("histories", [])
            if not isinstance(hists, list):
                continue
            curves = [c for c in (_interp100(h) for h in hists) if c is not None]
            if not curves:
                continue
            n_with_hist += 1
            alpha = 0.25 + 0.75 * (i / max(1, len(sub) - 1))
            ax.plot(np.mean(curves, axis=0),
                    color=colors[i % len(colors)], alpha=alpha, linewidth=0.9)
        ax.axhline(hash_fn.threshold, color="red", linestyle="--",
                   linewidth=1.5, label=f"threshold={hash_fn.threshold:.0f}")
        if n_with_hist == 0:
            ax.text(0.5, 0.5, "no histories\n(run _collect_histories first)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
        ax.set_title(atk_id, fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalised iterations →")
        ax.set_ylabel("Hash Distance")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for ax in axes[len(attacks):]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(0, max(n_trials - 1, 1)))
    # constrained_layout (set on the figure) reserves space for the colorbar so it
    # no longer overlaps the subplots; tight_layout would fight it and cause overlap.
    fig.colorbar(sm, ax=list(axes[:n]), label="Trial rank (worst→best)", shrink=0.6)
    fig.suptitle(f"Hash Distance Curves [{cfg.hash_id}]",
                 fontsize=14, fontweight="bold")
    out = cfg.results_dir / "hash_distance_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_param_importance_numeric(df: pd.DataFrame, cfg: SearchConfig):
    param_cols = [c for c in df.columns if c.startswith("p_")
                  and pd.api.types.is_numeric_dtype(df[c])]
    attacks = sorted(df["attack_id"].unique())
    n = len(attacks)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).ravel()

    print("Spearman ρ с mean_rel_impr  (* = p<0.05)")
    print("=" * 60)
    for ax_idx, atk_id in enumerate(attacks):
        ax  = axes[ax_idx]
        sub = df[df["attack_id"] == atk_id]
        if len(sub) < 5:
            ax.set_visible(False); continue
        corrs = []
        for pc in [c for c in param_cols
                   if c in sub and sub[c].notna().sum() >= 3]:
            v = sub[[pc, "mean_rel_impr"]].dropna()
            if len(v) < 3: continue
            try:
                rho, pval = spearmanr(v[pc], v["mean_rel_impr"])
                corrs.append((pc[2:], rho, pval))
            except Exception:
                pass
        if not corrs:
            ax.set_visible(False); continue
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        names = [c[0] for c in corrs]
        vals  = [c[1] for c in corrs]
        pvals = [c[2] for c in corrs]
        bars  = ax.barh(names, vals,
                        color=["#2ecc71" if v > 0 else "#e74c3c" for v in vals])
        for bar, h in zip(bars, ["" if p < 0.05 else "///" for p in pvals]):
            bar.set_hatch(h)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(atk_id, fontsize=11, fontweight="bold")
        ax.set_xlabel("Spearman ρ  (///=p>0.05)"); ax.set_xlim(-1, 1)
        print(f"\n{atk_id}:")
        for name, rho, pval in corrs:
            print(f"  {'*' if pval<0.05 else ' '} {name:<20} ρ={rho:+.3f}  p={pval:.3f}")

    for ax in axes[len(attacks):]:
        ax.set_visible(False)
    plt.suptitle(f"Numeric Param Importance [{cfg.hash_id}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = cfg.results_dir / "param_importance_numeric.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_param_importance_categorical(df: pd.DataFrame, cfg: SearchConfig):
    cat_items = []
    for atk_id, cat_params in cfg.categorical_params.items():
        sub = df[df["attack_id"] == atk_id]
        for cp in cat_params:
            col = f"p_{cp}"
            if col not in sub.columns: continue
            if sub[col].dropna().nunique() < 2: continue
            cat_items.append((atk_id, cp, col, sub))

    if not cat_items:
        print("Нет данных для категориального анализа.")
        return

    n = len(cat_items)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).ravel()

    print("CATEGORICAL PARAMETER ANALYSIS")
    print("=" * 60)
    for ax_idx, (atk_id, cp, col, sub) in enumerate(cat_items):
        ax  = axes[ax_idx]
        sub = sub[[col, "mean_rel_impr"]].dropna().copy()
        # None → строка "auto" для корректного отображения
        if sub[col].dtype == object or sub[col].isnull().any():
            sub[col] = sub[col].fillna("auto").astype(str)
        order = sorted(sub[col].unique(),
                       key=lambda v: -sub[sub[col] == v]["mean_rel_impr"].median())
        sns.boxplot(data=sub, x=col, y="mean_rel_impr",
                    order=order, ax=ax, palette="Set2")
        sns.stripplot(data=sub, x=col, y="mean_rel_impr",
                      order=order, ax=ax, color="black",
                      alpha=0.5, size=4, jitter=True)
        ax.axhline(1.0, color="green", linestyle="--", linewidth=1,
                   label="100% success")
        ax.set_title(f"{atk_id} — {cp}", fontsize=11, fontweight="bold")
        ax.set_xlabel(cp); ax.set_ylabel("mean_rel_impr")
        ax.legend(fontsize=7)
        print(f"\n{atk_id}.{cp}:")
        for v in order:
            grp = sub[sub[col] == v]["mean_rel_impr"]
            print(f"  {str(v):<12}  n={len(grp):3d}  "
                  f"median={grp.median():.3f}  max={grp.max():.3f}")

    for ax in axes[len(cat_items):]:
        ax.set_visible(False)
    plt.suptitle(f"Categorical Params [{cfg.hash_id}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = cfg.results_dir / "param_importance_categorical.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_validation_comparison(
    df_small: pd.DataFrame,
    df_256: pd.DataFrame,
    cfg: SearchConfig,
):
    """Bar chart: малое разрешение vs 256."""
    attacks = sorted(df_256["attack_id"].unique())
    if not attacks:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(len(attacks)); w = 0.35

    best_small = (
        df_small.groupby("attack_id")
        .apply(lambda g: g.sort_values(
            ["success_rate","mean_rel_impr"], ascending=False).iloc[0])
        .reset_index(drop=True)
        .set_index("attack_id")
    )
    best_256 = (
        df_256.groupby("attack_id")
        .apply(lambda g: g.sort_values(
            ["success_rate","mean_rel_impr"], ascending=False).iloc[0])
        .reset_index(drop=True)
        .set_index("attack_id")
    )

    for ax, (col, title) in zip(axes, [
        ("success_rate",  "Success Rate"),
        ("mean_rel_impr", "Relative Improvement"),
    ]):
        s = best_small.reindex(attacks)[col].fillna(0).values
        f = best_256.reindex(attacks)[col].fillna(0).values
        ax.bar(x_pos - w/2, s, w, label=f"{cfg.resize_small}px",
               alpha=0.85, color="steelblue")
        ax.bar(x_pos + w/2, f, w, label="256px",
               alpha=0.85, color="darkorange")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(attacks, rotation=25, ha="right")
        ax.set_title(f"{title}: {cfg.resize_small}px vs 256px [{cfg.hash_id}]")
        ax.set_ylabel(title); ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = cfg.results_dir / "validation_256_vs_small.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_all(df_small: pd.DataFrame, df_256: pd.DataFrame,
             cfg: SearchConfig, hash_fn):
    """Запускает все графики последовательно."""
    plot_hash_distance_curves(df_small, cfg, hash_fn)
    plot_param_importance_numeric(df_small, cfg)
    if cfg.categorical_params:
        plot_param_importance_categorical(df_small, cfg)
    if not df_256.empty:
        plot_validation_comparison(df_small, df_256, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_report(df_best: pd.DataFrame, cfg: SearchConfig):
    """
    Печатает copy-paste конфиги для 256×256.
    Формат: готовый Python-dict для каждой атаки.
    """
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  ФИНАЛЬНЫЕ КОНФИГИ ДЛЯ 256×256  [{cfg.hash_id.upper()}]")
    print(f"  Метрика: quality_score = pixel_l2 + queries / {cfg.query_divisor:.0f}")
    print(f"{sep}\n")

    for atk_id in sorted(df_best["attack_id"].unique()):
        sub = df_best[df_best["attack_id"] == atk_id].copy()
        print(f"{'─'*40} {atk_id} {'─'*40}")
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            params = row["params"] if isinstance(row["params"], dict) \
                else json.loads(row["params"])
            sr     = row["success_rate"]
            l2     = row.get("mean_pixel_l2", float("nan"))
            q      = row.get("mean_queries", float("nan"))
            dist   = row.get("mean_final_dist", float("nan"))
            qs     = row.get("quality_score", l2 + q / cfg.query_divisor)

            status = "✅ SR=100%" if sr >= 1.0 else f"⚠️  SR={sr:.0%}"
            print(f"\n  #{rank}  {status}  |  "
                  f"dist={dist:.1f}  |  L2={l2:.4f}  |  "
                  f"queries={q:.0f}  |  score={qs:.4f}")
            print(f"  # Скопируй этот конфиг:")

            # Pretty-print dict
            lines = [f'ATTACK_CONFIGS["{cfg.hash_id}"]["{atk_id}"] = {{']
            for k, v in params.items():
                if isinstance(v, float):
                    lines.append(f'    "{k}": {v:.6g},')
                elif isinstance(v, bool):
                    lines.append(f'    "{k}": {v},')
                elif isinstance(v, str):
                    lines.append(f'    "{k}": "{v}",')
                else:
                    lines.append(f'    "{k}": {v},')
            lines.append("}")
            print("\n".join(f"  {l}" for l in lines))
        print()

    # Сводная таблица
    print(f"\n{'─'*80}")
    print("СВОДНАЯ ТАБЛИЦА (только лучший конфиг per attack):\n")
    summary = []
    for atk_id in sorted(df_best["attack_id"].unique()):
        sub = df_best[df_best["attack_id"] == atk_id]
        row = sub.iloc[0]
        summary.append({
            "attack":        atk_id,
            "SR@256":        f"{row['success_rate']:.0%}",
            "rel_impr":      f"{row.get('mean_rel_impr', 0)*100:.1f}%",
            "dist":          f"{row.get('mean_final_dist', 0):.1f}",
            "pixel_L2":      f"{row.get('mean_pixel_l2', 0):.4f}",
            "queries":       f"{row.get('mean_queries', 0):.0f}",
            "quality_score": f"{row.get('quality_score', 0):.4f}",
        })
    print(pd.DataFrame(summary).set_index("attack").to_string())
    print(f"\n{'═'*80}\n")

    # Сохраняем CSV
    out = cfg.results_dir / "best_configs_256.csv"
    df_best.to_csv(out, index=False)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> pd.DataFrame:
    """Загружает jsonl, разворачивает params в колонки p_*."""
    if not path.exists():
        return pd.DataFrame()
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            flat = {f"p_{k}": v for k, v in rec.get("params", {}).items()}
            records.append({
                **{k: v for k, v in rec.items()
                   if k not in ("timestamp",)},
                **flat,
            })
    return pd.DataFrame(records)


def load_all_results(cfg: SearchConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает и объединяет random + local search (малое разрешение).
    Возвращает (df_small, df_256).
    """
    dfs = []
    for name in ("trials.jsonl", "local_search.jsonl"):
        p = cfg.results_dir / name
        if p.exists():
            dfs.append(_load_jsonl(p))
    df_small = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    df_256 = _load_jsonl(cfg.results_dir / "validation_256.jsonl")
    return df_small, df_256


def print_summary_table(df: pd.DataFrame, label: str = ""):
    """Быстрая сводка по атакам."""
    if df.empty:
        print("No data.")
        return
    cols = [c for c in ["mean_final_dist","mean_rel_impr","success_rate"]
            if c in df.columns]
    print(f"\n{'─'*60} {label}")
    print(df.groupby("attack_id")[cols].agg(["mean","max"]).round(3).to_string())
