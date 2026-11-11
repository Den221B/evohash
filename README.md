# EvoHash

EvoHash is a research codebase for evaluating and improving black-box attacks
against perceptual image hashing systems. It includes wrappers for several hash
functions, baseline attacks, hyperparameter search, LLM-guided evolution, and
benchmark tooling.

The repository intentionally keeps generated artifacts, research scratch
notebooks, thesis files, and local experiment outputs out of git. Tracked are
the datasets (`data/`), curated attack configuration files (`results/`), and
one example notebook per stage (see [Notebooks](#notebooks)).

## What Is Included

- `hashes/` - hash wrappers for pHash, PDQ, NeuralHash, PhotoDNA, and defense wrappers.
- `attacks/` - baseline black-box attack implementations.
- `evaluator.py`, `parallel_eval.py` - single-run and parallel evaluation loops.
- `optimizer.py` - hyperparameter optimization around baseline attacks.
- `stage1_find_ASR/` - search utilities for attack success-rate screening.
- `stage3_common/` - shared helpers for LLM-generated attack programs.
- `stage3_gigaevolve/` - GigaEvolve integration for program evolution.
- `stage3_openevolve/` - OpenEvolve integration for program evolution.
- `stage4_benchmark/` - benchmark, comparison, defense, and plotting utilities.
- `evolved_programs/` - selected evolved attack programs.
- `results/attack_configs.json` - curated attack configs.
- `results/attack_configs_optimized.json` - optimized curated attack configs.
- `third_party/` - setup notes and patch for local third-party dependencies.

## Local-Only Files

The following are intentionally ignored:

- other datasets and archives (`*.zip`, `data/` except the two tracked archives)
- notebooks (`*.ipynb`) except the curated stage notebooks listed below
- experiment outputs under `outputs/`
- generated results under `results/*`, except curated config JSON files
- thesis/defense/figure legacy workspaces
- checked-out third-party source trees

If a notebook should become part of the repository, add it explicitly:

```powershell
git add -f path\to\notebook.ipynb
```

## Setup

Tested with Python 3.12 on Windows (PhotoDNA needs Windows: it ships as a DLL).

The code is imported as the package `evohash`, so the checkout folder must be
named `evohash` (the default for `git clone`). The GigaEvolve checkout must
exist before installing requirements, because `requirements.txt` installs it in
editable mode.

```powershell
git clone https://github.com/Den221B/evohash.git
cd evohash

# GigaEvolve core + EvoHash patch (details: third_party/README.md)
git clone https://github.com/FusionBrainLab/gigaevo-core third_party/gigaevolve_core
git -C third_party/gigaevolve_core checkout 7a3ab40
git -C third_party/gigaevolve_core apply ../gigaevo-core-evohash.patch

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the CPU build of torch from PyPI. That is enough for
everything here (torch is only used for LPIPS; NeuralHash runs on onnxruntime).

GigaEvolve stores its population in Redis:

```powershell
docker compose up -d redis
```

LLM keys go into `.env` in the repo root (see `.env.example`).

### Datasets

Both dataset archives are tracked under `data/`:

- `data/dataset_v1_public.zip` - public split, 30 pairs (tuning, evolution).
- `data/dataset_v1_full.zip` - public + secret split, 100 secret pairs (final benchmark).

When running modules from the command line, make sure Python can import
`evohash` by putting the parent directory on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Get-Location).Parent.FullName
```

## Notebooks

One notebook per stage; each has a config cell at the top (`HASH_ID`, budget,
pairs, LLM). Open them from their own folder.

- `stage2_optimize/optimize.ipynb` - Optuna tuning of baseline attack configs
  (`pdq`, `phash`, `photodna`).
- `stage2_optimize/NeuralHash_optimize.ipynb` - the same for NeuralHash
  (separate search space and process settings).
- `stage3_openevolve/OpenEvolve.ipynb` - LLM-guided evolution with OpenEvolve.
- `stage3_gigaevolve/GigaEvolve.ipynb` - LLM-guided evolution with GigaEvolve
  (needs Redis).
- `stage4_benchmark/Benchmark.ipynb` - final benchmark: original vs optimized
  configs vs evolved programs (`split="public"` or `"secret"`).

## Quick Checks

Run from the repository folder with `PYTHONPATH` set as above.

Compile the tracked Python modules:

```powershell
python -m compileall -q attacks hashes stage1_find_ASR stage3_common stage3_gigaevolve stage3_openevolve stage4_benchmark evolved_programs evaluator.py metrics.py optimizer.py parallel_eval.py
```

Run a dry GigaEvolve package generation:

```powershell
python -m evohash.stage3_gigaevolve.run --hash phash --dataset data/dataset_v1_public.zip --dry-run
```

Run a dry OpenEvolve package generation:

```powershell
python -m evohash.stage3_openevolve.run --hash phash --dataset data/dataset_v1_public.zip --llm-source none --dry-run
```

## Metrics

README should not be the source of truth for headline numbers. Exact results
depend on dataset split, resize settings, budget, random seed, backend, and hash
thresholds. Keep reproducible result tables in benchmark outputs or paper/release
artifacts.

The main metrics reported by the evaluation pipeline are:

- `success` / ASR - fraction of runs where final hash distance is below the threshold.
- `initial_hash_l1`, `final_hash_l1`, `best_hash_l1` - hash-space distances.
- `relative_improvement` - normalized distance reduction from source to target hash.
- `pixel_l2` and `pixel_l2_raw` - image-space perturbation costs.
- `queries` - number of hash queries spent by the attack.
- `time_sec` - wall-clock runtime.
- `lpips` - optional perceptual distance, computed only when explicitly enabled.

Use `stage4_benchmark/` for reproducible comparisons and keep only curated,
portable configuration summaries in git.

## Third-party Notes

`third_party/gigaevolve_core/` is not vendored into this repository. The tracked
patch in `third_party/gigaevo-core-evohash.patch` documents the temporary local
changes needed by EvoHash. See `third_party/README.md` for exact commands.

Longer term, the preferred solution is a fork or submodule pinned to a reviewed
GigaEvolve commit.
