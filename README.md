# EvoHash

EvoHash is a research codebase for evaluating and improving black-box attacks
against perceptual image hashing systems. It includes wrappers for several hash
functions, baseline attacks, hyperparameter search, LLM-guided evolution, and
benchmark tooling.

The repository intentionally keeps generated artifacts, notebooks, datasets,
thesis files, and local experiment outputs out of git. Curated attack
configuration files are tracked under `results/`.

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

- datasets under `data/`
- notebooks (`*.ipynb`) unless manually force-added
- experiment outputs under `outputs/`
- generated results under `results/*`, except curated config JSON files
- thesis/defense/figure legacy workspaces
- checked-out third-party source trees

If a notebook should become part of the repository, add it explicitly:

```powershell
git add -f path\to\notebook.ipynb
```

## Setup

Create a Python environment, install the project dependencies, and prepare the
local GigaEvolve checkout described in `third_party/README.md`.

Typical local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GigaEvolve-based stage3 runs:

```powershell
Get-Content third_party\README.md
```

If you use GigaEvolve with Redis:

```powershell
docker compose up -d redis
```

Datasets are local. Put the public dataset archive under `data/`, for example:

```text
data/dataset_v1_public.zip
```

When running modules from this package directory, make sure Python can import
`evohash`; from PowerShell this usually means setting `PYTHONPATH` to the parent
directory:

```powershell
$env:PYTHONPATH = (Get-Location).Parent.FullName
```

## Quick Checks

Compile the tracked Python modules:

```powershell
python -m compileall -q attacks hashes stage1_find_ASR stage3_common stage3_gigaevolve stage3_openevolve stage4_benchmark evolved_programs evaluator.py metrics.py optimizer.py parallel_eval.py
```

Run a dry GigaEvolve package generation:

```powershell
python -m evohash.stage3_gigaevolve.run --hash phash --dataset data/dataset_v1_public.zip --dry-run
```

Run an OpenEvolve package generation:

```powershell
python -m evohash.stage3_openevolve.run --hash phash --dataset data/dataset_v1_public.zip --dry-run
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
