# Third-party setup

This repository does not vendor third-party source trees directly. The
`gigaevolve_core` checkout is kept local and ignored by git, while the small
EvoHash-specific downstream patch is tracked here.

## GigaEvolve Core

Clone upstream at the version used during EvoHash experiments:

```powershell
git clone https://github.com/FusionBrainLab/gigaevo-core third_party/gigaevolve_core
git -C third_party/gigaevolve_core checkout 8d1354d
git -C third_party/gigaevolve_core apply ../gigaevo-core-evohash.patch
python -m pip install -e third_party/gigaevolve_core
```

The patch adds the local/OpenRouter LLM configs and a few runtime fixes needed
by the stage3 EvoHash evolution pipeline. It is a temporary lightweight
alternative to a proper fork/submodule. If these changes become long-lived,
promote them to a forked branch and pin that commit from the main repo.

Useful environment variables:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENROUTER_API_KEY = "..."
```
