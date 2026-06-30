"""Build inlined seed attack programs for evolution."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

from evohash.attacks import build_attack_registry


def build_inlined_attack_source(
    *,
    attack_id: str,
    params: dict[str, Any],
    use_evolve_block: bool = False,
    include_compat_run: bool = False,
) -> str:
    """Return the actual attack module source as a mutable seed program.

    The generated program keeps the production ``run_attack`` contract but
    replaces DEFAULT_PARAMS with the optimized seed params.  The old notebook
    compatibility ``run(context, ...)`` helper is omitted by default to reduce
    prompt noise and keep OpenEvolve programs below code-length limits.
    """
    module_name = build_attack_registry().get(attack_id).run_attack.__module__
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ValueError(f"Cannot locate source file for attack {attack_id!r}")

    source = Path(module_file).read_text(encoding="utf-8", errors="replace")
    source = _rewrite_source(
        source,
        attack_id=attack_id,
        params=params,
        include_compat_run=include_compat_run,
    )
    header = (
        f"# Inlined EvoHash seed attack: {attack_id}\n"
        "# Generated for evolution so the LLM can mutate the actual attack logic.\n"
        "# The evaluator calls run_attack(x_source, oracle, params, budget, seed).\n"
    )
    source = header + source.lstrip()
    if use_evolve_block:
        source = "# EVOLVE-BLOCK-START\n" + source.rstrip() + "\n# EVOLVE-BLOCK-END\n"
    return source.rstrip() + "\n"


def _rewrite_source(
    source: str,
    *,
    attack_id: str,
    params: dict[str, Any],
    include_compat_run: bool,
) -> str:
    tree = ast.parse(source)
    edits: list[tuple[int, int, list[str]]] = []
    for node in tree.body:
        if _assigns_name(node, "ATTACK_ID"):
            edits.append((node.lineno - 1, node.end_lineno or node.lineno, [f"ATTACK_ID = {attack_id!r}"]))
        elif _assigns_name(node, "DEFAULT_PARAMS"):
            edits.append(
                (
                    node.lineno - 1,
                    node.end_lineno or node.lineno,
                    [f"DEFAULT_PARAMS: dict[str, Any] = {_pretty_params(params)}"],
                )
            )
        elif (
            not include_compat_run
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ):
            edits.append((node.lineno - 1, node.end_lineno or node.lineno, []))

    lines = source.splitlines()
    for start, end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
        lines[start:end] = replacement
    out = "\n".join(lines).rstrip() + "\n"
    ast.parse(out)
    return out


def _assigns_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    return False


def _pretty_params(params: dict[str, Any]) -> str:
    if not params:
        return "{}"
    parts = [f"\n    {key!r}: {value!r}," for key, value in sorted(params.items())]
    return "{" + "".join(parts) + "\n}"
