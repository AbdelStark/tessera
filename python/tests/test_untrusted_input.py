# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Checkpoints are untrusted (06 §3, TB-5): no module of the package can deserialize by pickle.

Every source file is parsed, and none may import a pickle-based module or call `torch.load`,
`numpy.load`, `eval` or `exec`, whatever their arguments.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "tessera_quant"
MODULES = {"pickle", "cPickle", "_pickle", "dill", "joblib", "marshal", "shelve", "cloudpickle"}
LOADERS = {("torch", "load"), ("numpy", "load"), ("np", "load")}


def violations(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in MODULES]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in MODULES:
                found.append(root)
            if root in {"torch", "numpy"}:
                found += [f"{root}.{a.name}" for a in node.names if a.name == "load"]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in LOADERS:
                found.append(f"{node.value.id}.{node.attr}")
        elif isinstance(node, ast.Name) and node.id in {"eval", "exec"}:
            found.append(node.id)
    return found


def test_no_module_deserializes_by_pickle() -> None:
    sources = sorted(PACKAGE.rglob("*.py"))
    assert sources, "the package's sources are found"
    for path in sources:
        assert violations(path.read_text(encoding="utf-8")) == [], path.name


def test_the_scan_finds_each_form() -> None:
    for source in [
        "import pickle",
        "from pickle import loads",
        "import dill.settings",
        "import torch\ntorch.load(f)",
        "from torch import load",
        "import numpy as np\nnp.load(f)",
        "eval('1')",
        "exec(code)",
    ]:
        assert violations(source), source
    assert violations("from safetensors import safe_open\nimport torch\ntorch.save") == []
