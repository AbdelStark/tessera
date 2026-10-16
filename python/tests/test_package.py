# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The installed distribution and its lock (RFC-0005 'Python packaging')."""

import tomllib
from importlib import metadata, resources
from pathlib import Path

import tessera_quant

LOCK = Path(__file__).parents[1] / "uv.lock"


def test_the_package_is_typed_and_versioned_by_its_distribution() -> None:
    assert tessera_quant.__version__ == metadata.version("tessera-quant")
    assert resources.files(tessera_quant).joinpath("py.typed").is_file()


def test_cpu_and_cuda_torch_are_one_release() -> None:
    # Calibration on CPU and on GPU must run the same PyTorch, or the two could derive different
    # scales and so different integer models D.
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    torch = [p for p in lock["package"] if p["name"] == "torch"]
    indexes = {p["source"]["registry"] for p in torch}
    assert indexes == {
        "https://download.pytorch.org/whl/cpu",
        "https://download.pytorch.org/whl/cu130",
    }
    releases = {p["version"].split("+")[0] for p in torch}
    assert len(releases) == 1, f"torch releases differ between the extras: {sorted(releases)}"
