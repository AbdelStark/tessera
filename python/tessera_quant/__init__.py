# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tessera's quantization front-end: from a Llama-3.1 checkpoint to the integer model `D`.

RFC-0005 specifies `D`, its quantization and its calibration. This version is the package's
skeleton; the checkpoint import, calibration and quantization modules follow.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__: str = version("tessera-quant")
"""The prover release train's version (09 §1), read from the installed distribution."""
