# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tessera's quantization front-end: from a Llama-3.1 checkpoint to the integer model `D`.

RFC-0005 specifies `D`, its quantization and its calibration. `import_checkpoint` reads a
checkpoint, untrusted input, from safetensors files only and checks it against `ArchSpec`
(RFC-0005 'Pipeline', step 1). `tessera_quant.calibration` draws the calibration set exactly
from a seed and a pinned dataset, and checks evaluation data disjoint from it (RFC-0005
'Calibration'). `tessera_quant.capture` runs the calibration set through the BF16 reference
and keeps each quantization point's per-channel maxima in a secret statistics file, with a
`tessera.calibration/1` report beside it. Quantization follows.
"""

from importlib.metadata import version

from tessera_quant.arch import ArchSpec, RopeSpec, read_config, tensor_shapes
from tessera_quant.checkpoint import FloatModel, import_checkpoint
from tessera_quant.errors import ArchitectureError, CalibrationError, TesseraQuantError

__all__ = [
    "ArchSpec",
    "ArchitectureError",
    "CalibrationError",
    "FloatModel",
    "RopeSpec",
    "TesseraQuantError",
    "__version__",
    "import_checkpoint",
    "read_config",
    "tensor_shapes",
]

__version__: str = version("tessera-quant")
"""The prover release train's version (09 §1), read from the installed distribution."""
