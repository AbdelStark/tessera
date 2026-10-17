# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tessera's quantization front-end: from a Llama-3.1 checkpoint to the integer model `D`.

RFC-0005 specifies `D`, its quantization and its calibration. `import_checkpoint` reads a
checkpoint, untrusted input, from safetensors files only and checks it against `ArchSpec`
(RFC-0005 'Pipeline', step 1). `tessera_quant.calibration` draws the calibration set exactly
from a seed and a pinned dataset, and checks evaluation data disjoint from it (RFC-0005
'Calibration'). `tessera_quant.capture` runs the calibration set through the BF16 reference
and keeps each quantization point's per-channel maxima in a secret statistics file, with a
`tessera.calibration/1` report beside it. `tessera_quant.migration` folds the RMSNorm gains,
equalizes queries and keys, and migrates activation outliers into the weights, with the
statistics that follow (RFC-0005 'Pipeline', steps 2 and 4). `tessera_quant.quantize` chooses
`D`'s scales and shifts and rounds its weights to INT8 (RFC-0005 'Scale policy', step 5).

`D` reaches the Rust tools as a TCE artifact (RFC-0003): `tessera_quant.tce` reads and writes
them, and `tessera_quant.integer_model` defines `D`. `tessera_quant.tables` generates the lookup
tables `D` uses and their digests, equal to the Rust generators' (RFC-0008).
"""

from importlib.metadata import version

from tessera_quant.arch import ArchFamily, ArchSpec, RopeSpec, read_config, tensor_shapes
from tessera_quant.checkpoint import FloatModel, import_checkpoint
from tessera_quant.errors import (
    ArchitectureError,
    CalibrationError,
    CodecError,
    EncodeError,
    TableError,
    TesseraQuantError,
)

__all__ = [
    "ArchFamily",
    "ArchSpec",
    "ArchitectureError",
    "CalibrationError",
    "CodecError",
    "EncodeError",
    "FloatModel",
    "RopeSpec",
    "TableError",
    "TesseraQuantError",
    "__version__",
    "import_checkpoint",
    "read_config",
    "tensor_shapes",
]

__version__: str = version("tessera-quant")
"""The prover release train's version (09 §1), read from the installed distribution."""
