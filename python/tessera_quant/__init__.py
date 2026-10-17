# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tessera's quantization front-end: from a Llama-3.1 checkpoint to the integer model `D`.

RFC-0005 specifies `D`, its quantization and its calibration. `import_checkpoint` reads a
checkpoint, untrusted input, from safetensors files only and checks it against `ArchSpec`
(RFC-0005 'Pipeline', step 1). `tessera_quant.calibration` draws the calibration set exactly
from a seed and a pinned dataset, and checks evaluation data disjoint from it (RFC-0005
'Calibration'). `calibrate` runs the calibration set through the BF16 reference and keeps each
quantization point's per-channel maxima, which `tessera_quant.capture.write_calibration` writes
to a secret statistics file with a `tessera.calibration/1` report beside it.
`tessera_quant.migration` folds the RMSNorm gains, equalizes queries and keys, and migrates
activation outliers into the weights, with the statistics that follow (RFC-0005 'Pipeline',
steps 2 and 4). `tessera_quant.scales` chooses `D`'s scales and shifts and rounds its weights to
INT8 (RFC-0005 'Scale policy', step 5). `quantize` chooses the migration strength and assembles
`D` with its provenance, and `export` writes it (steps 4 to 6). `evaluate_fidelity` measures how
closely `D` follows the BF16 reference, with confidence intervals (RFC-0005 'Fidelity
evaluation').

`D` reaches the Rust tools as a TCE artifact (RFC-0003): `tessera_quant.tce` reads and writes
them, and `tessera_quant.integer_model` defines `D`. `tessera_quant.tables` generates the lookup
tables `D` uses and their digests, equal to the Rust generators' (RFC-0008). `run_reference`, from
`tessera_quant.reference`, computes `D`'s logits in int64, the integer reference that the Rust
engine is checked against bit for bit (SC-7).

`tessera_quant.cli` is the `tessera-quant` command line, which runs the pipeline from files (02
§2.1).
"""

from importlib.metadata import version

from tessera_quant.arch import ArchFamily, ArchSpec, RopeSpec, read_config, tensor_shapes
from tessera_quant.assemble import export, quantize
from tessera_quant.capture import CalibrationResult, calibrate
from tessera_quant.checkpoint import FloatModel, import_checkpoint
from tessera_quant.errors import (
    ArchitectureError,
    CalibrationError,
    CodecError,
    EncodeError,
    TableError,
    TesseraQuantError,
    UsageError,
)
from tessera_quant.fidelity import EvalSet, EvalSpec, FidelityReport, evaluate_fidelity
from tessera_quant.reference import run_reference

__all__ = [
    "ArchFamily",
    "ArchSpec",
    "ArchitectureError",
    "CalibrationError",
    "CalibrationResult",
    "CodecError",
    "EncodeError",
    "EvalSet",
    "EvalSpec",
    "FidelityReport",
    "FloatModel",
    "RopeSpec",
    "TableError",
    "TesseraQuantError",
    "UsageError",
    "__version__",
    "calibrate",
    "evaluate_fidelity",
    "export",
    "import_checkpoint",
    "quantize",
    "read_config",
    "run_reference",
    "tensor_shapes",
]

__version__: str = version("tessera-quant")
"""The prover release train's version (09 §1), read from the installed distribution."""
