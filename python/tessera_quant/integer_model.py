# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The integer model `D` (03 §2): artifact `0x0101`, and its TCE codec (RFC-0003).

`IntegerModel` is what the front-end writes and the Rust tools read: the architecture, the
dataflow profile, every quantization constant, the INT8 weights and the provenance, each an
integer or an exact rational (INV-DM-6). Its classes follow 03 §2 field for field, and
`INTEGER_MODEL` reads and writes them in that order, as `spec/vectors/model/` pins. Write one
of any size with `tce.write_artifact(path, INTEGER_MODEL, model, limits=INTEGER_MODEL_LIMITS)`.

Weights and channel parameters are secret-class (05 §3): no `repr` shows their entries.
"""

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from tessera_quant.arch import ArchFamily, ArchSpec, RopeSpec
from tessera_quant.tce import (
    ASCII,
    DIGEST,
    I8,
    RATIONAL,
    U8,
    U16,
    U32,
    U64,
    IntVec,
    MapOf,
    Pair,
    Schema,
    Seq,
    Struct,
    Tagged,
    UnitEnum,
    Variant,
)


class Profile(Enum):
    """The dataflow of `D` (RFC-0005 'Profiles')."""

    Servable = 0
    """Every matmul operand INT8 and every accumulator INT32: exact mode is available."""
    Minimal = 1
    """The source's commitment-minimal dataflow."""


class WeightType(Enum):
    """A weight matrix of `D` (03 §2): the seven of a layer, the embedding and the LM head."""

    Embed = 0
    Q = 1
    K = 2
    V = 3
    # 03 §2 names the output projection `O`.
    O = 4  # noqa: E741
    Gate = 5
    Up = 6
    Down = 7
    LmHead = 8


class Site(Enum):
    """A rescale of a layer, in the order of RFC-0005 'Integer semantics' (03 §2)."""

    AttnNorm = 0
    """The attention norm's output."""
    RopeQ = 1
    """The queries after RoPE; their classes follow `W_Q`'s output channels."""
    RopeK = 2
    """The keys after RoPE; their classes follow `W_K`'s output channels."""
    V = 3
    """The values, `v = rq(acc, r_v,c)`; classes from `W_V`."""
    AttnOut = 4
    """The attention output, `o8 = rq(o, r_o)`; servable only."""
    AttnResidual = 5
    """The O projection's accumulator into the residual; classes from `W_O`."""
    MlpNorm = 6
    """The MLP norm's output."""
    Silu = 7
    """SiLU of the gate's accumulator, through the `Silu` table; classes from `W_Gate`."""
    Up = 8
    """The up projection, `u8 = rq(u, r_u)`; servable only; classes from `W_Up`."""
    Product = 9
    """The product, `h = rq(a · u8, r_h)`; servable only."""
    MlpResidual = 10
    """The down projection's accumulator into the residual; classes from `W_Down`."""


@dataclass(frozen=True)
class EmbedPoint:
    """`QuantPoint::Embed`: the INT8 embedding rows `x0`."""


@dataclass(frozen=True)
class LayerPoint:
    """`QuantPoint::Layer`: a rescale of a layer."""

    layer: int
    site: Site


@dataclass(frozen=True)
class FinalNormPoint:
    """`QuantPoint::FinalNorm`: the final norm's output, the LM head's input."""


QuantPoint: TypeAlias = EmbedPoint | LayerPoint | FinalNormPoint
"""A quantized or rescaled tensor of `D` (03 §2), ordered as declared."""


@dataclass(frozen=True)
class PublicQuantSpec:
    """The quantization data the claim list reveals (RFC-0007 'Schema', 03 §2 'Quantization')."""

    act_exponents: list[tuple[QuantPoint, int]]
    """The exponent `e` of each INT8 tensor's scale `2^e`, sorted by point."""
    residual_exponent: int
    residual_bits: int
    rsqrt_kappa: int
    rms_eps_int: int
    """RMSNorm's `ε` in the residual's integer units, rounded up (RFC-0005)."""
    exp_sigma: int
    score_scales: list[tuple[int, Fraction]]
    """Each layer's attention score scale, exactly."""
    shift_class_sets: list[tuple[QuantPoint, list[int]]]
    """Each rescale site's class shifts `r`, from 1 to 63, at most four."""
    rope: RopeSpec
    logits_bits: int


@dataclass(frozen=True, eq=False)
class ChannelParams:
    """A weight's hidden per-output-channel parameters (RFC-0012 'Layout'): secret-class.

    `class_` holds the field 03 §2 names `class`, a Python keyword.
    """

    ty: WeightType
    layer: int
    class_: npt.NDArray[np.uint8] = field(repr=False)
    """Each output channel's index into its site's shift classes."""
    multiplier: npt.NDArray[np.uint16] = field(repr=False)
    """Each output channel's multiplier `M_c` in `affine` mode; empty in `pot` mode."""


@dataclass(frozen=True, eq=False)
class QuantSpec:
    """Every quantization constant of `D` (03 §2 'Quantization')."""

    public: PublicQuantSpec
    silu_in_exponents: list[tuple[int, int]]
    """Each layer's `Silu` table input exponent, sorted by layer: public, in the table's spec."""
    channels: list[ChannelParams]
    """The hidden part, sorted by `(ty, layer)`."""


@dataclass(frozen=True, eq=False)
class WeightTensor:
    """One INT8 weight matrix, row-major, as a flat array: secret-class.

    Its rows are its output channels, so `Embed` holds the transposed embedding: `d_model` rows of
    `vocab` entries.
    """

    ty: WeightType
    layer: int
    """The layer, 0 for `Embed` and `LmHead`."""
    rows: int
    """The output channels."""
    cols: int
    """The input channels."""
    data: npt.NDArray[np.int8] = field(repr=False)
    """`rows · cols` entries in `[-127, 127]`."""


@dataclass(frozen=True)
class CalibrationIdentity:
    """The calibration data's identity (RFC-0005 'Calibration')."""

    dataset: str
    config: str
    split: str
    revision: str
    """The dataset's commit, 40 hex digits."""
    files: list[str]
    """The files read, in order."""
    seed: int
    sequences: int
    held_out: int
    tokens_per_sequence: int
    selection: bytes
    """The selection digest."""


@dataclass(frozen=True)
class Provenance:
    """Where `D` comes from (RFC-0005 'Determinism')."""

    tessera_quant: str
    """The version of `tessera-quant` that wrote `D`."""
    checkpoint: list[tuple[str, bytes]]
    """Each safetensors file's name and BLAKE3 digest, sorted by name."""
    calibration: CalibrationIdentity
    statistics: bytes
    """The statistics digest of RFC-0005 'Calibration'."""
    alpha: Fraction
    """The migration strength chosen."""
    libraries: dict[str, str]
    """Library versions, by name."""


@dataclass(frozen=True, eq=False)
class IntegerModel:
    """The integer model `D`, artifact `0x0101` (03 §2)."""

    arch: ArchSpec
    profile: Profile
    quant: QuantSpec
    weights: list[WeightTensor]
    """The weights, ordered by `(ty, layer)`."""
    provenance: Provenance


_SITE = UnitEnum(Site)
_WEIGHT_TYPE = UnitEnum(WeightType)

_QUANT_POINT = Tagged[QuantPoint](
    [
        Variant("Embed", EmbedPoint),
        Variant("Layer", LayerPoint, [("layer", U32), ("site", _SITE)]),
        Variant("FinalNorm", FinalNormPoint),
    ]
)

_ROPE = Struct(
    RopeSpec,
    [
        ("theta", U32),
        ("factor", U16),
        ("low_freq_factor", U16),
        ("high_freq_factor", U16),
        ("original_max", U32),
        ("bits", U8),
    ],
)

_ARCH = Struct(
    ArchSpec,
    [
        ("family", UnitEnum(ArchFamily)),
        ("n_layers", U32),
        ("d_model", U32),
        ("n_heads", U32),
        ("n_kv_heads", U32),
        ("d_head", U32),
        ("d_ff", U32),
        ("vocab", U32),
        ("max_context", U32),
        ("rope", _ROPE),
        ("rms_eps", RATIONAL),
    ],
)

_PUBLIC = Struct(
    PublicQuantSpec,
    [
        ("act_exponents", Seq(Pair(_QUANT_POINT, I8))),
        ("residual_exponent", I8),
        ("residual_bits", U8),
        ("rsqrt_kappa", U8),
        ("rms_eps_int", U64),
        ("exp_sigma", U8),
        ("score_scales", Seq(Pair(U32, RATIONAL))),
        ("shift_class_sets", Seq(Pair(_QUANT_POINT, Seq(I8)))),
        ("rope", _ROPE),
        ("logits_bits", U8),
    ],
)

_CHANNELS = Struct(
    ChannelParams,
    [
        ("ty", _WEIGHT_TYPE),
        ("layer", U32),
        ("class_", IntVec(np.uint8)),
        ("multiplier", IntVec(np.uint16)),
    ],
)

_WEIGHT = Struct(
    WeightTensor,
    [
        ("ty", _WEIGHT_TYPE),
        ("layer", U32),
        ("rows", U32),
        ("cols", U32),
        ("data", IntVec(np.int8)),
    ],
)

_CALIBRATION = Struct(
    CalibrationIdentity,
    [
        ("dataset", ASCII),
        ("config", ASCII),
        ("split", ASCII),
        ("revision", ASCII),
        ("files", Seq(ASCII)),
        ("seed", U64),
        ("sequences", U32),
        ("held_out", U32),
        ("tokens_per_sequence", U32),
        ("selection", DIGEST),
    ],
)

_PROVENANCE = Struct(
    Provenance,
    [
        ("tessera_quant", ASCII),
        ("checkpoint", Seq(Pair(ASCII, DIGEST))),
        ("calibration", _CALIBRATION),
        ("statistics", DIGEST),
        ("alpha", RATIONAL),
        ("libraries", MapOf(ASCII, ASCII)),
    ],
)

INTEGER_MODEL: Schema[IntegerModel] = Schema(
    "IntegerModel",
    0x0101,
    1,
    Struct(
        IntegerModel,
        [
            ("arch", _ARCH),
            ("profile", UnitEnum(Profile)),
            (
                "quant",
                Struct(
                    QuantSpec,
                    [
                        ("public", _PUBLIC),
                        ("silu_in_exponents", Seq(Pair(U32, I8))),
                        ("channels", Seq(_CHANNELS)),
                    ],
                ),
            ),
            ("weights", Seq(_WEIGHT)),
            ("provenance", _PROVENANCE),
        ],
    ),
)
"""`IntegerModel`, as its writers emit it: read and write it under `INTEGER_MODEL_LIMITS`."""
