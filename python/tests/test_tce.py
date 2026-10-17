# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The Tessera Canonical Encoding in Python (RFC-0003), against the shared vectors.

The reader decodes every golden vector to its value and the writer encodes it back to its bytes;
every non-canonical input is refused with the code and offset RFC-0003 fixes, and every envelope
is checked in the order it fixes, as the prover's and the verifier's readers do. The
`IntegerModel` vectors that the Rust types write decode to their values, and Python writes them
back byte for byte, in memory and streamed to a secret file. The writer refuses what its readers
would refuse, where they would, casts nothing, and streams a payload without copying it.
"""

import dataclasses
import stat
import tracemalloc
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pytest
from tce_vectors import from_json, limits, load, parse, to_json

from tessera_quant import CodecError, EncodeError
from tessera_quant.integer_model import INTEGER_MODEL, IntegerModel
from tessera_quant.tce import (
    INTEGER_MODEL_LIMITS,
    LOGITS_FILE,
    MESSAGE_LIMITS,
    TOKEN_FILE,
    U8,
    Limits,
    LogitsFile,
    MapOf,
    Schema,
    Struct,
    TokenFile,
    decode,
    encode,
    open_envelope,
    read_artifact,
    seal,
    unseal,
    write_artifact,
)

GOLDEN = load("codec/golden.json")
NONCANONICAL = load("codec/noncanonical.json")
ENVELOPES = load("codec/envelope.json")
MODELS = load("model/integer-model.json")

NAMED_LIMITS = {"message": MESSAGE_LIMITS, "integer-model": INTEGER_MODEL_LIMITS}

T = TypeVar("T")


def case_id(case: dict[str, Any]) -> str:
    return f"{case['type']}:{case['hex'][:40]}"


def test_every_vector_is_read() -> None:
    # An empty or renamed list would leave its parametrized tests silently empty.
    counts = [len(GOLDEN["cases"]), len(NONCANONICAL["cases"]), len(NONCANONICAL["accepted"])]
    assert counts == [51, 43, 11]
    assert [len(ENVELOPES["cases"]), len(MODELS["cases"])] == [52, 2]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=case_id)
def test_golden_values_decode_and_encode_back_to_their_bytes(case: dict[str, Any]) -> None:
    codec = parse(case["type"])
    data = bytes.fromhex(case["hex"])
    bounds = limits(GOLDEN["limits"])
    assert to_json(codec, decode(codec, data, limits=bounds)) == case["value"]
    assert encode(codec, from_json(codec, case["value"]), limits=bounds) == data
    for cut in range(len(data)):
        with pytest.raises(CodecError) as refused:
            decode(codec, data[:cut], limits=bounds)
        assert refused.value.code == "ERR-CODEC-TRUNCATED", cut


@pytest.mark.parametrize("case", NONCANONICAL["cases"], ids=case_id)
def test_non_canonical_inputs_are_refused_where_rfc_0003_says(case: dict[str, Any]) -> None:
    codec = parse(case["type"])
    with pytest.raises(CodecError) as refused:
        decode(codec, bytes.fromhex(case["hex"]), limits=limits(NONCANONICAL["limits"]))
    found = (refused.value.code, refused.value.offset)
    assert found == (case["code"], case["offset"]), case["why"]


@pytest.mark.parametrize("case", NONCANONICAL["accepted"], ids=case_id)
def test_inputs_exactly_at_a_limit_are_accepted(case: dict[str, Any]) -> None:
    codec = parse(case["type"])
    data = bytes.fromhex(case["hex"])
    bounds = limits(NONCANONICAL["limits"])
    assert encode(codec, decode(codec, data, limits=bounds), limits=bounds) == data


def test_the_named_limits_are_rfc_0003s_defaults() -> None:
    named = {name: limits(entry) for name, entry in ENVELOPES["named_limits"].items()}
    assert named == NAMED_LIMITS


@pytest.mark.parametrize("case", ENVELOPES["cases"], ids=case_id)
def test_envelopes_are_checked_in_the_order_rfc_0003_fixes(case: dict[str, Any]) -> None:
    schema = Schema("probe", case["schema"], case["version"], parse(case["type"]))
    bounds = NAMED_LIMITS[case["limits"]] if "limits" in case else limits(ENVELOPES["limits"])
    data = bytes.fromhex(case["hex"])
    if "code" in case:
        with pytest.raises(CodecError) as refused:
            open_envelope(data, schema, limits=bounds).decode(schema.codec, limits=bounds)
        found = (refused.value.code, refused.value.offset)
        assert found == (case["code"], case["offset"]), case["why"]
        return
    envelope = open_envelope(data, schema, limits=bounds)
    value = envelope.decode(schema.codec, limits=bounds)
    assert encode(schema.codec, value) == bytes.fromhex(case["payload"])
    if envelope.version == schema.version:
        # The writer's header and digest are the Rust writer's.
        assert seal(schema, value, limits=bounds) == data


def model_case(name: str) -> dict[str, Any]:
    return next(case for case in MODELS["cases"] if case["name"] == name)


def model(name: str = "servable-pot") -> IntegerModel:
    value: IntegerModel = from_json(INTEGER_MODEL.codec, model_case(name)["value"])
    return value


@pytest.mark.parametrize("name", ["servable-pot", "minimal-affine"])
def test_the_rust_models_decode_to_their_values_and_python_writes_their_bytes(
    tmp_path: Path, name: str
) -> None:
    assert (MODELS["schema"], MODELS["version"]) == (INTEGER_MODEL.schema_id, INTEGER_MODEL.version)
    assert NAMED_LIMITS[MODELS["limits"]] == INTEGER_MODEL_LIMITS
    case = model_case(name)
    data = bytes.fromhex(case["hex"])
    decoded = unseal(data, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    assert to_json(INTEGER_MODEL.codec, decoded) == case["value"]

    written = model(name)
    assert seal(INTEGER_MODEL, written, limits=INTEGER_MODEL_LIMITS) == data
    # A map encodes in its keys' canonical order, whatever the dict's order.
    libraries = dict(reversed(written.provenance.libraries.items()))
    shuffled = dataclasses.replace(
        written, provenance=dataclasses.replace(written.provenance, libraries=libraries)
    )
    assert seal(INTEGER_MODEL, shuffled, limits=INTEGER_MODEL_LIMITS) == data

    path = tmp_path / "model.d"
    digest = write_artifact(path, INTEGER_MODEL, written, limits=INTEGER_MODEL_LIMITS)
    assert path.read_bytes() == data
    assert digest == data[-32:]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    read = read_artifact(path, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    assert to_json(INTEGER_MODEL.codec, read) == case["value"]
    # Decoded arrays are views of the input, and never a way to write to it.
    mutable = unseal(bytearray(data), INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    assert not any(w.data.flags.writeable for w in [*read.weights, *mutable.weights])


def test_a_corrupted_file_is_refused_before_anything_is_decoded(tmp_path: Path) -> None:
    data = bytearray.fromhex(model_case("servable-pot")["hex"])
    data[-100] ^= 1
    path = tmp_path / "model.d"
    path.write_bytes(data)
    with pytest.raises(CodecError) as refused:
        read_artifact(path, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    assert (refused.value.code, refused.value.offset) == ("ERR-CODEC-DIGEST", len(data) - 32)
    path.write_bytes(b"")
    with pytest.raises(CodecError) as empty:
        read_artifact(path, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    assert (empty.value.code, empty.value.offset) == ("ERR-CODEC-TRUNCATED", 0)


def test_a_memory_mapped_weight_is_streamed_and_read_back_without_a_copy(tmp_path: Path) -> None:
    rows, cols = 4096, 16384
    weight = np.memmap(tmp_path / "embed.bin", dtype=np.int8, mode="w+", shape=(rows * cols,))
    weight[:] = (np.arange(rows * cols) % 255 - 127).astype(np.int8)
    small = model()
    embed = dataclasses.replace(small.weights[0], rows=rows, cols=cols, data=weight)
    large = dataclasses.replace(small, weights=[embed, *small.weights[1:]])
    path = tmp_path / "model.d"

    tracemalloc.start()
    try:
        write_artifact(path, INTEGER_MODEL, large, limits=INTEGER_MODEL_LIMITS)
        _, written_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        read = read_artifact(path, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
        _, read_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # 64 MiB of weights went through a chunk of memory each way: about 11 and 15 KiB measured.
    assert path.stat().st_size > rows * cols
    assert written_peak < 1 << 20
    assert read_peak < 1 << 20
    assert (read.weights[0].data == weight).all()


@pytest.mark.parametrize(
    ("sequences", "bounds", "offset"),
    [
        # Five sequences: the outer length, at the payload's first byte.
        ([[1]] * 5, Limits(1000, 4, 8), 20),
        # Five tokens: the first sequence's length, after the outer one.
        ([[1, 2, 3, 4, 5]], Limits(1000, 4, 8), 28),
        # The tokens of a sequence are a third level.
        ([[7]], Limits(1000, 4, 2), 28),
        # 104 bytes of payload: a 156-byte envelope.
        ([[1, 2, 3, 4]] * 4, Limits(100, 4, 8), 12),
        # Both: the envelope's size is checked first.
        ([[1]] * 5, Limits(100, 4, 8), 12),
    ],
)
def test_the_writer_refuses_what_its_readers_would_where_they_would(
    tmp_path: Path, sequences: list[list[int]], bounds: Limits, offset: int
) -> None:
    tokens = TokenFile([np.array(s, dtype=np.uint32) for s in sequences])
    with pytest.raises(CodecError) as refused:
        write_artifact(tmp_path / "tokens.ttk", TOKEN_FILE, tokens, limits=bounds)
    assert (refused.value.code, refused.value.offset) == ("ERR-CODEC-LIMIT", offset)
    assert list(tmp_path.iterdir()) == [], "neither the file nor a temporary one"
    sealed = seal(TOKEN_FILE, tokens, limits=MESSAGE_LIMITS)
    with pytest.raises(CodecError) as read:
        unseal(sealed, TOKEN_FILE, limits=bounds)
    assert (read.value.code, read.value.offset) == ("ERR-CODEC-LIMIT", offset)


def int16_weights(m: IntegerModel) -> IntegerModel:
    data = m.weights[0].data.astype(np.int16)
    return dataclasses.replace(m, weights=[dataclasses.replace(m.weights[0], data=data)])


def matrix_weights(m: IntegerModel) -> IntegerModel:
    w = m.weights[0]
    data = w.data.reshape(w.rows, w.cols)
    return dataclasses.replace(m, weights=[dataclasses.replace(w, data=data)])


def wide_layers(m: IntegerModel) -> IntegerModel:
    return dataclasses.replace(m, arch=dataclasses.replace(m.arch, n_layers=1 << 32))


def bool_bits(m: IntegerModel) -> IntegerModel:
    public = dataclasses.replace(m.quant.public, residual_bits=True)
    return dataclasses.replace(m, quant=dataclasses.replace(m.quant, public=public))


def float_epsilon(m: IntegerModel) -> IntegerModel:
    return dataclasses.replace(m, arch=dataclasses.replace(m.arch, rms_eps=1e-05))


def accented_version(m: IntegerModel) -> IntegerModel:
    provenance = dataclasses.replace(m.provenance, tessera_quant="0.1.0-é")
    return dataclasses.replace(m, provenance=provenance)


def short_digest(m: IntegerModel) -> IntegerModel:
    return dataclasses.replace(m, provenance=dataclasses.replace(m.provenance, statistics=b"x"))


def tuple_weights(m: IntegerModel) -> IntegerModel:
    return dataclasses.replace(m, weights=tuple(m.weights))


def bare_point(m: IntegerModel) -> IntegerModel:
    public = dataclasses.replace(m.quant.public, act_exponents=[(-3, "Embed")])
    return dataclasses.replace(m, quant=dataclasses.replace(m.quant, public=public))


def foreign_arch(m: IntegerModel) -> IntegerModel:
    return dataclasses.replace(m, arch=m.arch.rope)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (int16_weights, r"Vec<i8> is a one-dimensional array of int8"),
        (matrix_weights, r"Vec<i8> is a one-dimensional array"),
        (wide_layers, r"u32 lies in \[0, 4294967295\]"),
        (bool_bits, r"u8 is an integer, not a bool"),
        (float_epsilon, r"Rational is a Fraction, never a float"),
        (accented_version, r"character 6 is not"),
        (short_digest, r"\[u8; 32\] holds 32 bytes"),
        (tuple_weights, r"a Vec is a list"),
        (bare_point, r"a value of none of the enum's variants"),
        (foreign_arch, r"ArchSpec: a value of another class"),
    ],
)
def test_a_value_its_type_cannot_hold_is_refused_and_nothing_is_written(
    tmp_path: Path, change: Any, reason: str
) -> None:
    with pytest.raises(EncodeError, match=reason) as refused:
        write_artifact(tmp_path / "model.d", INTEGER_MODEL, change(model()), limits=MESSAGE_LIMITS)
    assert refused.value.code == "ERR-INTERNAL"
    assert list(tmp_path.iterdir()) == []


@dataclasses.dataclass(frozen=True, eq=False)
class Key:
    n: int


def test_a_map_whose_keys_encode_the_same_is_refused() -> None:
    # Two keys Python tells apart but whose encodings are equal: the map would not be canonical.
    entries = MapOf(Struct(Key, [("n", U8)]), U8)
    assert encode(entries, {Key(2): 0, Key(1): 0}) == bytes.fromhex("020000000000000001000200")
    with pytest.raises(EncodeError, match="two keys of a map encode the same"):
        encode(entries, {Key(1): 0, Key(1): 1})


def round_trip(path: Path, schema: Schema[T], value: T) -> None:
    write_artifact(path, schema, value, limits=MESSAGE_LIMITS)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    read = read_artifact(path, schema, limits=MESSAGE_LIMITS)
    assert to_json(schema.codec, read) == to_json(schema.codec, value)


def test_token_and_logits_files_round_trip_as_secret_artifacts(tmp_path: Path) -> None:
    tokens = TokenFile([np.array([128000, 9906, 1917], dtype=np.uint32), np.array([], np.uint32)])
    round_trip(tmp_path / "tokens.ttk", TOKEN_FILE, tokens)
    logits = LogitsFile(4, [np.array([-(1 << 40), 0, 7, (1 << 40) - 1], dtype=np.int64)])
    round_trip(tmp_path / "logits.tlg", LOGITS_FILE, logits)


def test_no_repr_shows_a_weight_a_channel_parameter_a_token_or_a_logit() -> None:
    m = model("minimal-affine")
    shown = repr(m)
    for hidden in ("data=", "class_=", "multiplier=", "array("):
        assert hidden not in shown
    assert "rows=6" in shown
    secret = TokenFile([np.array([123456789], dtype=np.uint32)])
    assert "123456789" not in repr(secret)
    assert "987654321" not in repr(LogitsFile(1, [np.array([987654321], dtype=np.int64)]))
