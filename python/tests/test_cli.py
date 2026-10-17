# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The `tessera-quant` command line (02 §2.1) on a tiny Llama-3.1 checkpoint and its tokenizer.

- **compile** draws the calibration set its specification names with `--seed` and writes `D`, a
  secret artifact whose digest it prints, with the calibration's report. The same seed gives the
  same bytes, and another seed another calibration set.
- **eval** measures `D` against its checkpoint. Its perplexity documents come from the very file
  `D` was calibrated on: nine stories, long enough for a calibration sequence, which the
  calibration set therefore holds whatever its seed, and eight notes too short for one. The
  command redraws the calibration set from `D`'s provenance and keeps its documents out, so the
  eight documents drawn are the notes, as the report's token count shows. A source that does not
  give the calibration set back is refused.
- **reference** writes, as a `LogitsFile`, `run_reference`'s logits for every position of every
  sequence of a `TokenFile`, in order.
- **Failures** exit with their code's status (04 §3): usage 64, a malformed artifact or a
  reference other than `D`'s checkpoint 65, an unwritable output 20. The table is the golden
  file's, `spec/vectors/errors/codes.json`.
- **RFC-0005's calibration** is the committed `docs/perf/examples/calibration-spec-c4.json`.
- **`python -m tessera_quant`** runs the same command line and exits with its status.
"""

import gzip
import io
import json
import stat
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from checkpoints import TINY, char_tokenizer, llama_like, write

from tessera_quant.calibration import (
    C4_DATASET,
    C4_REVISION,
    C4_VALIDATION_FILES,
    DEFAULT_COUNT,
    DEFAULT_HELD_OUT,
    DEFAULT_LENGTH,
)
from tessera_quant.cli import EXIT_CODES, main, read_calibration_spec
from tessera_quant.integer_model import INTEGER_MODEL
from tessera_quant.reference import run_reference
from tessera_quant.tce import (
    INTEGER_MODEL_LIMITS,
    LOGITS_FILE,
    MESSAGE_LIMITS,
    TOKEN_FILE,
    TokenFile,
    read_artifact,
    write_artifact,
)

ROOT = Path(__file__).parents[2]
SOURCE = {"name": "fixture", "config": "en", "split": "validation", "revision": "0" * 40}
STORIES = [f"story {i}: the quick brown fox jumps over the lazy dog." for i in range(9)]
"""Nine documents of at least 15 tokens: the calibration set's six and three held out."""
NOTES = [f"note {i}." for i in range(8)]
"""Eight documents of 7 tokens, too short for a calibration sequence of 16."""


def run(*argv: str | Path) -> tuple[int, dict[str, Any]]:
    """The command line's exit status on `argv`, with `--json`, and the object it printed."""
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        status = main([*map(str, argv), "--json"])
    return status, json.loads(out.getvalue())


def jsonl(path: Path, texts: list[str]) -> None:
    lines = "".join(json.dumps({"text": t}) + "\n" for t in texts)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(lines.encode("utf-8")))
    else:
        path.write_text(lines, encoding="utf-8")


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with its tokenizer, the datasets, and the specifications."""
    root = tmp_path_factory.mktemp("cli")
    checkpoint = write(root / "checkpoint", TINY | {"bos_token_id": 1}, llama_like(5))
    char_tokenizer(checkpoint)
    mixed = [text for pair in zip(STORIES, [*NOTES, ""], strict=True) for text in pair if text]
    jsonl(root / "calibration.jsonl", mixed)
    jsonl(
        root / "prompts.jsonl.gz", [f"{i} lorem ipsum dolor sit amet, {i * 7}" for i in range(12)]
    )
    calibration = SOURCE | {"files": ["calibration.jsonl"]}
    spec = {"schema": "tessera.calibration-spec/1", "source": calibration}
    sizes = {"sequences": 6, "tokens_per_sequence": 16, "held_out": 3}
    (root / "calibration.json").write_text(json.dumps(spec | sizes), encoding="utf-8")
    evaluation = {
        "schema": "tessera.eval-spec/1",
        "calibration": calibration,
        "documents": {"source": calibration, "count": 8, "max_tokens": 24},
        "prompts": {
            "source": SOURCE | {"files": ["prompts.jsonl.gz"]},
            "count": 6,
            "tokens": 8,
        },
        "continuation": 4,
        "resamples": 200,
    }
    (root / "eval.json").write_text(json.dumps(evaluation), encoding="utf-8")
    return root


def compile_d(tree: Path, out: str, seed: int, *extra: str | Path) -> tuple[int, dict[str, Any]]:
    return run(
        "compile",
        "--checkpoint",
        tree / "checkpoint",
        "--calibration",
        tree / "calibration.json",
        "--profile",
        "servable",
        "--scales",
        "pot",
        "--out",
        tree / out,
        "--seed",
        str(seed),
        *extra,
    )


@pytest.fixture(scope="module")
def compiled(tree: Path) -> dict[str, Any]:
    status, result = compile_d(tree, "model.d", 5, "--report", tree / "calibration-report.json")
    assert status == 0, result
    return result


def test_compile_writes_d_reproducibly_from_its_seed(tree: Path, compiled: dict[str, Any]) -> None:
    data = (tree / "model.d").read_bytes()
    assert compiled["digest"] == data[-32:].hex()
    assert stat.S_IMODE((tree / "model.d").stat().st_mode) == 0o600
    d = read_artifact(tree / "model.d", INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    calibration = compiled["calibration"]
    assert (calibration["seed"], calibration["sequences"], calibration["held_out"]) == (5, 6, 3)
    assert d.provenance.calibration.selection.hex() == calibration["selection_digest"]
    assert d.provenance.statistics.hex() == calibration["statistics_digest"]
    report = json.loads((tree / "calibration-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.calibration/1"
    assert report["selection_digest"] == calibration["selection_digest"]
    assert report["dataset"]["files"] == ["calibration.jsonl"]
    # The same seed gives the same bytes; another seed draws another calibration set.
    status, _ = compile_d(tree, "again.d", 5)
    assert status == 0
    assert (tree / "again.d").read_bytes() == data
    status, other = compile_d(tree, "other.d", 6)
    assert status == 0
    assert other["calibration"]["selection_digest"] != calibration["selection_digest"]


def test_compile_takes_affine_scales(tree: Path) -> None:
    status, result = run(
        "compile",
        "--checkpoint",
        tree / "checkpoint",
        "--calibration",
        tree / "calibration.json",
        "--profile",
        "servable",
        "--scales",
        "affine",
        "--out",
        tree / "affine.d",
    )
    assert status == 0, result
    assert result["scales"] == "affine"
    d = read_artifact(tree / "affine.d", INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    # Every site after Q, K, V, O, up and down, in both layers (RFC-0025).
    assert len(d.quant.public.affine_sites) == 2 * 6


def evaluate(tree: Path, spec: str) -> tuple[int, dict[str, Any]]:
    return run(
        "eval",
        "--model",
        tree / "model.d",
        "--reference",
        tree / "checkpoint",
        "--dataset",
        tree / spec,
        "--out",
        tree / "fidelity.json",
        "--seed",
        "9",
    )


def test_eval_measures_d_on_data_kept_apart_from_its_calibration(
    tree: Path, compiled: dict[str, Any]
) -> None:
    status, result = evaluate(tree, "eval.json")
    assert status == 0, result
    report = json.loads((tree / "fidelity.json").read_text(encoding="utf-8"))
    assert report["schema"] == "tessera.fidelity/1"
    assert report["seed"] == 9
    assert report["model"]["calibration_selection"] == compiled["calibration"]["selection_digest"]
    perplexity, agreement = report["perplexity"], report["agreement"]
    assert perplexity["dataset"]["files"] == ["calibration.jsonl"]
    # The notes, each a beginning-of-text token and one token per character.
    assert (perplexity["sequences"], perplexity["tokens"]) == (8, sum(len(n) for n in NOTES))
    assert (agreement["sequences"], agreement["tokens"]) == (6, 6 * 4)
    assert result["perplexity_ratio"] == perplexity["ratio"]
    # A file under the calibration's name that does not give its calibration set back.
    elsewhere = tree / "elsewhere"
    elsewhere.mkdir()
    jsonl(elsewhere / "calibration.jsonl", [f"another story {i}, " * 4 for i in range(40)])
    jsonl(elsewhere / "prompts.jsonl.gz", [f"{i} lorem ipsum dolor sit amet" for i in range(12)])
    spec = json.loads((tree / "eval.json").read_text(encoding="utf-8"))
    (elsewhere / "eval.json").write_text(json.dumps(spec), encoding="utf-8")
    status, refused = evaluate(tree, "elsewhere/eval.json")
    assert (status, refused["error"]["code"]) == (20, "ERR-QUANT-CALIB")
    assert "calibration set back" in refused["error"]["message"]


def test_reference_writes_the_logits_of_every_position_in_order(
    tree: Path, compiled: dict[str, Any]
) -> None:
    sequences = [np.array([1, 7, 9], dtype=np.uint32), np.array([1, 2, 3, 4, 40], dtype=np.uint32)]
    write_artifact(tree / "tokens.ttk", TOKEN_FILE, TokenFile(sequences), limits=MESSAGE_LIMITS)
    status, result = run(
        "reference",
        "--model",
        tree / "model.d",
        "--tokens",
        tree / "tokens.ttk",
        "--out",
        tree / "logits.tlg",
    )
    assert status == 0, result
    logits = read_artifact(tree / "logits.tlg", LOGITS_FILE, limits=MESSAGE_LIMITS)
    d = read_artifact(tree / "model.d", INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    expected: list[Any] = [row.tolist() for s in sequences for row in run_reference(d, s)]
    assert logits.vocab == d.arch.vocab
    assert [row.tolist() for row in logits.positions] == expected
    assert (result["sequences"], result["positions"]) == (2, 8)
    assert result["digest"] == (tree / "logits.tlg").read_bytes()[-32:].hex()


def test_failures_exit_with_their_codes_status(tree: Path, compiled: dict[str, Any]) -> None:
    def refused(*argv: str | Path) -> tuple[int, str]:
        status, result = run(*argv)
        return status, result["error"]["code"]

    # Arguments: missing, and out of range.
    assert refused("compile") == (64, "ERR-USAGE")
    assert refused("reference", "--model", "m", "--tokens", "t", "--out", "o", "--seed", "1") == (
        64,
        "ERR-USAGE",
    )
    assert compile_d(tree, "negative.d", -1)[1]["error"]["code"] == "ERR-USAGE"
    # Specifications: another schema, a key the schema lacks.
    spec = json.loads((tree / "calibration.json").read_text(encoding="utf-8"))
    for wrong in ({**spec, "schema": "tessera.eval-spec/1"}, {**spec, "seed": 3}):
        (tree / "wrong.json").write_text(json.dumps(wrong), encoding="utf-8")
        assert refused(
            "compile",
            "--checkpoint",
            tree / "checkpoint",
            "--calibration",
            tree / "wrong.json",
            "--profile",
            "servable",
            "--scales",
            "pot",
            "--out",
            tree / "wrong.d",
        ) == (64, "ERR-USAGE")
    # Inputs: bytes that are no artifact, and a reference other than D's checkpoint.
    (tree / "garbage.d").write_bytes(b"not an artifact")
    garbage = ("reference", "--model", tree / "garbage.d", "--tokens", tree / "garbage.d")
    assert refused(*garbage, "--out", tree / "x.tlg") == (65, "ERR-CODEC-MAGIC")
    other = write(tree / "other", TINY | {"bos_token_id": 1}, llama_like(6))
    char_tokenizer(other)
    assert refused(
        "eval",
        "--model",
        tree / "model.d",
        "--reference",
        other,
        "--dataset",
        tree / "eval.json",
        "--out",
        tree / "x.json",
    ) == (65, "ERR-QUANT-ARCH")
    # Resources: an output in a directory that does not exist.
    write_artifact(
        tree / "one.ttk",
        TOKEN_FILE,
        TokenFile([np.array([1, 2], dtype=np.uint32)]),
        limits=MESSAGE_LIMITS,
    )
    tokens = ("--model", tree / "model.d", "--tokens", tree / "one.ttk")
    assert refused("reference", *tokens, "--out", tree / "absent" / "x.tlg") == (20, "ERR-RES-IO")


def test_exit_statuses_are_the_golden_files() -> None:
    golden = json.loads((ROOT / "spec/vectors/errors/codes.json").read_text(encoding="utf-8"))
    assert {entry["code"]: entry["exit"] for entry in golden["codes"]} == EXIT_CODES


def test_the_committed_default_calibration_is_rfc_0005s() -> None:
    spec = read_calibration_spec(ROOT / "docs/perf/examples/calibration-spec-c4.json")
    identity = spec.source.identity
    assert (identity.name, identity.revision, identity.files) == (
        C4_DATASET,
        C4_REVISION,
        C4_VALIDATION_FILES,
    )
    assert spec.source.files is None
    assert (spec.sequences, spec.tokens_per_sequence, spec.held_out) == (
        DEFAULT_COUNT,
        DEFAULT_LENGTH,
        DEFAULT_HELD_OUT,
    )


def test_python_m_runs_the_command_line() -> None:
    found = subprocess.run(
        [sys.executable, "-m", "tessera_quant", "reference"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode == 64
    assert "ERR-USAGE" in found.stderr
