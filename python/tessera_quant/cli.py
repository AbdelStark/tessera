# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The `tessera-quant` command line (02 §2.1).

- `compile` draws the calibration set a specification names, captures its statistics, quantizes
  the checkpoint into `D` and exports it. `--report` writes the calibration's public report.
- `eval` measures `D` against the checkpoint it was compiled from, on evaluation data a
  specification names. It redraws `D`'s calibration set from its provenance, checked by the
  selection digest, and keeps those documents out of the evaluation data.
- `reference` runs `D`'s integer reference on a `TokenFile` and writes the logits of every
  position of every sequence, in order, as a `LogitsFile`.

Every command prints its progress to stderr and, with `--json`, one JSON object to stdout: its
result, or its error's code and message. The exit status is the error code's (04 §3): 0 on
success, 64 for usage, 65 for a malformed input, 20 for a calibration or I/O failure, 70 for a
bug. Messages hold public data only (05 §3); an unexpected error is reported by its type alone.

A specification is a JSON file, `tessera.calibration-spec/1` or `tessera.eval-spec/1`
(`spec/schemas/`), whose paths are relative to it. A dataset source is `"c4"`, the pinned C4
validation split of RFC-0005, or JSON-lines files of `text` fields with their identity.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from tessera_quant.assemble import check_options, checkpoint_digests, export, quantize
from tessera_quant.calibration import (
    C4_DATASET,
    C4_REVISION,
    C4_VALIDATION_FILES,
    DatasetIdentity,
    SplitMix64,
    draw,
    fetch_c4_validation,
    read_json_lines,
)
from tessera_quant.capture import calibrate, calibration_report
from tessera_quant.checkpoint import import_checkpoint, read_tokenizer
from tessera_quant.errors import ArchitectureError, CalibrationError, TesseraQuantError, UsageError
from tessera_quant.fidelity import RESAMPLES, EvalSpec, evaluate_fidelity, select
from tessera_quant.integer_model import INTEGER_MODEL, IntegerModel
from tessera_quant.reference import run_reference
from tessera_quant.tce import (
    INTEGER_MODEL_LIMITS,
    LOGITS_FILE,
    MESSAGE_LIMITS,
    TOKEN_FILE,
    LogitsFile,
    read_artifact,
    write_artifact,
)

CALIBRATION_SPEC = "tessera.calibration-spec/1"
"""The calibration specification's schema, `spec/schemas/calibration-spec-v1.schema.json`."""

EVAL_SPEC = "tessera.eval-spec/1"
"""The evaluation specification's schema, `spec/schemas/eval-spec-v1.schema.json`."""

EXIT_CODES: dict[str, int] = {
    "ERR-USAGE": 64,
    "ERR-CODEC-MAGIC": 65,
    "ERR-CODEC-VERSION": 65,
    "ERR-CODEC-NONCANONICAL": 65,
    "ERR-CODEC-TRUNCATED": 65,
    "ERR-CODEC-DIGEST": 65,
    "ERR-CODEC-LIMIT": 65,
    "ERR-QUANT-ARCH": 65,
    "ERR-QUANT-CALIB": 20,
    "ERR-COMPILE-RANGE": 20,
    "ERR-COMPILE-WRAP": 20,
    "ERR-COMPILE-PROFILE": 20,
    "ERR-WITNESS-EXACT-MISMATCH": 20,
    "ERR-WITNESS-ANCHOR": 20,
    "ERR-WITNESS-BUDGET": 20,
    "ERR-PROTO-ORDER": 20,
    "ERR-PROTO-VERSION": 20,
    "ERR-PROTO-COINTOSS": 20,
    "ERR-PROTO-REUSE": 70,
    "ERR-PROTO-TIMEOUT": 20,
    "ERR-PROTO-QUERIES": 20,
    "ERR-RES-MEMORY": 20,
    "ERR-RES-DEVICE": 20,
    "ERR-RES-IO": 20,
    "ERR-RES-TIME": 20,
    "ERR-INTERNAL": 70,
}
"""Each error code's exit status, as 04 §3 lists them."""

_SEEDS = 1 << 64


# ----- Specifications ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A dataset a specification names: its identity, and its files, or none for C4."""

    identity: DatasetIdentity
    files: tuple[Path, ...] | None
    """The JSON-lines files, read in order; `None` for C4, fetched from the Hub."""

    def documents(self) -> list[str]:
        """The `text` field of every line of every file, in order.

        Raises:
            CalibrationError: as `read_json_lines`.
        """
        if self.files is None:
            return fetch_c4_validation()[1]
        return [text for file in self.files for text in read_json_lines(file)]


@dataclass(frozen=True)
class CalibrationSpec:
    """A `tessera.calibration-spec/1` file: what `compile` draws its calibration set from."""

    source: Source
    sequences: int
    tokens_per_sequence: int
    held_out: int


@dataclass(frozen=True)
class Drawn:
    """Evaluation sequences a specification asks for: drawn from `source`, `count` of them."""

    source: Source
    count: int
    tokens: int
    """The sequence length, the beginning-of-text token included: the most a document gives, or
    each prompt's."""


@dataclass(frozen=True)
class EvalDataSpec:
    """A `tessera.eval-spec/1` file: what `eval` measures `D` on."""

    calibration: Source
    documents: Drawn
    prompts: Drawn
    continuation: int
    resamples: int


C4 = Source(DatasetIdentity(C4_DATASET, "en", "validation", C4_REVISION, C4_VALIDATION_FILES), None)
"""The default calibration and evaluation source: C4's `en` validation split, pinned."""


def _fields(value: object, where: str, required: set[str], optional: set[str]) -> dict[str, Any]:
    """`value` as a JSON object with the `required` keys, and no key but those and `optional`."""
    if not isinstance(value, dict):
        raise UsageError(f"{where} is a JSON object")
    found = cast(dict[str, Any], value)
    missing = sorted(required - found.keys())
    unknown = sorted(found.keys() - required - optional)
    if missing or unknown:
        raise UsageError(f"{where}: missing {missing}, unknown {unknown}")
    return found


def _integer(value: object, where: str, low: int) -> int:
    """`value` as a JSON integer of at least `low`."""
    if not isinstance(value, int) or isinstance(value, bool) or value < low:
        raise UsageError(f"{where} is an integer of at least {low}")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise UsageError(f"{where} is a non-empty string")
    return value


def _source(value: object, where: str, base: Path) -> Source:
    """A dataset source: `"c4"`, or local JSON-lines files with their identity."""
    if value == "c4":
        return C4
    keys = {"name", "config", "split", "revision", "files"}
    fields = _fields(value, where, keys, set())
    files: object = fields["files"]
    if not isinstance(files, list) or not files:
        raise UsageError(f"{where}.files lists at least one file")
    names = tuple(_text(f, f"{where}.files") for f in cast(list[object], files))
    identity = DatasetIdentity(
        name=_text(fields["name"], f"{where}.name"),
        config=_text(fields["config"], f"{where}.config"),
        split=_text(fields["split"], f"{where}.split"),
        revision=_text(fields["revision"], f"{where}.revision"),
        files=names,
    )
    return Source(identity, tuple(base / name for name in names))


def _document(path: Path, schema: str) -> dict[str, Any]:
    """The JSON object of the specification file `path`, whose schema must be `schema`."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UsageError(f"no specification file {path}") from None
    try:
        value: object = json.loads(text)
    except ValueError as error:
        raise UsageError(f"{path.name} is not JSON: {error}") from None
    if not isinstance(value, dict) or cast(dict[str, object], value).get("schema") != schema:
        raise UsageError(f"{path.name} is not a {schema} specification")
    return cast(dict[str, Any], value)


def read_calibration_spec(path: Path) -> CalibrationSpec:
    """The calibration specification in the file `path`.

    Raises:
        UsageError: for a file that is not one.
    """
    keys = {"schema", "source", "sequences", "tokens_per_sequence", "held_out"}
    fields = _fields(_document(path, CALIBRATION_SPEC), path.name, keys, set())
    return CalibrationSpec(
        source=_source(fields["source"], "source", path.parent),
        sequences=_integer(fields["sequences"], "sequences", 1),
        tokens_per_sequence=_integer(fields["tokens_per_sequence"], "tokens_per_sequence", 2),
        held_out=_integer(fields["held_out"], "held_out", 1),
    )


def read_eval_spec(path: Path) -> EvalDataSpec:
    """The evaluation specification in the file `path`.

    Raises:
        UsageError: for a file that is not one.
    """
    keys = {"schema", "calibration", "documents", "prompts", "continuation"}
    fields = _fields(_document(path, EVAL_SPEC), path.name, keys, {"resamples"})
    base = path.parent
    documents = _fields(fields["documents"], "documents", {"source", "count", "max_tokens"}, set())
    prompts = _fields(fields["prompts"], "prompts", {"source", "count", "tokens"}, set())
    return EvalDataSpec(
        calibration=_source(fields["calibration"], "calibration", base),
        documents=Drawn(
            _source(documents["source"], "documents.source", base),
            _integer(documents["count"], "documents.count", 1),
            _integer(documents["max_tokens"], "documents.max_tokens", 2),
        ),
        prompts=Drawn(
            _source(prompts["source"], "prompts.source", base),
            _integer(prompts["count"], "prompts.count", 1),
            _integer(prompts["tokens"], "prompts.tokens", 2),
        ),
        continuation=_integer(fields["continuation"], "continuation", 1),
        resamples=_integer(fields.get("resamples", RESAMPLES), "resamples", 1),
    )


# ----- Commands ----------------------------------------------------------------------------------


def _say(args: argparse.Namespace, message: str) -> None:
    print(f"tessera-quant {args.command}: {message}", file=sys.stderr, flush=True)


def _compile(args: argparse.Namespace) -> dict[str, Any]:
    check_options(args.profile, args.scales)
    spec = read_calibration_spec(args.calibration)
    _say(args, f"checking the checkpoint in {args.checkpoint}")
    model = import_checkpoint(args.checkpoint)
    tokenizer = read_tokenizer(args.checkpoint, model.arch)
    if spec.tokens_per_sequence > model.arch.max_context:
        raise UsageError(f"a sequence holds at most the context, {model.arch.max_context} tokens")
    _say(args, f"drawing the calibration set with seed {args.seed}")
    calibration = draw(
        spec.source.documents(),
        tokenizer.encode,
        tokenizer.bos,
        source=spec.source.identity,
        seed=args.seed,
        count=spec.sequences,
        length=spec.tokens_per_sequence,
        held_out=spec.held_out,
    )
    _say(args, f"capturing the statistics of {spec.sequences} sequences on {args.device}")
    result = calibrate(model, calibration, args.device)
    _say(args, f"quantizing ({args.profile}, {args.scales})")
    d = quantize(model, result, profile=args.profile, scales=args.scales, device=args.device)
    digest = export(d, args.out)
    _say(args, f"wrote {args.out}")
    if args.report is not None:
        report = calibration_report(model, calibration, result.data, args.device)
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        args.report.write_text(text, encoding="utf-8")
    alpha = d.provenance.alpha
    return {
        "out": str(args.out),
        "digest": digest.hex(),
        "profile": args.profile,
        "scales": args.scales,
        "alpha": f"{alpha.numerator}/{alpha.denominator}",
        "calibration": {
            "seed": args.seed,
            "sequences": len(calibration.calibration),
            "held_out": len(calibration.held_out),
            "tokens_per_sequence": calibration.length,
            "selection_digest": calibration.selection_digest.hex(),
            "statistics_digest": result.digest.hex(),
        },
        "report": None if args.report is None else str(args.report),
    }


def _calibration_hashes(
    d: IntegerModel,
    source: Source,
    documents: list[str],
    encode: Callable[[str], Sequence[int]],
    bos: int,
) -> frozenset[bytes]:
    """The content hashes of `D`'s calibration documents, redrawn from its provenance."""
    recorded = d.provenance.calibration
    identity = source.identity
    named = (
        identity.name,
        identity.config,
        identity.split,
        identity.revision,
        list(identity.files),
    )
    if named != (
        recorded.dataset,
        recorded.config,
        recorded.split,
        recorded.revision,
        recorded.files,
    ):
        raise CalibrationError("the calibration source is not the dataset D was calibrated on")
    calibration = draw(
        documents,
        encode,
        bos,
        source=identity,
        seed=recorded.seed,
        count=recorded.sequences,
        length=recorded.tokens_per_sequence,
        held_out=recorded.held_out,
    )
    if calibration.selection_digest != recorded.selection:
        raise CalibrationError("the calibration source does not give D's calibration set back")
    return calibration.document_hashes


def _eval(args: argparse.Namespace) -> dict[str, Any]:
    spec = read_eval_spec(args.dataset)
    d = read_artifact(args.model, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    _say(args, f"checking the reference in {args.reference}")
    reference = import_checkpoint(args.reference, expected=d.arch)
    if checkpoint_digests(reference) != d.provenance.checkpoint:
        raise ArchitectureError("the reference is not the checkpoint D was compiled from")
    tokenizer = read_tokenizer(args.reference, reference.arch)
    context = d.arch.max_context
    if spec.documents.tokens > context or spec.prompts.tokens + spec.continuation > context:
        raise UsageError(f"a document, or a prompt with its continuation, exceeds {context} tokens")
    read: dict[Source, list[str]] = {}

    def documents(source: Source) -> list[str]:
        if source not in read:
            read[source] = source.documents()
        return read[source]

    _say(args, "redrawing D's calibration set")
    excluded = _calibration_hashes(
        d, spec.calibration, documents(spec.calibration), tokenizer.encode, tokenizer.bos
    )
    _say(args, f"drawing the evaluation data with seed {args.seed}")
    rng = SplitMix64(args.seed)
    drawn = [
        select(
            documents(want.source),
            tokenizer.encode,
            tokenizer.bos,
            source=want.source.identity,
            rng=rng,
            count=want.count,
            length=want.tokens,
            minimum=minimum,
            exclude=excluded,
        )
        for want, minimum in ((spec.documents, 1), (spec.prompts, spec.prompts.tokens - 1))
    ]
    data = EvalSpec(drawn[0], drawn[1], spec.continuation, excluded, spec.resamples)
    _say(args, f"evaluating on {args.device}")
    report = evaluate_fidelity(d, reference, data, seed=args.seed, device=args.device).as_json()
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.out.write_text(text, encoding="utf-8")
    _say(args, f"wrote {args.out}")
    agreement = report["agreement"]
    return {
        "out": str(args.out),
        "perplexity_ratio": report["perplexity"]["ratio"],
        "top1": agreement["top1"],
        "kl": agreement["kl"],
    }


def _reference(args: argparse.Namespace) -> dict[str, Any]:
    d = read_artifact(args.model, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    tokens = read_artifact(args.tokens, TOKEN_FILE, limits=MESSAGE_LIMITS)
    if not tokens.sequences:
        raise UsageError("the token file holds no sequence")
    _say(args, f"running D on {len(tokens.sequences)} sequences")
    positions = [row for sequence in tokens.sequences for row in run_reference(d, sequence)]
    digest = write_artifact(
        args.out, LOGITS_FILE, LogitsFile(d.arch.vocab, positions), limits=MESSAGE_LIMITS
    )
    _say(args, f"wrote {args.out}")
    return {
        "out": str(args.out),
        "digest": digest.hex(),
        "sequences": len(tokens.sequences),
        "positions": len(positions),
    }


_COMMANDS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "compile": _compile,
    "eval": _eval,
    "reference": _reference,
}


# ----- The parser --------------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """A parser whose errors are `ERR-USAGE`, not argparse's exit status 2."""

    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def _seed(text: str) -> int:
    try:
        seed = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if not 0 <= seed < _SEEDS:
        raise argparse.ArgumentTypeError("a seed lies in [0, 2^64)")
    return seed


def parser() -> argparse.ArgumentParser:
    """The command line of 02 §2.1, with `--device` and `--json`."""
    top = _Parser(prog="tessera-quant", description="Tessera's quantization front-end (RFC-0005).")
    commands = top.add_subparsers(dest="command", required=True)

    def command(name: str, summary: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.add_argument("--json", action="store_true", help="print the result as JSON to stdout")
        return sub

    compile_ = command("compile", "quantize a checkpoint into the integer model D")
    compile_.add_argument("--checkpoint", type=Path, required=True, metavar="DIR")
    compile_.add_argument("--calibration", type=Path, required=True, metavar="SPEC.json")
    compile_.add_argument("--profile", choices=("servable", "minimal"), required=True)
    compile_.add_argument("--scales", choices=("pot", "affine"), required=True)
    compile_.add_argument("--out", type=Path, required=True, metavar="MODEL.d")
    compile_.add_argument("--report", type=Path, metavar="CALIB.json")
    compile_.add_argument("--seed", type=_seed, default=0, metavar="INT")
    compile_.add_argument("--device", default="cpu")

    evaluate = command("eval", "measure D against its BF16 reference")
    evaluate.add_argument("--model", type=Path, required=True, metavar="MODEL.d")
    evaluate.add_argument("--reference", type=Path, required=True, metavar="DIR")
    evaluate.add_argument("--dataset", type=Path, required=True, metavar="SPEC.json")
    evaluate.add_argument("--out", type=Path, required=True, metavar="FIDELITY.json")
    evaluate.add_argument("--seed", type=_seed, default=0, metavar="INT")
    evaluate.add_argument("--device", default="cpu")

    reference = command("reference", "run D's integer reference on token sequences")
    reference.add_argument("--model", type=Path, required=True, metavar="MODEL.d")
    reference.add_argument("--tokens", type=Path, required=True, metavar="TOKENS.ttk")
    reference.add_argument("--out", type=Path, required=True, metavar="LOGITS.tlg")
    return top


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line on `argv`, or the process's arguments, and returns its exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = next((a for a in arguments if a in _COMMANDS), None)
    as_json = "--json" in arguments
    try:
        args = parser().parse_args(arguments)
        result = {"command": args.command} | _COMMANDS[args.command](args)
    except TesseraQuantError as error:
        return _fail(command, as_json, error.code, str(error))
    except OSError as error:
        where = f" {error.filename}" if error.filename else ""
        return _fail(
            command, as_json, "ERR-RES-IO", f"{error.strerror or type(error).__name__}{where}"
        )
    except MemoryError:
        return _fail(command, as_json, "ERR-RES-MEMORY", "out of memory")
    except Exception as error:  # every other failure is a bug, named by its type alone (05 §3)
        return _fail(command, as_json, "ERR-INTERNAL", type(error).__name__)
    if as_json:
        print(json.dumps(result, ensure_ascii=False))
    return 0


def _fail(command: str | None, as_json: bool, code: str, message: str) -> int:
    print(f"tessera-quant: {code}: {message}", file=sys.stderr, flush=True)
    if as_json:
        print(json.dumps({"command": command, "error": {"code": code, "message": message}}))
    return EXIT_CODES.get(code, 70)
