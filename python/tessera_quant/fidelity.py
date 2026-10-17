# SPDX-License-Identifier: Apache-2.0 OR MIT
"""How closely `D` follows its BF16 reference (RFC-0005 'Fidelity evaluation').

`evaluate_fidelity` runs `D` through the integer reference, its logits at their real scale
`2^logits_exponent` (RFC-0027), and the BF16 reference through Hugging Face's Llama, on data
disjoint from the calibration:

| Metric | Data | Interval, 95% |
|---|---|---|
| Perplexity ratio `ppl(D) / ppl(ref)` | documents | bootstrap over documents |
| Teacher-forced top-1 agreement | prompts, then the reference's greedy continuation | Wilson |
| Mean per-token `KL(ref ‖ D)` | the same continuations | bootstrap over prompts |

A document's perplexity counts every token it predicts, the first excepted. On a prompt, `D`
reads the prompt and the reference's continuation, and each continuation token is a trial: `D`
agrees when its most likely next token is the reference's. Bootstrap intervals are the 2.5% and
97.5% percentiles of `resamples` resamples, drawn from `seed`.

The report, `tessera.fidelity/1`, records the seeds, the data's identities and digests, the
library versions and `D`'s own provenance digests. Bounded mode's `Û` joins it with RFC-0017's
predictor.

`select` draws evaluation sequences from a dataset's documents as the calibration set is drawn:
in the order of a seeded shuffle, each document tokenized without special tokens, and none of
them a calibration document.
"""

import math
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import version
from typing import Any

import blake3
import numpy as np
import numpy.typing as npt
import torch

from tessera_quant.calibration import DatasetIdentity, SplitMix64, document_hash
from tessera_quant.capture import continue_greedily, logits_of, reference_model
from tessera_quant.checkpoint import FloatModel
from tessera_quant.errors import ArchitectureError, CalibrationError, TesseraQuantError
from tessera_quant.integer_model import IntegerModel
from tessera_quant.reference import trace

SCHEMA = "tessera.fidelity/1"
"""The report's schema (03 §5), `spec/schemas/fidelity-v1.schema.json`."""

Z95 = 1.959963984540054
"""The standard normal's 97.5% quantile: two-sided 95% intervals."""

RESAMPLES = 1000
"""Bootstrap resamples, by default."""

_DATA_CONTEXT = "tessera-quant 2026-10-17 evaluation data v1"


@dataclass(frozen=True)
class EvalSet:
    """Token sequences for evaluation, and what identifies them (RFC-0005 'Fidelity evaluation').

    `document_hashes` are the content hashes of the documents the sequences come from, which must
    be disjoint from the calibration's (`tessera_quant.calibration.document_hash`).
    """

    source: DatasetIdentity
    sequences: tuple[tuple[int, ...], ...]
    document_hashes: frozenset[bytes]

    @property
    def digest(self) -> bytes:
        """BLAKE3 of the sequences, under its own key-derivation context.

        It hashes their count, then each one's length and tokens, as u64 and u32 little-endian.
        """
        hasher = blake3.blake3(derive_key_context=_DATA_CONTEXT)
        hasher.update(len(self.sequences).to_bytes(8, "little"))
        for sequence in self.sequences:
            hasher.update(len(sequence).to_bytes(8, "little"))
            hasher.update(np.asarray(sequence, dtype="<u4").tobytes())
        return hasher.digest()


def select(
    documents: Sequence[str],
    encode: Callable[[str], Sequence[int]],
    bos: int,
    *,
    source: DatasetIdentity,
    rng: SplitMix64,
    count: int,
    length: int,
    minimum: int,
    exclude: frozenset[bytes],
) -> EvalSet:
    """`count` evaluation sequences from `documents`, none of them a document of `exclude`.

    The documents are visited in the order of `rng`'s shuffle of their indices
    (`SplitMix64.shuffled`). A document whose content hash is in `exclude`, the calibration's, is
    skipped, as is one of fewer than `minimum` tokens, tokenized without special tokens. Each
    other gives the beginning-of-text token `bos`, then its first `length - 1` tokens, until
    `count` are drawn. No document gives two sequences.

    Raises:
        CalibrationError: when the documents give fewer than `count` sequences
            (`ERR-QUANT-CALIB`).
        ValueError: for a count below 1, a length below 2 or a minimum below 1.
    """
    if count < 1 or length < 2 or minimum < 1:
        raise ValueError("a selection draws sequences of at least 2 tokens, from 1 token up")
    sequences: list[tuple[int, ...]] = []
    hashes: set[bytes] = set()
    for index in rng.shuffled(len(documents)):
        if len(sequences) == count:
            break
        text = documents[index]
        digest = document_hash(text)
        if digest in exclude:
            continue
        tokens = encode(text)
        if len(tokens) < minimum:
            continue
        sequences.append((bos, *tokens[: length - 1]))
        hashes.add(digest)
    if len(sequences) < count:
        raise CalibrationError(
            f"{len(documents)} documents give {len(sequences)} evaluation sequences, "
            f"not the {count} drawn"
        )
    return EvalSet(source, tuple(sequences), frozenset(hashes))


@dataclass(frozen=True)
class EvalSpec:
    """What `evaluate_fidelity` measures on (02 §4)."""

    documents: EvalSet
    """The perplexity documents: WikiText-2 test, and held-out C4 validation documents."""
    prompts: EvalSet
    """The agreement prompts: 512, by RFC-0005."""
    continuation: int
    """The reference's greedy tokens after each prompt: 256, by RFC-0005."""
    calibration_hashes: frozenset[bytes]
    """The calibration's documents, by content hash: no evaluation document may be one."""
    resamples: int = RESAMPLES


@dataclass(frozen=True)
class Interval:
    """An estimate with its 95% interval."""

    estimate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        """The interval as JSON: its estimate and bounds."""
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


@dataclass(frozen=True)
class Totals:
    """Per-unit sums: negative log-likelihoods, KL divergences, agreements and trials."""

    reference_nll: npt.NDArray[np.float64]
    model_nll: npt.NDArray[np.float64]
    kl: npt.NDArray[np.float64]
    agreeing: npt.NDArray[np.int64]
    tokens: npt.NDArray[np.int64]


@dataclass(frozen=True)
class FidelityReport:
    """`D` against its reference: the three metrics and their intervals (RFC-0005)."""

    perplexity_ratio: Interval
    reference_perplexity: float
    model_perplexity: float
    agreement: Interval
    agreeing: int
    trials: int
    kl: Interval
    spec: EvalSpec
    model: IntegerModel
    seed: int
    device: str

    def as_json(self) -> dict[str, Any]:
        """The `tessera.fidelity/1` report."""
        p = self.model.provenance

        def data(s: EvalSet, tokens: int) -> dict[str, Any]:
            source = s.source
            return {
                "dataset": {
                    "name": source.name,
                    "config": source.config,
                    "split": source.split,
                    "revision": source.revision,
                    "files": list(source.files),
                },
                "sequences": len(s.sequences),
                "tokens": tokens,
                "digest": s.digest.hex(),
            }

        libraries = ("blake3", "numpy", "safetensors", "torch", "transformers")
        return {
            "schema": SCHEMA,
            "tessera_quant": version("tessera-quant"),
            "model": {
                "profile": self.model.profile.name.lower(),
                "layers": self.model.arch.n_layers,
                "d_model": self.model.arch.d_model,
                "vocab": self.model.arch.vocab,
                "alpha": f"{p.alpha.numerator}/{p.alpha.denominator}",
                "statistics_digest": p.statistics.hex(),
                "calibration_selection": p.calibration.selection.hex(),
            },
            "seed": self.seed,
            "resamples": self.spec.resamples,
            "perplexity": {
                **data(self.spec.documents, _predicted(self.spec.documents)),
                "reference": self.reference_perplexity,
                "model": self.model_perplexity,
                "ratio": self.perplexity_ratio.as_dict(),
            },
            "agreement": {
                **data(self.spec.prompts, self.trials),
                "continuation": self.spec.continuation,
                "top1": {**self.agreement.as_dict(), "agreeing": self.agreeing},
                "kl": self.kl.as_dict(),
            },
            "environment": {
                "device": self.device,
                "python": platform.python_version(),
                "libraries": {name: version(name) for name in libraries},
            },
        }


def wilson(successes: int, trials: int, z: float = Z95) -> Interval:
    """The rate `successes / trials` with its Wilson score interval.

    Raises:
        TesseraQuantError: for no trial, or successes outside `[0, trials]`.
    """
    if trials < 1 or not 0 <= successes <= trials:
        raise TesseraQuantError("a rate needs at least one trial, and successes among them")
    p = successes / trials
    z2 = z * z
    centre = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / (1 + z2 / trials)
    # At no success the interval starts at 0, and at all successes it ends at 1, exactly.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return Interval(p, low, high)


def bootstrap_ratio(
    numerators: npt.NDArray[np.float64],
    denominators: npt.NDArray[np.float64],
    resamples: int,
    rng: np.random.Generator,
) -> Interval:
    """`Σ numerators / Σ denominators`, with its bootstrap percentile interval.

    The interval comes from `resamples` resamples of the units, drawn with replacement.

    Raises:
        TesseraQuantError: for no unit, no resample, or a total denominator of 0.
    """
    units = len(numerators)
    if units == 0 or resamples < 1 or float(denominators.sum()) <= 0:
        raise TesseraQuantError("a bootstrap needs units, resamples and a positive denominator")
    estimate = float(numerators.sum() / denominators.sum())
    picks = rng.integers(0, units, size=(resamples, units))
    ratios: npt.NDArray[np.float64] = numerators[picks].sum(axis=1) / np.maximum(
        denominators[picks].sum(axis=1), 1e-300
    )
    ordered = ratios.copy()
    ordered.sort()
    # The linear percentiles of numpy's default method, written out: 2.5% and 97.5%.
    low, high = (_percentile(ordered, q) for q in (0.025, 0.975))
    return Interval(estimate, low, high)


def token_totals(
    reference: Sequence[torch.Tensor],
    model: Sequence[torch.Tensor],
    targets: Sequence[Sequence[int]],
    starts: Sequence[int],
) -> Totals:
    """Per-sequence sums over the positions that predict `targets[s][i]`, `i >= starts[s]`.

    Position `i - 1` of each sequence's logits predicts its token `i`. `reference` and `model`
    hold each sequence's logits at their real scale, `[length, vocab]`.
    """
    out: dict[str, list[float | int]] = {k: [] for k in ("ref", "model", "kl", "agree", "n")}
    for ref, mod, tokens, start in zip(reference, model, targets, starts, strict=True):
        indices = range(max(start, 1) - 1, len(tokens) - 1)
        positions = torch.tensor(list(indices), dtype=torch.long)
        target = torch.tensor([tokens[i + 1] for i in indices], dtype=torch.long)
        log_p = torch.log_softmax(ref[positions].to(torch.float64), dim=-1)
        log_q = torch.log_softmax(mod[positions].to(torch.float64), dim=-1)
        picked = target[:, None]
        out["ref"].append(float(-log_p.gather(1, picked).sum()))
        out["model"].append(float(-log_q.gather(1, picked).sum()))
        out["kl"].append(float((log_p.exp() * (log_p - log_q)).sum()))
        out["agree"].append(int((log_q.argmax(dim=-1) == target).sum()))
        out["n"].append(len(target))
    return Totals(
        np.array(out["ref"], dtype=np.float64),
        np.array(out["model"], dtype=np.float64),
        np.array(out["kl"], dtype=np.float64),
        np.array(out["agree"], dtype=np.int64),
        np.array(out["n"], dtype=np.int64),
    )


def evaluate_fidelity(
    model: IntegerModel,
    reference: FloatModel,
    spec: EvalSpec,
    *,
    seed: int = 0,
    device: str = "cpu",
) -> FidelityReport:
    """`D` against its BF16 reference, by RFC-0005's metrics (02 §4).

    Raises:
        CalibrationError: when an evaluation document is a calibration document, or the data is
            empty (`ERR-QUANT-CALIB`: no evaluation data may inform a choice).
        ArchitectureError: for a reference of another architecture.
        TesseraQuantError: as `trace`.
    """
    for name, data in (("documents", spec.documents), ("prompts", spec.prompts)):
        if not data.sequences:
            raise CalibrationError(f"the evaluation {name} are empty")
        if data.document_hashes & spec.calibration_hashes:
            raise CalibrationError(f"the evaluation {name} share a document with the calibration")
    if spec.continuation < 1:
        raise CalibrationError("each prompt is followed by at least one reference token")
    if reference.arch != model.arch:
        raise ArchitectureError("the reference has another architecture than D")
    llama = reference_model(reference, device)
    scale = Fraction(2) ** model.quant.public.logits_exponent

    def d_logits(sequence: Sequence[int]) -> torch.Tensor:
        found = trace(model, np.asarray(sequence, dtype=np.uint32)).logits
        return found.to(torch.float64) * float(scale)

    documents = spec.documents.sequences
    totals = token_totals(
        logits_of(llama, documents, seed, device),
        [d_logits(s) for s in documents],
        documents,
        [0] * len(documents),
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    tokens = totals.tokens.astype(np.float64)
    log_ratio = bootstrap_ratio(
        totals.model_nll - totals.reference_nll, tokens, spec.resamples, rng
    )
    ratio = Interval(
        math.exp(log_ratio.estimate), math.exp(log_ratio.low), math.exp(log_ratio.high)
    )
    prompts = spec.prompts.sequences
    continuations = continue_greedily(llama, prompts, spec.continuation, seed, device)
    full = [tuple(p) + c for p, c in zip(prompts, continuations, strict=True)]
    turns = token_totals(
        logits_of(llama, full, seed, device),
        [d_logits(s) for s in full],
        full,
        [len(p) for p in prompts],
    )
    agreeing, trials = int(turns.agreeing.sum()), int(turns.tokens.sum())
    kl = bootstrap_ratio(turns.kl, turns.tokens.astype(np.float64), spec.resamples, rng)
    predicted = float(totals.tokens.sum())
    return FidelityReport(
        perplexity_ratio=ratio,
        reference_perplexity=math.exp(float(totals.reference_nll.sum()) / predicted),
        model_perplexity=math.exp(float(totals.model_nll.sum()) / predicted),
        agreement=wilson(agreeing, trials),
        agreeing=agreeing,
        trials=trials,
        kl=kl,
        spec=spec,
        model=model,
        seed=seed,
        device=device,
    )


def _percentile(ordered: npt.NDArray[np.float64], q: float) -> float:
    """The `q` percentile of sorted values, interpolated linearly between ranks."""
    rank = q * (len(ordered) - 1)
    below = math.floor(rank)
    above = min(below + 1, len(ordered) - 1)
    return float(ordered[below] + (rank - below) * (ordered[above] - ordered[below]))


def _predicted(data: EvalSet) -> int:
    """The tokens a perplexity counts: every token of a document but its first."""
    return sum(max(len(s) - 1, 0) for s in data.sequences)
