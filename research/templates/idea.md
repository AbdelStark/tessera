# IDEA-NNNN: <title>

| | |
|---|---|
| Status | `proposed` |
| Origin | <SUR-NNNN finding, RISK-n, OQ-nnnn-n, source §9 problem k, or "new"> |
| Lever | <terms moved: `c_c`, `c_ℓ`, `c_π`, `C_lin`, `K_lin`, `A`, `regen`, `stream`, `τ`, `size`, `verify`> |
| Spec surfaces | <RFCs and sections that change; tier A (02 §1)? then a superseding RFC is required> |
| Priority | <score from the rubric; filled at triage> |

## Mechanism

What changes, precisely enough that someone could implement it: the relation, the table, the pass, the parameter. A formula or a small diagram where words are ambiguous.

## Why it could work

The evidence, each item with a bibliography key and an evidence level.

## Expected gain

The cost-model scenario that expresses the idea, its command, and the result:

```sh
python3 research/tools/cost_model.py --set <name>=<value> ...
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | | | |
| 4K | | | |
| 16K | | | |
| 128K | | | |

State the evidence level of every input. A gain estimated from a paper's number on other hardware is E0 with a `claim` source until an experiment here replaces it.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | | |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | | |
| Zero knowledge (G-5, RFC-0013) | | |
| Verifier independence (SP-6) | | |
| Post-quantum (hash-based commitments) | | |

A `fail` anywhere rejects the idea. An `unknown` is the first thing the experiment or the survey must settle.

## Prior rejection check

Which "Alternatives Considered" item of which RFC, or which row of [baseline.md](../baseline.md) "Already rejected", covers this idea, and what is new now (a new result, a new measurement, a changed assumption). "None" is an acceptable answer only after the check was made.

## Kill criterion

The single observation that would refute the idea, stated before any experiment.

## Cheapest experiment

Hypothesis, method, the evidence level reachable on the hardware at hand, and the effort (`s`, `m`, `l` as in CONTRIBUTING's labels). What stays unknown until an H100 run.

## Refutations

Filled at triage, one row per refuter lens (soundness, prior art, cost model, feasibility).

| Lens | Verdict | Argument |
|---|---|---|
| | survives / refuted / weakened | |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| <Δμ at the dominant context, or the scoreboard metric moved> | <0.1–0.9, with the reason> | <s/m/l, hardware, RFC needed> | <rubric in `.claude/skills/autoresearch/ideate.md`> |

## Decision and links

Experiments: EXP-NNNN. RFC or issue: . Ledger line updated on: .
