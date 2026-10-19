# IDEA-0013: Rate 1/4 for the advice commitment: fewer opened columns against a doubled encode-and-hash volume

| | |
|---|---|
| Status | `refuted` by [EXP-0004](../experiments/EXP-0004-advice-rate-quarter/README.md) (E0) as a default; kept on record as a size knob for a verifier-bound deployment (see "Decision and links") |
| Origin | Ledger queue item 5 (bootstrap); [baseline.md §6](../baseline.md#6-levers-which-technique-class-moves-which-term) (rate moves `τ` against `c_c`); [SUR-0001](../surveys/SUR-0001-proximity.md) question 3; RISK-4, RISK-8 |
| Lever | `τ` and proof size against `c_c` (encoded and hashed volume per committed element) |
| Spec surfaces | RFC-0014 'Parameter sets' (advice rate, `τ_A`), RFC-0011 'Encoding' (coset of order `n = 4k`): tier A, a superseding RFC or a new parameter set |
| Priority | 0 on `Δμ` (negative gain); as a size knob, 1.4 × 0.9 / 2 ≈ 0.6 on proof size for deployments that pay for verification rather than proving |

## Mechanism

Encode each advice row on a coset of order `4k` instead of `2k` (`ρ' = (k + P_A)/4k ≈ 1/4`). Under unique decoding the per-query bits rise from `0.415` to `0.678`, so `τ_A` falls from 155 to 95 and the opened columns from 0.72 to 0.44 GB; under the Johnson-regime bounds of [IDEA-0012](IDEA-0012-johnson-regime-tau-linear-error.md), from 129 to 65 (0.30 GB). The prover's coset NTT grows from size `2k` to `4k` and the column hasher sees twice the symbols per committed element, so `c_c` rises; the block buffer of the streaming committer (`B · n · 8` bytes, 2 GiB today) and the chain state double as well.

## Why it could work

- E0, own tool ([tools/proximity.py](../tools/proximity.py) `--sweep-rates`): `τ_A` 155 → 95 → 78 → 71 at rates 1/2, 1/4, 1/8, 1/16 under unique decoding; opened columns 0.719 → 0.441 → 0.362 → 0.329 GB; encode-and-hash volume 2× → 4× → 8× → 16× of the message.
- E0, own tool, Johnson regime ([ArkLib26] bound, `η = 0.005`): at rate 1/4, `τ_A = 65` with 84.0 bits, 0.302 GB.
- E0, cost model: the per-element cost `c_c` is bandwidth-bound encode plus hashing (source §5); at rate 1/4 the coset NTT doubles (size `4k` against `2k`; the inverse NTT of size `k` is unchanged) and the hashed symbols double. With the encoder-to-hash split of 63:37 that [IDEA-0004](IDEA-0004-calibrate-commit-cost.md) measured on CPU (E1), the factor on `c_c` is about `0.63 × (11 + 46)/34 + 0.37 × 2 ≈ 1.8`: `c_c` 30 → about 54 ps, bracketed by the 45 and 60 ps scenarios the experiment runs.

## Expected gain

```sh
python3 -I research/tools/proximity.py --sweep-rates
python3 -I research/tools/cost_model.py --set commit_ps=60
python3 -I research/tools/cost_model.py --set commit_ps=45
```

| Context | Δμ (`c_c` 60 ps / 45 ps, corrected profile) | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | +20.6% / +10.3% | −14% (opened columns −0.28 GB of 2 GB; −0.42 GB under the Johnson bounds, −21%) | column hashing and inner products −39% (unique decoding) or −50% (Johnson); coset NTTs of the combined rows 2× longer |
| 4K | +21.5% / +10.8% | same | same |
| 16K | +21.2% / +10.6% | same | same |
| 128K | +19.2% / +9.6% | same | same |

Inputs: E0 from the two tools; the `c_c` factor is an operation-count estimate anchored on IDEA-0004's E1 split.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4) | unchanged targets; `τ_A = 95` gives 64.4 bits under unique decoding, field term 104 bits | pass |
| Uniqueness upstream of the logits (SP-3) | unchanged | pass |
| Zero knowledge (G-5) | fewer columns opened than mask coefficients (`95 ≤ 160`) | pass |
| Verifier independence (SP-6) | unchanged | pass |
| Post-quantum | unchanged | pass |

## Prior rejection check

RFC-0014 'Parameter sets' fixes rate 1/2 for the advice and 1/4 for the weights without an Alternatives Considered entry on the advice rate; the source (§7) chose rate 1/2 for the advice as "bandwidth bound" encoding. RISK-8 states the hash budget at rate 1/2 as about 530 GB/s per H100; rate 1/4 doubles it to about 1,060 GB/s, which is why the knob was never the default. No prior experiment quantified the trade.

## Kill criterion

`Δμ` at 16K exceeds +10% for a proof-size gain below 15%: the primary metric loses more than the secondary gains.

## Cheapest experiment

E0, both tools, effort `s`: [EXP-0004](../experiments/EXP-0004-advice-rate-quarter/README.md).

## Refutations

Self-triage by the authoring agent (one agent, four lenses, flagged as such).

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | A pure parameter change inside the analysis RFC-0014 already uses. |
| prior art | weakened | The trade is the textbook rate trade-off of every Ligero-family scheme; nothing new, only quantified here. |
| cost model | refuted | `c_c` is 19–21% of `μ` at every context; doubling the volume costs 10–21% of `μ` for 14% of the proof, and the hash budget of RISK-8 doubles to about 1 TB/s per GPU. |
| feasibility | survives as a knob | A new parameter set is cheap to specify; the streaming committer's buffer doubles to 4 GiB per block (RFC-0011 'Streaming procedure'), which still fits. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| `Δμ` +10% to +21% (negative); proof size −14% to −21% | 0.9 | `m` (a parameter set) | 0 on `Δμ`; 0.6 on size for verifier-bound deployments |

## Decision and links

Experiments: [EXP-0004](../experiments/EXP-0004-advice-rate-quarter/README.md), `done: refuted`. Not a default: `μ` is the primary metric and the SC-2 ceilings are already at risk under the corrected profile. Shelved as a size knob: a `live-64-small`-style parameter set would be the way to offer it, and [IDEA-0012](IDEA-0012-johnson-regime-tau-linear-error.md) delivers 0.12 GB of the same saving at no prover cost first. Reopens if `c_c` is measured well below 30 ps on H100 (RFC-0020 gate) so that the volume is cheap, or if a deployment states proof size as its binding constraint. Ledger line updated on 2026-10-16.
