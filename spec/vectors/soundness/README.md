# Soundness calculator vectors

`soundness.json` pins the soundness calculator of [RFC-0014 'Soundness calculator'](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#soundness-calculator): every row of the soundness table, with its exact error bound, its bits as a FixedBits lower bound, its target and whether it reaches it. Both calculators must reproduce it bit for bit: `tessera-protocol`'s ([`crates/tessera-protocol/tests/soundness.rs`](../../../crates/tessera-protocol/tests/soundness.rs), and `tessera params check --json` in [`crates/tessera-cli/tests/params_check.rs`](../../../crates/tessera-cli/tests/params_check.rs)) and the verifier's (#110).

[`generate.py`](generate.py) writes it. The script is the independent oracle: it shares no code with `crates/`, `verifier/` or `python/`, and is written from RFC-0014, RFC-0049 'The calculator' (a weight matrix's count in `tessera-protocol/1`), [RFC-0026 'Schema'](../../../docs/rfcs/RFC-0026-claim-list-layout.md#schema), [RFC-0040 'The matrices'](../../../docs/rfcs/RFC-0040-weight-commitment-artifacts.md#the-matrices), [RFC-0030 'Cost'](../../../docs/rfcs/RFC-0030-masked-exponentials.md#cost), [RFC-0054 'Soundness'](../../../docs/rfcs/RFC-0054-token-groups.md#soundness) (the leaf stage in token groups) and SP-1 ([06 §4](../../../docs/spec/06-security.md#4-security-properties)). It states the 405B lookups of the compiler's report, [`docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json`](../../../docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json), and the lowering's 53,489 split values, which [`crates/tessera-bench/tests/commitment_report.rs`](../../../crates/tessera-bench/tests/commitment_report.rs) counts in both workspaces, since it runs on `spec/vectors/` alone, and a `tessera-protocol` test checks the calculator's default against the report itself. It derives the 405B weight matrices from the architecture by RFC-0040's rules. Its random inputs come from SplitMix64 with the seed the file states, so `python3 spec/vectors/soundness/generate.py` reproduces the file byte for byte.

## How the oracle checks itself

Every bit count is computed twice:

- **by LOG2**, RFC-0014's algorithm, with Python's integers: the FixedBits both calculators must give;
- **as the true value** `λ = g + max(0, −log2 ε)`, with 60-digit decimals that never go through LOG2.

**The boundary pairs.** For the smallest `a` with `log2(a/b) ≥ k/2^F`, `b` near `2^62` and `F` up to 8, `a − 1` lies about `2^-62` below the boundary. Rounding a step of LOG2 up instead of down, its normalization, a squaring or a halving, crosses some of these boundaries. The oracle keeps four pairs `(a − 1, a)` for each step, found by running LOG2 with that step rounded up, which it uses only to choose them. A calculator that rounds any step up gives some pair a result above the oracle's.

Before it writes anything, the script checks that every FixedBits `r` satisfies `r ≤ 256·λ < r + 1 + 2^-20`, and every per-query count `r'` satisfies `r' ≤ 256·b(ρ') < r' + 1 + 2^-37`. It decides every pass or fail with exact integers, `c·2^T ≤ q` and `A^τ ≥ 2^T·B^τ`, and checks each decision against the true value wherever that lies further than `10^-30` from the target. It also checks the claims below:

- `live-64/1` and `ni-100/1` meet their targets at the 405B statement under both protocols, and `test-small/1` is held to none;
- under `tessera-protocol/1` every weight matrix counts `ϑ_T = ⌈log2 m_T⌉` and no mask, and under `tessera-protocol/2` its own blinding rows `B_T` and `P_W` mask coefficients: `B_W`, and for `ChannelParams` `B_W + 2·(e_CP − 1)·⌊max(B_W − 4, 0)/6⌋` from its claims `e_CP` (RFC-0065), so that its `ϑ` is 14 at 405B;
- each adversarial case fails exactly the row it aims at, which is then the minimum;
- at the 405B statement the rounds row is the merge's batching `P + G`, and the union row the ungrouped one's plus `G` copies of the leaf stage's levels and the merge's `2·n_max + P + G`;
- the edge cases decide as their names say.

## Sections

| Section | A case | Cases |
|---|---|---|
| `field` | `p` and `q = p^2`, decimal strings | one |
| `log2` | LOG2's `precision` `W` and its two output fractions | one |
| `log2_cases` | `a`, `b`, `fraction` and `result`, decimal strings | its edges; `q` against `2^128 − 1`; 24 random 128-bit inputs; and 12 pairs at a boundary, below |
| `parameter_sets` | a parameter set, as a report's `params` states it | RFC-0014's three sets |
| `cases` | `name`, `why`, `report` (a `tessera.soundness/1` report) and `true` (per row, `⌊256·λ·10^30⌋` as `bits`, and for a queries row `⌊256·b(ρ')·10^30⌋` as `per_query`) | 92, below |

The cases:

| Name | What it is |
|---|---|
| `405b/{set}` | each set at the 405B 16K-token, 16K-context statement, the CLI's default, under `tessera-protocol/1`: its weights carry no mask and no blinding row (RFC-0049) |
| `405b-protocol-2/{set}` | the same statement under `tessera-protocol/2`, whose weights count their `P_W` mask coefficients and their own blinding rows, `B_W` and `ChannelParams`' 12,280 at `e_CP = 7`: RFC-0014's `64.0` and `80.5` bits for `τ_W` |
| `adversarial/{set}/{row}` | a size or parameter that makes exactly `row` miss its target, at `live-64/1` and `ni-100/1`; the rounds row fails with the union, which counts its terms. A weight matrix's queries row misses its target by its width only under `tessera-protocol/2`: without a mask the rate is `1/4` at every width |
| `edge/numerator-pass`, `edge/numerator-fail` | the opening row at `T_g = ⌊q/2^100⌋` and `2^28`: `100 + 4.7·10^-9` bits passes and `100 − 6.7·10^-10` fails |
| `edge/logup-2^50-live` | `N + T = 2^50` at `live-64/1`: 78 one-shot bits pass its 64-bit target |
| `edge/protocol-2` | the 405B statement under `tessera-protocol/2` |
| `edge/empty` | no sumcheck, forest, instance, group or matrix: only the advice and the hash have terms |
| `edge/rate-one` | weights at rate 1, `ρ' ≥ 1`: their queries give no bit |
| `edge/line-only`, `edge/mask-only`, `edge/zero-count` | one term of the rounds row alone: a tree level's line challenge, a stage's mask in `tessera-protocol/2`, and rounds of count 0, which count nothing |
| `edge/hash-128` | targets of 128 bits, which only the hash row reaches |
| `edge/q-below-2^128` | `N + T = 2^64 − 1` at `live-64/1`: `64 − 6.7·10^-10` bits fail 64, where `2^128 − 1` in place of `q` would pass them |
| `edge/channel-params-at-a-power`, `edge/channel-params-past-a-power` | `ChannelParams` under `tessera-protocol/2` at test-small's `B_W = 32` and `e_CP = 7`, `B_CP = 80`: 944 data rows reach `2^{10}` (`ϑ = 10`), and 945 pass it (`ϑ = 11`), so a `B_CP` one row off either way changes a `ϑ` |
| `edge/blinding-below-4` | a parameter set of `B_W = 3`: `⌊max(B_W − 4, 0)/6⌋ = 0`, so `B_CP = B_W` at `e_CP = 2` and `ChannelParams`' `2 + 3` rows give `ϑ = 3` |
| `edge/empty-channel-params` | no weight has channel parameters: under `tessera-protocol/1` `ChannelParams` commits no row, so it has no queries row and no field term, and is not refused (RFC-0049) |
| `edge/groups-none` | the 405B statement were no grouping accepted (`groups.count` 1): the ungrouped rounds and union rows, 121.99 and 111.84 bits |
| `edge/groups-union-miss` | 6,000 level rounds per group at `G = 2^{14}`: the union row alone misses 100 bits at the largest grouping, which the statement without groups would pass (RFC-0054 'Testing Strategy') |
| `edge/groups-merge-only` | two groups, no level and a one-round merge of one claim: the rounds row is `P + G = 3` and the union `2·1 + 1 + 2 = 5` |
| `random/{i}` | random sizes under random parameter sets, or RFC-0014's: under `tessera-protocol/1` a random grouping of `2^0` to `2^{14}` groups, under `tessera-protocol/2` none |

## The report

A report is a `tessera.soundness/1` object ([schema](../../schemas/soundness-v1.schema.json)): `params`, `statement`, `rows`, `minimum` (an index into `rows`) and `meets_targets`. Every number in it is an exact integer, and a numerator, which reaches `2^128`, is a decimal string. A row holds:

| Field | Meaning |
|---|---|
| `row`, `commitment` | `rounds`, `union`, `forests`, `logup`, `openings`, `queries`, `field`, `linking` or `hash`; `advice` or a matrix's name for `queries` and `field` |
| `numerator` | `c` of a field-form bound `c/q` |
| `codeword`, `message`, `mask`, `queries` | `n`, `k`, `P` and `τ` of a queries bound `((n + k + P)/(2n))^τ` |
| `theta` | a field term's `ϑ` |
| `per_query_bits` | a queries row's `b(ρ') = −log2((1 + ρ')/2)`, FixedBits |
| `bits` | `λ` as a FixedBits lower bound, units of `2^-8` bits; `null` for no term |
| `grinding`, `covered_by` | the bits of the proof of work that covers the row, `seed-pow` or `query-pow` |
| `target`, `margin`, `pass` | the target in bits, `bits − 256·target`, and the exact decision; `null` under a set without targets |
