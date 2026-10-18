# Soundness calculator vectors

`soundness.json` pins the soundness calculator of [RFC-0014 'Soundness calculator'](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#soundness-calculator): every row of the soundness table, with its exact error bound, its bits as a FixedBits lower bound, its target and whether it reaches it. Both calculators must reproduce it bit for bit: `tessera-protocol`'s ([`crates/tessera-protocol/tests/soundness.rs`](../../../crates/tessera-protocol/tests/soundness.rs), and `tessera params check --json` in [`crates/tessera-cli/tests/params_check.rs`](../../../crates/tessera-cli/tests/params_check.rs)) and the verifier's (#110).

[`generate.py`](generate.py) writes it. The script is the independent oracle: it shares no code with `crates/`, `verifier/` or `python/`, and is written from RFC-0014, [RFC-0026 'Schema'](../../../docs/rfcs/RFC-0026-claim-list-layout.md#schema), [RFC-0040 'The matrices'](../../../docs/rfcs/RFC-0040-weight-commitment-artifacts.md#the-matrices), [RFC-0030 'Cost'](../../../docs/rfcs/RFC-0030-masked-exponentials.md#cost) and SP-1 ([06 §4](../../../docs/spec/06-security.md#4-security-properties)). It states the 405B lookups of the compiler's report, [`docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json`](../../../docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json), since it runs on `spec/vectors/` alone, and a `tessera-protocol` test checks the calculator's default against the report itself. It derives the 405B weight matrices from the architecture by RFC-0040's rules. Its random inputs come from SplitMix64 with the seed the file states, so `python3 spec/vectors/soundness/generate.py` reproduces the file byte for byte.

## How the oracle checks itself

Every bit count is computed twice:

- **by LOG2**, RFC-0014's algorithm, with Python's integers: the FixedBits both calculators must give;
- **as the true value** `λ = g + max(0, −log2 ε)`, with 60-digit decimals that never go through LOG2.

**The boundary pairs.** For the smallest `a` with `log2(a/b) ≥ k/2^F`, `b` near `2^62` and `F` up to 8, `a − 1` lies about `2^-62` below the boundary. Rounding a step of LOG2 up instead of down, its normalization, a squaring or a halving, crosses some of these boundaries. The oracle keeps four pairs `(a − 1, a)` for each step, found by running LOG2 with that step rounded up, which it uses only to choose them. A calculator that rounds any step up gives some pair a result above the oracle's.

Before it writes anything, the script checks that every FixedBits `r` satisfies `r ≤ 256·λ < r + 1 + 2^-20`, and every per-query count `r'` satisfies `r' ≤ 256·b(ρ') < r' + 1 + 2^-37`. It decides every pass or fail with exact integers, `c·2^T ≤ q` and `A^τ ≥ 2^T·B^τ`, and checks each decision against the true value wherever that lies further than `10^-30` from the target. It also checks the claims below:

- `live-64/1` and `ni-100/1` meet their targets at the 405B statement, and `test-small/1` is held to none;
- each adversarial case fails exactly the row it aims at, which is then the minimum;
- the edge cases decide as their names say.

## Sections

| Section | A case | Cases |
|---|---|---|
| `field` | `p` and `q = p^2`, decimal strings | one |
| `log2` | LOG2's `precision` `W` and its two output fractions | one |
| `log2_cases` | `a`, `b`, `fraction` and `result`, decimal strings | its edges; `q` against `2^128 − 1`; 24 random 128-bit inputs; and 12 pairs at a boundary, below |
| `parameter_sets` | a parameter set, as a report's `params` states it | RFC-0014's three sets |
| `cases` | `name`, `why`, `report` (a `tessera.soundness/1` report) and `true` (per row, `⌊256·λ·10^30⌋` as `bits`, and for a queries row `⌊256·b(ρ')·10^30⌋` as `per_query`) | 82, below |

The cases:

| Name | What it is |
|---|---|
| `405b/{set}` | each set at the 405B 16K-token, 16K-context statement, the CLI's default |
| `adversarial/{set}/{row}` | a size or parameter that makes exactly `row` miss its target, at `live-64/1` and `ni-100/1`; the rounds row fails with the union, which counts its terms |
| `edge/numerator-pass`, `edge/numerator-fail` | the opening row at `T_g = ⌊q/2^100⌋` and `2^28`: `100 + 4.7·10^-9` bits passes and `100 − 6.7·10^-10` fails |
| `edge/logup-2^50-live` | `N + T = 2^50` at `live-64/1`: 78 one-shot bits pass its 64-bit target |
| `edge/protocol-2` | the 405B statement under `tessera-protocol/2` |
| `edge/empty` | no sumcheck, forest, instance, group or matrix: only the advice and the hash have terms |
| `edge/rate-one` | weights at rate 1, `ρ' ≥ 1`: their queries give no bit |
| `edge/line-only`, `edge/mask-only`, `edge/zero-count` | one term of the rounds row alone: a tree level's line challenge, a stage's mask in `tessera-protocol/2`, and rounds of count 0, which count nothing |
| `edge/hash-128` | targets of 128 bits, which only the hash row reaches |
| `edge/q-below-2^128` | `N + T = 2^64 − 1` at `live-64/1`: `64 − 6.7·10^-10` bits fail 64, where `2^128 − 1` in place of `q` would pass them |
| `random/{i}` | random sizes under random parameter sets, or RFC-0014's |

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
