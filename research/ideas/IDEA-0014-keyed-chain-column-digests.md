# IDEA-0014: Keyed-chain column digests: the chain value as the BLAKE3 key so every 32-row block fills whole compression blocks

| | |
|---|---|
| Status | `triaged` (2026-10-16) |
| Origin | [SUR-0006](../surveys/SUR-0006-hash.md#findings) finding [BLAKE3ref20] and synthesis 3, candidate idea "Chain value as the BLAKE3 key (free-start chaining)"; supersedes [EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md)'s extra finding `B = 28` (ledger queue item 18); RISK-8 |
| Lever | `c_c` (the hash share: BLAKE3 compressions per committed element 0.3125 → 0.25, the rate-1/2 floor of [E0-hash]); the block buffer, `B` and the 32-byte chain state are unchanged |
| Spec surfaces | [RFC-0004 'Hash function and domains'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#hash-function-and-domains) (the `column-step` row leaves the `derive_key` table and becomes a keyed-mode rule), ['Column digests'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests) (the step formula), 'Testing Strategy' (official `keyed_hash` vectors next to the `derive_key` ones); [RFC-0011 'Column digests and root'](../../docs/rfcs/RFC-0011-advice-commitment.md#column-digests-and-root) (`B = 32` stays, the digest rule it cites changes); [RFC-0018](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md) kernel row "BLAKE3 column-chain" (byte-identical to the new CPU reference); [RFC-0019](../../docs/rfcs/RFC-0019-verifier.md) `tv-hash` exposes keyed mode. The chain layout is part of the commitment ([RFC-0004 Drawbacks](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#drawbacks)): the commitment descriptor, `commitment_id` and every leaf change, so this is a tier-A change ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)) needing a superseding RFC and a protocol version, cheap before the `tessera-protocol/1` freeze at v0.3 and best bundled with any other descriptor change ([IDEA-0002](IDEA-0002-coefficient-form-rows.md)) |
| Priority | 0.21 (−1.4% × 0.6 / 4 at 16K under the `counted` profile: a tier-A superseding RFC by the rubric; 0.84 at the marginal cost `s` if bundled into the descriptor RFC of [IDEA-0002](IDEA-0002-coefficient-form-rows.md)) |

## Mechanism

**Today** ([RFC-0004 'Column digests'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests), read 2026-10-16):

```text
d_j[0]   = H("column-init",  commitment_id ‖ u64(j))
d_j[i+1] = H("column-step",  d_j[i] ‖ symbols of rows iB .. min((i+1)B, m) − 1 at column j)
```

with `H(domain, bytes) = BLAKE3.derive_key("tessera/v1/" + domain, bytes)`. At `B = 32` the step's key material is `32 + 8 × 32 = 288` bytes = 4.5 BLAKE3 blocks, so each step costs 5 compressions (the fifth over a half-empty block), 0.3125 compressions per committed element at rate 1/2 ([EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md) compression table: `288 → 5`, `256 → 4`; E0 count fitted to E1). The chain value is paid for as input bytes.

**Change.** The chain value enters as BLAKE3's key, that is as the initial chaining value of the compression, which is where BLAKE3 itself puts the previous block's output inside a chunk ([BLAKE3ref20]: `compress(chaining_value, block, counter, block_len, flags)`, `new_keyed` replaces the IV by the 32-byte key with the `KEYED_HASH` flag):

```text
d_j[0]   = H("column-init", commitment_id ‖ u64(j))                            (unchanged: derive_key)
d_j[i+1] = BLAKE3.keyed_hash(key = d_j[i], message = TCE(symbols of rows iB .. min((i+1)B, m) − 1 at column j))
leaf_j   = d_j[ceil(m / B)]                                                     (unchanged)
```

- The message is exactly the symbols' 8-byte TCE encodings, concatenated, no length prefix (as today: `B` and `m` fix how many there are). A full block is 256 bytes = 4 blocks of one BLAKE3 chunk = 4 compressions, the last with `CHUNK_END | ROOT`; a partial last block (`m mod B ≠ 0`) is `8 × (m mod B)` bytes, as today.
- Domain separation moves to two places: `column-init` keeps `derive_key` with `commitment_id`, so chains of different commitments and different columns start from independent keys; every step output carries the `KEYED_HASH` flag, which no `derive_key` output in RFC-0004's table carries (`DERIVE_KEY_MATERIAL`), so a step digest can never be confused with a `merkle-node`, `column-merge` or transcript digest on equal bytes. `column-merge` ([RFC-0011](../../docs/rfcs/RFC-0011-advice-commitment.md#column-digests-and-root)) and the Merkle nodes do not change.
- `B` stays 32 and the chain state stays 32 bytes per column per segment, so [RFC-0011](../../docs/rfcs/RFC-0011-advice-commitment.md#column-digests-and-root)'s `StreamingCommitter` state, the 2 GiB block buffer and the 256 MiB of chain values ([RFC-0004](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests)) are untouched. The queued `B = 28` idea (ledger item 18), which only reaches 0.286 compressions per element by filling blocks under the `derive_key` layout, is superseded: the floor 0.25 is reached at the specified `B`.
- **Prover.** `tessera_hash::ColumnChains::absorb_block` (`crates/tessera-hash/src/columns.rs` lines 84–114, read 2026-10-16) builds one contiguous `chain ‖ symbols` buffer per column and calls `hash(Domain::ColumnStep, &input)`; it becomes `blake3::Hasher::new_keyed(chain).update(symbols).finalize()` with the same gather loop (the `blake3` 1.x crate the prover already uses has `new_keyed`). The precomputed `derive_key` context keys of `domain.rs` (a `LazyLock`, line 83) lose the `ColumnStep` entry. The CUDA kernel of RFC-0018 does the same: a keyed compression is the plain compression with a different initial state and flag word.
- **Verifier.** `tv-hash` implements BLAKE3 from the specification; its `Hasher::new(key, flags)` already takes `KEYED_HASH` (`verifier/crates/tv-hash/src/blake3.rs` lines 41 and 900–909, read 2026-10-16; the constructor is `pub(crate)`), so keyed mode is a public function plus the official keyed vectors, written from the RFC. The per-column recomputation of an opened column ([RFC-0004](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests)) gets one fewer compression per full block.

## Why it could work

- Compressions per committed element 0.3125 → 0.25, −20% of the hash: E0, primary. 64-byte blocks and the single-chunk structure for inputs below 1,024 bytes are the BLAKE3 reference implementation ([BLAKE3ref20]); the counts `256 B → 4`, `288 B → 5` are [EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md)'s compression table (E0, fitted to E1 within 5 ns per call). [E0-hash] ([SUR-0006](../surveys/SUR-0006-hash.md#findings)) names 0.25 as the rate-1/2 floor: every block full of symbols.
- Hash share of `c_c`: 248 of 969 (low translation) to 300 of 1,574 (high) int32 operations per committed element, 19–26% ([EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md), E0 counts fitted to E1 on an Apple M4 Max, primary). −20% of that is −3.8% to −5.1% of `c_c` (E0).
- CPU anchor: a `column-step` call costs 5.71 ns plus 50.13 ns per compression at 288 bytes ([EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md), E1, primary, Apple M4 Max, one core), and `absorb_block` over 2^23 columns at `B = 32` measured 19.22 ns per committed element against a predicted `(5.71 + 5 × 50.13) / 16 = 16.02` ns. Removing one compression per 16 committed elements saves `50.13 / 16 = 3.13` ns: the prediction falls to 12.89 ns (−19.6%) and the measurement, if the gather cost is unchanged, to about 16.1 ns (−16%); [IDEA-0002](IDEA-0002-coefficient-form-rows.md)'s independent 19.24 ns per committed element for the same kernel would fall to about 16.1 ns (E0 derived from E1).
- GPU frame: [E0-hash] puts the hash alone at 24.8–30.0 ps per element at 60% utilization of the H100 integer pipeline, 50–100% of the whole 30 ps `c_c` budget (E0, primary counts over documentation figures); [SUR-0006](../surveys/SUR-0006-hash.md#synthesis) synthesis 1 and the hashcat anchor [OpenBench25] (secondary, H100) make the integer pipe the binding resource for the hash, so every compression removed is budget recovered where it is scarcest.
- Nothing else moves the hashed volume at fixed rate: the Merkle tree is 10^-5 of the hashing, arity and caps move only the multiproof, column batching is what the chained digest already is ([SUR-0006](../surveys/SUR-0006-hash.md#synthesis) synthesis 3, E0); rate is priced separately by [EXP-0004](../experiments/EXP-0004-advice-rate-quarter/README.md). Hashing only systematic symbols (OQ-0004-1) is unsound (synthesis 4). This idea is the last volume lever at rate 1/2.

## Expected gain

The idea moves `commit_ps` by the hash share times −20%. Under the `counted` profile (`c_c = 94` ps, EXP-0002) the bracket −3.8% to −5.1% is `commit_ps` 90.4 to 89.2; the centre 89.5 (−4.8%) is the scenario:

```sh
python3 -I research/tools/cost_model.py --profile counted --set commit_ps=89.5
```

```text
baseline: profile 'counted'
   S   linear  attention  weights  regen  streaming    total
   1K     9.20       3.70    0.021    4.0       0.42     17.3
   4K     9.06      14.59    0.021    4.0       2.32     30.0
  16K     8.54      54.98    0.020    4.0      11.24     78.8
 128K     5.55     285.91    0.013    4.0      77.93    373.4

scenario: commit_ps=89.5
   S   linear  attention  weights  regen  streaming    total
   1K     8.99       3.64    0.021    4.0       0.42     17.1   Δ   -1.6%
   4K     8.85      14.34    0.021    4.0       2.32     29.5   Δ   -1.5%
  16K     8.34      54.06    0.020    4.0      11.24     77.7   Δ   -1.4%
 128K     5.42     281.11    0.013    4.0      77.93    368.5   Δ   -1.3%
```

Bracket ends, same command with `--set commit_ps=89.2` (hash share 26%): −1.7 / −1.6 / −1.5 / −1.4%; with `--set commit_ps=90.4` (share 19%): −1.3 / −1.2 / −1.1 / −1.1% at 1K / 4K / 16K / 128K (E0, run 2026-10-16). Under the `corrected` profile (`c_c = 30` ps assumed), `python3 -I research/tools/cost_model.py --set commit_ps=28.6` (−4.8%) gives −1.0 / −1.0 / −1.0 / −0.9% (E0, run 2026-10-16).

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | −1.6% counted (bracket −1.3% to −1.7%); −1.0% corrected | 0 (the opened columns send all `m` symbols either way) | −20% of the column-chain compressions the verifier recomputes over the opened columns (0.72 GB live, [IDEA-0001](IDEA-0001-list-decoding-proximity.md)); E0 |
| 4K | −1.5% counted (−1.2% to −1.6%); −1.0% corrected | 0 | same |
| 16K | −1.4% counted (−1.1% to −1.5%); −1.0% corrected | 0 | same |
| 128K | −1.3% counted (−1.1% to −1.4%); −0.9% corrected | 0 | same |

Evidence levels of the inputs: compression counts E0 (primary, [BLAKE3ref20], EXP-0002); hash share 19–26% E0 fitted to E1 (EXP-0002, Apple M4 Max; the GPU share is unknown until the v0.1 gate, E2); `c_c = 94` ps counted (E0 from E1) and 30 ps corrected (assumed, RFC-0020 gate); the cost model's formula and the other unit costs as in [baseline.md §2](../baseline.md#2-the-cost-model). All of it is a projection (08 §1). The gain is bounded by the hash share: if the fused kernel hides the hash behind the encoder's FMA-pipe work ([SUR-0006](../surveys/SUR-0006-hash.md#candidate-ideas) candidate "Hash on the integer pipe"), the saving shrinks toward 0; if the hash is the binding pipe, it is realized in full.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Binding of a column leaf changes from collision resistance of `derive_key(column-step, chain ‖ symbols)` to collision resistance of BLAKE3's keyed mode as a function of the pair `(key, message)`, where the key is always a prior hash output (never attacker-chosen directly): a leaf collision needs two pairs with equal output, either under one key (a plain 256-bit-output collision) or across keys (the free-start form of the compression function with the `KEYED_HASH` flag set), which is exactly the assumption BLAKE3's own intra-chunk chaining makes ([BLAKE3ref20]). Both are generic at 2^128, the level [RFC-0014](../../docs/rfcs/RFC-0014-interaction-and-soundness.md) already books for "hash collisions, BLAKE3, 256-bit output: 128 bits". Domain separation from every other digest is by flag (`KEYED_HASH` against `DERIVE_KEY_MATERIAL`) and from other columns and commitments by `column-init`'s `derive_key` over `commitment_id ‖ u64(j)`. No transcript change: INV-FS-1..4 untouched, no challenge moves. The superseding RFC must state the assumption in these words and the `soundness-review` skill must read it. | pass (conditional on the restatement being accepted in review) |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Commitment layer only; no relation, table, range or witness changes. | pass |
| Zero knowledge (G-5, RFC-0013) | The leaf of a column of hidden symbols is a hash of the same symbols as today under a key that is itself public-derivable from `commitment_id`; mask rows and blinding are unchanged, and the digest leaks the same (negligible) information as today. The verifier's hashing is on opened columns only, as before. | pass |
| Verifier independence (SP-6) | `tv-hash` implements BLAKE3 from the specification; keyed mode is part of that specification with official vectors, and its hasher already takes the `KEYED_HASH` flag (`verifier/crates/tv-hash/src/blake3.rs`). A public `keyed_hash` written from the RFC, no prover code read. `allowlist.toml` unchanged. | pass |
| Post-quantum (hash-based commitments) | Still a hash-based commitment over a 256-bit state; Grover's bound on second preimages and the BHT collision bound are unchanged. | pass |

Robustness (SP-8, not a gate row): the verifier's recomputation gets strictly simpler (fixed 256-byte messages for full blocks, no `chain ‖ symbols` buffer). Redaction (INV-ARCH-6): no new secret flow; the chain values were and remain commitment-class data.

## Prior rejection check

- [RFC-0004 Alternatives Considered](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#alternatives-considered) 1–6 (SHA-256, SHA-3, Poseidon2, BLAKE2s, STROBE, one hasher per column) choose the primitive, not the layout of the step input; none covers the chain value as key. Alternative 6's objection (1.9 KB hasher state per column, 16 GB) does not arise: the chain state stays 32 bytes, and the keyed hasher is constructed and dropped within one step as `derive_key` is today.
- [RFC-0004 Drawbacks](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#drawbacks) acknowledges that the chain layout and `B` are part of the commitment, hence the tier-A cost; it does not reject changing them before the v0.3 label freeze.
- [RFC-0011 Alternatives Considered](../../docs/rfcs/RFC-0011-advice-commitment.md) 1–5 (storing the encoded witness, FRI-family PCS, Brakedown, systematic encoding, per-tensor commitments) do not touch the digest rule.
- [baseline.md §7](../baseline.md#7-already-rejected) rows "SHA-256, SHA-3, BLAKE2s, Poseidon2 for the commitment path" and "One streaming hasher per column" are the same two items; neither has a "Reopens if" that this idea needs.
- Not previously rejected: none. What is new: EXP-0002's compression table (E0/E1) showing that the specified layout pays a fifth, half-empty compression per step, and SUR-0006's reading of the reference implementation ([BLAKE3ref20]) that the chaining value is a free input. It supersedes the queued `B = 28` note (ledger item 18), which reaches 0.286 compressions per element against 0.25 here at the specified `B = 32`.

## Kill criterion

On the same core and harness as EXP-0002's chain kernel, `keyed_hash` over 256 bytes at `B = 32` is not at least 10% faster per symbol than `derive_key` over 288 bytes, because the per-call overhead or the column gather, not the compression count, bounds the kernel ([IDEA-0003](IDEA-0003-fused-encode-hash-tiles.md) measured the gather binding `ColumnChains` at about −5% of `c_c` for a −15% compression change). Independent killers: the soundness review judges the free-start chaining assumption weaker than RFC-0004's current statement and accepts no restatement at 128 bits; or the v0.1 gate (E2) shows the hash below 10% of the fused kernel's time, making the saving invisible in `c_c`.

## Cheapest experiment

**Hypothesis.** At `B = 32`, replacing `derive_key(column-step, chain ‖ symbols)` by `keyed_hash(chain, symbols)` removes one of five compressions per step (E0) and lowers the chain kernel's time per committed element by 10% to 20% on the CPU (E1), with byte-identical results between an implementation over the `blake3` crate and an independent recomputation.

**Method** (E1 here, effort `s`): a scratch Cargo workspace (empty `[workspace]` table) depending on `tessera-hash` and `tessera-field` by path, one Apple M4 Max performance core, the RFC-0020 protocol as EXP-0002 used it (fastest of 5, pinned sizes, in-cache unit costs reported beside the full-size run):

1. Compression-count check: 256-byte keyed and 288-byte `derive_key` calls against the reference structure of [BLAKE3ref20] (E0), and ns per call for both at 64 to 1,024 bytes to refit the `a + b × compressions` line of EXP-0002.
2. ns per symbol for the chain step at `B = 32` under both layouts over 2^21 columns, through the `ColumnChains::absorb_block` gather path and through a column-major tile layout (as IDEA-0003's E1 sweep separated gather from compression), so the kill criterion is tested on the kernel and not on the gather.
3. An independent recomputation of 64 sampled leaves under the keyed layout (a second implementation from the formula above, not the harness's), to pin the specification text, the partial-last-block case (`m mod B ≠ 0`) and the official keyed vectors.

**Expected.** −20% compressions (E0); −10% to −20% chain time (E1). **Unknown until an H100 run** (E2, the v0.1 gate): the hash's share of the fused commit kernel on the integer pipe, hence the realized share of the −3.8% to −5.1% of `c_c`; [SUR-0006](../surveys/SUR-0006-hash.md#candidate-ideas)'s seed "Separate hash and encoder in the RFC-0020 benchmark, with a keyed-chain variant" asks for exactly this variant to be measured there.

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict, with the cost-model reruns the scoring agent made to check disputed cells noted as such.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | A leaf collision under the keyed chain needs either a same-key message collision or a (key, message) pair collision of the compression with the `KEYED_HASH` flag, the free-start form BLAKE3's own intra-chunk chaining already assumes (SUR-0006; [BLAKE3ref20] `self.chaining_value = first_8_words(compress(&self.chaining_value, …))`); both are generic 2^128 events, the level RFC-0014 already books for 256-bit collisions, so no bound moves and no challenge is reordered (the chain is outside the transcript, RFC-0004 'Column digests'). Domain separation is structural: every step output carries `KEYED_HASH` while every other digest in RFC-0004's table carries `DERIVE_KEY_MATERIAL`, `block_len` is a compression input so a partial last block (`m mod B ≠ 0`) cannot collide with a full one, and `B`, `m` and `commitment_id` are public, so ADV-P cannot shorten or re-split a column. ADV-V learns nothing new: the digest covers the same symbols under a key derivable from public data, masks and blinding rows untouched. Verified in code: the prover builds `chain ‖ symbols` and calls `hash(Domain::ColumnStep, …)` (crates/tessera-hash/src/columns.rs:103-113), and `tv-hash` already has `pub(crate) fn new(key, flags)` with `KEYED_HASH` and tests the official `keyed_hash` vectors (verifier/crates/tv-hash/src/blake3.rs:41, 902, 1072), so the verifier side is a visibility change written from the spec. Condition, not a kill: the superseding RFC states the free-start assumption in words and keeps `column-init` as the per-column, per-commitment separator. |
| prior art | survives | RFC-0004's Alternatives Considered 1–6 choose the primitive and reject one 1.9 KB hasher state per column; none addresses the layout of the `column-step` input, and the keyed form keeps the 32-byte chain state Alternative 6 demanded, so no prior rejection covers it. The rule is fixed at RFC-0004 'Column digests' as `derive_key(column-step, chain ‖ symbols)` and its security sentence rests on `derive_key` domain separation, so the idea is a genuine tier-A change (RFC-0004 Drawbacks) rather than something already specified; the verifier's `KEYED_HASH` flag exists only under `#[cfg(test)]` (blake3.rs:40-41), confirming keyed mode is not yet a specified surface. The compression counts are EXP-0002's measured table (E0 fitted to E1, primary), and the note correctly supersedes the queued `B = 28` item (ledger 18: 0.286 against 0.25 compressions per element). New: the chain value as BLAKE3's initial chaining value, which SUR-0006 seeded and nothing in baseline.md §7 rejects. |
| cost model | weakened | The scenario reproduces (counted `commit_ps` 94 → 89.5: Δμ −1.6 / −1.5 / −1.4 / −1.3% at 1K / 4K / 16K / 128K; bracket 90.4 gives −1.1% at 16K; corrected 28.6 gives −1.0%), and `c_c` matters (29.7% of μ(16K) under `counted`). But the move is 4.8% of a term whose own input spans 2.7× (EXP-0002: `c_c ∈ [57.9, 156.8]` ps) and whose hash share is itself a 19–26% bracket, so the gain is an order of magnitude below the uncertainty of every input it multiplies; the note concedes it shrinks toward 0 if the hash hides behind the encoder's pipe, and IDEA-0003 measured the gather, not the compression count, binding `ColumnChains`. E1 support is real but indirect: EXP-0002's `B` sweep shows chain time tracking compressions per symbol (`B = 28`: −8.6% compressions, −8.7% time), so the −20% compression count should land on the CPU, yet a −1.4% μ change is invisible to the cost model at any profile: a free tier-A tweak to bundle, not an idea to score on Δμ. |
| feasibility | survives | The prover's step is one `derive_key` call over a `chain ‖ symbols` buffer (columns.rs:104-113) and `blake3 = "1.8"` (crates/tessera-hash/Cargo.toml:12) has `Hasher::new_keyed`, so the prover change is a few lines; on the verifier `tv-hash` implements keyed mode but gates the flag `#[cfg(test)]` and keeps `Hasher::new` `pub(crate)` (blake3.rs:40-41, 902), so exposing it is a small, spec-written addition with the official `keyed_hash` vectors already in the test table (blake3.rs:1072). The count 288 B → 5 against 256 B → 4 compressions is E0 primary (BLAKE3 64-byte blocks, EXP-0002's table), so the −20% figure is sound. The feasibility cost is the tier-A hit, correctly acknowledged: the leaf rule changes `commitment_id` and every leaf (02 §1 row A), hence a superseding RFC on RFC-0004/RFC-0011 and a protocol version, cheap only if bundled before the v0.3 freeze. The cheapest experiment reaches E1 on this CPU as claimed (EXP-0002's harness), though IDEA-0003's E1 finding that the gather binds `ColumnChains` (about −5% of `c_c` for a −15% compression change) makes the 10% kill threshold a real risk; the realized GPU share remains E2. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −1.4% μ(16K), −1.6% at 1K (E0 cost model at `commit_ps=89.5` under `counted`; bracket −1.1% to −1.5%; −1.0% under `corrected`); compressions per element 0.3125 → 0.25 is E0 primary ([BLAKE3ref20], EXP-0002); the hash share 19–26% of `c_c` is E0 fitted to E1 (EXP-0002, Apple M4 Max) | 0.6: the mechanism and the compression count are exact and primary and the CPU anchor is E1, but the realized share depends on whether the hash binds the fused GPU kernel (E2) and IDEA-0003 found the gather, not the compression count, binding `ColumnChains` | `l` by the rubric: a tier-A superseding RFC on RFC-0004 'Column digests' and RFC-0011 'Column digests and root' plus a protocol version; the E1 experiment itself is `s`, and the marginal cost is `s` when bundled into IDEA-0002's descriptor RFC | 1.4 × 0.6 / 4 = 0.21 (0.84 at the bundled marginal cost `s`) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.21. The gain is inside the cost model's noise and no experiment here can lift it above it (the hash's share of the fused GPU kernel is E2), so it earns no experiment of its own; its place is a clause in the superseding RFC that changes the commitment descriptor anyway (IDEA-0002's coefficient form), where it costs one sentence, the free-start restatement of RFC-0004's binding assumption that the soundness lens requires, and buys one compression per full block. Its E1 step (keyed against `derive_key` through `ColumnChains` and a tile layout) rides on EXP-0002's harness when that RFC is drafted. The queued `B = 28` item (ledger 18) is superseded by this note. Experiments: none. RFC or issue: none of its own (a clause for the superseding RFC of IDEA-0002 on RFC-0004 'Column digests' and RFC-0011 'Column digests and root'). Ledger line updated on: pending consolidation (2026-10-16).
