# IDEA-0003: Fused encode-and-hash on row-block tiles so the codeword never round-trips HBM (with `B = 128`)

| | |
|---|---|
| Status | `shelved` (H100) (2026-10-16): the decisive measurement is an unfused/fused kernel pair on an H100 (E2, the RFC-0020 v0.1 gate); the consolidation decision is recorded in [LEDGER.md](../LEDGER.md) |
| Origin | new; seeded by [SUR-0003](../surveys/SUR-0003-passes.md) synthesis 6 and its candidate "Re-derive `p_s` from bytes" ([DMABKGR25]: sumcheck-style kernels are paid in bytes), [RFC-0011 'Streaming procedure'](../../docs/rfcs/RFC-0011-advice-commitment.md#streaming-procedure-witness-pass) (the block buffer) and its v0.1 "GPU encode-and-hash microkernel" ([Migration / Rollout](../../docs/rfcs/RFC-0011-advice-commitment.md#migration--rollout)), RISK-8, OQ-0004-1 |
| Lever | `c_c` (HBM bytes per committed element and BLAKE3 compressions per symbol); frees the 2 GiB block buffer of RFC-0011 'Streaming procedure' (`stream`, marginal: 2 GiB against 0.2 TB free) |
| Spec surfaces | At `B = 32`: none; a prover kernel under [RFC-0018 'Kernels'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#kernels) and a wording change to RFC-0011 'Streaming procedure' (the block buffer becomes an implementation choice). At `B = 128`: [RFC-0004 'Column digests'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests) and [RFC-0011 'Column digests and root'](../../docs/rfcs/RFC-0011-advice-commitment.md#column-digests-and-root) (`B` is in the commitment descriptor and `commitment_id`; "changing `B` changes the commitment", RFC-0004 Drawbacks): a parameter-set change, tier A ([02 §5](../../docs/spec/02-public-api.md#5-parameter-sets-tier-a)), superseding RFC required |
| Priority | 0.08 (−1.1% × 0.3 / 4); the cost-model lens refutes the gain against the baseline |

## Mechanism

Today ([RFC-0011 'Streaming procedure'](../../docs/rfcs/RFC-0011-advice-commitment.md#streaming-procedure-witness-pass)): for each non-zero row, regenerate, mask, encode (inverse NTT of size `k = 2^22`, coset NTT of size `n = 2^23`, ['Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding)), append the codeword to a block buffer of `B · n · 8` bytes (2 GiB at `B = 32`), and every `B` rows advance all `n` column chains from that buffer ([RFC-0004 'Column digests'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests): `d_j[i+1] = H("column-step", d_j[i] ‖ B symbols)`). The codeword is written to HBM and read back: 16 B written and 16 B read per committed element (two 8-byte symbols at rate 1/2), on top of the NTT's own intermediates.

Change, in three parts:

1. **Batch the block.** Encode the `B` rows of a block as one batched transform with the row index innermost, so every intermediate is stored as tiles `(columns × B)`. Each transform of size `2^22` or `2^23` is a multi-step (four-step) NTT `n = n_1 × n_2`; the sub-transforms of a step run in shared memory and the step boundaries are the only HBM round trips.
2. **Hash from the tile.** The last step of the coset NTT produces, per thread block, `n_2` output columns for each of the `B` rows: a tile of `B × n_2 × 8` bytes in shared memory (`B = 32, n_2 = 256`: 64 KB; `B = 128, n_2 = 64`: 64 KB; the H100 allows up to 227 KB per block). The column-step hash reads the `B` symbols of each of the `n_2` columns straight from the tile, in row order as RFC-0004 requires, and writes only the 32-byte chain value. The block buffer disappears, the codeword never reaches HBM, and the chain state stays `32 · n` bytes (RFC-0004 Alternatives Considered 6 respected).
3. **Raise `B` to 128.** With the buffer gone, `B` is bounded by the shared-memory tile (`B · n_2 · 8 ≤ ~100 KB`), not by HBM. A column step then hashes `32 + 1024 = 1056` B (17 BLAKE3 compressions per 128 symbols) instead of `288` B (5 per 32 symbols), and the chain traffic falls from 2 B to 0.5 B per symbol. `B` is a public descriptor parameter, so this part changes the commitment (tier A); parts 1 and 2 do not.

Determinism: a column chain depends only on the ordered symbols of its column, so the tile decomposition and launch order cannot change a digest; the CPU reference (`tessera-hash::ColumnChains`) stays the oracle ([RFC-0004 'Testing Strategy'](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#testing-strategy): CPU, GPU and verifier chains agree).

The decisive implementation constraint, stated here because it decides the kill criterion: the tile bound `n_2 ≤ 256` (at `B = 32`) leaves `n_1 = 2^15` for the first step of the `2^23`-point transform, 256 KB per sub-transform, more than one block's shared memory. Either a Hopper thread-block cluster computes it in distributed shared memory (2 × 227 KB), or the first step splits again and adds one HBM round trip. At `B = 128`, `n_2 ≤ 64` and `n_1 = 2^17` (1 MB): the extra round trip is unavoidable, so part 3 costs part 2 some of its bytes.

## Why it could work

- **The specified pipeline is at or above the byte budget by itself.** E0, primary (own count, script `idea0003_bytes.py` in the session scratchpad; assumptions in the table below): per committed element the unfused pipeline moves about 108 B through HBM, which is 32 ps at the H100's 3.35 TB/s [NVIDIA-H100] (secondary, vendor table) and 40 ps at 80% achieved bandwidth, above the 30 ps `c_c` of [baseline.md §2](../baseline.md#2-the-cost-model). The block buffer alone is 32 B of the 108 (30%).

  | Item, per committed element (2 symbols) | Unfused (spec) | Fused, `B = 32` | Fused, `B = 128` | Assumption |
  |---|---|---|---|---|
  | Row data read | 8 B | 8 B | 8 B | read once |
  | `2^22`-point inverse NTT intermediate | 16 B | 16 B | 16 B | two steps, sub-transforms ≤ 2^12 points in shared memory |
  | Coefficients written and re-read | 16 B | 16 B | 16 B | the two transforms do not fuse through shared memory |
  | `2^23`-point coset NTT intermediate | 32 B | 32 B | 32 B | two steps (one round trip of 2 symbols) |
  | Extra NTT step when `n_1` does not fit | 0 | 0 or +32 B | +32 B | see Mechanism |
  | Codeword to block buffer and back | 32 B | 0 | 0 | the fusion |
  | Chain state read and written | 4 B | 4 B | 1 B | 64 B per column step, amortized over `B` symbols |
  | **Total** | **108 B** | **76 B (108 with the extra step)** | **73 B (105 with the extra step)** | |
  | At 3.35 TB/s / at 80% | 32 / 40 ps | 23 / 28 ps (32 / 40) | 22 / 27 ps (31 / 39) | E0 |

- **The kernel's compute side is above the budget too, so bytes are necessary, not sufficient.** E0, primary (same script): BLAKE3 at about 15 int32 operations per byte (RFC-0004 Alternatives Considered 1, secondary) over 16 payload bytes padded to whole compressions is 300 ops at `B = 32` and 255 at `B = 128`; the 34 butterflies per committed element (11 for the inverse NTT, 23 for the coset NTT) at 18 int32 ops per Goldilocks multiplication ([SUR-0002](../surveys/SUR-0002-lookups.md) [E0-this]) plus 6 for add, sub and reductions are 816 ops; total 1,070–1,120 ops, which is 64–67 ps at the int32 peak of 1.67×10^13 ops/s (132 SMs × 64 lanes × 1.98 GHz) and 107–111 ps at the 60% utilization ZKProphet measures for field-multiplication kernels ([VYTA25] via SUR-0002, primary). With a 12-op multiplication: 52–55 ps at peak. Under these assumptions the fused kernel is compute-bound and the fusion's saving is invisible; see Kill criterion. The assumptions are the thing to measure.
- **`B = 128` lowers the hash work per symbol.** E0: compressions per symbol 5/32 → 17/128, −15%. E1, primary (this session, Apple M4 Max, one core, `blake3` 1.8.7 `derive_key` with the `tessera/v1/column-step` context, median of 7, scratchpad program `bsweep`): 9.79 ns per symbol at 288 B, 7.98 at 544 B, 7.31 at 1056 B, 6.86 at 2080 B: −25% from `B = 32` to `128` (the generation pass measured 10.04 → 7.27, −28%, same machine). Through `tessera-hash::ColumnChains::absorb_block` (4096 columns × 2048 rows, same machine): 8.45 → 7.77 → 7.99 → 8.08 ns per symbol for `B = 32, 64, 128, 256` (E1, primary): only −5%, because the CPU path gathers one 8-byte symbol per column from row-major rows (a 32 KB stride) and that gather, not the compression, bounds it. This is the CPU form of the thesis: the chain hash is paid in data movement, and the layout (the tile) matters more than the compression count.
- **The shape is already in the specification for the leaves.** [RFC-0018 'Pass structure'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure) ("Fusion") fuses P2's first streamed round with the bottom-up tree build so a level is regenerated into registers and never written; [SUR-0003](../surveys/SUR-0003-passes.md) synthesis 6 reads [DMABKGR25] (primary, E0 modelling on a 255-bit field) as saying such kernels are paid in bytes. This idea applies the same principle to the commitment, whose per-element budget (30 ps) is tighter than the leaf budget's.

## Expected gain

The cost model sees `c_c` only, so the idea is expressed as the `commit_ps` it protects or lowers. Three scenarios, all E0, baseline profile `corrected`:

1. **What the fusion removes.** An unfused kernel pinned at its HBM floor at 80% achieved bandwidth is `c_c ≈ 40 ps` (own count) to 45 ps (the generation pass's count, which charged the `2^23`-point NTT two full passes); the fusion brings the floor back under 30 ps. Against the 30 ps baseline the gain is 0 by construction; against the unfused kernel it is the penalty below.

```sh
python3 -I research/tools/cost_model.py --set commit_ps=45
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | +10.3% (unfused at 45 ps; fusion removes it) | 0 | 0 |
| 4K | +10.8% | 0 | 0 |
| 16K | +10.6% | 0 | 0 |
| 128K | +9.6% | 0 | 0 |

2. **`B = 128` alone.** Hash compressions −15% (E0); with BLAKE3 at about 37% of `c_c` (the CPU ratio of compression to gather in the E1 numbers above; the GPU ratio is unknown) that is about −5% of `c_c`: `commit_ps = 28.5`.

```sh
python3 -I research/tools/cost_model.py --set commit_ps=28.5
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | −1.0% | 0 (`B` does not change the opened columns: all `m` symbols are sent either way) | 0 (the verifier recomputes 4× fewer, 4× longer chain steps per opened column: about −15% of that hashing, E0) |
| 4K | −1.1% | 0 | same |
| 16K | −1.1% | 0 | same |
| 128K | −1.0% | 0 | same |

3. **Context: if the kernel is compute-bound at the op count above** (`c_c ≈ 65 ps` at the int32 peak), `python3 -I research/tools/cost_model.py --set commit_ps=65` gives Δμ +24.0% / +25.1% / +24.8% / +22.4% at 1K / 4K / 16K / 128K (E0), and neither part of this idea changes it. That scenario is the kill criterion, not the expected gain.

Inputs: bytes and operation counts E0, primary (own count, assumptions listed); H100 bandwidth and SM count secondary (vendor documentation); ops per byte of BLAKE3 secondary (RFC-0004); ops per Goldilocks multiplication E0 (SUR-0002); B-sweep E1, primary (this machine, which says nothing about the GPU's ratio). The gain of part 2 is a gain against a kernel that has not been written; it becomes a number only at E2.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Same code, same rate, same `τ`, same digests (at `B = 32` byte-identical; at `B = 128` a different but equally binding chain: each step is a BLAKE3 call over the chain value and the ordered symbols, collision resistance unchanged). The proximity analysis does not see `B`. | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Untouched: commitment layout and witness relations are unchanged. | pass |
| Zero knowledge (G-5, RFC-0013) | Untouched: the mask coefficients and blinding rows are drawn and encoded exactly as before; which memory holds a codeword symbol is not visible to the verifier. | pass |
| Verifier independence (SP-6) | The verifier already recomputes chains with `B` read from the commitment descriptor (RFC-0004 'Column digests'); no prover code or layout assumption crosses. At `B = 128` the verifier's parameter table changes through the superseding RFC, which is the normal path. | pass |
| Post-quantum (hash-based commitments) | Unchanged: BLAKE3 chains and Merkle tree. | pass |
| Redaction (05 §3, INV-ARCH-6) | Codeword symbols are secret-class and now stay in shared memory instead of an HBM buffer; the kernel emits chain values only, which are public commitment data. Nothing new reaches a log. | pass |

## Prior rejection check

- [RFC-0004 Alternatives Considered](../../docs/rfcs/RFC-0004-hashing-and-transcript.md#alternatives-considered) 6, "one streaming hasher per column" (16 GB of state; also the row "One streaming hasher per column" of [baseline.md §7](../baseline.md#7-already-rejected)): not reopened. The fusion keeps 32-byte chains; it changes where the step's input lives, not the chain construction.
- [RFC-0011 Alternatives Considered](../../docs/rfcs/RFC-0011-advice-commitment.md#alternatives-considered) 1, "storing the encoded witness": not reopened; nothing is stored, less is (the 2 GiB block buffer goes).
- [RFC-0018 'Pass structure'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure) "Fusion" lists P2/P3 pass fusions, not encode-and-hash; [RFC-0011 'Migration / Rollout'](../../docs/rfcs/RFC-0011-advice-commitment.md#migration--rollout) names a v0.1 "GPU encode-and-hash microkernel" while 'Streaming procedure' specifies the block buffer this fusion removes. The specification therefore neither rejects nor requires the fusion; it is silent on the kernel's memory shape.
- `B = 32` is stated in RFC-0004 and RFC-0011 without an Alternatives entry explaining the choice; no rejection of a larger `B` was found in either RFC or in baseline §7 (checked 2026-10-16: RFC-0004 §Alternatives 1–6, RFC-0011 §Alternatives 1–6, baseline §7 rows for RFC-0004 and RFC-0011).
- New: the per-element byte count showing that the block buffer round trip is 30% of the unfused pipeline's HBM traffic and that the unfused pipeline exceeds the 30 ps budget on bytes alone; and the E1 B-sweep. None of the above rejects it; none covers it.

## Kill criterion

One of:

1. The v0.1 Nsight Compute roofline of the unfused encode-and-hash microkernel ([RFC-0020 'Measurement protocol'](../../docs/rfcs/RFC-0020-cost-model.md#measurement-protocol-v01-gate), "Report") shows it already compute-bound (int32 pipe utilization above about 80%, achieved DRAM bandwidth below about 50% of 3.35 TB/s): removing 32 B per committed element then saves nothing, and the compute count above (64–67 ps at peak) is the problem to solve first.
2. On the H100, the first step of the `2^23`-point transform cannot run in one block or one cluster at the `n_1` the tile bound imposes (`n_1 = 2^15` at `B = 32`), so the fused pipeline needs the extra HBM round trip: its byte count returns to 108 B, equal to the unfused pipeline's, and the fusion only moves bytes from the buffer into the NTT. At `B = 128` this branch fires by construction (`n_1 = 2^17`), so part 3 then survives only on its −15% compressions.

## Cheapest experiment

- **E0 here, effort `s` (done in this note, to be moved into the experiment's directory):** the byte-and-operation table above as a script with every assumption a parameter (steps per transform size, shared-memory capacity, L2 residency, ops per butterfly, ops per BLAKE3 byte, achieved bandwidth and utilization), so the v0.1 roofline can be read against it line by line. Hypothesis: the unfused pipeline exceeds 30 ps on bytes alone at ≤ 80% achieved bandwidth; the fused one does not. Success criterion: the roofline's achieved DRAM bytes per committed element lie within 25% of 108 B for the unfused microkernel.
- **E1 here, effort `s` (first points measured today):** the `B` sweep {32, 64, 128, 256} through `tessera-hash::ColumnChains` and raw BLAKE3 on the M4 Max, plus a column-major (tile-layout) variant of `absorb_block` to separate the gather cost from the compression cost. Hypothesis: with the tile layout the CPU path follows the raw BLAKE3 curve (−25% from `B = 32` to `128`) instead of the row-major curve (−5%). Success criterion: the tile-layout variant at `B = 128` is at least 20% faster per symbol than row-major at `B = 32`.
- **Unknown until an H100 run (E2, effort `m` for the fused microkernel next to the v0.1 unfused one, under the RFC-0020 protocol):** the ratio of compute to bytes in the real kernel; whether a thread-block cluster carries the `2^15`-point first step; the achieved bandwidth; and therefore whether `c_c` lands under 30 ps at all.

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | Parts 1 and 2 change only where codeword symbols live before the chain step reads them; the chain d_j[i+1] = H("column-step", d_j[i] ‖ symbols of rows iB..min((i+1)B, m) − 1) of RFC-0004 'Column digests' depends only on the ordered symbols, so the digests are byte-identical and the CPU reference stays the oracle. Part 3 (B = 128) gives a different but equally binding chain: each step is one domain-separated BLAKE3 call over the previous chain value and a symbol count fixed by the public B and m, the verifier reads B from the descriptor bound into commitment_id, and the proximity bound of RFC-0011 never sees B. ADV-V gains nothing: the symbols are secret-class and the kernel emits only chain values, already public; τ_A opened columns reveal all m symbols either way. No bound in RFC-0014 moves. |
| prior art | survives | Neither specified nor rejected: RFC-0004 states B = 32 with no reason and RFC-0011 repeats it, the only visible constraint being the B·n·8 = 2 GiB block buffer of the streaming procedure that the fusion removes; RFC-0018 'Kernels' lists the RS-encode and BLAKE3 column-chain kernels as separate rows and names fusion only for the fraction-tree build and the P2/P3 passes, so the kernel's memory shape is open. RFC-0004 Alternatives Considered 6 (per-column hasher state) is respected by 32-byte chains in a shared-memory tile; RFC-0011 Alternative 1 (storing the encoded witness) is the opposite of what is proposed. Fused LDE-and-hash kernels are common GPU-prover practice, so the technique is not new; new are the per-element byte table (108 B unfused, above the 30 ps budget on bytes alone, E0) and the E1 B sweep. Part 3 touches the descriptor and so a tier-A parameter set. |
| cost model | refuted | Against the corrected baseline the fusion's Δμ is 0 by construction (it defends the assumed 30 ps) and the only positive scenario, B = 128 at `commit_ps=28.5`, reruns to −1.0 / −1.1 / −1.1 / −1.0%, an order of magnitude below the uncertainty of c_c (30 to 119 ps per IDEA-0004; `commit_ps=60` alone is +21%). The note's own operation count puts the fused kernel at 52–67 ps at the int32 peak and 107–111 ps at measured utilization, above both the unfused HBM floor (32–40 ps) and the fused one (23–28 ps), so in every case the note lists the kernel is compute-bound and removing 32 B per element changes c_c by nothing; the −15% compressions of B = 128 are worth −5% of c_c only under a CPU compression-to-gather ratio the note itself says does not describe the GPU. The byte-and-op accounting is useful, but it belongs as a line in IDEA-0004's calibration, not as a gain; the tier-A B change buys about −1%, inside noise. |
| feasibility | weakened | Parts 1 and 2 are claims about an H100 kernel's memory shape (shared-memory tiles, thread-block clusters, DSMEM) with no CPU analogue: `tessera-gpu` holds only a `vector_add` template and needs nvcc/CUDA 12.8 (`crates/tessera-gpu/src/lib.rs`, `src/device.rs`), so no encode-and-hash kernel exists to fuse and the v0.1 microkernel issues (#55, #66) are open. The proposed E1 (B sweep through `ColumnChains::absorb_block`, whose row-major gather is confirmed at `crates/tessera-hash/src/columns.rs`) measures a CPU gather artifact the note says says nothing about the GPU ratio. Part 3 (B = 128) is a tier-A parameter-set change justified only by a 15% compression count. Reachable here: the E0 byte script and an E1 BLAKE3 curve; everything decisive is E2 at effort `m` on hardware not present, so the idea is a shelve-until-H100 candidate rather than an experiment. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −1.1% μ(16K), −1.0% at 1K, from B = 128 alone (E0 cost model at `commit_ps=28.5`; E1 BLAKE3 curve on one M4 Max core); parts 1 and 2 are 0 against the baseline by construction (E0 byte count) | 0.3: the note's own operation count makes the kernel compute-bound, in which case the byte saving is worth nothing; the B sweep is E1 on a CPU gather path that does not transfer | `l`: the decisive measurement is an unfused/fused kernel pair on an H100 before anything is known (E2, `m`), and part 3 is a tier-A parameter-set change; the E0 script and the E1 BLAKE3 curve are `s` here | 1.1 × 0.3 / 4 = 0.08 |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.08; the cost-model lens refutes the gain against the baseline and the feasibility lens finds nothing decisive measurable here. Recommendation for consolidation: `shelved` (H100), with the per-element byte table carried into IDEA-0004's calibration as a line item and the B = 128 question deferred to the RFC-0020 v0.1 roofline. Experiments: none yet (candidate EXP: the E0 script and the E1 `B` sweep above; scratchpad `idea0003_bytes.py` and `bsweep/` hold today's versions). RFC or issue: none; a superseding RFC is needed only for part 3 (`B = 128`). Bibliography keys this note uses that the consolidation agent must resolve in [bibliography.md](../bibliography.md) from [SUR-0003](../surveys/SUR-0003-passes.md) and [SUR-0002](../surveys/SUR-0002-lookups.md): [DMABKGR25], [NVIDIA-H100], [VYTA25]. Shared-memory capacity (227 KB per block on compute capability 9.0) is from the CUDA Programming Guide (secondary, vendor documentation), not yet in the bibliography. Ledger line updated on: pending consolidation.
