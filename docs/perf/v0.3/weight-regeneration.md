# Regenerated protocol/1 weights

Issue [#607](https://github.com/AbdelStark/tessera/issues/607) adds
`tessera_pcs::RegeneratedWeightCommitment` and uses it in the CPU driver and
`tessera commit-weights`. The resident `WeightCommitment` remains the equivalence oracle.
This implements the unchanged encoding, descriptor, row order and chains of
[RFC-0040](../../rfcs/RFC-0040-weight-commitment-artifacts.md) with bounded scratch.

## Storage and checks

Each matrix is read through validated `WeightLayout::rows`. One clearing input row and a
clearing block of at most `B` codewords are encoded with the existing unmasked `RowEncoder`.
Only public Merkle trees and layout metadata survive construction. Empty ChannelParams
matrices retain their initial column digests, as the resident implementation does.

P5 regenerates codewords and extracts the distinct requested columns. It checks both the
entire regenerated root and the actual extracted columns with their canonical multiproof
before copying them out of clearing buffers into protocol messages. Query range and model
shape checks precede candidate allocation; a mismatch returns a typed error without sending
candidate values. No new transcript label, schema, field operation or verifier code is used.

The host estimate includes retained trees, one encoder/block and transient tree/chain
metadata, folding buffers, and two copies of all possible queried columns. It overestimates
ChannelParams by charging every possible position. The caller's INT8 model is excluded, as
`MemoryBudget` documents. `PassPlan` still charges graph tensors, resident witness data,
lookup trees and stage working sets separately. This change does not remove those floors.

## Measurement

Measured 2026-10-29 on an Apple M1 Max, 64 GiB RAM, Rust 1.95.0, one benchmark thread.
Fuzzing and model calibration were running concurrently; timings are not isolated throughput.
Five paired runs use the same synthetic model: 16 layers, all nine weight shapes `128 × 128`,
1,867,776 INT8 entries, `k = 128`, rate 1/4, block 4, no ChannelParams and 32 distinct queries
per matrix. Input model data and initialized metrics are outside the allocation counter.
The counter measures current-thread component heap, not RSS.

| Operation | Resident | Regenerated |
|---|---|---|
| Construction peak heap | 75,344,504 B | 346,800 B |
| Retained heap | 75,321,776 B | 327,680 B |
| Median construction time | 0.263 s | 0.328 s |
| Construction time range | 0.244–0.289 s | 0.314–0.404 s |
| P5 additional peak heap, all matrices and returned columns | 3,765,504 B | 4,286,096 B |
| Median P5 time | 0.00133 s | 0.359 s |
| P5 time range | 0.00115–0.00158 s | 0.351–0.376 s |

Construction peak heap falls by 99.54%; retained heap falls by 99.56%. Regeneration costs
additional computation: P5 re-encodes the model rather than reading stored codewords. This
backend prioritizes capacity and does not meet a measured frontier throughput target.

There are 14,592 encoded rows in this case. Each row's inverse length-128 and forward
length-512 transforms perform 2,752 radix-2 butterflies, plus spreading and column hashing.
The block's working set fits in cache; removing the 75 MB resident matrix reduces capacity
requirements without removing those field and hash operations.

Reproduce:

```sh
cargo bench -p tessera-pcs --bench weights_regenerated --locked
```

Every sample also asserts equal public roots, all column values and multiproof siblings.
The same benchmark runs a small smoke case under `cargo test --benches`.

## Verification

The equivalence cases exercise one and several layers, non-power-of-two dimensions, inputs
wider than the nominal row width, empty ChannelParams, classes and multipliers, multiple
epochs, blocks 1/3/32, and repeated/out-of-order queries. Changed weights, out-of-range and
empty queries, corrupted candidates, corrupted multiproofs and wrong candidate shapes have
exact typed refusal tests. Allocation tests increase the row count sixteenfold while holding
code widths fixed, and compare retained/peak heap with the resident oracle.

Four manually planted faults in an isolated worktree are caught on test assertions:
omitting the regenerated-root check, omitting the candidate-column proof check, allocating a
full codeword matrix, and shifting the epoch in the commitment descriptor. The unmodified
control passes. The allocation mutant's custom panic message prints its measured peaks;
the failure location is the test's explicit row-count independence assertion.

## Remaining scale work

These are component measurements and tiny-model proof checks. They do not establish a
billion-parameter proof or a production memory bound. The CPU witness still materializes
weights as `i64`, `PassPlan` charges resident graph tensors conservatively, and dense lookup
working sets remain large. The real-model ramp must qualify those paths and produce an
independently verified proof with certifying parameters. Small diagnostic calibrations of
the real 1B and 8B checkpoints initially refused unsupported rescale shifts. The
[subsequent real-model ramp](../v0.2/reference-int8-tiles.md#actual-1b-candidate-and-proof-status)
completed full 1B C4 calibration, exported an affine candidate and committed its actual weights;
its proof attempt failed during leaf planning. Full fidelity requirements of
[#93](https://github.com/AbdelStark/tessera/issues/93) remain separate work. Protocol/1 is non-hiding.
