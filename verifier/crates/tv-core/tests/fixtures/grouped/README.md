# Recorded CPU grouped-proof fixtures

Public non-interactive proofs of a grouped leaf stage (RFC-0054), captured from
`tessera-prover/tests/model_leaf.rs`'s `cpu_api::a_forced_budget_groups_the_leaf_stage`
test. They are producer-derived integration data, not independent conformance oracles.

The synthetic model is the conformance bounded model with three layers (padded to four),
width 2, sequence length 5 and prompt length 4, under `test-small/1`, whose parameters
certify nothing. Each folder holds one proof whose memory budget forced its grouping:

| Folder | `leaf_groups` (`g_T`, `g_L`) | Groups |
|---|---|---:|
| `g1` | (0, 1) | 2 |
| `g2` | (0, 2) | 4 |
| `g3` | (1, 2) | 8 |

- `claims.tcl`: canonical claim-list envelope.
- `weights.tce`: public epoch commitment payload.
- `ni.tnp`: complete canonical non-interactive proof envelope; its statement is the one the
  verifier pins.

`SHA256SUMS` pins the capture. To capture again, set `TESSERA_CAPTURE_GROUPED_FIXTURE` to a
new absolute directory and run the named test; it refuses to overwrite a capture. The masks
come from the OS, so a new capture changes every hash.
