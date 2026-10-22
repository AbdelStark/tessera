# Recorded CPU driver integration fixture

These are public messages captured from
`tessera-prover/tests/model_leaf.rs`'s
`complete_runtime_cpu_proofs_discharge_original_sources_in_both_contexts` test.
They are producer-derived integration data, not independent conformance oracles.
The independent protocol, field, table and reduction oracles remain separate.

The synthetic model has width 2, sequence length 5 and prompt length 4. Its
`test-small/1` parameters certify nothing; its replayable masks provide no zero
knowledge. All data here is synthetic and public. `SHA256SUMS` pins the capture.

- `claims.tcl`: canonical claim-list envelope.
- `weights.tce`: public epoch commitment payload.
- `ni.tnp`: complete canonical non-interactive proof envelope.
- `live-messages.tnp`: the recorded Live Round1/2/3 in an NI envelope used only
  as a storage container. Its nonce fields are placeholders; it is not an NI proof.
- `live-seed.bin`: the public coin-toss digest after both shares were revealed.
- `queries.tce`: the public late OS-random Live query message.

The private replay test verifies each complete transcript 100 times and compares
report bytes. Public `LiveVerifier` sessions remain one-shot and obtain fresh
entropy. The root integration test separately completes six fresh real sessions.

To capture again, set `TESSERA_CAPTURE_DRIVER_FIXTURE` to a new absolute directory
and run the named root test with `--features tv-core/test-params`. The test refuses
to overwrite a capture. Regeneration changes the Live randomness and therefore
the fixture hashes; review both the new messages and their provenance.
