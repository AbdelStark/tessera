# Initial controls and harness corrections

The first finite algebra run passed the frozen cases; its output is retained in
`../initial-result.json`. All ten initial mutation controls failed on assertions
and both no-op controls passed semantically. A separately written audit then
found that the no-op source replacement rewrote `FAULT = ""` with single quotes,
so the control/restored source hashes agreed with each other but not with the
unmodified file. The original source and receipts are retained here. The runner
now writes the original bytes for empty-fault controls, and the audit requires
control/restored hashes to match the actual source. No domain, predicate,
planner formula or success criterion changed; the full algebra run and mutation
controls were rerun after correction.

The first Rust adapter build missed the `Table` trait import for `.spec()`.
Its compiler error is retained in `../initial-probe-build.log`; the import was
added. Neither correction changes production code or an experiment gate.
