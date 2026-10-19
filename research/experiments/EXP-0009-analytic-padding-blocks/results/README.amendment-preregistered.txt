## Amendment 2026-10-19: amortized ragged loop

Status: `planned`. The original hypothesis, criterion, methods, results and verdict above remain the historical experiment. This amendment tests a new `analytic-amortized` arm; the four original harness arms and the production anchor remain runnable without selecting the new arm.

### Hypothesis and unchanged criterion

Sorting the processing order of each height class by real prefix length makes equal-length trees adjacent. Sharing their prefix/boundary metadata and closed-form `TAIL` contribution, and folding odd tails together across such runs, removes enough per-tree bookkeeping that `analytic-amortized` satisfies both original clauses on S1, S2, S3 and S4. Tree indices in the transcript and the tree-coordinate polynomial keep their original order; sorting is an internal traversal permutation, reversed before the tree rounds.

The baseline remains `dense`; only the candidate is replaced by `analytic-amortized` in `R_same`, `R_prod` and `a(c)`. The unpadded reference remains `u = dense(U23)` nanoseconds per leaf. **Validated** means every judged configuration has `0.90 <= R_same / f <= 1.10` and `a(c) / u <= 1.05`, every proof/message/evaluation/leaf claim and final transcript state equals production exactly, and all arms needed for those ratios have CV at most 3% after at most one rerun. The original refutation thresholds `R_same <= 1 + (f - 1)/2` remain reported; a failed or noisy amendment leaves EXP-0009 and IDEA-0019 inconclusive, as the handover specifies. A mismatched output is a harness defect to fix before judging, never a performance result. No threshold, configuration or seed is selected after measurement.

### Method and reproduction plan

The new arm retains the original materialized per-tree `eq` prefixes (it does not silently substitute the contextual shared-`eq` optimization). At each position round, groups are traversed in runs with equal current even, odd and `eq` prefix lengths. One run shares its real-pair count, boundary column and `TAIL` coefficient. The fully padded contribution is accumulated with the sum of its tree scales. Full pairs retain the original arithmetic; odd final folds are traversed across a run with its parity decided outside that traversal. The original order is used again when assembling the tree-coordinate tables. Sorting, metadata construction, allocations, level building and folds remain inside the timed proof.

Inputs and session context are exactly those above: SplitMix64 leaves seed `0x7E55_E7A0_0009_0001`, checks seed `0x7E55_E7A0_0009_0002`; S1-S4 and U23 decide the result, with U24 and the two 1K cases as context. All original hand cases, small ragged shapes, independent `tv-core` acceptance and direct leaf-MLE checks run before timing. The new arm must equal the production proof and digest on all these shapes. Extra repeated-length and unsorted real-length shapes exercise traversal permutation and odd tails.

The protocol is 3 warmups and 10 measured samples per arm, median/minimum/maximum/CV, rerunning an arm once if CV exceeds 3%; each sample repeats the complete fresh-transcript proof until elapsed time is at least 20 ms, reporting time per proof. Raw sample durations and repetition counts, including a first attempt that triggers a rerun, are retained. Only one experiment benchmarks at a time. CPU/OS/toolchain/date/git SHA are captured with the raw output. Evidence is E1 CPU wall clock and exact transcript agreement; `count.py --amendment` will emit E0 counts of tree-rounds, equal-length runs, odd-tail folds and shared tail terms for all configurations. No GPU number is inferred.

Commands from this directory (after committing this preregistration):

```sh
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --all --check
cargo run --release --locked -- amendment results/amendment-run.json > results/amendment-run.txt
python3 -I count.py --amendment > results/amendment-count.txt
```

The `all` command preserves the original arms. The new `amendment` command includes all original arms plus `analytic-amortized`, applies the unchanged criterion to the new arm, and writes separate raw files. H100 follow-up: run the same causal-row shapes under RFC-0020 with block-skipping and grouped odd boundary columns, to establish whether tile alignment and partially idle warps preserve the CPU saving. No production crate, verifier, transcript label or acceptance condition changes.
