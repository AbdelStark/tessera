# EXP-0028: Complete weight-mask draw accounting

| | |
|---|---|
| Status | `planned` |
| Idea | issue #568 ([source](https://github.com/AbdelStark/tessera/issues/568)), a specification/accounting audit from [SUR-0016](../../surveys/SUR-0016-weight-mask-draw-cost.md), not a new masking protocol |
| Evidence level | E0: exact integers, finite enumeration and rational cost scenarios |
| Hardware | CPU execution of public metadata only; no GPU timing |
| Pre-registered on | 2026-10-29 |

## Hypothesis

For the frozen public cases below, independent literal row/support enumeration agrees
with closed-form per-matrix stream accounting, including reserved rows, distinct
consumers, matrix widths and repeated passes; this distinguishes the incomplete
data-mask subtotal from two different hypothetical blinder-exclusion scenarios.

## Success criterion

`validated` only if every condition holds:

1. All five models and all ten matrices of `spec/vectors/weights/weights.json`, and
   all eight cases of `spec/vectors/weights-opening/openings.json`, reproduce the
   recorded message widths and committed row indices from public geometry alone.
2. Every synthetic case and both 405B channel layouts have identical literal and
   formula counts for FULL, SELECTED and MASK-ALL-SELECTED-VALUES support. Record
   masks and blinding values separately. Every valid support configuration's store
   read bytes equal its distinct accepted stream bytes; this is a byte equality,
   not a time or secrecy claim.
3. FULL with the historical common-width 405B example reproduces exactly
   654,109,048,832 accepted bytes. SELECTED with 72 reserved base rows reproduces
   216,735,416,320; MASK-ALL-SELECTED-VALUES reproduces 242,463,277,056. The tighter
   ChannelParams case is derived independently, not required to equal these figures.
4. A stored-codeword P4-only run counts one draw. A no-codeword-store P4/P5 run counts
   two independent generations of the same required streams, not one union across
   passes. A two-epoch full-support linking scenario counts both epochs. Within one
   generation, consumers of the same stream entry never double-count its production.
5. Invalid reservation/width/rate inputs are explicitly classified before computing
   costs. Zero/no-claim ChannelParams contributes no blinding rows, reservation or
   draws. Expected candidate words use exact rejection probability; the complete
   enumeration of 8-bit candidates with prime 251 agrees with that expectation.
6. Each of the ten faults listed in Method changes an applicable result or guard and
   is caught; the unmodified control passes. Record the concrete counterexample per
   fault. A fault that makes no behavioral change is not counted as caught.

`refuted` if a surviving mathematical counterexample disproves the hypothesis for a
frozen valid case. `inconclusive` if the harness cannot finish or be independently
checked, a fixture cannot be interpreted unambiguously, a control fails, or a fault
survives. Correctable implementation errors retain their first failure records; a
corrected run may validate only if inputs and criteria above are unchanged.

No timing threshold, throughput gain, security approval, complete compiled Protocol
Two schedule or 405B deployment qualification enters this verdict.

## Method

Implement one standard-library Python harness with two separately written paths:

- **Literal oracle:** enumerate which public row indices have live output/channel
  positions. For small cases enumerate each consumer's sets of encoding-mask and
  blinding-value entries, then union within a generation and sum across generations.
- **Candidate:** count intervals/ceilings for public row geometry; use integer
  cardinalities and closed-form per-matrix bytes without allocating row-sized masks.
  In the large 405B case the oracle may enumerate rows and multiply their widths,
  but never allocate a tensor, codeword or stream-sized buffer.

Neither path imports the existing vector generator or production implementation.
Fixture inputs use only architecture, channel presence, layout and expected public
geometry; synthetic tensor values in those files are ignored. Save input file digests
and the source revision in the result. Existing vector tests in both workspaces are
separate conformance evidence, not a masked-layout implementation.

Domains, fixed before code:

- All five model geometries in `weights.json` and all eight opening geometries in
  `openings.json`, with their recorded `k_W` and per-matrix widths.
- Synthetic single-weight geometry: `out` in {1,3,4,9}, `inputs` in {1,2,5}, `layers`
  in {1,3}, and `k_W` in {2,8,16}. For each, use `P_W` in {0,1,4}, `B_W` in {0,4,8,32}
  and reservation claim count `e` in {0,1,3,7}. Treat `e=0` as an empty/unopened
  ChannelParams scenario rather than an unclaimed nonempty ordinary weight matrix.
  Classify protocol-two narrow widths and insufficient row budgets as invalid.
- 200 additional public geometries from SplitMix64 seed `0x5680028`: output 1..33,
  input 1..65, layers 1..7, `k_W` one of powers 2^0..2^8, `P_W` one of {0,1,4,16},
  `B_W` one of {0,4,8,32}, `e` one of {0,1,3,7}. Draw these fields in that order.
- Llama-3.1-405B public architecture: 126 layers, residual width 16,384, 128 query
  heads, 8 KV heads, head width 128, FF width 53,248, vocabulary 128,256; common
  `k_W=2^20`, `P_W=2^16`, `B_W=4096`, `e_CP=7`. Derive the nine weight shapes from
  RFC-0012/0040. The tight channel layout has Q,K,V,O,Gate,Up,Down class parameters
  at every real layer; derive its rows from the flattened `[16,L_pad,C]` positions.
  The historical channel upper bound is 882 data rows at the same width, explicitly
  a count scenario rather than a different claimed tensor layout.

Use RFC-0065's `B_CP = B_W + 2*(e_CP-1)*floor(max(B_W-4,0)/6)` for positive `e_CP`,
zero otherwise; ordinary matrices have `B_W`. A proof's reservation is `2*e+4` base
rows when opened. Its row availability excludes the link's last four rows; otherwise
mark the proof reservation unfit and do not report it as a feasible selected run.
Geometry/accounting counts may still be reported for an unfit hypothetical layout.

Support definitions, without asserting #394's security:

- FULL: all data and blinding rows' masks, and every blinding row's values.
- SELECTED: all data masks plus reserved rows' masks and values, conditional on every
  other blinder having zero coefficients in every relevant combination.
- MASK-ALL-SELECTED-VALUES: all row masks and only reserved blinding values; retain
  this as a conservative accounting comparison, not an alternative encoding theorem.

Consumer records name P4 evaluation, extension-mask and proximity work; the distinct
streams are the union of their requirements within a pass. With the stored codeword
default, P5 adds no generation. Without it, model P5's independent re-generation of
all rows whose codeword symbols are answered, even if a P4 support scenario was
smaller. Do not claim that the combined-row support alone prices a fallback P5.
Two independent epochs add two complete sets for the link scenario.

Rates are rational scenario inputs: 100, 530 and 1000 decimal GB/s. Report generation
and store-read scenarios at the same assumed effective rate, separately from the
existing 0.24 GPU-s core. No rate is a measurement or predicted store bandwidth.
Show ideal rejection expansion for Goldilocks and the exhaustive toy field, and keep
setup, block rounding, compaction, transfers and folding as unpriced work. A measured
all-inclusive `W` must replace a sum of components, never have that sum added again.

Ten behavioral faults, with a matching unmodified control:

1. Drop reserved rows' encoding masks.
2. Drop reserved rows' message values.
3. Replace all `k_T` with `k_W`, losing a wider input's message width.
4. Use `B_W` rather than the claim-dependent `B_CP`.
5. Charge a fixed six-row reservation for `e=7`.
6. Charge nonzero blinding/reservation work for empty ChannelParams.
7. Sum duplicate consumer requirements rather than union them within a pass.
8. Union identical streams across P4/P5 rather than summing two generations.
9. Count only one epoch in the two-epoch full-support link scenario.
10. Replace rejection expansion with 1 (the 8-bit exhaustive case must catch this).

Use exact integers/Fraction for calculations; JSON outputs contain integers, finite
scenario floats and numerator/denominator pairs where exact fractions are needed.
Retain stdout/errors, environment and mutant counterexamples. No timing samples,
warmups, CV test or GPU extrapolation: this is E0 enumeration only.

## Reproduction

After this report and its frozen copy are committed, implement and run:

```sh
python3 -I research/experiments/EXP-0028-weight-mask-draws/audit.py \
  --out research/experiments/EXP-0028-weight-mask-draws/results/run.json
python3 -I research/experiments/EXP-0028-weight-mask-draws/check.py \
  research/experiments/EXP-0028-weight-mask-draws/results/run.json
python3 -I research/tools/check_kb.py
cargo xtask lint
```

The output path must not exist before a run; failed attempts are retained under new
names. The second reader independently replays saved case inputs and totals; it does
not import `audit.py`. Freeze this report's bytes and SHA-256 before either script.

## Results

Not run. No experiment implementation exists at registration.

## Analysis

Pending. Finite metadata agreement will not establish GPU throughput, constant-time
secret handling, a joint hiding/rank argument, or completeness of a future Protocol
Two implementation's consumer list. Unsupported or absent consumers remain an open
implementation obligation, not a measured zero cost.

## Verdict

Pending; status `planned`.

## Follow-ups

If validated, draft #568's accounting/storage decision for the current full-support
generator, preserving its explicit #394/#145 security prerequisites and adding an
H100 XOF/rejection/all-consumer benchmark requirement. Any cost-model/budget amendment
must distinguish a conditional projection from a measured rate, retain the core-pass
gate, price repeated generation and leave blinder-exclusion savings uncredited.
