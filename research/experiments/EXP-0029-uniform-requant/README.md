# EXP-0029: Uniform public requant selection

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0041](../../ideas/IDEA-0041-uniform-requant.md), issue #617 |
| Evidence level | E0: exact public metadata, integer/field identities and finite cases |
| Hardware | CPU execution; no timing threshold or GPU claim |
| Pre-registered on | 2026-10-29 |

## Hypothesis

The common-family candidate of IDEA-0041 yields only valid exact requant gadgets,
retains uniform ordered read roles, and handles the fixed real-1B and small
Wide/Split reproductions without changing the integer model or removing the
RFC-0045 guard.

## Success criterion

`validated` only when every frozen gate passes:

1. Reproduce the original failure on the synthetic two-layer MLP residual with
   b=39, shifts 2 and 12, one class, saturation [-524288,524287], B=32; and reproduce
   the Wide/Split public real-1B site at B=30,31,32. Preserve the real public
   Requant metadata fixture and its source claim SHA-256. Candidate plans for
   these sites must satisfy the original leaf read agreement check in a
   diagnostic adapter, without modifying its implementation.
2. Enumerate singleton profiles b=1..62, shift=1..63, n=1..4, B in
   {8,16,30,31,32}, saturation [-128,127] or [-1048575,1048575]. For n>1 use
   sorted distinct shifts beginning at shift, successive shifts capped at 63;
   skip cases that cannot supply n distinct shifts. Also include explicit
   [1,63], [1,2,62], [1,2,62,63] multi-class extremes.
3. Test at least 512 deterministic multi-layer sites from these profiles with
   seed 617, plus explicit One/Split, One/Wide, Split/Wide, Wide with mixed
   low presence, differing class counts above one, tiny and real metadata
   cases. Preserve already-uniform selections; explicitly refuse invalid
   one-versus-several class mixtures, invalid widths/shifts/saturation and
   impossible common plans. Not every random profile has to be admissible.
4. Independently validate every admitted plan's public bounds, table widths,
   no-wrap and rounding preconditions and ordered read roles. Check outputs
   against direct integer rounding/clamping at endpoints, -1,0,1, low/middle
   boundaries, rounding ties and saturation boundaries that lie in range,
   for every class. Exhaust all keys for admitted b<=8; sample at least eight
   seeded additional keys for each larger admitted singleton. Use Goldilocks
   modular lookup identities as well as exact integer reconstruction.
5. Adversarial auxiliary changes must fail the actual lookup predicates or
   give the identical unique auxiliary tuple: perturb low, middle and
   saturation independently where present. Include the known four-class,
   b=62 Split wrap counterexample near packed key 0, and the below-rounding
   bit requirement. These are bounded alternative-witness tests, not a proof
   over every possible field assignment.
6. Run a passing no-op control and catch each of ten meaningful planted
   faults on an assertion, retaining its concrete failing case: original
   per-layer family selection, mixed zero/nonzero low presence, omitted
   packed-key bound, split at the rounding bit, omitted low range, shifted
   signed offset, omitted 62-bit limit, insufficient middle saturation
   width, ignored table bound and accepted incompatible class presence.
   A non-behavioral edit does not count as caught. Retain first failures and
   restored control evidence.

`refuted` if a mathematical counterexample survives under the correct candidate
or either required reproduction cannot be handled within the stated bounds.
`inconclusive` if code/oracles cannot be reconciled, controls fail, a meaningful
fault survives, or the run cannot finish. Correctable harness errors may be
fixed with their initial evidence retained; frozen inputs/criteria cannot be
silently relaxed. Any amended domain needs a dated separate registration.

No full proof, production adoption, independent review, global G-4, formal hiding,
throughput or peak-RSS conclusion is part of this finite E0 verdict.

## Method

One standard-library Python harness, integer oracle distinct from the candidate
planner/field validator. Candidate uses the explicit formulas in IDEA-0041;
oracle directly rounds/clamps and checks table membership, auxiliary uniqueness
on the specified perturbations and every precondition. Publish raw JSON with
case counts, refusals, hashes and concrete fault counterexamples. No model
weights, prompt tokens, activations, private calibration statistics or fake
proofs enter research artifacts.

A separate Rust diagnostic adapter may load the original public lowering and
substitute candidate gadget/table metadata solely to call unmodified
LeafLayout::model. This adapter is not a canonical claim-list constructor,
selection.reads rebuilder, witness implementation or verifier, and must never
export a proof or an altered purported production claim list. Use synthetic
IntegerModel conformance data and committed public metadata, not real weights.

The original baseline diagnostics predate this registration. Candidate planner,
validator and experiment code are written only after the frozen plan has been
committed and checked against git show HEAD. Freeze README.preregistered.txt,
its SHA-256 and UTC timestamp under results/ before that commit.

## Reproduction

From repository root, after implementation:

```sh
python3 -I research/experiments/EXP-0029-uniform-requant/run.py
python3 -I research/experiments/EXP-0029-uniform-requant/run.py --mutations
python3 -I research/experiments/EXP-0029-uniform-requant/audit.py
python3 -I research/experiments/EXP-0029-uniform-requant/export_plans.py
python3 -I research/experiments/EXP-0029-uniform-requant/probe.py
python3 -I research/tools/check_kb.py
cargo xtask lint
```

The adapter's exact command and required public inputs will be recorded with
its results. Record OS, CPU and Python/Rust versions. E0 uses no timed iteration
statistic and makes no GPU extrapolation.

## Results

All results here are **E0, primary, finite metadata/algebra observations** on
Apple M1 Max, 64 GiB, macOS 15.6.1, Python 3.13.18 and rustc 1.95.0. No time or
memory comparison is made.

| Frozen gate | Observed result |
|---|---|
| Singleton profiles | 154,380; independent domain recount agrees |
| Complete sites including random/explicit/real cases | 154,900 |
| Admitted / refused | 109,568 / 45,332 |
| Preserved already-uniform selections | 109,496 |
| Distinct semantic plans checked | 41,919 |
| Integer/Goldilocks field cases | 4,590,008 |
| Independently perturbed committed auxiliaries rejected | 9,881,680 |
| Meaningful physical source faults | 10/10 caught; control/restored pass with original source bytes |
| Tiny canonical lowering baseline | exact original leaf-read refusal |
| Tiny candidate, unchanged leaf guard | pass; two layers promoted, 1,029 split values |
| Real public 1B lowering, B=30,31,32 baseline | exact original leaf-read refusal at each policy |
| Real public 1B candidate, B=30,31,32 | pass at each policy; sixteen layers promoted, 6,895 split values |

Raw records: [result.json](results/result.json), [audit.json](results/audit.json),
[mutations.json](results/mutations.json), [layout receipt](results/layout-receipt.json),
[tiny log](results/layout-tiny.log), [real log](results/layout-real.log) and
[toolchain](results/toolchain.txt). The algebra digest is
`faaaa77c59c2c7991b1c8ae0cc831cbf3da7164cdac2acaed4cef5a27edfe1c7`;
it is identical across the initial, corrected-control and final formatted runs.

The public claim fixture SHA-256 is
`af7d31a9e853cde6fdf4e2a8a195e6d312ae3ae6fc2fdb2e8700cb963ab3d5aa`.
No weights, tokens, activations or private calibration statistics are present.
The original `LeafLayout::model` and `LeafLayout::new` source hashes agree before
and after the scratch executions. Only read-cache visibility was changed in
that scratch compiler; ordered ids were supplied alongside candidate gadgets.
The initial scratch visibility change emitted a missing-doc warning; the
reproduction script now documents that field, and a separate
[release check](results/layout-warning-clean.log) passed without warnings.
No production source changed.

Initial implementation errors are retained: [control notes](results/initial-mutations/NOTES.md),
[missing trait import](results/initial-probe-build.log), and the correct
read-cache refusals in [tiny](results/initial-cache-tiny.log) and
[real](results/initial-cache-real.log). These did not change a criterion or domain;
the cache-method correction was committed at `2ff6ab9` before its implementation.
The original registration was committed at `e56dda7` before candidate code.

## Analysis

The candidate passes every frozen gate, including the unchanged full model leaf
constructor at all required real public policies. The independent recount and
source-restoration audit pass. This settles the frozen compatibility/finite
integer experiment, not production soundness or a complete verifier.

Only finite algebra/metadata coverage is reachable here. Full actual-model
advice/weight layout, runtime budget, a soundness proof, independent verifier
implementation/review and certifying-parameter proof/verification remain owed.

## Verdict

`validated` within the frozen E0 scope. Common-family promotion resolves this
public leaf-layout incompatibility without deleting its guard. IDEA-0041 is
validated for the finite experiment; its security/count/integration obligations
remain explicit and block automatic production or RFC acceptance. No complete
proof, global G-4, formal hiding, runtime or GPU gain is credited.

## Follow-ups

If the frozen experiment validates, write a superseding RFC draft with exact
canonical rules, migration and full count/review obligations. Do not adopt the
candidate in production before the RFC decision. If refuted, retain the failing
case and consider a distinct registered rule; do not weaken the success gate.

## Method amendment: complete diagnostic selection metadata (2026-10-29)

The initial adapter changed public gadgets/tables but retained Selection's private
ordered lookup cache. LeafLayout::model correctly rejected the inconsistent cache
before reaching read agreement. The initial Method's statement that the adapter
was not a selection.reads rebuilder omitted this required part of metadata
substitution. No mathematical counterexample or criterion change is inferred.

Keep the original cases and success criteria, and still call unmodified
LeafLayout::model with every production guard intact. For this diagnostic only,
create a scratch source snapshot of main a407755bc2c03736466b0c1b45dae38c2fb2bcb5,
changing the visibility of Selection.reads from private to public. Change no
algorithm, leaf code or check. Supply the exact ordered candidate lookup ids
with the substituted gadgets/tables; leave all other cached reads unchanged.
The accessor change and adapter exist only in the scratch experiment build,
never in production source or a proof. Compare leaf source bytes against the
original git object before and after execution. Preserve initial refused-probe
logs. This method correction must be frozen and committed before implementing
the scratch overlay/accessor or read-cache substitution.
