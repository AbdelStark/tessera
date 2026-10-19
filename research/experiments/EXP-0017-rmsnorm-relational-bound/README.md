# EXP-0017: RMSNorm relational bound under current precision and counts

| | |
|---|---|
| Status | `done: refuted` |
| Idea | [IDEA-0011](../../ideas/IDEA-0011-rsqrt-fact-norm-key.md) |
| Evidence level | E0 exact integer bounds and compiler-derived counts |
| Hardware | none for the claim; CPU runs exact arithmetic and the compiler probe, without timing |
| Pre-registered on | 2026-10-19 |

## Hypothesis

The verifier-derivable joint bound on an RMSNorm key `res_i * rho` removes one committed split limb and one lookup per element at both layer norms of the current 405B servable fixture, saving exactly 32,768 commitments and 32,768 lookups per layer per token under RFC-0023's current precision policy.

## Success criterion

The hypothesis is the idea's count-saving claim, rather than merely the validity of its algebraic bound. Set `B_max=32` and use the current compiler's signed-width convention, shift classes, balanced split, table sharing and counts. Compare baseline and candidate on the same public fixture.

- `validated`: all mandatory bound/oracle/adversarial checks pass, and both 405B layer norm sites switch from `RequantSplit` to `Requant1`, saving exactly 32,768 commitments and 32,768 lookups per layer per token. The final norm and first layer's embedding-derived input must also be counted explicitly; they cannot be silently treated as an ordinary saturated residual. The candidate's narrowed tables must reproduce every tested integer norm output exactly.
- `refuted`: an accepted integer witness violates the proposed joint bound; or all mandatory checks complete and the exact joint signed widths at both ordinary 405B layer norms remain greater than 32 bits, their selected gadgets remain `RequantSplit` with two commitments and two lookups per element, and the per-layer commitment and lookup reductions are both exactly zero. This fires the current-count kill criterion even if table rows decrease.
- `inconclusive`: all other outcomes, including missing baseline reproduction, incomplete required checks, a partial count reduction, or failure to connect a reported count to the actual current selection rules. A defect in the harness is fixed before judging; a disagreement between the specification and production is reported separately and blocks validation until resolved.

The bound-validity result, exact key intervals, table-row change, and count verdict are recorded separately. No table-row benefit, historical kappa=16 control, conditional cost-model saving or favourable toy case can replace the primary current-count test. Finite checks support the stated arithmetic argument; they do not establish a new full-protocol soundness theorem.

## Method

Preparation is restricted to this preregistration. No harness code, execution or production changes are authorized in this unit. A later, separately authorized implementation must first verify this frozen text in a parent-created commit, as research rule 3 requires. Scope after that authorization is this experiment directory and the IDEA-0011 result update; shared bookkeeping belongs to the parent.

The normative source revision is main `709c37c622caf0351df5ccb9acebcaa6cc09ab6f`. Record the executing checkout and hashes of all source inputs. If relevant norm/range/selection code differs from this revision, describe the difference and amend the registration before running. Reuse the existing public compiler APIs and report fixture wherever possible, with no verifier dependency on prover code. The compiler probe is a separate safe Rust workspace; an independent Python integer oracle reads its public JSON output.

1. **Current baseline.** Read RFC-0005 integer semantics, RFC-0006 range analysis/selection, RFC-0009 RsqrtRow and Lemma 0, RFC-0022 residual saturation and root range, RFC-0023 precision, RFC-0028 balanced splitting and RFC-0035 padding. Reproduce the 405B servable c2/B32 report's `C=45,566,660`, `K=45,440,678`, `M=7,693,467,648` and per-layer `C=360,232`, `K=360,380` from the same public fixture used by `crates/tessera-bench/tests/commitment_report.rs`. Preserve its approximate RsqrtRow cost and omitted range-table multiplicities; they cancel between arms, rather than becoming exact gadget costs. Record every norm node, its input/root/key ranges, dimension, epsilon, kappa, shift classes, selected gadget and table definitions. Verify these against the committed report, not historical EXP-0011's hand table.
2. **Precision and intervals.** Independently derive `e=ceil(d * 2^14 / 100000)` and `kappa=min(31,max{k : floor(d * 2^(2k) / e)<2^64})` for the pinned residual exponent -7. Required current fixtures are 8B (`d=4096`, 32 layers, `e=672`) and 405B (`d=16384`, 126 layers, `e=2685`); both use kappa=30, root-check width 32 and one norm shift class `r=kappa-3=27`. Do not propagate the two weight-site classes into the norms. Record the actual first embedding-derived norm input separately from all later INT20 residuals, including the final norm. Controls use kappa=16 and adjusted norm shift13 on the same dimensions/epsilon, and cap case kappa=31 at public `e=d`. These controls cannot change the primary verdict.
3. **Exact integer bound.** On a real row, `u=sum(res_j^2)+e`, `e>=1`, `N=d*2^(2*kappa)` and the exact root bracket gives `rho^2*u<=N<(rho+1)^2*u`. For rho>=1, `(rho*res_i)^2<=N-rho^2*e<=N-e`; for rho=0 the key is zero. Thus `abs(key)<=isqrt(N-e)` when `N>=e`, otherwise all roots and keys are zero. Also report the weaker strict bound `abs(key)<2^kappa*sqrt(d)` using integers only. Derive the minimal signed width from both asymmetric endpoints, rather than treating an exclusive magnitude bound as a signed width. This already predicts a current 405B width near38 rather than the historical24/25; the harness must compute the exact value.
4. **Attainability and tighter oracle.** For every magnitude `x=0..2^19` of a saturated INT20 coordinate, compute `x*isqrt(N//(x*x+e))`. Other channels can only decrease rho, so their zero assignment maximizes the key at each x. Record the exact extremum and its full one-nonzero-coordinate witness, independently checking both root inequalities. Use `x<=2^19-1` for the positive endpoint and `x<=2^19` for the negative endpoint. Run this complete magnitude enumeration for both dimensions at kappa16 and30 and the kappa31 cap controls. For the narrower first norm use its actual compiler interval and embedding representability; do not call an arbitrary INT20 witness attainable through the embedding. Compare Python `isqrt` against a separately written integer binary-search root routine on every extremizing witness and all boundary controls.
5. **Finite accepted-witness exhaustion.** Enumerate every row of dimensions1,2,3,4 over signed3-bit residuals `[-4,3]`, epsilon1,2,3,4 and kappa0,1,2,3,4. Enumerate every candidate rho from0 through `isqrt(N//e)+1` against the direct pair of integer bracket inequalities. There must be exactly one accepted root, equal to the independent integer-square-root oracle; every accepted key must satisfy both the universal and magnitude-enumerated bound. Include zero rows, maximally negative/positive rows, one nonzero channel, exact-square brackets and immediate neighbours. Test padded `psi=0,u=0,rho=0` and reject a nonzero padded rho by its explicit zero pin. Negative controls must show that dropping epsilon positivity invalidates the zero-row root-uniqueness premise, dropping either root inequality admits an incorrect root, and dropping the padding pin admits a second padded root. These controls are separate from the intact candidate.
6. **No-wrap and semantics.** Establish from the current declared residual/root ranges that each original key is a true centered Goldilocks integer before applying a joint bound. Check `u<2^55`, kappa<=31, root width<=32 and the exact big-integer comparisons; do not infer a small integer from a field congruence. Check zero/nonpositive epsilon rejection and root-width33 rejection. The range refinement is derived from the existing constraints; it introduces no witness, lookup, challenge, opening or transcript change. Report any extra precondition needed by the derivation. A standalone congruence or small-field test cannot substitute for the exact bracket.
7. **Selection and table rows.** Feed baseline and refined public key intervals with their original shifts/classes into production `select`, using small synthetic graphs if the public `Ranges` API prevents replacing an existing range. Cross-check the rule with an independently written integer implementation: Requant1 iff `b+ceil(log2(classes))<=32`; otherwise use all current split preconditions and the balanced split. Reconstruct the full candidate Selection with only norm entries replaced, deduplicate tables by exact definition, and pass it to the production report, or independently reproduce the complete report with a documented node/table mapping. Baseline reproduction is mandatory. Count unique tables over the whole model: shrinking a norm table does not remove an old Range table still used elsewhere. Include the first and final norms and unchanged affine/weight/row gadgets. Record C,K,M deltas, per-layer and per-token totals, and conditional cost-model projections only after these exact counts are known. Projections are E0 and must retain current model omissions; no GPU or wall-clock improvement is claimed.
8. **Additional synthetic controls.** SplitMix64 seed `0x7E55_E7A0_0011_0017` supplies1,000 rows each at dimensions4,16,128 with signed8-bit entries, and100 sparse/boundary-pattern rows each at dimensions4096 and16384 within INT20. Test all endpoints, zeros and alternating signs. Candidate Requant1/RequantSplit output is compared with direct integer rounded-and-saturated requantization at each tested key; include ties, saturation boundaries and adjacent keys. All saved data is synthetic and public. Serialize the parameters, extrema, coverage counts, all failed checks and representative accepted/rejected witnesses to raw JSON.

This is E0. Warmups, timing batches and CV thresholds are inapplicable; no CPU benchmark is run and no timing slot is consumed. Record OS, Python/Rust versions, UTC time, executing SHA and input/harness SHA-256 values. A single deterministic command must reconstruct the verdict from raw count/bound records. The independent checker must fail when an extremum, width, split multiplicity, unique-table row total or verdict is deliberately altered in a scratch result, and pass a no-op control.

## Reproduction

After preregistration commit and separate implementation authorization, from the repository root:

```sh
cargo run --release --locked --manifest-path research/experiments/EXP-0017-rmsnorm-relational-bound/Cargo.toml -- > research/experiments/EXP-0017-rmsnorm-relational-bound/results/compiler.json
python3 -I research/experiments/EXP-0017-rmsnorm-relational-bound/check.py
```

The first command will emit public baseline/candidate selection records; the second will independently check exact bounds, counts and frozen verdict conditions. Those files do not exist at preregistration. `results/README.preregistered.txt` freezes this report, and `results/preregistration.txt` records its SHA-256 and UTC time.

## Results

Executed after parent committed the preregistration at `d4bca74`; the committed frozen text was compared byte for byte before code. Its SHA-256 is `b4271d0ac83c136e77b1adb94c4aebc909b6c643f0285036e795c7cfaeada654`. The current source inputs for norm lowering, range analysis, selection, reporting and the 405B fixture match main `709c37c` exactly. A later main merge updates unrelated RFC-0047 layout validation; [environment.json](results/environment.json) records the executing checkout and exact source hashes.

The committed 405B c2/B32 baseline is reproduced exactly: per token `C=45,566,660`, `K=45,440,678`, `M=7,693,467,648`; first layer per token `C=360,232`, `K=360,380`. The independent Python checker also reconstructs every baseline gadget's commitment/lookup contribution, including unchanged row scalars and the embedding checkpoint.

| Dimension / precision | Ordinary interval width → exact joint width | Balanced split before → after | Exact ordinary key magnitude | Whole-token ΔC = ΔK | Unique table rows before → after |
|---|---|---|---|---|---|
| 8B, current κ30/e672 | 52 → 37 | 26 → 19 | 68,719,476,636 | 0 | 2,043,740,160 → 1,909,260,288 |
| 405B, current κ30/e2685 | 52 → 38 | 26 → 19 | 137,438,952,750 | 0 | 7,693,467,648 → 7,559,249,920 |
| 8B, historical κ16/e672 | 38 → 23 | 12 → single | 4,194,303 | −262,144 | 2,109,804,544 → 1,916,862,464 |
| 405B, historical κ16/e2685 | 38 → 24 | 12 → single | 8,388,607 | −4,128,768 | 7,759,532,032 → 7,574,978,560 |
| 8B, κ31 cap/e4096 | 52 → 38 | 26 → 19 | 137,438,952,441 | 0 | 2,043,740,160 → 1,909,522,432 |
| 405B, κ31 cap/e16384 | 52 → 39 | 26 → 20 | 274,877,898,699 | 0 | 7,693,467,648 → 7,558,725,632 |

At current 405B the universal epsilon-positive bound is `abs(key)<=137,438,953,471`, while complete coordinate-magnitude enumeration sharpens it to `137,438,952,750`. Both signs attain the latter at a row with one coordinate `±524,250`, all other coordinates zero, and `rho=262,163`. The independent checker records the exact lower/upper bracket slacks and a sparse full-row witness of declared dimension. This is attainability in the declared residual-row domain; no claim is made that one fixed pretrained model reaches every such row.

The first embedding-derived norm has even coordinates in `[-254,254]`, rather than arbitrary INT20 values. Its current 405B key interval has width 41 and split 21 before refinement, and exact width 38/split 19 afterwards. Its extrema are `±134,665,295,312`, with one coordinate `±254` and `rho=530,178,328`. The final norm is the 253rd norm and uses the ordinary INT20 bound; neither it nor the first norm changes multiplicity. In the historical κ16 control the first norm was already single; the remaining 252 norms become single, saving `252*16384=4,128,768` values/lookups per token. Its first-layer saving is only 16,384, which illustrates why the first norm cannot be silently replaced by an ordinary layer average.

Current 405B table sharing gives this exact difference:

| Table definition | Change in rows |
|---|---|
| `Requant{b20,r6,lo−128,hi127}` (old first norm high table) | −1,048,576 |
| `Requant{b26,r1,lo−128,hi127}` (old ordinary norm high table) | −67,108,864 |
| `Range{26}` | −67,108,864 |
| `Requant{b19,r8,lo−128,hi127}` (all refined norms) | +524,288 |
| `Range{19}` | +524,288 |
| Net | −134,217,728 |

The old first norm's `Range{21}` remains used elsewhere; the program keeps it. Every other retained definition and all table IDs/definitions/digests are recorded in the raw compiler output and [count.json](results/count.json). The candidate's commitment and lookup counts remain exactly `45,566,660` and `45,440,678`, including first and final norms. No production implementation was changed.

Required checks complete:

- Complete magnitude enumeration for both dimensions and all three precision settings, including positive/negative INT20 endpoints and even embedding coordinates. Python independently reproduces every Rust extremum; a separate binary-search root checks each extremizing witness.
- 93,600 toy rows, 954,440 candidate roots and 361,120 accepted keys, with exactly one accepted root per row/epsilon/precision combination. Coverage spans all registered dimensions, signed3-bit rows, epsilon1–4 and κ0–4.
- 3,200 seeded/boundary synthetic rows: 1,000 each at dimensions 4/16/128 and 100 each at 4096/16384. All norm inequalities and no-wrap bounds pass. Fifteen production-table request groups compare 160,877 distinct within-group key outputs against direct integer rounding/saturation, including ties and saturation neighbours, current/control precisions and embedding intervals.
- Zero/nonpositive epsilon controls, an actual root-width33 rejection, exact-square neighbours, zero rows, lower/upper bracket deletion controls, and nonzero padded-root rejection with the padding pin. Dropping the pin admits a second padded root. No candidate field residue is treated as an integer without the original centered-range bound.
- Six scratch saved-record mutations are caught (extremum, signed width, split multiplicity, unique table rows, verdict and requant coverage); the no-op control passes. These are data-audit mutations, not production-code or cryptographic protocol fault injection.
- Narrow Rust formatting and clippy pass. Repository lint passes; provenance records8 harness files,36 source inputs and the saved raw outputs. Critical source inputs match normative main709c37c.

Raw evidence: [compiler.json](results/compiler.json), [audit.json](results/audit.json), [lookup-requests.json](results/lookup-requests.json), [lookup-responses.json](results/lookup-responses.json), [mutations.json](results/mutations.json), [count.json](results/count.json), and [SHA256SUMS.json](results/SHA256SUMS.json). The binary's synthetic table-lookup mode is driven by `check.py`; default `check.py` regenerates the oracle records, while `--audit <file>` independently rechecks saved records. The checker runs every finite enumeration again; no coverage claim is inferred from a saved success flag.

Additional commands from the repository root:

```sh
python3 -I research/experiments/EXP-0017-rmsnorm-relational-bound/check.py --audit research/experiments/EXP-0017-rmsnorm-relational-bound/results/audit.json
python3 -I research/experiments/EXP-0017-rmsnorm-relational-bound/mutate.py
python3 -I research/experiments/EXP-0017-rmsnorm-relational-bound/count.py
cargo fmt --manifest-path research/experiments/EXP-0017-rmsnorm-relational-bound/Cargo.toml --check
cargo clippy --all-targets --locked --manifest-path research/experiments/EXP-0017-rmsnorm-relational-bound/Cargo.toml -- -D warnings
cargo xtask lint
python3 -I research/experiments/EXP-0017-rmsnorm-relational-bound/provenance.py
```

## Analysis

The integer implication survives, but the registered current count-saving claim does not. At κ30 the best exact 405B row-domain bound is still38 signed bits, so both ordinary norm sites and the final norm remain split. The old37-to25-bit narrative predates RFC-0023. Its κ16 control produces24 bits at 405B and23 at 8B, independently explaining the old count saving without assigning it to current quantization.

Table rows decrease by1.7445674% of the counted 405B table set. That is an E0 logical-row result. In the current conditional `counted` model, which amortizes table commitments/lookups over 16K tokens, μ changes from 32.2941033 to32.2914299 at1K (−0.0082784%), and 102.4439173 to102.4414366 at16K (−0.0024214%). The `corrected` reductions are −0.0065297% and−0.0019512%, respectively. These calculations replace only the table-row count; commitment/lookup counts and every unit-cost assumption remain fixed. They are partial planning projections with the current P4/P5/streaming omissions or bounds, not full GPU bounds. They cannot justify the historical conditional−5.92% current-count claim.

The argument requires exact integer `u`, positive public epsilon, exact BigIntCompare brackets and a padded-rho zero pin. The original interval product is already a true centered integer (`abs(key)<2^51` in the current fixtures), so refinement does not infer integer semantics from a small field residue. The candidate introduces no new relation or witness in this E0 model. This establishes the arithmetic implication and row-domain extrema under those premises, not an endorsement of every implementation of the root comparator or the entire protocol's soundness/zero knowledge. There is no independent verifier integration or proof benchmark here. The full compiler report retains its approximate 20/30 RsqrtRow per-row cost and omitted row-gadget range-table rows.

## Verdict

`refuted` for IDEA-0011's current commitment/lookup count-saving hypothesis: both ordinary 405B norms keep two commitments and two lookups per element and per-layer ΔC=ΔK=0. The bound itself remains a valid algebraic refinement under the stated exact-integer premises. IDEA-0011 moves to `refuted` for this scoped count claim, priority0, confidence 0.95 in the current count refutation. A distinct table-row optimization remains unpromoted and unscored.

## Follow-ups

A new scoped proposal could test the logical table-row reduction, implementation cost and verifier re-derivation. It would need a specification amendment and independent verifier work, rather than assuming the old four-lens reviews cover today's precision or a production change. H100 measurements could price table construction, multiplicity storage and lookup throughput; they cannot change the exact count or establish the integer bound. No broad sweep or unattended schedule was installed.
