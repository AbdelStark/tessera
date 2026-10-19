# EXP-0016: Argmax-read certificate uniqueness and merged-instance routing

| | |
|---|---|
| Status | `done: refuted` (literal merged-instance proposal); local scoped finite case passes |
| Idea | [IDEA-0020](../../ideas/IDEA-0020-argmax-read-max-certificate.md) |
| Evidence level | E0 exact finite arithmetic, exhaustive witnesses and symbolic bounds; no performance evidence |
| Hardware | CPU execution of exact Python integer/F65537 checks; no benchmark |
| Pre-registered on | 2026-10-19 |

## Hypothesis

IDEA-0020's proposed chunked argmax-read certificate has exactly one witness equal to the true causal maximum and first maximizing index, including when its committed read tables are merged across layers and KV heads. The finite checks below test that claim and identify which additional namespace, table-totality and padding conditions a general theorem would require.

## Success criterion

`validated` means every specified literal finite-domain test is complete and unique, all merged-instance certificates are correctly routed, and every prescribed boundary/attack is rejected. `refuted` means an explicit wrong or second witness satisfies the stated relations, with a complete balanced merged LogUp multiset where routing is involved, or an honest witness is impossible under the literal relations. `inconclusive` means neither condition is established or some required enumeration/check is incomplete. A corrected variant passing its finite checks cannot validate the original untagged merged-instance proposal. Finite success is not a theorem over all architectures, fields, transcript challenges or committed tables, and cannot justify an RFC by itself.

## Method

No production changes. One exact-arithmetic Python program models the candidate relations and an independently written direct causal argmax oracle. Integer checks and congruences over F65537 are both recorded. Synthetic domains are fixed below; no wall-clock speed-up, GPU or end-to-end proof claim is made.

1. **Local row exhaustion.** Every length-3 and length-4 integer row in [-8,8], every real query prefix q=0..length-1, and every selected j=0..3 (including the padded fourth position for length3). Two chunks of two entries encode j. The faithful realizable embedding is d=1, Q=1 and K_i=score_i; an auxiliary d=2 suite enumerates Q and both K rows in {-1,0,1}^2 for a two-position context. For each candidate j, read membership determines Kread and the dot equation determines c as a field element; no a-priori integer maximum assumption is used. Every passing witness must match the independent max/first-index oracle and the base Exp values.
2. **Full field factors, not an infeasible Cartesian product.** Enumerate all 65537 field values for a chunk-entry booleanity equation, leaving only its exact roots. Enumerate all 16 Boolean assignments of the four chunk entries and filter the two sum-one equations. For all i,j in 0..3 enumerate every field-valued candidate lt against its defining linear equation. Enumerate every field-valued candidate c and Kread for each scalar value in [-8,8] against the fixed-key read and dot equations. Enumerate each Exp2 key in [0,128) and field-valued candidate outputs for each distinct table value; a fixed key in a functional table has exactly one output. These factorizations are part of the method, not a post-hoc reduction of the witness space. Pointwise relations are idealized: random zerocheck/LogUp challenge failure probabilities are not exhausted here.
3. **Tie and causality controls.** All equal rows, zero rows, negative rows, later tied indices, future j>q, padded columns, zero/one chunk sums, two-hot chunks, arbitrary non-Boolean field values, and forged lt are included. Disabling causality and parity in separate controls must admit the predicted alternative witnesses; the intact routed local variant must reject them.
4. **Exp2 variants.** Use the current RFC-0008 integer Exp generator at sigma4, scale1 and key_bits6 (last base key maps to0). The note's mathematical variant is `Exp2[t]=Exp[ceil(t/2)]`; make its otherwise undefined top odd row explicit with zero-tail continuation Exp[64]=0. Separately test the note's contradictory claim that the value depends on `t>>1`; odd-key outputs must be compared to the original Exp semantics. Neither choice is silently substituted for the other. Migrated masked score positions use the doubled table's public last key127 and output0, as RFC-0030 requires.
5. **Merged routing exhaustion.** Enumerate two layers, two KV heads, one position, d=1; every committed K entry in {-1,0,1}, every per-reader scalar Q in {-1,0,1}, and all 24 permutations assigning the four readers to the four source instances. One reader per instance, all j=0, m=1 per source position. Check the entire lookup multiset against the entire weighted table multiset, modulo F65537 as well as over integers. Record every accepted wrong maximum for untagged keys i*d+d_index, and require the fully tagged key (((layer*H+kv)*S+i)*d+d_index) to reject each wrong read; permutations that only exchange equal values are not false witnesses.
6. **Full nondegenerate merged attacks.** Construct complete, mutually balanced read batches for: cross-layer swaps, cross-KV-head swaps, their composition, and omitted dimension keys. Use two layers, two KV heads, two query heads per KV group, two positions and d=2. Each reader's mandated KV head is query_head//2. Each row reads every dimension at its selected causal index; the table side uses one shared m(layer,kv,index) for both dimensions, as the idea requires. The examples include nonconstant scores and reciprocal brackets, tied/zero rows and masked future entries. The program records all Q/K/one-hot/c/lt/key'/e/R/m data, every row relation, exact tuple Counters and matching sums of fractions at fixed nonsingular challenge pairs. The attack must be an actual global identity with balanced m, not a permissive union-membership oracle. Repeat each attack with omitted layer, omitted head, omitted both and complete namespace; fully scoped tables or independent identities are the repair controls.
7. **Padding migration.** Test a length3 sequence padded to4, plus a padded layer and padded KV/query head. Keep current RFC-0030 C=0 and zero-tail masked lookups, and RFC-0035 c/R zero pins. Test the literal new chunks' sum-one rule on an all-zero padded witness (completeness); test a naive sum=psi/read-gated migration without a Kread padding pin (alternative padded reads); and an explicit repair with chunk sums=psi, gated reads, (1-psi)*Kread=0, c=psi*dot, causal check gated by psi, and masked exponentials0. Enumerate padded scalar Kread over all F65537 values and Boolean chunk assignments. A repaired result is reported separately, never attributed to the literal proposal.
8. **Goldilocks obligations.** Derive score bounds from full INT8 endpoints (including attained +2^21 at d=128), the doubled key's range, and no-wrap conditions for chunk sums, index recomposition and namespaced memory keys. Check 405B parameters L=126,Hkv=8,S=2^17,d=128 and maximum public type-shaped parameters L=Hkv=65535,S=2^17,d=65534, explicitly separating syntactic shapes from any resource/statement acceptance that is not checked. Probe Goldilocks aliases when a packed namespace reaches P=2^64-2^32+1, and all-field one-hot wrap when a chunk can hold P ones. A general theorem must require namespace injectivity belowP or use explicit tuple coordinates, small chunk counts belowP, correct per-instance routing, functional total Exp2, masked/padded witness pins, fixed commitments before compression challenges, and the ordinary statistical LogUp/zerocheck bounds. The full protocol is not implemented by this experiment.

The current normative context is main `6d9de83`, including RFC-0048's distinction between model fidelity and uniqueness under arbitrary committed values, and current RFC-0009/RFC-0035 Argmax vocabulary/padding selectors. Q and K in this proposed read certificate are post-Requant INT8 outputs, not arbitrary unbounded accumulators; no general input-range assumption is silently borrowed from trusted weights. The compiler's GQA routing and the committed table's layer/head identity must be part of any repair.

## Reproduction

From the repository root:

```sh
python3 -I research/experiments/EXP-0016-argmax-read-lemma/lemma.py
```

The run saves results/run.json and explicit counterexamples under results/. Before code, results/README.preregistered.txt stores this text and results/preregistration.txt stores its SHA-256 and UTC time. No harness code or production edit is included in the preregistration commit.

## Results

**E0, exact arithmetic. The literal merged-instance proposal is refuted.** This is a candidate-relation counterexample, not a vulnerability in Tessera's existing product-tree certificate and not a forged production proof.

Reproduce with the command above. [results/run.json](results/run.json) and [results/run-final-namespace-support.txt](results/run-final-namespace-support.txt) contain the final exact run; the environment is recorded in the JSON. All inputs are exhaustive synthetic domains or explicit integer fixtures, without randomness or performance claims.

| Finite check (E0) | Cases | Result |
|---|---|---|
| Length3 rows in [-8,8] | 4,913 rows; 14,739 causal prefixes; 58,956 candidate indices | Exactly one scoped witness per prefix |
| Length4 rows in [-8,8] | 83,521 rows; 334,084 causal prefixes; 1,336,336 candidate indices | Exactly one scoped witness per prefix |
| d2 actual dot products, coordinates in {-1,0,1} | 1,458 query-prefix cases | Exactly the direct oracle's first maximum |
| Full F65537 factors | 65,537 Booleanity candidates; 16×65,537 lt candidates; 17×65,537 each c and scalar Kread candidates; 5×65,537 output candidates | Only the required roots/outputs survive |
| Two-layer/two-head scalar merged assignments | 157,464 complete table/query/permutation configurations | 38,448 wrong-maximum sublemma certificates; 49,248 alternative-read certificates |
| Scalar alternatives including reciprocal brackets | 10,800 | Alternative Kread remains possible; no need to omit the bracket |
| Fully tagged scalar repair | Same finite configurations | No wrong certificate |
| Padding selector grid | 64 layer/head/query rows, of which 27 real | All repaired rows complete; padded fields uniquely zero |

The factor checks cover complete field-valued scalar domains without enumerating their Cartesian product: Booleanity first restricts each chunk entry to 0/1; sum-one leaves four chunk assignments; the read and dot equations then fix Kread and c, and each defining lt/output equation has one field root. The independent oracle uses direct integer max/first-index selection. This does not assume a malicious maximum is an integer in advance. It idealizes pointwise constraints, rather than sampling a zerocheck challenge.

### Complete balanced routing counterexample

The layer attack has two layers, two KV heads, two query heads per KV group, two positions and two dimensions. The required GQA mapping is always `kv_head=query_head//2`.

- `K[layer,kv,i]=(layer,-i)`.
- The first query head of each group uses `Q=(1,0)` in layer0 and `Q=(-1,0)` in layer1. The second uses `Q=(0,1)` and is honest; these benign rows have nonconstant scores `(0,-1)`.
- Every reader selects j=0 with the valid chunk witnesses. Every q=0 reader remains honest. Only q=1 of the first query head reads position0 from the other layer's same KV head. The causal bound is satisfied.
- The shared table multiplicity is `m[layer,kv,0]=4`, `m[layer,kv,1]=0`, used identically for both dimension columns. Three readers per instance stay local and the fourth is exchanged with the other layer; every source still has four readers.
- In layer0 the attacked row has true scores `(0,0)` but c=1. In layer1 its scores are `(-1,-1)` but c=0. In either row lt=0, both valid doubled lookup keys are2 and both exponentials are6, rather than the honest16. Thus s=12, R=21, a=4, b=7. Every range and reciprocal bracket passes; all future/padded exponentials are0. The removed product-tree certificate would have root1, and rejects.

This is four wrong rows in a complete16-row batch. [results/counterexample-layer.json](results/counterexample-layer.json) includes every query, key row, read, chunk, c/lt/key/e/R/slack/limb witness, shared multiplicity and exact multiset. The complete untagged key/value lookup multiset equals the complete weighted table multiset, not merely a union-membership predicate. Therefore their LogUp rational functions are identical for **every** compression challenge and every non-pole denominator challenge. The saved F65537/Goldilocks fraction evaluations at two challenge pairs are independent arithmetic diagnostics, not the reason the identity holds. No challenge grinding is needed.

[results/counterexample-head.json](results/counterexample-head.json) exchanges KV heads instead, [results/counterexample-both.json](results/counterexample-both.json) exchanges both axes, and [results/counterexample-dimension.json](results/counterexample-dimension.json) is the extra omitted-dimension negative control. Every batch has four wrong rows and all reciprocal brackets pass. Each missing relevant axis admits its corresponding attack. The original proposal already includes the dimension index; the dimension control does not accuse it of omitting that index.

The fully scoped address rejects each example. Its wrong read tuple is absent from the **entire** routed table support, so changing arbitrary field-valued m cannot repair it. This is stronger than testing just the chosen multiplicities. Keeping separate instance identities also prevents exchanging one instance's table for another's, subject to ordinary randomized batching soundness.

### Direct Goldilocks lift and current validation

The additional direct lifts are analytic zero-extensions of the finite counterexample, not a new exhaustive domain: [layer](results/counterexample-Goldilocks-layer.json), [head](results/counterexample-Goldilocks-head.json), [both](results/counterexample-Goldilocks-both.json). They use d_head4, four positions, two layers, four query heads, two KV heads and the same nonzero coordinates. Extra query/key coordinates are0; only q=1 of the first head cheats. Chunk sizes2×2 now match the four-position axis. All query rows, including honest q=2/3 rows, and all dimension reads are included. Shared position0 multiplicities are8. All reciprocal brackets pass over the integers, and all wrong rows have nonzero current product roots.

The scale1 is compatible with integer-model score scaling at d_head4 and e_q+e_k=1: `2/sqrt(4)=1`. The public INT8 spread is130,560, so the base key width is17 and the doubled table has18 bits; its last key262,143 maps to0. All real keys are tiny and no field alias is used. This is a constraint-level lift with compatible public parameters, not a constructed complete model/claim list or an end-to-end proof.

Upstream709c37c/#388 changed compiler lowering, tv-core validation, tests and specification: it enforces `(2*max_context-1)*2^(2*exp_sigma)<P`, by a bit bound in the compiler and independently checked exact u128 arithmetic in the verifier. At sigma4 and contexts2/4, the left sides are768/1,792, below both fields. No relation-harness semantic change is needed. The attack has s=12<2^sigma because the replacement maximum is false; the new rule does not separately enforce the row-sum lower bound, which is derived from a sound maximum certificate. The routing verdict is unchanged.

### Exp2, padding and no-wrap obligations

The note's two Exp2 descriptions disagree. At key1, ceil-based Exp2 gives Exp[1]=6 while `key>>1` gives Exp[0]=16. The honest row `(0,1,0)`, q=1, j=1 therefore gets the wrong first exponential under the shift-based description. The top odd key127 also requests Exp[64], outside the base table. The checked ceil variant explicitly completes that row with the zero tail; an RFC must define it, and cannot reuse the aligned low-bit-elimination claim unchanged.

On zero public padded witness rows, the literal chunks' sum-one rule is incomplete. A naive migration that changes the sum to psi and gates reads still leaves65,537 choices per padded scalar Kread if nothing else pins it. The explicitly repaired candidate adds a zero pin (or would need equally explicit structural public-zero padding). It has exactly one such value. This is a hypothetical candidate migration, not a claim that current normative padding permits free values. The program evaluates C/psi for real and padded query, layer and KV/query-head rows, including length3 padded to4.

Goldilocks bounds use the full INT8 endpoints, including attained score +2^21 at d128. The 405B packed namespace `126*8*2^17*128=16,911,433,728` is belowP. A syntactic extreme shape `(65535,65535,131072,65534)` has36,891,236,390,554,828,800 rows: addresses `(0,0,0,0)` and `(32769,32770,65542,13)` both map to0 moduloP. Full statement/resource acceptance of that extreme model was not tested; it establishes the need for an explicit packing bound or distinct tuple coordinates. A one-hot chunk also needs length<P: P Boolean ones sum to0, and P+1 ones sum to1. Current small chunks are below that bound. F65537 exhaustion discharges none of these general Goldilocks obligations by itself.

## Analysis

The structural failure is selecting from the union of committed tensors while claiming to have read one mandated tensor. LogUp itself faithfully proves that union multiset; it cannot infer a missing layer/KV-head identity. The attack uses ordinary positive multiplicities and valid INT8 inputs, includes every row and dimension, and passes the unchanged reciprocal brackets. This refutes the literal merged certificate, independently of any projected cost.

The scoped local arithmetic survives every finite case, which explains why single-row checks alone would miss the defect. A general repaired theorem still needs globally injective namespaces or independent instances, table commitments tied to the correct post-Requant tensor, functional total ceil-based Exp2, unique padding, small chunk counts, no-wrap bounds, and complete challenge/opening/masking integration. RFC-0048 does not supply a hidden range assumption: the Q/K used here are bounded post-Requant outputs; arbitrary malformed weight commitments do not justify weakening those obligations.

Finite enumeration is E0 evidence, not a full theorem, a statistical LogUp/zerocheck soundness calibration, a verifier implementation or a complete production proof. H1's historical cost scenario is not repriced in this security-first unit; no H100 or performance claim is made. Existing production retains the product certificate and rejects every wrong row in these fixtures.

Artifact provenance: development E0 runs reused the unversioned output files before preservation was requested. The available final post-#388 run was copied to run-final-before-mutations.{json,txt}; the final extended-support output and unmodified mutation control are retained separately. Earlier timestamps are not reconstructed or represented as original raw evidence. Counts are deterministic exact results, not timing samples.

## Verdict

`refuted`. IDEA-0020 becomes `rejected` for the literal untagged merged-instance mechanism under research rule1 (SP-3 fails), priority1.21→0. Confidence0.3→1.0 is confidence in this explicit structural counterexample, not in any repaired scheme. The tagged local variant has finite evidence only and remains an unproven proposal.

## Follow-ups

If refuted, record the exact missing clauses and reject promotion of the literal mechanism; any repaired proposal needs a new scoped theorem and superseding RFC. H100 costs, opening groups, full protocol/zero-knowledge integration and current cost repricing remain separate work. No scheduler or unattended job is installed.

## Validation and review

The full exact run and a no-op mutation control pass. [results/mutations.json](results/mutations.json) records four caught relation faults: dropping routing tags, dropping parity, dropping the padded-read zero pin, and replacing ceil by floor. Raw assertion outputs are retained; no production file is mutated.

```sh
python3 -I research/experiments/EXP-0016-argmax-read-lemma/lemma.py
python3 -I research/experiments/EXP-0016-argmax-read-lemma/mutate.py
```

Adversarial review checked: full-input finite factors rather than an assumed max range; exact score/namespace bounds; whole-batch tuple equality and shared dimension multiplicities; GQA routing; ties and future indices; functional lookup values; every reciprocal/range bracket; public masking and candidate padding; and the source of residual statistical soundness. The independent integer max oracle, source-table support check and current product root are distinct rejection oracles. No verifier independence surface or secret-handling production code changes.
