# EXP-0016: Argmax-read certificate uniqueness and merged-instance routing

| | |
|---|---|
| Status | `planned` |
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

Pending.

## Analysis

Finite arithmetic and global multiset identities establish finite counterexamples or finite checks; they do not certify Fiat-Shamir soundness, zero knowledge, verifier integration, all possible dimensions, or performance. No stochastic false-accept measurement replaces a structural routing attack. The Exp generator is reused as the normative function; correctness of its implementation is not what this experiment measures.

## Verdict

Pending the fixed checks.

## Follow-ups

If refuted, record the exact missing clauses and reject promotion of the literal mechanism; any repaired proposal needs a new scoped theorem and superseding RFC. H100 costs, opening groups, full protocol/zero-knowledge integration and current cost repricing remain separate work. No scheduler or unattended job is installed.
