# RFC-0058: InverseCdf Columns and Comparisons

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.4
- Supersedes in the listed sections:
  - [RFC-0009](RFC-0009-gadgets.md#inversecdf-temperature-sampling) 'InverseCdf': the partition's bound, the integers' representation, the committed values, their order and the range requests;
  - [RFC-0009](RFC-0009-gadgets.md#summary-of-costs) 'Summary of costs': the `InverseCdf` row.
  - [RFC-0035](RFC-0035-padded-rows.md#the-padding-zero-check) 'The padding zero-check': `InverseCdf`'s virtual partition `Z` joins the values pinned on a padded row.
- Scope: the gadget, in `tessera-gadgets`. The vocabulary certificate, the PRF circuit and the gadget's place in the schedule and the advice layout are [#152](https://github.com/AbdelStark/tessera/issues/152)'s; the verifier's checks are [#160](https://github.com/AbdelStark/tessera/issues/160)'s.
- Origin: [#144](https://github.com/AbdelStark/tessera/issues/144)

## Summary

RFC-0009 certifies a temperature-sampled token `ŷ` by two comparisons of integers that exceed the field, `C_<·2^32 ≤ u·Z` and `u·Z + ψ ≤ C_≤·2^32` (Lemma 10). It leaves open how those integers enter `BigIntCompare`: their limbs, which of them are committed, the comparisons' limb count and the order of the values and range requests. Two correct implementations could commit different columns, and the verifier could not derive the advice layout from public data (INV-PCS-1, SP-7).

This RFC fixes them, as RFC-0050 did for `RsqrtRow`:
- **Three vocabulary sums.** `Z`, `B = C_<` and `S = e_ŷ`, proven over the vocabulary. `C_≤` is `B + S` and is not committed.
- **Two decompositions** in radix `2^16`: the draw `u` in two limbs, and `Z` and `B` in `k = ⌈W/16⌉` limbs each, where `W` is the width of the partition's public bound `vocab·2^σ`. One limb of each is virtual.
- **The comparisons.** Both use radix `2^16` and `m = k + 2` limbs. Their coefficients are sums of limb products, below `2^33`.
- **The order** of the committed values and of the range requests, and their counts: `1 + 2(k − 1) + 12(k + 2)` values and `2 + 2k + 12(k + 2)` range requests per output token, plus the sampled token's indicator in bounded mode.
- **What this gadget proves itself.**
  - Bounded mode's indicator gets OneHot's booleanity, vocabulary and count relations.
  - A padded row's partition gets a zero-check.

It also corrects RFC-0009's bound on the partition, `Z < 2^{σ+18}`. A vocabulary of `2^18` tokens whose logits are all equal reaches `Z = 2^{σ+18}`; the bound is `Z ≤ vocab·2^σ`.

## Motivation

[#144](https://github.com/AbdelStark/tessera/issues/144) implements `InverseCdf`. RFC-0009 gives its relations at the level of integers:

- `C_≤ = Σ_v e_v Σ_{w ≥ v} ĥ_w` and `C_< = C_≤ − Σ_v ĥ_v e_v`, virtual;
- `BigIntCompare(C_<·2^32 ≤ u·Z)` and `BigIntCompare(u·Z + ψ ≤ C_≤·2^32)`.

`BigIntCompare` takes vectors of limb coefficients below `2^40` (RFC-0050 'Complete comparison schema'). The integers here reach `2^82`, and `Z` alone `2^50`, so each needs limbs, and a product `u·Z` needs the limbs of both factors. RFC-0009's cost row, "`vocab` + about 20" commitments, predates any such choice. RFC-0050 replaced the corresponding estimate for `RsqrtRow`, about 20, with an exact 187 at a 32-bit root.

## Goals

- A representation that the prover and the verifier derive from public data alone: `vocab`, the vocabulary table's `σ`, and the mode.
- Exactly one passing witness per output token, real or padded, so that Lemma 10's uniqueness carries over to the committed values (SP-3).
- No change to RFC-0009's relations as integer statements, to the tables, to the messages or to the labels.

## Non-Goals

- **The vocabulary certificate.** The exponentials `e_v`, their max certificate over the vocabulary and the `Exp_T` tables are RFC-0017's and #152's.
- **The draw.** `u` is committed by the PRF circuit of RFC-0017 (#142, #152). This gadget reads it.
- **The schedule.** Which stage proves each relation, and the advice layout's segments, are the compiler's (RFC-0010, RFC-0041), when #152 wires the gadget in.

## Proposed Design

### Parameters and inputs

The public parameters are:
- the vocabulary `V`, with `1 ≤ V ≤ 2^18` (RFC-0007 'Verifier validation'), padded to `V' = 2^n`;
- the vocabulary table's `σ`, with `0 ≤ σ ≤ 32` (RFC-0008 'Table specifications', `Exp`);
- the mode: exact or bounded (RFC-0017 'Predictors').

A row's `σ` is its vocabulary table's. Rows whose temperatures' tables differ in `σ` have different `W` and `k`, and #152 groups them by table.

`C_v = [v < V]` is the public vocabulary indicator, as in `Argmax`. Write `W = bit_length(V·2^σ)`, which is at most 51 since `V·2^σ ≤ 2^50`, and `k = ⌈W/16⌉`, at most 4.

Each output token's row reads:
- **The exponentials** `e_v`, `v < V'`. The vocabulary certificate gives true integers in `[0, 2^σ]` for `v < V`. Entries at `v ≥ V` are never read unmasked.
- **The draw** `u`, a committed value.
- **The indicator** `ĥ`, with count `ψ` (RFC-0009 Lemma 8, RFC-0035):
  - **In exact mode**, the next recorded token's indicator (teacher forcing, RFC-0009 'OneHot'). Its own four OneHot obligations, proven where it is committed, bind it.
  - **In bounded mode**, a second indicator, for `ŷ`, which this gadget commits. This gadget proves OneHot's booleanity, vocabulary and count relations on it.
    - **Why they are needed.** Without them the row's obligations admit a non-Boolean `ĥ` with count 1, or one on the vocabulary's padding (Testing Strategy).
    - **No index relation.** Bounded mode commits no token value: the indicator itself carries `ŷ`, which the mixture's match bit reads (RFC-0017 'In-circuit scoring').
- **The row selector** `ψ`, public (RFC-0035).

### The vocabulary sums

Three virtual integers per row, each a contraction over the vocabulary's `n` variables:

| Value | Relation | Degree, one row | Meaning on a real row |
|---|---|---|---|
| `Z` | `Σ_v C_v·e_v` | 2 | the partition, `Σ_{v < V} e_v` |
| `B` | `Σ_v e_v·L_v` | 2 | `C(ŷ − 1) = Σ_{v < ŷ} e_v`, which is `C_<` |
| `S` | `Σ_v ĥ_v·e_v` | 2 | `e_ŷ` |

The degrees are a single row's, over the vocabulary's variables, with the public `C_v` counted as a factor. Batched over the rows, the claim at a random row point `r` gains `eq(r, t)`: `Z̃(r) = Σ_{t,v} eq(r, t)·C_v·e(t, v)`, of degree 3 over the row and vocabulary variables, and so for `B` and `S`. #152 places them in the schedule with those degrees.

`L_v = Σ_{w > v} ĥ_w` is `Argmax`'s strict suffix. A claim `L̃(ρ)` reduces to `Σ_v w_v(ρ)·ĥ_v` with `w_v(ρ) = Σ_{i < v} eq(ρ, i)`. The verifier evaluates the weights' extension in `O(n)` time with `Argmax`'s triangular kernel (RFC-0009 'Argmax'). `C_≤ = B + S` is never formed as a value of its own.

The mask `C_v` keeps vocabulary padding out of `Z`. `B` and `S` need none: OneHot's vocabulary relation makes `ĥ_v = 0` at `v ≥ V` (#378), so `L_v`, a sum of `ĥ_w` over `w > v ≥ V`, is `0` there too.

`Z`, `B` and `S` are virtual tensors over the rows, each with its own multilinear-extension claim, which its contraction reduces to claims on `e` and `ĥ`. As for `RsqrtRow`'s input (RFC-0050 'Padding and binding obligations'), the extension of a product is not in general the product of extensions at the final point. The row relations below are pointwise polynomials in these values, not identities between extensions.

### The decompositions

Each uses RFC-0009 Lemma 3's limbs, in RFC-0050's schema: least significant first, every limb committed except the top one, which is virtual. A decomposition into one limb commits nothing.

- **The draw.** `u = u_0 + 2^16·u_1`. `u_0` is committed and requests Range(16). The virtual top `u_1 = (u − u_0)/2^16` requests Range(16).
- **The partition.** `Z = Σ_{j < k} Z_j·2^{16j}`. `Z_0` to `Z_{k−2}` are committed, each requesting Range(16). The virtual top `Z_{k−1} = (Z − Σ_{j < k−1} Z_j·2^{16j})/2^{16(k−1)}` requests Range(`W − 16(k − 1)`).
- **The prefix.** `B` is decomposed as `Z` is, in `k` limbs `B_j`, in the same box.

The divisions are field inverses. The complete boxes, `2^32` and `2^W ≤ 2^51`, lie below `(P − 1)/2`, so the lookups force the unique integer decomposition (Lemma 0). `B` takes `Z`'s box because `B ≤ Z` whenever `ĥ` is an indicator. The decomposition would refuse a value of `2^W` or more.

### The comparisons

Both comparisons follow RFC-0050 'Complete comparison schema', with radix `2^16` and `m = k + 2` limbs. That holds `u·Z < 2^{32+W}` and `(B + S)·2^32` on every passing row: a side of `2^{16m}` or more leaves a final carry, which has no witness.

The input coefficients, `j` from `0` to `m − 1`, with `B_{−1} = B_{−2} = 0` and every limb outside its decomposition `0`:

```text
P_j     = Σ_{i + h = j, i ∈ {0,1}, 0 ≤ h < k} u_i·Z_h              the product u·Z
lower:  X_j = B_{j−2}                    ≤   Y_j = P_j
upper:  X_j = P_j + [j = 0]·ψ            ≤   Y_j = B_{j−2} + [j = 2]·S
```

That is, lower is `B·2^32 ≤ u·Z` and upper is `u·Z + ψ ≤ (B + S)·2^32`, RFC-0009's two relations with `C_< = B` and `C_≤ = B + S`.

- **The coefficients** are sums of products of limbs, never limbs of a product already reduced in the field.
- **Their bounds** are derived from the limb boxes: `2^16 − 1` for the low limbs, `2^{16} − 1` for `u_1`, and the narrowed top boxes. With `ψ ≤ 1` and `S ≤ 2^σ`, every coefficient lies below `2^33`, inside RFC-0050's `2^40`. The carry widths, the equation boxes and the Lemma 0 check follow from these bounds as RFC-0050 specifies.
- **`S` sits in one coefficient**, `Y_2`, below `2^16 + 2^32`. So `C_≤` needs no decomposition of its own.

### Committed values, range requests and counts

A row commits, in this order, after any graph output of the node (RFC-0006 gives `InverseCdf` none):

1. in bounded mode, the sampled token's indicator `ĥ`, `V'` entries;
2. `u_0`;
3. `Z_0` to `Z_{k−2}`;
4. `B_0` to `B_{k−2}`;
5. the lower comparison's six arrays, in RFC-0050's array order;
6. the upper comparison's six arrays.

The range requests follow the same order, and each decomposition's virtual top follows its low limbs:
1. `u_0` and `u_1`, each Range(16);
2. `Z`'s low limbs and its top;
3. `B`'s low limbs and its top;
4. the lower comparison's requests;
5. the upper comparison's requests.

Per row that makes `1 + 2(k − 1) + 12(k + 2)` committed values, plus `V'` in bounded mode, and `2 + 2k + 12(k + 2)` range requests. The indicator requests none. It is a vocabulary-shaped column, committed as OneHot's is, so a row's own list starts at `u_0` in both modes.

The row's relations come in this order, `6k + 19` in all:
1. the lower comparison's `3m + 3` equalities, in RFC-0050's order: the left normalization chain and its final carry, the right chain and its final carry, the subtraction chain and its final borrow;
2. the upper comparison's `3m + 3`;
3. the padding zero-check of 'Padded rows'.

The three contractions and, in bounded mode, OneHot's three relations are vocabulary relations, outside this list.

| `V` | `σ` | `W` | `k` | `m` | Committed | Requests |
|---:|---:|---:|---:|---:|---:|---:|
| 128,256 | 16 | 33 | 3 | 5 | 65 | 68 |
| 128,256 | 32 | 49 | 4 | 6 | 79 | 82 |
| 2^18 | 32 | 51 | 4 | 6 | 79 | 82 |

These replace RFC-0009's "about 20" and "about 30" for the comparisons. The vocabulary certificate's own costs, `vocab` lookups, commitments and product leaves (RFC-0009 'InverseCdf'), come on top. The three contractions and the suffix's dot product commit nothing.

### Padded rows

At a padded row, `ψ = 0`, and OneHot's count gives `ĥ = 0`, so `B = S = 0`. The partition is pinned by this gadget's own zero-check, in RFC-0035's stage-1 `ZeroCheck`:

```text
(1 − ψ)·Z = 0
```

- **Why it is needed.** The comparisons alone accept any `Z` on a padded row when `u = 0`: lower is `0 ≤ 0·Z`, and upper is `0·Z + 0 ≤ 0`. Without the zero-check, the gadget's witness on a padded row would be unique only if the exponentials were pinned there. RFC-0009 'InverseCdf' pins them to 0 through the vocabulary certificate's masked keys, RFC-0030's `K_0` with `Exp[K_0] = 0`. That is a property of the certificate, which this gadget does not see, and RFC-0035's padding zero-check pins only the certificate's maximum. With this zero-check, a padded row has a witness exactly when `Z = 0`, whatever the certificate does.
- **What #152 must still check.** Completeness needs `Exp_T[K_0] = 0` for every vocabulary table, or a padded row has `Z = V·Exp_T[K_0] ≠ 0` and no witness. RFC-0007 validates it for `SoftmaxRow`'s tables only. #152 adds the same validation for the vocabulary tables.
- **What it costs.** It adds no column: one row relation of degree 2, 3 with the zero-check's `eq` factor.

Then:
- **The sums.** `Z = B = S = 0`, so every limb of `Z` and `B` is `0`.
- **The comparisons.** Lower is `0 ≤ u·0` and upper is `u·0 + 0 ≤ 0`. Every digit, carry, difference and borrow is `0`.
- **The draw.** `u_0` is the low limb of whatever `u` its producer commits.

The witness is unique. On a real row the zero-check is `0·Z = 0`.

### The honest prover

The prover finds `ŷ` with a scan over the vocabulary's `V` entries:
- the prefix sums are kept in 128-bit integers, `u·Z < 2^82`;
- the first entry whose prefix crosses `u·Z` is selected by masks, without a branch on the data (INV-ARCH-2);
- the scan's addresses depend on `V` alone.

In exact mode a recorded token other than `ŷ` has no witness. The gadget refuses it with its own error, distinct from an input outside the domain, and the prover reports `ERR-WITNESS-EXACT-MISMATCH` (RFC-0017 'Predictors'). The token agrees with RFC-0017's reference sampler, `tessera_scoring::inverse_cdf`, on every distribution of true integers in `[0, 2^σ]`.

## Security Argument

**Lemma 10′.** Let `V ≤ 2^18`, `σ ≤ 32`, `e_v` true integers in `[0, 2^σ]` for `v < V`, and `ĥ` an assignment satisfying OneHot's booleanity, vocabulary and count obligations with count `ψ`: in exact mode as the recorded token's indicator, in bounded mode by this RFC's own. Consider a real row, `ψ = 1`. The obligations of this RFC hold if and only if:
- `u ∈ [0, 2^32)`;
- `Z ≥ 1`;
- `ĥ` is the indicator of `ŷ = min{v < V : C(v)·2^32 > u·Z}`, with `C(v) = Σ_{w ≤ v} e_w`.

Every committed value is then determined by `e` and `u`. On a padded row, every committed value is determined by `u`, whatever `e` is.

*Proof.*
1. **The indicator.** Booleanity, the vocabulary relation and the count make `ĥ` the indicator of one `ŷ < V` on a real row, and `0` on a padded row (Lemma 8 without its index relation).
2. **The sums.** `Z`, `B` and `S` are sums of at most `2^18` products of a true integer in `[0, 2^σ]` with Boolean public or committed factors. The products at `v ≥ V` are `0`: `C_v = 0` in `Z`, and `ĥ_v = L_v = 0` in `S` and `B`. Each is a true integer of at most `2^50 < (P − 1)/2`, so the field sum is the integer sum: `Z = Σ_{v<V} e_v`, `B = C(ŷ − 1)` and `S = e_ŷ`. The contractions bind them as RFC-0010 binds any reduction, up to its sumcheck error.
3. **The decompositions.** By Lemma 3, `u ∈ [0, 2^32)` and `Z, B ∈ [0, 2^W)`, and every limb is the corresponding base-`2^16` digit.
4. **The coefficients.** Each coefficient is a true integer below `2^33`, so the four coefficient vectors represent the integers `B·2^32`, `u·Z`, `u·Z + ψ` and `(B + S)·2^32` exactly.
5. **The comparisons.** By Lemma 4, with RFC-0050's boxes, the two comparisons hold if and only if `C(ŷ − 1)·2^32 ≤ u·Z` and `u·Z + 1 ≤ C(ŷ)·2^32`. Then `C(ŷ − 1) < C(ŷ)`, so `e_ŷ > 0`, and `Z ≥ C(ŷ) > 0`.
6. **The token.** The prefix sums are non-decreasing. So `C(v)·2^32 ≤ C(ŷ − 1)·2^32 ≤ u·Z` for every `v < ŷ`, and `C(ŷ)·2^32 > u·Z`: `ŷ` is the least such `v`. Conversely, that least `v` satisfies both comparisons.
7. **Uniqueness.** `ĥ` is fixed by `ŷ`, the sums by `ĥ` and `e`, the limbs by Lemma 3, and the comparison arrays by Lemma 4. On a padded row, step 1 gives `ĥ = 0` and so `B = S = 0`; the padding zero-check gives `Z = 0`; and the argument of 'Padded rows' fixes the rest.

The comparisons cannot be satisfied when `Z = 0`, which a real row's vocabulary certificate excludes (`e = 2^σ` at the maximal logit), and never select a token of weight zero. The reference sampler refuses the first (`NoMass`) and never returns the second.

**Soundness accounting.** The three contractions, the suffix's dot product, bounded mode's three OneHot relations and the row zero-checks, the padding zero-check among them, are stage instances like any other. RFC-0014's calculator counts them from the derived schedule once #152 places them, and the lookups join the LogUp error term with their count.

**Zero knowledge.** Every value this RFC commits is advice, masked as RFC-0013 masks advice. Nothing new is revealed.

## Alternatives Considered

1. **Byte radix, as in `RsqrtRow`.** That gives `m = ⌈(W + 32)/8⌉`, 11 at `W = 49`, so 132 comparison columns where radix `2^16` uses 72. Radix `2^16` keeps every product below `2^32`, and `RsqrtRow`'s triple products are what needed bytes there. Rejected on cost.
2. **A decomposition of `C_≤`.** That costs `k − 1` more columns and `k` requests per row. `S` fits as one coefficient instead. Rejected.
3. **The draw's limbs from the PRF circuit's output bytes**, `u_0 = b_0 + 2^8·b_1`. That saves one column and two requests per row, but it ties this gadget's layout to the AES circuit's. Left to #152 (OQ-0058-1).
4. **Radix `2^20`.** A limb product reaches `2^40`, beyond RFC-0050's input bound. Rejected.

## Drawbacks

- **Per-row cost.** 79 committed values and 82 range requests per output token at `σ = 32` and a vocabulary of `2^16` or more, against RFC-0009's estimate of about 20 and 30. The vocabulary certificate's `V` lookups, commitments and product leaves, and in bounded mode the `V'` indicator entries, still dominate.

## Migration / Rollout

- **#144 (this RFC).** `tessera_gadgets::InverseCdf`:
  - the parameters, the honest sampler and the witness;
  - the row constraints, the padding zero-check included;
  - the three contractions and bounded mode's OneHot relations, as sumcheck relations of one row.
- **#152.** The vocabulary certificate, the PRF and the wiring: the compiler's advice roles and requests for `InverseCdf`, from this RFC's order, and the gadget's place in the schedule.
- **#160.** The verifier re-derives `W`, `k`, the columns and the requests from `V`, `σ` and the mode.

## Testing Strategy

- **Agreement.** On 10^5 random distributions of true integers in `[0, 2^σ]`, the gadget's honest token equals `tessera_scoring::inverse_cdf`'s, and every equality and range request passes.
- **Neighbours rejected.** For `ŷ + 1` the lower comparison fails, and for `ŷ − 1` the upper comparison fails, each at its final borrow. The other obligations are computed honestly from the wrong token.
- **Each value pinned.** Every committed value placed outside its table fails its own range request, and one moved by one inside it fails an equality.
- **Bounded mode's indicator.** Two assignments pass every row obligation and are refused only by OneHot's relations:
  - a non-Boolean `ĥ = (0, 1, 1, −1)` with count 1, refused by booleanity;
  - an indicator on the vocabulary's padding, refused by the vocabulary relation.
- **Padded rows.** A padded row with `u = 0` and `Z ≠ 0` fails the padding zero-check alone.
- **The edges.**
  - A token of weight zero.
  - A draw of `2^32` or more, which the draw's top limb rejects.
  - A partition `Z = 0` on a real row, which the upper comparison rejects.
  - A padded row with an all-zero witness.
  - The largest partition, `V = 2^18` and `σ = 32`.
  - The test field `F65537`, refused for these boxes.
- **Shared vectors.** The rows of `spec/vectors/scoring/inverse_cdf.txt` whose weights fit `[0, 2^σ]` give the same token through the gadget.
- **Load-bearing relations.** For each relation that a row's obligations cannot replace, an assignment that passes everything but it:
  - bounded mode's count, against a two-hot indicator and a padded row's one;
  - the suffix's dot product, against a non-suffix `L′` whose prefix the row accepts.
- **Bounds and widths.** The whole request sequence, the comparisons' carry widths included, matches the widths RFC-0050 derives from this RFC's bounds, and a row at its coefficient bounds has its witness.
- **Counts and order.** The committed values and range requests match this RFC's formulas at every `W` from 1 to 51, and their order matches a list written out from this RFC.

## Open Questions

- **OQ-0058-1.** Should the draw's limbs come from the PRF circuit's output bytes? Owner: @AbdelStark. Resolve in #152, by amending 'The decompositions' if it saves the column.

## References

- [RFC-0009](RFC-0009-gadgets.md) Lemmas 0, 3, 4, 8 and 10; 'Argmax'; 'InverseCdf'.
- [RFC-0050](RFC-0050-gadget-advice-columns.md) 'Complete comparison schema', 'Padding and binding obligations'.
- [RFC-0017](RFC-0017-scoring.md) 'Sampler PRF', 'Predictors'.
- [RFC-0035](RFC-0035-padded-rows.md) 'The relations', 'The padding zero-check'.
- [RFC-0008](RFC-0008-logup-and-tables.md) 'Table kinds'.
