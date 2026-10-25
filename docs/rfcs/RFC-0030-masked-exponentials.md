# RFC-0030: `SoftmaxRow` Pins Its Masked Exponentials to Zero

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.3
- Amended by: [RFC-0034](RFC-0034-causal-blocks.md) ('Cost': who pays for the masked half), [RFC-0071](RFC-0071-request-slots.md) ('The causal selector' and the root claim: per request, in a batch)
- Supersedes, on acceptance:
  - [RFC-0009](RFC-0009-gadgets.md#softmaxrow) 'SoftmaxRow': relation 1, the reduction of the trees' leaf claim, and Lemma 5;
  - [RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions) 'Multilinear extensions': the causal selector; 'Tree levels': `SoftmaxRow`'s public root values;
  - [RFC-0006](RFC-0006-compiler-ir.md#range-analysis) 'Range analysis': the width of `SoftmaxRow`'s `Exp` table, and the ground of the softmax fact;
  - [RFC-0007](RFC-0007-claim-list.md#verifier-validation) 'Verifier validation', step 5: `SoftmaxRow`'s table.
- Origin: [#279](https://github.com/AbdelStark/tessera/issues/279)

## Summary

`SoftmaxRow` commits its exponentials `e` at the scores' full shape, and RFC-0006 says `e` is 0 past the causal mask. Nothing proved it. The lookups covered the valid positions only, while `PV = MatMulAct(e, v)` sums every key position. A prover could therefore add `Σ_{j > q} e_qj·v_j` to any attention output and pass every check.

This RFC looks up every position of the score tensor:

- a valid position's key is `c_q − x_qi`, as before;
- a masked position's key is the `Exp` table's last key, which the table maps to 0.

Every exponential then has exactly one passing value: `Exp[c_q − x_qi]` at a valid position, 0 at a masked one. The row sum, `PV` and range analysis's fact read `e` exactly as they are written. The RFC also corrects the reduction of the trees' leaf claim, which RFC-0009 wrote as a product of extensions.

## Motivation

`SoftmaxRow`'s checks on `e` cover the valid positions `i ∈ V_q` only ([RFC-0009](RFC-0009-gadgets.md#softmaxrow)):

- the lookups `(k_qi, e_qi) ∈ Exp` pin `e` there;
- the max certificate's trees read the keys, with the constant 1 at masked positions;
- the bracket reads `s_q = Σ_{i ∈ V_q} e_qi`.

The attention product does not stop there. RFC-0006 lowers `PV` to `MatMulAct(e, v)`, whose relation `Z̃(r_a, r_c) = Σ_j Ã(r_a, j)·B̃(j, r_c)` runs over every key position `j` ([RFC-0010](RFC-0010-proof-schedule.md#relations)). The masked terms `e_qj`, `j > q`, are committed values that no relation constrains. A cheating prover can commit any values there: every check passes, while `PV_q` gains `Σ_{j > q} e_qj·v_j` and the logits depart from `D`'s. That breaks the uniqueness of every committed value upstream of the logits (SP-3, RFC-0009 'Goals').

The same gap leaves two other statements without ground:

- **The row sum.** RFC-0010 proves sums as `Linear` relations over a whole row. That is `Σ_{i ∈ V_q} e_qi` only if the masked `e` vanish.
- **The lookups' layout.** A LogUp instance's lookup side is a polynomial over the score tensor's positions, and every leaf's tuple must be a row of the table ([RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree)). "Every `i ∈ V_q`" says nothing about the other positions. A masked key `c_q − x_qi` is negative wherever a later score exceeds the row's causal maximum, so the honest prover would not complete.

RFC-0009 also says the stage producing the keys "proves `L̃(r) = C̃(r)·k̃(r) + 1 − C̃(r)`". That holds on the hypercube but not between extensions, since `Σ_x eq(r, x)·C(x)·k(x) ≠ C̃(r)·k̃(r)` in general, and as written it would fail the honest prover.

## Proposed Design

### The causal selector

For a sequence of `n` positions, laid out on `2^m ≥ n` positions per axis ([RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) 'Canonical advice layout'), the causal selector is

```text
C(q, i) = [i ≤ q < n]
```

at every position of one of the model's layers and heads. A score tensor stacked over layers or heads also holds the positions of padded layers and heads, when `n_layers` or `n_heads` is not a power of two; there `C = 0`. A position is **valid** where `C = 1` and **masked** where `C = 0`. Masked positions are therefore:

- the causal mask's, `i > q`;
- every position of a padded row (`q ≥ n`) or column (`i ≥ n > q`);
- every position of a padded layer or head.

At every masked position the honest exponential is 0: committed as 0 where the mask cuts a real row, and public padding elsewhere.

- RFC-0010's selector, `[i ≤ q]` alone, made a padded query row valid, and a padded layer's or head's causal positions too. Their lookups would read `(0, 0)`, which `Exp` refuses: `Exp[0] = 2^σ`.
- `C̃(r_q, r_i)` has an `O(m)` closed form, which the prover and the verifier each evaluate. It is a recursion over the bits, most significant first, that carries two comparisons: `q` against `n`, and `i` against `q`.
- Over a tensor stacked over layers and heads, the extension is `LT(r_ℓ, n_layers)·LT(r_h, n_heads)·C̃(r_q, r_i)`; a tensor of one layer drops the first factor.

### Relation 1: every position looked up

For every position `(q, i)` of the padded score tensor, `(κ_qi, e_qi) ∈ Exp { key_bits, σ, scale }`, where

```text
κ_qi = C(q, i)·(c_q − x_qi) + (1 − C(q, i))·K_0,        K_0 = 2^{key_bits} − 1
```

- **The lookup side.** `SoftmaxRow`'s lookups are the positions of its score tensor, in that tensor's index order. The value polynomial is `e` itself, committed advice, so the value claim of RFC-0008 'Fraction tree' lands on `ẽ`.
- **The key polynomial `κ`** is virtual. Its leaf claim `κ̃(ρ)` goes to the stage that produces the keys. That stage proves an `Elementwise` relation of degree 3, `Σ_x eq(ρ, x)·C(x)·k(x) = κ̃(ρ) − K_0·(1 − C̃(ρ))`. The verifier evaluates its public factor `C` at the stage's final point, and the relation ends on `k̃`, where `k = c − x` is `Linear`.
- **The precondition.** `Exp[K_0] = 0`: the table's last key maps to 0.
  - A non-empty zero tail ([RFC-0008](RFC-0008-logup-and-tables.md#fixed-point-generators) 'Fixed-point generators') ensures it, but does not follow from it: values can round to 0 at keys below the tail's first, `z`. The check reads the value, not `z`.
  - **The rule.** `key_bits` is the narrowest width, from the spread's up, whose last key maps to 0 ('Range analysis' below). A value at key `k` does not depend on the table's width, so the rule is well defined.
  - **The bound.** RFC-0008 refuses a zero tail that starts past key `2^24`, so a table the rule widens and RFC-0008 allows has at most 24 bits: values round to 0 just before the zero tail starts, so a table widened to 25 bits has its tail past `2^{24}`. RFC-0007 step 2 checks that the width exists and that RFC-0008 allows its table. A scale that would need more is refused, as its table is.

### Relation 2: the trees

The leaves are unchanged: `L_qi = C(q, i)·k_qi + 1 − C(q, i)`, the constant 1 at every masked position. The leaf claim's reduction is the sum it stands for. The stage that produces the keys proves

```text
Σ_x eq(r, x)·C(x)·k(x) = L̃(r) − 1 + C̃(r)
```

This is an `Elementwise` relation of degree 3, whose public factor `C` the verifier evaluates at the stage's final point. It and relation 1's reduction are two instances of that stage, each at its own point, and each ends on `k̃`.

A padded row's tree has every leaf 1, so its root is 1. The forest's public root values are therefore:

- 0 for the trees of a sequence's rows, layers and heads;
- 1 for every other tree: those of padded rows, and those of padded layers and heads, which pad the forest to a power of two.

The root claim `Σ_y v_y·eq(r, y)` is the extension of a public predicate over the tree index, with an `O(log)` closed form ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels'). For one sequence stacked over layers and heads, it is `1 − LT(r_ℓ, n_layers)·LT(r_h, n_heads)·LT(r_q, n)`. For a batch of several requests ([RFC-0071](RFC-0071-request-slots.md#the-public-selectors)), `LT(r_q, n)` becomes `Σ_{r<R} eq(r_b, r)·LT(r_p, n_r)`, with `r_p` the query's position variables within its slot, low, and `r_b` its request's, high, and the causal selector is `C(τ, i) = [i ≤ τ mod S < n_{⌊τ/S⌋}]`, keys counted within the query's request.

### Lemma 5, extended

Relations 1 and 2 hold if and only if:

- for every row `q < n` of a layer and head, `c_q = max_{i ≤ q} x_qi`;
- `e_qi = Exp[c_q − x_qi]` at every valid position;
- `e_qi = 0` at every masked position.

Proof:

- **A masked position.** Its key is `K_0`, a public constant: `κ` is a fixed combination of public `C`, the constant `K_0` and the virtual `k`, and the reduction above pins its extension. Relation 1 puts `(K_0, e_qi)` in `Exp`. The table's keys are distinct, so `e_qi = Exp[K_0] = 0`.
- **A valid position.** `κ_qi = c_q − x_qi`, and RFC-0009's proof of Lemma 5 applies unchanged: the lookup bounds every key, and the tree puts a zero among them.

A padded row's `c_q` enters no relation of its positions, since `C = 0` multiplies it in `κ` and in the leaves. Nor does its `R_q` reach the output: `PV_q` is 0 there. Whether its bracket is satisfiable is [#322](https://github.com/AbdelStark/tessera/issues/322)'s question.

### What reads `e`

With `e = 0` at every masked position, each consumer reads `e` as written, over every position:

- **The row sum.** `s_q = Σ_i e_qi`, over the padded row, equals `Σ_{i ∈ V_q} e_qi`. It is a `Linear` relation of RFC-0010.
- **The attention product.** `PV = MatMulAct(e, v)` sums every key position, and its masked terms vanish: `PV_q = Σ_{i ≤ q} e_qi·v_i`, `D`'s attention ([RFC-0005](RFC-0005-integer-model.md#integer-semantics)).
- **The softmax fact.** Range analysis's `R_q · Σ_i e_qi ≤ 2^{2σ}` ([RFC-0006](RFC-0006-compiler-ir.md#range-analysis)) holds over every `i`, because the masked terms are 0 by relation 1.

No other relation needs the selector.

### Range analysis and validation

- **RFC-0006.** `SoftmaxRow`'s `Exp` table takes the narrowest `key_bits` from `bits(128 · 255 · d_head)` up whose last key maps to 0. The compiler emits it, and the verifier's lowering derives the same width from public data.
- **RFC-0007 step 5.** It rejects a claim list whose `SoftmaxRow` reads an `Exp` table that maps its last key to anything but 0 (`Reject(StatementMismatch)`). Without that check, a claim list whose table lacked a zero would certify attention outputs with the constant `Exp[K_0]` at every masked position: a function other than `D`.

### Cost

- **Lookups.** The `Exp` instance's lookup side covers the padded square instead of the causal half. Every masked lookup is the same tuple `(K_0, 0)`, so its leaf is the constant `(1, α − K_0)`. The masked entries of `e` were committed before this RFC too, since `e` has the scores' full shape.
- **Who pays for the masked half.** [RFC-0034](RFC-0034-causal-blocks.md) settles it. The advice layout skips the blocks above the diagonal, and the tree and lookup provers evaluate constant subtrees in closed form. What remains is committed: the diagonal blocks' masked halves, `A·b` elements per token, about 0.2 on `μ` at 405B. Paid in full, the masked half would have doubled 405B's attention term: +1.13 on `μ(1K)` and +16.8 on `μ(16K)`.
- **The proof.** One more `Elementwise` instance in the stage that produces the keys: no stage, challenge or commitment is added.
- **Soundness.** The instance adds its sumcheck's error. LogUp's `(N + T)/|Fp2|` counts the `Exp` instance's `N` over the padded square, about `2^{42}` lookups for one 405B request at 16K. The soundness calculator counts them ([RFC-0014](RFC-0014-interaction-and-soundness.md)).

### Unchanged

`Exp`'s definition, generator and digest; the trees' leaves; relation 3, the bracket; `R`'s range; the `MatMulAct` and `Linear` relations; the IR's semantics, in which `e` was already 0 at masked positions (RFC-0006 'Operation set').

## Alternatives Considered

1. **The selector in the consumers**, `PV_qc = Σ_j C(q, j)·e_qj·v_jc` and `s_q = Σ_i C(q, i)·e_qi` (#279, option 1). Rejected, for three reasons:
   - The masked `e` stay free committed values.
   - Both consumers read the virtual product `C ⊙ e`, an `Elementwise` reduction over the whole square. That is one more relation on the attention chain, and a fourth stage for `servable` ([RFC-0010](RFC-0010-proof-schedule.md#stages) 'Stages').
   - The lookups still need a rule at masked positions.
2. **A zero-check**, `Σ_x eq(r, x)·(1 − C(x))·e(x) = 0` (#279, option 3). Rejected: it pins the masked `e` to 0 with one more instance, but leaves the lookups' rule at masked positions to be written. The masked lookup does both with one rule.
3. **Lookup numerators `C` instead of 1**, so that masked leaves are `(0, α − a)`. Rejected:
   - RFC-0008's check `p = 1` would become `p = C̃(ρ)` for this instance;
   - masked keys and values would become free, so alternative 2's zero-check would still be needed.
4. **Lookups at valid positions only, packed.** Rejected: the lookup index would no longer be the score tensor's index. The key polynomial would then not be a function of `k̃` that a sumcheck reduces.
5. **Masked lookups at row 0's tuple `(0, 2^σ)`**, with value polynomial `C ⊙ e + (1 − C)·2^σ`. It needs no zero tail. Rejected: the masked `e` stay free, and the value side needs a reduction as well.

## Drawbacks

- `SoftmaxRow`'s lookup side covers the padded square, and its masked half's cost rests on [#321](https://github.com/AbdelStark/tessera/issues/321).
- `Exp` tables need a zero among their keys.
  - The compiler widens `key_bits` when the spread's width lacks one.
  - No model in the repository needs it: the conformance models' zero tails start between keys 7,193 and 28,312, of `2^19` or `2^20`, and 405B's near key `2 · 10^5`, of `2^22`.
- One more `Elementwise` instance per proof.

## Migration / Rollout

v0.3, before `SoftmaxRow` ([#122](https://github.com/AbdelStark/tessera/issues/122)), the stage schedule (#113, #114) and the verifier's final checks (#124) are implemented. Nothing implements relations 1 and 2 yet. The compiler's width rule lands with this RFC. The witness engine and the oracle already write 0 at masked positions, and no claim list changes.

## Testing Strategy

- **The width rule** (`tessera-compiler`):
  - a score scale whose zero tail starts past the spread's width widens `key_bits` to the narrowest width whose last key maps to 0;
  - a scale whose zero tail starts within it keeps the spread's width, as 405B's shapes do.
- **Conformance.** The oracle's and the engine's exponentials are 0 at every masked position: `traces.json` and the differential tests, as today.
- **Adversarial**, planned for #122 and #124, each rejected:
  - a nonzero exponential at a masked position: `(K_0, e)` is not in `Exp`, `Reject(Lookup)`;
  - a masked position's key replaced by its real key `c_q − x_qi`, with the matching exponential: the key's reduction fails, `Reject(Sumcheck)`;
  - a padded row's leaves other than 1, or its root value 0: `Reject(Sumcheck)`;
  - a claim list whose `SoftmaxRow` table maps its last key to a nonzero value: `Reject(StatementMismatch)` (#107).
- **The selector.** `C̃` equals the brute-force sum of `eq(r_q, q')·eq(r_i, i')` over `i' ≤ q' < n`, at random points, for `n` a power of two and not (#113, #114).

## Open Questions

None. Padded rows' brackets are [#322](https://github.com/AbdelStark/tessera/issues/322)'s, and the masked half's cost [#321](https://github.com/AbdelStark/tessera/issues/321)'s.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0014](RFC-0014-interaction-and-soundness.md).
- [#279](https://github.com/AbdelStark/tessera/issues/279), [#321](https://github.com/AbdelStark/tessera/issues/321), [#322](https://github.com/AbdelStark/tessera/issues/322), [#122](https://github.com/AbdelStark/tessera/issues/122), [#247](https://github.com/AbdelStark/tessera/issues/247).
