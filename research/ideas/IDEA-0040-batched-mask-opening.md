# IDEA-0040: One masked batching sumcheck opens every mask claim at one point

| | |
|---|---|
| Status | `validated` ([EXP-0027](../experiments/EXP-0027-batched-mask-opening/README.md)) |
| Origin | [#489](https://github.com/AbdelStark/tessera/issues/489) and its design-notes comment (2026-10-25, point 3) [TesseraMask26], from [`docs/zk/simulator.md`](../../docs/zk/simulator.md) §4.2 and R11; [SUR-0008](../surveys/SUR-0008-zk.md) (Libra masks [XZZPS19], [Xie24]; the "fold a random instance first" shape of [VegaDocs26], [KS25]) |
| Lever | `size` (combined rows, one per opening group), `verify` (`(G + 1)·τ_A` inner products, `G + 1` coset NTTs), zero knowledge (closes R11) |
| Spec surfaces | `tessera-protocol/2` only, not yet frozen: RFC-0013 'Masked sumcheck' step 4, 'Cost', 'Simulator argument', INV-ZK-2; RFC-0011 'Opening protocol' step 1 (one more group); RFC-0032 'Masked extensions that spare the pair bit' and 'Labels' (the mask claims `R̃_t(r'_t)` stop being opened alone); RFC-0068 step 6 (`G(ρ')`); RFC-0014 'Transcript schedule' (new rows) and 'Soundness calculator' (new terms); `docs/zk/simulator.md` R11, Lemma 5. Tier A of `tessera-protocol/2` (02 §1), which no release has frozen, so an amending RFC suffices, as RFC-0065 and RFC-0068 did |
| Priority | Correctness prerequisite of `tessera-protocol/2` (R11): without an opening scheme the protocol is not sound. Size gain over per-claim groups below, E0 |

## Mechanism

**The problem.** Two kinds of claim on the advice have no opening the budget affords (#489):
- each masked sumcheck's `G_k(r_k) = a_0 + Σ_j g_j(r_{k,j})`, whose column functional `(r_{k,j}^m)` is not an `eq` vector (RFC-0011 opens `eq` functionals only);
- each tree level's mask claim `R̃_t(r'_t)` (RFC-0032), an `eq` functional at its own level's final point, so no two of them share a column sub-point.

One opening group each costs a blinding pair, a blinding claim and a combined row of `k = 2^22` `Fp2` values, 64 MiB.

**The candidate.**

1. **The mask block `M`.** A logical cube of `m = log2 k + ⌈log2 R_M⌉` variables, the column variables lowest. Its `R_M` `Fp2` rows hold every masking polynomial's coefficients, in 8-slot blocks per round, and every tree-level mask row `R_t`, packed in aligned sub-blocks (OQ-0013-1 admits tree levels' masks). Every other position of the cube is a public zero.
2. **The pad `S`.** In Round 1 the prover also commits `S`, a fresh `Fp2` block from the per-proof PRG (INV-ZK-3), laid out exactly like `M`: uniform at every position where `M`'s layout holds a mask value, zero elsewhere. The layout is public and fixed before any challenge, so `S`'s support does not depend on `C`.
3. **Batching.** After the last masked sumcheck, every mask claim `v_i` is fixed. The verifier squeezes `α`. Each claim is a linear functional `⟨c_i, M⟩`:
   - for `G_k(r_k)`, `c_i` is `r_{k,j}^m` at slot `(k, j, m)` and `1` at `a_0`'s slot;
   - for `R̃_t(p_t)`, it is `eq(p_t, ·)` on `R_t`'s sub-block.

   With `C = Σ_i α^i c_i`, public and sparse, the claims combine to `v = Σ_i α^i v_i = ⟨C, M⟩`.
4. **The masked batching sumcheck.** The prover sends `Γ_S = ⟨C, S⟩`, and the verifier squeezes `ρ`. A sumcheck of degree 2 over the `m` variables, the lowest first, proves `Σ_x C̃(x)·z̃(x) = v + ρ·Γ_S` with `z = M + ρ·S`.
5. **Its end.**
   - The prover sends `y = z̃(r*)`.
   - The verifier computes `C̃(r*)` from public data in `O(#coefficients + #rows·m)` and checks the last round against `C̃(r*)·y`.
   - `y` joins the advice pool as one claim at column sub-point `r*_{1..22}`: `M`'s rows with row coefficient `eq(r*_row, ·)`, `S`'s rows with `ρ·eq(r*_row, ·)`. That is one more RFC-0011 group.
6. **What is never sent.** `M̃(r*)` and `S̃(r*)` are never sent apart, and no individual `G_k(r_k)` or `R̃_t(r'_t)` is opened.

**Transcript order.** Last masked sumcheck, `α`, `Γ_S`, `ρ`, then the rounds and their challenges, then `y`, then `pcs/advice/blinding`. Proposed labels: `zk/batch/alpha`, `zk/batch/zk-sum`, `zk/batch/zk-rho`, `zk/batch/round/{j}`, `zk/batch/r/{j}`, `zk/batch/evals`.

## Why it could work

- **Lemma B (zero knowledge), statement and proof sketch, E0.**
  - **Setup.** Let `T` be the positions where `M` is not a public constant. Let `S` be uniform on `F^T`, zero off `T`, and independent of everything sent before `Γ_S`.
  - **Claim.** Given the earlier view and `ρ ≠ 0`, the batching transcript depends on the secrets only through `v`, which the earlier view already fixes. A simulator exists for every `C` and every challenge strategy.
  - **Proof.**
    - Given `Γ_S`, `S_T` is uniform on the hyperplane `⟨C_T, ·⟩ = Γ_S`.
    - So `z_T = M_T + ρ·S_T` is uniform on `{⟨C_T, z_T⟩ = v − ⟨C_{T^c}, M_{T^c}⟩ + ρ·Γ_S}`, a set fixed by public values.
    - `z` off `T` is `M`'s public part.
    - Every later message is a function of `z`, `C` and the challenges.
    - So the simulator samples `Γ_S` uniformly, then `z_T` on that hyperplane, then runs the honest prover.
  - **What it does not need.** No rank condition on `C`: any dependence among the functionals the rounds reveal is shared by `M`'s side, so it is a public relation.
  - **Its error.** `ρ = 0`, `1/q`.
- **What the issue's open question becomes.** The issue asks whether `S ↦ (Γ_S, every round's non-constant coefficients, the final value)` has full rank. The answer is that the rank need not be full. What `S` needs is coverage: `S`'s support must contain every non-public position of `M`'s cube.
  - **Restricting `S` to `C`'s support.** That suffices only when `C`'s support contains every non-public position of the cube.
  - **Sub-cube supports.** A `C` supported on a sub-cube `{x_j = b}` loses one direction of the M-side map in round `j`. Its restriction loses two. A position `M` holds off that support would then be revealed unmasked.
  - **Measurement.** EXP-0027 measures these losses.
- **The shape has precedent.** A sumcheck on `M + ρ·S` with `Γ_S` sent first is a fold of the real instance with a random one used as a one-time pad, the shape of BlindFold and Vega ([VegaDocs26], [KS25], [a16z26zk]; `secondary`). Here it uses hash-committed rows, so it stays post-quantum.
- **Libra's mask would recurse.** Libra's `a_0 + Σ_j g_j(x_j)` ([Xie24] §2.4.1) is size-optimal, but its final `G(r*)` is again a claim of one point per block (#489, note 1). `C̃·S̃` ends at the same point `r*` as `M̃`, so `y` is one ordinary claim.

## Expected gain

No cost-model knob prices opening groups. The scenario is a count, from the compiler's schedule in EXP-0027 (E0).

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 16K tokens, 405B (the only context counted) | `S`'s committed elements, about the tree-level mask rows' size: under 0.5% of committed elements by #489's estimate, E0 | `G_mask × 64 MiB` → `64 MiB` plus `S`'s opened symbols. #489 projects `G_mask` near 200, about 12.5 GiB; EXP-0027 recounts it | `G_mask·τ_A` inner products and `G_mask` coset NTTs → 1 each; one sumcheck of `m` rounds |

Every input is E0: the issue's projection, until EXP-0027 counts it from the compiler (`crates/tessera-compiler`) at 405B.

### EXP-0027's count, 2026-10-25

E0, from the compiler's 405B report at 16K tokens and 16K context, `servable`, two classes:

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 16K tokens, 405B | `S` is 1,558 `Fp` rows packed, 0.163% of the advice's 953,580 non-zero rows; 6.5·10^9 committed elements. The batching prover does at most 2.6·10^10 `Fp2` multiplications | 2,667 mask claims, not about 200. Per claim: 2,667 groups, 166.7 GiB. Per column point: 612 groups, 38.25 GiB. The candidate: one group, 65.85 MiB in all (`live-64`). That is −99.83% of the mask claims' opening bytes against the per-point baseline | 612 groups' `τ_A` inner products and coset NTTs → one group's; one sumcheck of 32 rounds of degree 2; `C̃(r*)` in about 8.8·10^4 terms |

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Adds at most `(N_claims − 1)/q` over `α`, `2m/q` over the rounds and `1/q` over `ρ`. `α` comes after every `v_i`, and `ρ` after `Γ_S`. With about 2,500 claims and `m ≈ 33` at 405B, about `2^-116` at `q = p^2` (E0) | pass (E0 bound; EXP-0027 (c) measures it over F65537) |
| Uniqueness upstream of the logits (SP-3) | none: masks only | pass |
| Zero knowledge (G-5, RFC-0013) | Lemma B for the batching view. Conditions: `S` covers `M`'s non-public positions and is uniform over `Fp2`; `ρ ≠ 0`; `M̃(r*)` never sent alone. Lemma 5's proximity row keeps its free rows, now with `S`'s rows added (argued, not tested) | unknown → EXP-0027 (b) |
| Verifier independence (SP-6) | none | pass |
| Post-quantum | hash-based rows only | pass |

## Prior rejection check

- **RFC-0013.**
  - Alternative 2 (commit every round message) is a different mechanism.
  - Alternatives 1, 3, 4 and 5 do not concern openings.
- **RFC-0032.** Alternative 6 (commit the top levels apart) is different.
- **RFC-0011 and baseline §7.** Neither lists a batching of claims at distinct points.
- **`docs/zk/simulator.md` §4.2.** It names this route as "the alternative: a batching sumcheck that reduces these claims to shared column points". It records that such a sumcheck "needs its own masking polynomial (Lemma 4) or a rank argument of its own". This idea supplies the mask `S` and Lemma B in place of a rank argument.
- **The issue comment.**
  - Its point 2 rejects transposed layouts, which cost a free column in about `4·10^4` rows, or `w` groups.
  - Its point 1, the `eq` form of a univariate's evaluation, leaves one point per block.
  - Neither covers this idea.

None of these rejects the idea.

## Kill criterion

Any one of these kills it:
- **Zero knowledge.** Under the candidate layout, two witnesses whose earlier views are identically distributed give distinguishable batching views on any registered configuration.
- **Soundness.** A cheating prover beats the analytic bound.
- **Size.** At 405B, `S`'s cost in opened symbols exceeds the combined rows it saves.

## Cheapest experiment

[EXP-0027](../experiments/EXP-0027-batched-mask-opening/README.md), E0, effort `s`:
- exact rank measurements of the revealed functionals over F65537 and Goldilocks;
- an exact distinguisher over F65537, on an executable toy protocol with masked stages, tree-level-like mask rows and the batching;
- a Monte Carlo soundness measurement over F65537;
- the 405B count from the compiler.

Nothing here needs an H100. The batching prover's cost is an operation count.

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| Soundness | survives | **As ADV-P.**<br>- `S` is committed in Round 1 and `M`'s coefficients before each stage's `ρ_k`.<br>- Every `v_i` is fixed before `α`, and `Γ_S` before `ρ`.<br>- A false `v_i` makes `E(α) = Σ α^i(v_i − ⟨c_i, M⟩)` a non-zero polynomial of degree `N − 1`. A false `Γ_S` then cancels it only at one `ρ`.<br>- The rounds are a degree-2 sumcheck, and `y` is bound by the advice opening.<br>- Grinding: in non-interactive mode each challenge costs a hash, so the union bound over grinding attempts applies as for every stage (RFC-0014).<br>- The layout's public zeros are harmless: `C` vanishes there, so values a prover puts there change neither `v` nor the sumcheck.<br><br>**Order matters.** A `Γ_S` sent after `ρ`, or an `α` drawn before the claims, would let the prover cancel any error. EXP-0027's negative controls check both. |
| Prior art | survives | The fold-with-a-random-instance shape is BlindFold's and Vega's ([VegaDocs26], [KS25]), there with homomorphic commitments. Libra's mask ([XZZPS19]) is cheaper but recurses on its own opening. No source batches mask claims at distinct points under a hash-based commitment; SUR-0008 found none. |
| Cost model | survives | **The cost.**<br>- `S` doubles the mask rows `M` occupies: by #489 a small fraction of RFC-0013's 0.6% of advice.<br>- One group of 64 MiB, and one sumcheck of about 30 rounds of degree 2: about 1 KiB of messages and prover multiplications linear in `M`'s size.<br><br>**The comparison.** About 200 groups by #489's count, 12.5 GiB, against a 2 GiB proof target (SC-3). The gain exceeds every uncertainty of the inputs by orders of magnitude. |
| Feasibility | survives | **Within `tessera-protocol/2`.**<br>- New labels and one group, within a protocol version no release has frozen. No new primitive.<br>- The prover regenerates `M` and `S` from the PRG and evaluates `C` from the claim points.<br>- The verifier's `C̃(r*)` is `O(#coefficients + #rows·m)`.<br><br>**Its risk.** The simulator argument must take Lemma B, and Lemma 5 must cover `S`'s rows: a written argument, as RISK-6 requires. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| Closes R11 (correctness prerequisite). Size: about 12.5 GiB of combined rows → 64 MiB (#489's projection, E0; EXP-0027 recounts) | 0.6: mechanism and lemma written here, numbers projected; an independent reading of the lemma is still owed | `m`: one RFC amending RFC-0013/0011/0032/0014 within `tessera-protocol/2`, no hardware | correctness prerequisite, ahead of the performance queue, as IDEA-0036 to IDEA-0038 were |
| Re-scored 2026-10-25 after EXP-0027: closes R11. Proof bytes of the mask claims' openings −99.83%, 38.25 GiB → 65.85 MiB (E0, compiler counts) | 0.8: E0 validated by EXP-0027; an independent reading found no break, two specification gaps for the RFC (`S`'s dependence on `M` after the batching; how `G_k(r_k)` is carried) and two missing soundness terms (`1/q` for `y`'s group, `S`'s rows in the proximity term) | `m`, unchanged | correctness prerequisite; the queue's first item |

## Result, 2026-10-25

[EXP-0027](../experiments/EXP-0027-batched-mask-opening/README.md), `done: validated` (E0).

**Zero knowledge.**
- **Lemma B holds exactly** on a toy protocol. The candidate's view never separated two witnesses in 300 draws.
- **Every exposure was shown.** The bare batching, `M̃(r*)` alone, and `S` laid out by `C`'s support each leaked on every draw of the configuration built to expose them.
- **The bare batching leaks on a protocol-like layout too.** It goes through a fully pinned root-level row. So `S` is required, not optional.

**The rank question.** The revealed functionals reach `2m`, the most the sumcheck's shape allows, for protocol-like `C` on every draw over F65537 and Goldilocks.
- **Where they lose rank.** Sub-cube supports lose `j` on `M`'s side and `2j` on `S`'s when `S` is restricted to `C`'s support. A single `eq` claim gives `m + 1`.
- **What decides zero knowledge.** Coverage of `M`'s layout, not full rank.

**Soundness.** The composite cheater is accepted at its exact predicted rate, under `(N + 2m + 1)/q`. Both order controls are accepted every time.

**At 405B.** 2,667 mask claims. One group of 64 MiB replaces 612 groups (38.25 GiB) or 2,667 (166.7 GiB), with an `S` of 0.163% of the advice rows and 32 rounds.

**The independent reading.** It found no break. Its two gaps and two missing soundness terms are in EXP-0027's report: they are what the RFC must settle.

## Decision and links

Experiments: [EXP-0027](../experiments/EXP-0027-batched-mask-opening/README.md) (`done: validated`). Issue: [#489](https://github.com/AbdelStark/tessera/issues/489), whose RFC is the next step. Ledger line updated on: 2026-10-25.
