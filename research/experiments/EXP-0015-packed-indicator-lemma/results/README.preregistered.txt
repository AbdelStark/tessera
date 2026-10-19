# EXP-0015: Finite-domain uniqueness of the packed-indicator max certificate

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0029](../../ideas/IDEA-0029-packed-indicator-max-certificate.md) |
| Evidence level | E0: exact integer arithmetic, exhaustive finite domains and analytic interval obligations; elapsed CPU time is execution metadata, not an E1 performance result |
| Hardware | CPU only; environment recorded with the run |
| Pre-registered on | 2026-10-19 |

## Hypothesis

For the finite domains below, the packed marker `f(k) = B E(k) + [k = 0]` gives exactly one passing row certificate and exactly one passing output decontamination witness, equal to an independent integer oracle. The Goldilocks parameter bounds imply the integer decompositions used in the proof sketch cannot wrap. The experiment explicitly tests the current specification's padding and signed endpoints, rather than treating F65537 as proof about Goldilocks.

## Success criterion

Written before harness code. Let `bad` count accepted witnesses differing from the integer oracle, `missing` count admissible inputs with no honest passing witness, and `multi` count inputs with more than one passing witness. A witness includes range-gadget limbs, not only its reconstructed scalar.

- `validated` for this **registered lemma-check unit**: `bad = missing = multi = 0` on every exhaustive domain; all registered Goldilocks honest boundary cases pass uniquely; every stated Goldilocks interval/non-wrap inequality passes exactly; all five negative-control classes below expose their specified failure and the primary checker rejects the same inadmissible witness at the named check; the original missing-output-range attack is reproduced. Any missing domain or incomplete run prevents validation.
- `refuted`: at least one `bad`, `missing` or `multi` occurs **inside the primary candidate's stated domains or full-size parameter bounds**, with the exact input, field residues, integer interpretation and check trace saved. Expected failures in deliberately weakened controls do not refute the repaired candidate.
- `inconclusive`: no primary counterexample, but any required domain, negative control, exact bound or independent cross-check is missing, unsupported or fails to run. Finite exhaustion alone does not establish the complete scheme's SP-3 claim.

This experiment does not validate Fiat–Shamir soundness, sumcheck reductions, table compression, zero-knowledge masking, a protocol implementation, GPU performance or a cost-model gain. A positive unit verdict will be recorded as such, with the idea's wider proof and integration obligations kept explicit.

Nothing in the sections above is edited after preregistration; amendments are dated and appended.

## Method

### The theorem actually checked

For a public valid-position set `V` and row selector `ψ = [V is nonempty]`, let each looked-up key be an integer and each table value satisfy `0 <= E(k) <= E_max`, `E(0) = E_max`, and `E(K_0) = 0` for masked positions. The candidate looks up masked positions at `K_0` as [RFC-0030](../../../docs/rfcs/RFC-0030-masked-exponentials.md#relation-1-every-position-looked-up) requires, so their packed values are also zero. Valid keys are the field differences `c - x_i`; it is not assumed that malicious `c` is already a true integer.

Define `F = sum f_i`, `S = sum E_i`, `Z = sum [k_i = 0]`. The certificate checks `n - ψ` in its range and `s = (F - n)/B` with `NonNeg`. On real rows `n >= 1`; on padded rows the adapted check permits `n = 0` and the existing [RFC-0035](../../../docs/rfcs/RFC-0035-padded-rows.md#the-padding-zero-check) rule pins the maximum to zero. This selector adaptation is declared now; the literal historical `n >= 1` variant is an expected completeness-negative control, not silently fixed after testing.

If `F` and `B s + n` are integers in a common interval of width less than the field prime, congruence implies integer equality. With `0 <= n, Z < B`, Euclidean decomposition yields `n = Z` and `s = S`. On a real row `n >= 1` supplies a zero key, which first pins `c` to an actual score. Only then does the score-spread bound identify every other key as the nonnegative integer `c - x_i`, proving maximality. This order avoids assuming an untrusted maximum's range.

For each output coordinate and chunk, define `O_prime = sum f_i v_i`, `O = sum E_i v_i`, `T = sum [k_i = 0] v_i`. The candidate checks the centered contamination `t` and `o = (O_prime - t)/B`, including `NonNeg(o + offset, bits)`. Non-wrap and `-B/2 <= t < B/2`, with the honest `T` strictly inside that interval, give the unique centered decomposition `t = T`, `o = O`. Chunk outputs sum as integers. These are algebraic premises to justify separately from finite exhaustion, not conclusions supplied by a small field run.

### Exhaustive domains (F65537)

The scaled primary parameters are `p = 65537`, `B = 64`, `sigma = 2`, `E_max = 4`, `n - ψ in [0, 4)`, `s in [0, 16)`, `t in [-32, 32)`, `o in [-32, 32)`. The toy deterministic table has keys `0..31`, `E(k) = max(4 - k, 0)`, and masked key `K_0 = 31`. It is an abstract bounded table, explicitly **not** a claim that the production exponential is linear. The algebraic lemma uses its stated range/marker properties; actual exponential endpoints receive a separate full-size check.

1. **Integrated real rows.** Every ordered score row of length 1, 2 or 3 with scores in `[-8, 8]` (5,219 rows), every value vector in `[-2, 1]^length`, and every admissible `n` and contamination `t`. Candidate maxima cover **every field residue that could pass even the first valid key lookup**, obtained by `c = x_first + k (mod p)` for all 32 keys; all other residues necessarily fail that lookup. Thus this pruning is exhaustive, not a sampled maximum range. Test every range witness through independently enumerated limb assignments.
2. **Masks and padded rows.** Three slots, every score vector in `[-2, 2]^3`, every value vector in `[-2, 1]^3`, all eight public valid-position masks (including empty), the same exhaustive maximum/witness domains and a fixed zero maximum on an empty row. This includes causal prefixes and stronger arbitrary public masks. Masked scores/values may differ from valid ones; the masked lookup must still pin `f = 0`.
3. **Abstract decomposition kernels.** For lengths 1–3, enumerate every `(E_i, z_i)` from `{(0,0), (1,0), (2,0), (3,0), (4,0), (4,1)}` and every value vector in `[-2,1]^length`. Check zero-count and output uniqueness directly, independently of a particular table or score ordering. Zero-marker inputs are a rejection class on real rows and the honest zero case on padded rows.
4. **NonNeg digits.** Toy limb width 2. Enumerate every admissible limb tuple for widths 4 and 6 and all reconstructed field residues; compare with the field-form gadget's virtual-high-limb rule. The witness finder must retain multiplicity, so two digit tuples cannot collapse into one scalar and hide a second witness.

The checker uses modular field identities and range-gadget predicates. The independent oracle uses Python integers, `max`, direct sums and integer digits; it does not call the modular witness finder. Enumerating contamination candidates once per distinct `(O_prime, expected O, expected T)` is permitted memoization, with the exhaustive input and witness counts retained. No random sample substitutes for these domains.

### Goldilocks obligations and boundary cases

Use the exact prime `P = 2^64 - 2^32 + 1`, `B = 2^23`, `E_max = 2^16`, row length at most `2^17`, chunk length at most `2^14`, `s in [0, 2^34)`, `n in [1, 2^17]` on real rows, `t in [-2^22, 2^22)`, `o in [-2^37, 2^37)`, and `v in [-128,127]`. Current scores obey `-2^21 < x <= 2^21`: the positive endpoint is attained by INT8 QK and must not be dropped.

- Check exact bounds for `F`, `B s + n`, `O_prime`, `B o + t`, the honest `S`, `T` and `O`, and their possible differences against `P`. In particular the negative honest endpoint `O = -2^37` is included, whereas the positive endpoint is strictly below `2^37`. Check `2^17 < B` and `|T| <= 2^21 < B/2`.
- Check all row-length endpoints `{1, 2, 2^14, 2^14+1, 2^17}`, chunk splits, extrema and ties; signed values `{-128,-1,0,1,127}`; score sets using `-2^21+1`, `0`, `2^21`; zero tails, masked slots and empty rows. Repeated runs are represented by exact multiplicities, so large boundary rows need not allocate per-score arrays.
- Evaluate the normative integer-only `Exp` oracle at its zero key, last key and selected boundary/core keys at `sigma = 16` and an explicit admitted rational scale; record parameters and the table digest/values used. This is a boundary check of the lemma's actual table premises, not an exhaustive table-generator audit.
- NonNeg's default limb width is 16 (RFC-0009 and `GadgetPolicy::default`). Widths 34 and 38 therefore use three digits, two committed limbs and three range lookups each. The old 258-commitment/387-lookup cost counted limb width 21; with the default the additions are 387 commitments and 516 lookups per row at one output chunk. Record this discrepancy without asserting a new measured cost gain.

### Negative controls and exact gates

1. **No output NonNeg.** Reproduce the historical fractional witness: one maximum with `E = 2^16`, `v = 0`, true `O_prime = O = t = 0`, fake `t = -2^22`, fake `o = 1/2` in Goldilocks. An even reciprocal `R = 2^16` turns `R o` into 32,768 and a shift-16 requant into 1 rather than 0. The primary candidate rejects at `PV_NONNEG`; record the corresponding toy control too.
2. **Missing positive zero count / oversized low part.** A maximum strictly above all scores passes when zero count is allowed to be 0; the primary real-row candidate rejects at `COUNT_RANGE`. Allowing `n` to grow by `B` permits `n_prime = n+B`, `s_prime = s-1`; the primary candidate rejects at the same range check. These are two cases in one control class.
3. **No chunking.** A full-size all-tie row with more than `2^14` values `-128` lies outside the proposed per-chunk output window. The literal unchunked version has no honest witness (`PV_NONNEG`); correctly split chunks pass. This tests the stated precondition, not a refutation inside it.
4. **Unsafe field-wrap parameters.** Change `m` to 34 only in a control: for `O_prime = 0`, `o = 2^30`, `t = -2^32+1` satisfy `2^34 o+t = P`, while both output/contamination range checks pass. The primary shift-23 equation rejects at `PV_IDENTITY`. Also record a row-sum alias obtained by widening `s` beyond its registered bound; the primary candidate rejects at `SUM_NONNEG`. This makes the Goldilocks wrap obligation executable rather than relying on F65537.
5. **Unadapted padding.** The literal historical `n >= 1` version rejects an empty padded row. The preregistered selector-adapted candidate uniquely accepts its zero semantic witness, with the fixed default NonNeg limb digits for the shifted output range. A nonzero padded maximum fails at `PADDING_MAX`.

The error tags above are research-checker predicates, **not** new production `RejectReason` variants. Counterexamples to a weakened local decomposition are not described as full false proofs unless all downstream relations are actually checked.

## Reproduction

After committing this preregistration, from this directory:

```sh
python3 -I check.py --self-test
python3 -I check.py --output results/run.json > results/run.txt
```

The program will have only standard-library dependencies and SPDX headers. Raw results retain exact domain sizes, checked witness counts, invariant checks, all negative controls, any primary counterexample and deterministic SHA-256 digests of canonical results. Record Python/OS/CPU/date/git commit and hashes of the frozen preregistration and harness. No Rust workspace, production code, verifier code, git writes or benchmark is needed.

## Results

Pending the committed preregistration and complete run.

## Verdict

Pending. A finite-domain unit verdict will remain separate from a proof of the complete modified protocol. Any remaining Goldilocks, padding, table, masking or proof-format obligations remain explicit in the idea note.

## Follow-ups

Only after the security unit: a superseding tier-A RFC must specify packed tables, selector-adapted row relations, masked zero tuples, chunking, transcript/claim changes, independent-verifier checks, protocol2 masks and the corrected limb counts. An H100 run could measure packed lookup and per-row range-gadget costs against the current split-eq forest; it cannot establish witness uniqueness.
