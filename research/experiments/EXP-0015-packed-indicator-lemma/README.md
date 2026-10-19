# EXP-0015: Finite-domain uniqueness of the packed-indicator max certificate

| | |
|---|---|
| Status | `done: validated` (registered E0 lemma-check unit only; the complete protocol security gate remains open) |
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

E0, primary. Preregistration commit `774b7c7`, before harness code; frozen SHA-256 `b905bfbeedb2e16def42c76361f4d94d0ffa0426f05eb3a7ce7334d412ed3250` in [results/preregistration.txt](results/preregistration.txt) and [results/README.preregistered.txt](results/README.preregistered.txt). Raw data: [results/run.json](results/run.json) and [results/run.txt](results/run.txt). The first run is retained in [results/run-initial.json](results/run-initial.json) and [results/run-initial.txt](results/run-initial.txt); the final run adds explicit score-spread/table-key interval checks and checks the reciprocal bracket in the already registered negative control. No domain or criterion changed.

| Registered finite domain | Coverage | Bad / missing / multiple |
|---|---|---|
| Real score rows | 5,219 ordered rows; all 319,124 score/value-vector combinations | 0 / 0 / 0 |
| Public masks, including empty padded rows | 1,000 score/mask rows; all 64,000 score/mask/value combinations | 0 / 0 / 0 |
| Abstract bounded exponential/marker kernels | All 14,424 value combinations; 103 positive-marker rows accepted, 155 zero-marker rows rejected as required | 0 / 0 / 0 |
| NonNeg digit boxes | All 65,537 field residues for each width, every committed low-limb tuple, and all 16 / 64 complete integer digit tuples | Exactly one tuple for each in-box residue; none outside |

There are 195,133 maximum candidates after exhaustive first-lookup pruning, 603,940 row witness candidates, and 8,960 contamination candidates over 140 distinct cached packed-output residues. The output cache does not sample: every one of the 397,548 registered input combinations is checked against its independent integer answer. Finite-result SHA-256: `c84a121d73ee905033a012f4428c7d9f04c0d4102d0641b6b4655a12fe4079cb`.

Goldilocks checks cover all 75 all-tie length/score/signed-value endpoint combinations, including lengths immediately above the chunk limit and the attained positive score endpoint, 625 mixed-extrema/zero-tail/masked value vectors, and empty rows. Every honest chunk has exactly one rectangle solution to `B o+t = O_prime (mod P)`; enumerating the bounded wrap multiples is complete because the analytic interval contains every admissible pair. The output values and their actual default-width limb digits are retained in the raw JSON.

The exact interval bounds are:

| Quantity | Bound, E0 |
|---|---|
| Maximum honest packed row sum | `2^17 (2^39+1) = 72,057,594,038,059,008` |
| Maximum candidate `B s+n` | `2^23 (2^34-1)+2^17 = 144,115,188,067,598,336` |
| Maximum absolute honest packed chunk output | `2^60+2^21 = 1,152,921,504,608,944,128` |
| Maximum absolute candidate `B o+t` | At most `2^60+2^22 = 1,152,921,504,611,041,280` |
| Honest sum and output endpoints | `S <= 2^33`; `-2^37 <= O < 2^37`; `abs(T) <= 2^21 < B/2` |

Both nonnegative row sides lie below `P`; the sum of the two absolute output bounds is below `P`, so their differences cannot wrap. After a zero key pins the maximum to a score, a negative score difference has residue greater than every admitted 32-bit table key; the positive spread `2^22-1` fits the selected 22-bit table. The NonNeg boxes are exactly 34 and 38 bits, below both `P` and the new RFC-0009 limb-box limit of 62 bits (#377). Padding and the upper INT8 score endpoint follow current main rather than the older idea's strict inequalities.

The actual integer Exp oracle uses `key_bits=22`, `sigma=16`, rational scale `1/16`. Its last key maps to zero and key zero to 65,536; core/tail boundary values and the base `Exp` TCE specification/digest are in the JSON. This digest is for the existing `Exp` table; a normative packed-table kind/digest is not invented by this experiment. The general table bound is a premise of the algebraic lemma, justified by RFC-0008's integer generator; sampled boundary values are not an exhaustive generator audit.

### Load-bearing negative controls

All five registered classes expose the intended defect or failed precondition. The complete predicate traces are saved, including predicates that pass.

| Control | Exact observation, E0 |
|---|---|
| Drop output NonNeg | Goldilocks `t=-2^22`, `o=1/2` satisfy the packed-output equation at zero. `R=2^16` and the honest reciprocal bracket still pass; `R o=32768` requantizes at shift 16 to 1 instead of 0. The repaired candidate rejects at `PV_NONNEG`. The scaled F65537 attack also changes its requant to 1. |
| Relax the count range | Allowing zero accepts a maximum above all scores; allowing `n+B` admits a second `(n,s)` decomposition. The repaired candidate rejects at `COUNT_RANGE`. |
| Omit chunking | Length 16,385, all maximum keys and values -128 give `O=-137,447,342,080`, below the permitted per-chunk lower endpoint. The literal unsplit witness fails `PV_NONNEG`; its prescribed split passes. |
| Unsafe full-size wrap bounds | At `m=34`, `o=2^30`, `t=-2^32+1` give integer right side exactly `P`, so every weakened local range/equality check passes at packed output zero. The primary shift-23 equation fails `PV_IDENTITY` and its narrower contamination range also fails. A widened-sum-window control similarly gives `B s'+n'=F+P`; the primary candidate fails `SUM_NONNEG`. |
| Literal padded-row count | Requiring `n>=1` makes the empty row unsatisfiable. The preregistered `n-ψ` range admits its unique zero semantic witness; a nonzero padded maximum fails `PADDING_MAX`. |

The no-output-range example checks the specified downstream reciprocal/requant arithmetic, but no control is represented as a constructed full false model proof. These are local certificate witnesses or completeness failures under deliberately weakened conditions.

## Analysis

The finite checks validate the registered local relation, and the exact Goldilocks bounds supply the non-wrap premise of its integer argument. In particular, a NonNeg decomposition of a virtual field expression fixes a bounded integer representative because its full digit box is below `P`; an untrusted field value is not assumed to have been an integer already. The maximum argument derives a zero key before using an ordering on the maximum. These two details prevent circular range reasoning.

Current-spec migration is material. The marker at the masked last key remains zero, so the all-position lookup must still pin masked packed values to zero. The selector-adapted count and existing padded-maximum pin are needed for completeness; the literal historical relation cannot handle an empty row. The default limb width is 16: sum width 34 uses digits `(16,16,2)`, and output width 38 uses `(16,16,6)`. Each adds two committed limbs and three range lookups. At one output chunk the corrected additions are **387 commitments and 516 lookups per row**, rather than the idea's historical 258 and 387. No new cost-model gain follows from this check.

Limits: the toy exponential is an abstract bounded table and toy values are INT2, not INT8. Goldilocks boundary cases are finite examples plus exact interval reasoning, not exhaustion of full production rows or a machine-checked theorem. The lookup arguments, sumcheck reductions, packing-table definition/digest, range analysis across all consumers, protocol2 masks, claim schedule and independent verifier are not implemented or audited here. Source-code assertions are evidence that this program checked its specified predicates, not proof of the full protocol's SP-3 property. All inputs are synthetic and deterministic.

Verification: `python3 -I check.py --self-test` passes, and the complete deterministic run reproduces every finite-domain count and the same result digest. `python3 -I mutate.py` runs a no-op control and four isolated faults; the control passes and all four faults fail: permitting zero count, removing the output range check, sending masked lookups to key zero, and widening the virtual top limb. Raw failures retain their exact counterexample/check trace in [results/mutations.json](results/mutations.json). These faults alter scratch copies only. The full-size named negative controls are separate and retained in `run.json`.

For an independent repeat without replacing the judged raw files, run `python3 -I check.py --output results/reproduction.json > results/reproduction.txt`. Environment, git commit and exact harness/preregistration hashes are recorded by each full run. [results/audit.json](results/audit.json) records the immutable preregistration, preserved hypothesis/method text, reproduced finite digest and fault-injection checks.

## Verdict

`validated` for the registered E0 lemma-check unit: every finite-domain witness agrees with the independent integer oracle, no required honest witness is missing, all exact Goldilocks obligations pass and the five negative-control classes are detected. This closes the queued finite-exhaustion unit. **IDEA-0029 stays `triaged`**, with its complete security gate open: the local check is not a proof of the modified protocol. Confidence remains 0.6 and priority 1.4 is retained only as the historical score; its costs and gain need revision for default limb width 16 and current main.

## Follow-ups

Only after the security unit: a superseding tier-A RFC must specify packed tables, selector-adapted row relations, masked zero tuples, chunking, transcript/claim changes, independent-verifier checks, protocol2 masks and the corrected limb counts. An H100 run could measure packed lookup and per-row range-gadget costs against the current split-eq forest; it cannot establish witness uniqueness.

### Current-spec review, 2026-10-19: reciprocal precision bound

Upstream `709c37c` includes `0710097` (#388), which gives RFC-0009 Lemma 6 the public precondition `(2 max_context - 1) 2^(2 sigma) < P` and requires claim-list validation to enforce it. At this experiment's fixed `sigma=16`, `max_context=2^17`, the left side is exactly 1,125,895,611,875,328, below Goldilocks. The updated honest sum bound is inclusive at `2^(sigma+17)`, as this experiment already checks.

No registered domain or criterion changes. This experiment neither exhausts reciprocal witnesses nor generalizes its fixed packing/output windows to other admitted `exp_sigma` values. In particular, a larger sigma changes the packed-sum and output bounds and can make the current sum/output windows incomplete. The full-protocol proof and superseding RFC must preserve the new reciprocal precondition and derive all packed-certificate bounds for each admitted precision. The fractional negative control uses the valid default sigma and remains applicable. The pinned review and exact arithmetic are recorded in `results/audit.json`.
