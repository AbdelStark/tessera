# RFC-0075: Account for Weight-Mask Regeneration in Every Proof

- Status: Draft. Amendments marked as this draft's take effect only on acceptance.
- Authors: @AbdelStark
- Created: 2026-10-29
- Target milestone: v0.5
- Amends on acceptance:
  - [RFC-0020](RFC-0020-cost-model.md#executable-cost-model): the weight-pass component, measurement and decision rule;
  - [08 Performance budget](../spec/08-performance-budget.md#2-unit-cost-targets): the distinction between the core weight pass and Protocol Two's complete weight work;
  - draft [RFC-0073](RFC-0073-weight-mask-expansion.md#cost-at-405b): complete draw counts and OQ-0073-5's storage/accounting choice.

## Summary

Choose **bounded GPU regeneration** as the planning/default execution path for Protocol
Two's per-proof weight-mask and blinding-row work. Retain RFC-0012's existing column-major
codeword store for P5 answers. Do not add a persistent raw mask/blinder store and do not
exclude blinding rows from the current proximity generator to meet a performance budget.

Every proof prices the complete reached streams in `W`, independently of its column
answers. Keep the old core-pass target/re-plan gate; add an explicit draw component and
report total weight work. The initial numerical scenario is an E0 projection using an
assumed effective rate, not a measured H100 XOF rate. Acceptance of this accounting
decision does not clear #145/#394's zero-knowledge prerequisites or activate Protocol Two.

## Motivation

[#568](https://github.com/AbdelStark/tessera/issues/568) identifies work missing from
08's 0.24 GPU-s weight-pass target. A stored codeword column answers a query, but it
does not give the combined message rows and encoding-mask polynomials needed before
queries. RFC-0012's proximity coefficients reach every committed row, so opening work
regenerates every encoding mask and every blinding row's values unless they are stored.

[SUR-0016](../../research/surveys/SUR-0016-weight-mask-draw-cost.md) also finds that
RFC-0004's 530 GB/s comes from 16 codeword bytes divided by an assumed 30 ps per committed
element. It is neither a measured XOF rate nor a rate for rejection, compaction and
combination. [EXP-0028](../../research/experiments/EXP-0028-weight-mask-draws/README.md)
validates finite public geometry/support/pass accounting (E0), not that throughput.
Counting only the data masks or pricing stored columns as all opening work hides costs.

## Goals

- One default storage/regeneration choice with complete, reproducible per-proof counts.
- An executable cost-model interpretation using its existing `weight_pass_gpu_s`, with
  no omitted or double-counted draw work.
- Separate core/draw/total targets, preserving the core gate and measurement honesty.
- A benchmark contract that can replace the projection and expose required re-planning.

## Non-Goals

- A new proximity generator, joint hiding argument, row budget, mask expansion, transcript,
  protocol identifier, proof format or verdict. #394 and #145 remain prerequisites.
- Implementing the masked GPU backend or a raw secret store in this change.
- Claiming an H100 rate, a complete Protocol Two cost forecast or a proved secrecy property.
- Changing historical/source Protocol One projections or treating their counts as Protocol Two.

## Proposed Design

### Storage and execution choice

The default has the epoch seed and existing codeword store, with **no raw mask store**.
During the opening work, generate each needed stream in bounded chunks and consume a
chunk into all its combined-row/mask-polynomial consumers before releasing it. The
public schedule fixes row addresses, consumers and chunk bounds. Count one generation
only when the implementation actually shares it; separate passes and resume/replay
work are charged separately. A nominally fused implementation must prove its count with
a pass/consumer trace, not infer it from the number of distinct row addresses.

Every live stream state, accepted/rejected block, intermediate and released allocation
obeys OBS-2/OBS-6 and the GPU allocator's clearing rules. Aggregate live storage across
workers must fit the configured memory budget. Do not allocate the whole output volume,
return secret blocks through public `Debug`/errors, or expose rejection positions in
metrics. Expansion, row addressing, exact rejection and seed/recovery rules remain
RFC-0004 and draft RFC-0073's. This is an implementation requirement, not evidence that
such a GPU kernel already exists.

P5 normally reads and authenticates queried columns from RFC-0012's codeword store. If
that store is absent, its fallback recomputes codeword symbols after the queries arrive.
That fallback has additional evaluation **and stream regeneration** work. A P4/P5
implementation which discards stream values at P4 cannot charge their generation once
merely because the epoch seed is the same. Storage reads, transfers and all extra
generations are included in measured `W`; overlap affects latency, not aggregate GPU work.

### Complete public count

For each matrix `T`, let `d_T` be its committed data rows, `b_T` its committed blinding
rows, `k_T` its message width, and `P_W` its encoding-mask coefficients per row. Derive
them from public geometry and the claim/reservation rules, never from tensor values.
Use RFC-0065's claim-dependent `B_CP`, including zero for empty ChannelParams. Widths
are per matrix; `k_T` may exceed `k_W`. Validate Protocol Two's width/rate restrictions
and the reservation budget separately; a hypothetical byte count is not admission.

For one generation, form the union of actual consumers' encoding-mask row support `Jm_T`
and blinding-value support `Jv_T`. Accepted stream bytes are

```text
B_generation = 8 * sum_T (P_W * |Jm_T| + k_T * |Jv_T|).
```

The current generator's conservative support is `Jm_T = all d_T+b_T rows` and
`Jv_T = all b_T blinders`. Thus

```text
B_full = 8 * sum_T ((d_T + b_T) * P_W + b_T * k_T).
```

Union consumers **within** one generation; sum separate generations and independent
epochs. A fallback P5 reaches all answered rows even if a hypothetical future P4 rule
has smaller support. Link work includes both epochs. Public degenerate challenges do
not reduce the planning envelope. Expected extra candidate words from rejection are
not a deterministic bound or permission to use modular reduction.

The effective measurement below includes rejection and all stream-consumer work; do
not multiply it again by a rejection factor. If instead reporting raw XOF throughput,
label it separately and add setup/rejection/compaction/folding/clearing costs explicitly.

### Executable model and report decomposition

RFC-0020's existing `UnitCosts.weight_pass_gpu_s` is the **total** once-per-proof weight
work, not a core-GEMM-only number. The contribution to `mu` remains `W/(T*native_s)`:

```text
W = W_core + W_draw + W_fallback + W_other
W_draw_projection = sum_generations B_generation / R_effective
```

`W_core` is the weight fold/GEMM work the old 0.24 GPU-s target priced. `W_draw` includes
stream setup, exact field rejection, compaction, all stream-consumer combinations,
temporary traffic and clearing. `W_fallback` includes extra column evaluation and
regeneration when there is no stored answer; account for each operation in one
component only. `W_other` names remaining weight work, including non-overlapped GPU
transfer work if applicable. Record host/storage latency and traffic separately too.

The all-inclusive measured `W` replaces this component sum; it is never increased by
adding a second estimate of draws already inside it. Glue does not secretly price
these missing components. `UnitCosts::SOURCE` remains a historical source-input profile,
whose 0.24 must not be presented as a complete masked-proof estimate. A Protocol Two
projection must supply total `W` or explicitly report unpriced weight work; missing
measurements cannot silently select zero. No new tier-A surface is introduced.

Every report claiming the amended budget records protocol/mode, matrix widths/data and
blinding rows, claim counts, reservation size, consumer fan-out, generation counts,
publicly derived accepted-byte counts, source/code revisions, storage choice, component
GPU-seconds and total `W`. Candidate-byte/block counts are reported only by public-seed
benchmarks: production secret streams disclose neither rejection counts nor positions.
Schema/report implementation follows acceptance; current runtime reports
must not advertise a masked benchmark that has not run.

### Initial 405B scenario and revised budget

These are **E0 projections**, not H100 results. The frozen experiment derives nine
weight matrices with 411,284 data rows. Seven channel types across 126 real layers give
56 tight ChannelParams rows. At `e_CP=7`, there are 49,144 blinding rows in all. With
common `k_T=2^20` and `P_W=2^16`:

| Layout case | Full-support accepted bytes per generation | Draw time at hypothetical `R_effective=530e9 B/s` |
|---|---:|---:|
| Tight: 411,340 data rows | 653,675,986,944 | 1.2333509187622642 GPU-s |
| Historical 882-channel-row upper bound: 412,166 data rows | 654,109,048,832 | 1.234168016664151 GPU-s |

Use the historical envelope for the initial planning gate, until the deployment's
derived inventory replaces it. The assumed 530e9 **effective accepted-byte** rate is
a deliberately explicit scenario based on RFC-0004's implied bandwidth, not a transfer
of a measured hash rate to XOF. Including setup/consumer work in that assumption is
unvalidated; failing to achieve it triggers re-planning rather than changing its label.

For a stored-codeword proof with one draw generation, the amended 08 §2 gate is:

| Component | Planning target | Re-plan threshold |
|---|---|---|
| Core weight work | unchanged: at most 0.24 GPU-s | unchanged: above 0.5 GPU-s |
| Complete weight work | `0.24 + B_full/530e9`, about 1.474 GPU-s under the envelope | above `0.5 + B_full/530e9`, about 1.734 GPU-s, **or** the core threshold fails |

An implementation cannot hide a failing core pass behind a faster draw kernel. Counts
or generation multiplicity that differ from this case require a stated re-fit, including
fallbacks; they cannot be advertised as meeting this one-generation target. No existing
CI/measurement gate is removed. This is a new masked-workload budget alongside the old
core budget, not a claim the original total-work projection was met.

The existing model can reproduce a sensitivity scenario by supplying total
`weight_pass_gpu_s = 1.474168016664151` under the historical envelope. The research tool
command in SUR-0016 does so; its other historical counts do **not** constitute a complete
Protocol Two model. Source/default charts remain source charts.

### Measurement required to replace the scenario

On an H100 SXM under RFC-0020's warmups, ten samples, statistics and CV/re-run rules:

1. Check every row stream against an independent CPU expansion, including rejection
   boundaries and seeks across output blocks. Check final combined mask/message rows,
   not just a digest of generated bytes. Use public synthetic data and fixed test seeds.
2. Measure the exact per-matrix widths, row counts, consumer fan-out and generation
   schedule used by the workload. Include stream initialization, field rejection,
   compaction, all consumer arithmetic, transfers and clearing in the effective rate.
3. Report isolated core and draw phases, then total weight work with their actual
   overlap. In this public-seed benchmark, count accepted bytes separately from physical
   bytes/blocks, temporary memory and storage traffic. Never export secret-stream rejection
   counts or positions in a production report. Check peak aggregate live secret memory
   against the budget.
4. Run stored-codeword and fallback scenarios; repeated P5/resume generations must
   appear in traces and cost. A microbenchmark which leaves full masks resident cannot
   qualify bounded regeneration. Publish raw samples, parameters and conformance checks.
5. Re-fit total `W` and apply both thresholds. A failure leads to an explicit plan/budget
   revision under RFC-0020, not a zero draw term or unreviewed blinder exclusion.

## Alternatives Considered

1. **Persistent raw mask/blinder store.** Not the default. It replaces generation by a
   full secret-class read of the reached values every proof, about 654 GB in the
   historical envelope (E0). Effective storage-to-consumer bandwidth, transfers,
   lifetime/permissions/clearing and crash/recovery semantics need their own design and
   matched benchmark. Existing codeword storage does not imply this raw store exists.
2. **Exclude unreserved blinders.** Not selected. #394 must fix and prove the complete
   adaptive view first, including proximity binding and joint hiding. The hypothetical
   historical selected-only count is 216,735,416,320 bytes, including 72 reserved rows'
   masks and values (E0). The 216 GB data-mask subtotal is incomplete; retaining all
   encoding masks but only selected values instead gives 242,463,277,056 bytes. Neither
   calculation approves an alternative generator or earns a production saving.
3. **Price it inside commitment or P5 costs.** Rejected: it is per-proof P4 work even
   when no epoch commitment is made and P5 reads stored columns. That would omit or
   double-count it depending on which phases a benchmark timed.
4. **Call the 530 GB/s rate measured.** Rejected: the source documents no comparable
   measurement. The projection is allowed to have an explicit unmeasured input; its
   evidence level is not allowed to change by repetition.

## Drawbacks

The complete projected weight cost is much greater than the old core-only number.
The effective-rate assumption may be optimistic, especially with many consumers.
Bounded regeneration requires a clearing implementation and careful sharing of output
within a pass; repeated generation can be materially more expensive. No E2 evidence
here chooses the fastest possible design. The choice avoids a new persistent secret
store while making that uncertainty and its measurement gate explicit.

## Migration / Rollout

1. Review this decision and its evidence; it remains Draft until accepted. Draft-marked
   changes to RFC-0020, 08 and RFC-0073 take effect only on acceptance.
2. Preserve the #145 hold and #394 joint-view obligations. No masked GPU path or proof
   capability is enabled by this RFC or its research scripts.
3. Implement report decomposition, generation traces and bounded clearing execution
   with the masked backend, after its security prerequisites. Keep Protocol One source
   defaults unchanged; mark unpriced Protocol Two reports explicitly until qualified.
4. Run the H100 measurement contract, publish the result and re-fit/apply the gates.

## Testing Strategy

EXP-0028's frozen E0 domain passes 3,671 cases, 25,576 literal/formula accounting
judgments and 28,500 independent saved-record checks. Ten accounting faults and eleven
reader corruptions are caught with passing controls. Its immutable registration,
earlier attempts, exact quantities and limits are linked in the experiment report.
These are accounting controls, not production mutations or a proof of zero knowledge.

Required implementation regressions include missing reserved masks/values, wrong
matrix width or channel claim count, charging empty matrices, duplicate consumers,
forgotten repeated passes/epochs, premature buffer release and uncleared allocations.
The benchmark contract adds expansion/combined-row equivalence and real memory/pass
checks. Current code changes no verifier predicates, transcript or gadget relations.

Security review: ADV-P gains no new proof format or unchecked relation. ADV-V receives
no new stream seed, raw store or transcript item. Public geometry drives accounting;
secret execution/clearing remains a future implementation gate. This is a non-regression
claim about the decision's scope, **not** an approval of the current Protocol Two
joint hiding argument. SP-6, INV-DM-6, exact field rejection and all existing checks remain.

## Open Questions

- **OQ-0075-1.** What effective H100 draw/consumer rate and peak live memory does the
  complete bounded implementation achieve? Owner: @AbdelStark; measurement before
  claiming the amended masked-workload budget.
- **OQ-0075-2.** Does #394 justify a different support/generator and epoch hiding rule?
  If so, re-derive every consumer/pass count and budget through a separate reviewed
  amendment; do not substitute the selected-only subtotal in this design.

## References

- [#568](https://github.com/AbdelStark/tessera/issues/568), [#394](https://github.com/AbdelStark/tessera/issues/394), [#145](https://github.com/AbdelStark/tessera/issues/145).
- [SUR-0016](../../research/surveys/SUR-0016-weight-mask-draw-cost.md), [EXP-0028](../../research/experiments/EXP-0028-weight-mask-draws/README.md), with primary XOF sources and retained raw evidence.
- RFC-0004, RFC-0012, RFC-0020, RFC-0065, draft RFC-0073 and 08 §2 as linked above.
