# SUR-0016: Weight-mask draw cost and the missing XOF rate

| | |
|---|---|
| Status | `current (2026-10-29)` |
| Agenda topic | `hash`, question 1, specialized to RFC-0073's weight streams for [#568](https://github.com/AbdelStark/tessera/issues/568) |
| Questions | Standing question: “Measured BLAKE3 throughput on an H100 in a column-hashing setting, against the 530 GB/s per GPU the budget implies (RISK-8); the same for Poseidon2, Monolith, Tip5, Skyscraper, Keccak and BLAKE2s (ICICLE, sppark, Plonky3, RISC Zero, Stwo numbers).” This bounded addendum asks whether that column-hash rate prices weight XOF output, what bytes each storage/regeneration option actually needs, and what #568 can decide before #394. It does not repeat the other-hash comparison in SUR-0006. |
| Period covered | BLAKE3's 2020 design through primary documentation/code checked 2026-10-29; local normative documents at `7f08aed075d118d1cf0dc53dab5f810d34f07e21` |
| Produced by | `autoresearch survey hash`, backlog #568, 2026-10-29 |

## Method

The term is `W`, `UnitCosts::weight_pass_gpu_s`, not `c_c`: it is work paid once per
proof unit and then divided by batch tokens and native cost. Read RFC-0012 'Column
answers', RFC-0065 'The reservation' and 'Cost at 405B', draft RFC-0073 'Cost at 405B'
and OQ-0073-5, RFC-0020's executable model and 08 §2. The earlier hash survey is
[SUR-0006](SUR-0006-hash.md); it already identifies the implied column-hashing budget
as a projection. No production stream kernel was changed and no secret data was used.

Searched the web, arXiv- and ePrint-restricted indexes and GitHub with `BLAKE3 GPU XOF
benchmark H100`, `BLAKE3 GPU extendable output throughput H100`, `BLAKE3 GPU extended
output performance`, `BLAKE3 official specification extendable output root output
blocks counter`, and `ingonyama icicle blake3 output_size`. Read the BLAKE3 authors'
IETF draft's output/concurrency sections, upstream C `output_root_bytes`, Rust
`OutputReader` documentation, and ICICLE's libraries/benchmark pages. Search results
for a T4 whole-file Vulkan hash and CPU XOF libraries were excluded as H100 rate
evidence. No applicable H100 long-XOF plus field-rejection/folding benchmark was found
in these sources; this is a search outcome, not proof none exists. Three primary
artifacts establish semantics/support; none supplies the missing rate.

The count below is exact integer arithmetic **conditional on the RFC's stated layout**.
It is not an experiment or a new compiler enumeration. The older common-width row
count includes up to 882 `ChannelParams` data rows; RFC-0073 also notes a tighter
56-row layout. Neither is silently treated as a newly measured current workload.
No timing or security experiment, protocol change, or gain is claimed.

## Findings

| Source | What it is | Mechanism | Numbers and evidence | Term | Applicability | Caveat |
|---|---|---|---|---|---|---|
| [B3XofDraft24] | BLAKE3 authors' algorithm draft, extendable output and implementation sections | Root output blocks vary a counter; they can be generated independently after the root state is fixed. | E0, primary: 64 output bytes per root compression; no H100 rate. | `W` | direct: count output blocks rather than input bytes hashed through a tree. | Parallelism is not throughput; domain/row initialization and exact field rejection remain. |
| [B3XofCode26] | Upstream C implementation, `output_root_bytes` | The aligned bulk path invokes `blake3_xof_many`; partial blocks use the XOF compression path. | E0, primary code: bulk output advances the counter by output length / 64. | `W` | direct: a long stream need not rehash its seed for each output block. | CPU implementation evidence; not a GPU kernel or Tessera secret-handling implementation. |
| [B3Output26] | Upstream Rust API documentation | `OutputReader` emits and seeks through extended output; shorter outputs are prefixes. | Primary documentation, no timing quantity. | `W`, `zk` | direct for stream semantics; adapt for implementation. | The public reader's traits/storage do not satisfy Tessera's clearing-reader requirements by themselves. Seeking by accepted field index also needs a rejection-position map. |
| [IcicleHash26] | ICICLE support and benchmark documentation | GPU-capable BLAKE3 hashing is listed, alongside CPU and other backends. | Primary support claim; no applicable H100 XOF/compaction/fold rate on the inspected pages. | `W` | adapt: possible implementation starting point, not cost evidence. | Hash API support does not establish RFC-0073's stream, secrecy, field output or workload. No dependency proposed. |
| [E0-SUR16] | This survey's count of RFC-0012/0065/0073's example | Sum encoding-mask coefficients over reached rows and message values over reached blinding rows. | E0, primary arithmetic: 654,109,048,832 accepted bytes for all rows; 216,735,416,320 for the hypothetical selected-only support below. | `W`, storage | direct for complete accounting. | Conditional counts, not a proof that blinder exclusion is sound or that an implementation draws each byte only once. |

### Complete draw sets

Let matrix `T` have `d_T` data rows, `b_T` blinding rows, message width `k_T` and `P_W`
encoding-mask coefficients per row. For the union `J_T` of rows reached by the proof's
combined-row coefficients, accepted field bytes are

```text
draw_bytes = 8 * sum_T (P_W * |J_T| + k_T * |J_T intersect blinding_rows_T|).
```

This is the distinct stream output needed if each value is generated once and reused
for every combination consuming it. Repeated passes/independent regeneration add work;
the formula does not promise it away. Data values are model weights, not PRG output.
Stored codeword column answers avoid P5 re-encoding; they do not supply P4's combined
message rows or encoding-mask combinations.

Under the specified proximity tensor, every committed row has a generically nonzero
coefficient, so the conservative support is all `d_T + b_T` rows. Public degenerate
challenges can reduce support but are not a planning discount. If #394 instead proves
and specifies zero coefficients for every unreserved blinding row in **every** relevant
combination, the support becomes data plus reserved blinders. Omitting a row's value
while retaining its encoding-mask coefficient is a different accounting scenario.

The RFC example uses `P_W = 2^16`, common `k_T = 2^20`, 461,310 total rows, 49,144
blinding rows and thus 412,166 data rows. RFC-0065 gives `e_CP = 7`; a proof reserves
`9*6 + (2*7 + 4) = 72` base-field blinding rows. These are **E0 conditional layout
inputs**, not measured values. The selected-only scenario includes all 72 rows' values
and encoding masks, even if a future support proof might show a smaller union.

| Component / scenario | Accepted bytes (E0) | Decimal GB (E0) |
|---|---:|---:|
| All rows' encoding masks | 241,859,297,280 | 241.859297280 |
| All blinding rows' message values | 412,249,751,552 | 412.249751552 |
| Current full-support generator: sum of preceding rows | 654,109,048,832 | 654.109048832 |
| Data rows' encoding masks only | 216,093,687,808 | 216.093687808 |
| Reserved 72 rows: encoding masks and values | 641,728,512 | 0.641728512 |
| Hypothetical selected-only support: preceding two rows | 216,735,416,320 | 216.735416320 |
| All encoding masks but only reserved blinding values | 242,463,277,056 | 242.463277056 |

The 216 GB in #568/draft RFC-0073 is the **data-mask subtotal**, missing reserved-row
work. The 242 GB in #394's comment corresponds to the last row's more conservative
mask support. They are not measurements of the same algorithm. Neither exclusion
scenario is the present generator or a validated hiding argument.

RFC-0004 rejects 64-bit candidates at or above `p`. With ideal uniform words, expected
candidate bytes for `L` accepted words are `8*L*2^64/p`. The expansion factor is
approximately `1.0000000002328306` (E0); buffer block rounding, row initialization and
compaction/folding must be counted separately. This tiny expected expansion is not a
deterministic cap, a permission to use modular reduction, or a reason to leak skip
positions (INV-FS-3, INV-ARCH-2, RFC-0073).

Reproduce the accepted-byte totals without model data:

```sh
python3 -I - <<'PY'
rows, blinders, p_w, k_w, reserved = 461310, 49144, 2**16, 2**20, 72
all_masks = rows * p_w * 8
all_values = blinders * k_w * 8
selected = (rows - blinders) * p_w * 8 + reserved * (p_w + k_w) * 8
print(all_masks, all_values, all_masks + all_values, selected)
PY
```

### Rate and storage comparisons

RFC-0004 derives roughly 530 GB/s from 16 codeword bytes / an **assumed** 30 ps per
committed element. It neither measures H100 XOF output nor isolates compression from
encoding and column hashing. RFC-0020 explicitly calls its source costs unmeasured.
Thus #568's phrase “measured stream rate” has no supporting measurement in that RFC.

At an explicitly hypothetical XOF output rate of 530e9 bytes/s, full support costs
`654109048832 / 530e9 = 1.234168016664151` GPU-s, plus initialization/rejection/folding
and any repeated generation. The selected-only scenario costs `0.408934747773585`
GPU-s at that same proxy. These are **E0 rate scenarios**, not E2 estimates justified by
a comparable kernel. Adding the first to the existing 0.24 core weight pass yields
1.474168016664151 GPU-s in a model sensitivity scenario:

```sh
python3 -I research/tools/cost_model.py --profile counted \
  --set weight_pass_gpu_s=1.474168016664151
```

This isolated change increases the model's weight term at 1K/4K/16K/128K context by
about 0.110/0.108/0.101/0.066 in `mu` (E0). The other terms remain those of the selected
historical profile; this is **not** a complete Protocol Two forecast or an updated
scoreboard result. A measured total `W` that already includes draws must not have this
scenario added a second time.

A raw per-epoch store holds the same distinct mask/blinding values and trades their
generation for a full secret-class read per proof. Its baseline read time is
`draw_bytes / effective_store_bytes_per_second`, plus folding and transfer costs, not
zero. The relevant rate is end-to-end bandwidth from the chosen store to consumers,
including contention with weights and existing codewords. Peak HBM bandwidth and a
codeword store's small queried-column reads are not that measurement. A store also
adds clearing, access, lifetime and crash-consistency obligations. GPU regeneration
can consume bounded chunks directly into combinations and avoid a full persistent
store, but needs its own measured kernel and secret-memory review.

## Synthesis

Settled analytically: current coefficient support requires all masks and blinding
values; column-answer storage alone cannot remove this work; 216 GB is incomplete;
530 GB/s is an assumed column-hash budget, not XOF evidence. RFC-0020's statement that
stored weight column answers have no extra term is about those answers, not all
Protocol Two opening work. The unqualified 08 weight-pass budget cannot price this
draw volume while retaining its old interpretation.

Unsettled: the complete support after a sound resolution of #394, H100 XOF plus
rejection/fold throughput, repeated-generation count, storage medium/bandwidth and
peak live secret storage. Neither the search nor arithmetic chooses between a store
and regeneration on measured grounds. Blinder exclusion changes the masking/proximity
argument and cannot be approved by this cost survey. No prior hash-alternative
rejection is reopened: the proposed work retains BLAKE3, domains, exact uniformity,
current coefficient support and SP-6.

## Candidate ideas

- **Price full-support GPU regeneration as the default planning path**, with bounded
  chunked generation fused into all consumers and an explicit measured draw/fold term
  in `W`; no new persistent secret store. Benefit is complete accounting and bounded
  storage, not a credited speedup. First killer: independent byte/count audit finds
  another consumer/pass, or measured cost cannot meet an explicitly amended budget.
- **Compare an optional secret mask store in the same harness**, including writes,
  reads, transfers and clearing; credit a gain only against full regeneration of the
  same streams/combined rows. First killer: storage bandwidth or lifetime/security
  obligations remove the saving. No gain is scored without measurement.
- **Treat blinder exclusion as a separately gated scenario**, deriving the union of
  coefficient support only after #394's joint-view proof fixes it. The conditional
  byte ratio is about 0.331 of full support (E0); it is not a security-approved saving.
  First killer: exclusion invalidates proximity/binding or joint hiding. Until then,
  no production implementation or promotion on this basis.

## Staleness triggers

- #394 fixes the joint coefficient/support rules or the epoch reservation changes.
- A public H100 benchmark measures the exact RFC-0073 streams, uniform field rejection,
  all consumers and memory traffic under RFC-0020's protocol.
- A compiler-derived weight-row inventory replaces the RFC's common-width example,
  including tight `ChannelParams`, every matrix width and per-proof claim counts.
- #568 chooses an accounting/storage design and the accepted budgets/model change.

## References

Keys resolve in [bibliography.md](../bibliography.md): [B3XofDraft24], [B3XofCode26],
[B3Output26], [IcicleHash26], [E0-SUR16]. Normative inputs are RFC-0004, RFC-0012,
RFC-0020, RFC-0065, draft RFC-0073 and 08 §2. This survey is not normative.
