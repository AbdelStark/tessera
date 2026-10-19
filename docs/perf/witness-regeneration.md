# CPU witness regeneration memory

The CPU source implements #112 for the accepted current-model lowering
(RFC-0018, RFC-0041, RFC-0050). It regenerates shares, all real gadget advice,
lookup keys/values and complete padded table counts. It does not produce a proof;
the driver must still bind the gadgets' relations and product forests.

## What is counted

A working tile includes the **entire call's** live inputs, outputs and scratch.
`TileWitness::bytes()` counts returned secret buffers only; it is not the peak.
The allocation test reports these quantities separately:

- `C`: retained checkpoint allocations, including their bookkeeping. Payload is
  exactly `(layers + 1) * n * d_model * 4` bytes.
- `P`: retained source metadata, prepared scalar gadgets and public table data
  allocated when constructing the source. Construction peak is measured too.
- `B(tile)`: peak live allocation during one `tile()` call, including all replay
  tensors, temporary keys/columns, public call metadata and returned buffers.
  Dropping the returned tile leaves zero retained allocations from that call.

The measured additional residency above borrowed inputs is
`max(checkpoint-build peak, C + source-build peak, C + P + max_tile B(tile))`.
The borrowed model/weights, tokens, lowering, selection and pre-built engine
(including its public table cache) are outside these figures. They are present
in both the resident and regenerated runs. These are thread-local allocator
bytes, not process RSS, CUDA memory or hardware throughput.

There is one request in flight. A replay releases a tensor at its last reader,
stops at its layer's checkpoint, and computes an attention core row by row.
Attention tiles retain only the requested query block's rows. Table counting
retains one column plus one contributing replay; its peak is bounded in the test
by that column's storage plus the largest ordinary regeneration peak and the
public visit metadata. It retains Q/K once per core and consumes one row's
certificate at a time. This is not a bound of `C + returned tile bytes`.

## Reproduce

```sh
cargo test -p tessera-witness --locked --test memory -- --nocapture
```

The test uses one CPU thread so the allocator sees every allocation. It checks
every planned tile and every selected table, Q=1 and Q=512, at the four models'
64-token context. It also checks n=32 versus n=64 at Q=1: each ordinary tile's
peak grows at most linearly. Corresponding layer positions have identical peaks
and total allocations; replay depth does not grow with the number of preceding
layers. The full accounted residency must stay below half the resident first
pass peak on these fixtures. This is a fixture regression bound, not a 405B
capacity claim.

Measured on Apple M4 Max, macOS 26.6.2 arm64, Rust 1.95.0, optimized test profile:

| Model | Checkpoint payload | C | P | Largest returned tile | Largest B | Accounted peak | Resident first-pass peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny-2-servable | 49,152 | 49,236 | 590,561 | 399,408 | 712,200 | 1,351,997 | 6,020,696 |
| tiny-2-minimal | 49,152 | 49,236 | 363,665 | 282,640 | 596,712 | 1,009,613 | 5,558,120 |
| tiny-2-affine | 49,152 | 49,236 | 468,031 | 399,408 | 791,064 | 1,308,331 | 6,315,752 |
| tiny-4-servable | 327,680 | 327,820 | 808,977 | 2,359,312 | 2,909,544 | 4,046,341 | 41,366,104 |

All figures are bytes, and identical for the two query-block choices at this
context because a table tile sets the largest peak. Construction peaks for P
are respectively 689,821 / 463,061 / 567,139 / 902,573 bytes; checkpoint build
peaks are 371,472 / 370,892 / 371,844 / 1,003,332 bytes. The test prints fresh
measurements; allocator/platform changes may alter these recorded observations.

## Wide canonical columns

A `Range { bits: 32 }` has 2^32 logical rows: dense u64 counts would require
32 GiB. That full domain is still committed, including zeros. The CPU column
chooses `min(8 * padded_table_rows, 16 * public_update_bound)` storage before
seeing keys. The update bound includes repeated requests, per-axis padding and
the final row-0 tail. Fixed initialized weighted entries are never compacted by
secret sparsity; reading a public chunk scans their entire public capacity and
scatters counts into the clearing output buffer. It never sorts secret keys or
uses a secret-dependent loop bound (INV-ARCH-2). Keyed addresses still require
the protocol's release schedule before externally observable messages.

Unit tests exercise 2^32-row logical columns through bounded chunks,
including its last row and zero holes. Integration tests compare every
contributed row and deterministic zero/padding chunks against padded first-pass
histograms. Root scalar arithmetic uses the existing gadget oracle; Softmax
keys and slack digits are reconstructed by independent integer arithmetic.
The Python advice oracle separately checks exact role identities and request
order. Deallocation tests inspect dense and weighted storage after clearing,
with an uncleared-copy negative control. GPU regeneration remains #165; neither
these CPU measurements nor EXP-0024 certify GPU performance or timing ZK.
