# CPU regenerated advice openings

The P4/P5 components implement [RFC-0011](../rfcs/RFC-0011-advice-commitment.md#opening-protocol)
without retaining a witness matrix, codeword matrix or `G × N` coefficient matrix.
The caller supplies public compiler-mapped claims, canonical row runs and a replay of the
same per-proof mask stream for P1, P4 and P5. Combined-row messages are released only after
checking a randomly selected column's merged leaf and each completed combination at it.

## Storage and declassification

P4 retains a block of `B` rows and masks, one reusable codeword, `G + 1` combined rows and
sampled-column sums. Coefficients and unreduced combination scratch are block-local.
All secret-derived allocations live in `SecretBuf`, including the shared kernel's
unreduced sums; aborts clear them. Final output buffers are allocated as zeros before any
private result is copied into a returned protocol message. No fallible operation follows
that copy. P5 similarly holds one codeword and the eventual output columns, then copies
those into the protocol message after the complete pass succeeds.

The public opening tree is a separate `32 × (2n − 1)`-byte allocation. During its
construction, P1's completed segment chains are still live caller-held input, adding
`32 × S × n` bytes; the tree counter starts after those chains were allocated. They can
be released once the tree exists. The root-only `merge_segments` retains its
[zero-heap behavior](streaming-advice-commitment.md).

Prepared public groups are also measured separately from the passes. They hold one
column point per group, one row point and weight per mapped image, and the proximity
point: `O(images × row-index bits + G × column bits)` data plus vector headers. This is
metadata per image, not per committed row. Original input claims can be released after
preparation, and prepared groups after P4. Layout, encoder tables, input row, mask source,
original mapped claims, prepared groups and previously returned messages are built before
the pass measurements; their memory is additional to the pass peaks below.

## Allocator measurements

Measured on a 64-bit ARM CPU, Darwin, Rust 1.95.0, using public synthetic data:

```sh
cargo test -p tessera-pcs --locked --test regenerated_memory -- --nocapture
cargo test -p tessera-gpu --locked --test checkpoint_clearing
```

The synchronous current-thread counter has a positive allocation control and no spawned
workers. Each P4 measurement includes construction, every push, block combinations,
self-check, final message copies and private-buffer drops. P5 includes construction,
all regenerated symbols, canonical multiproof construction and final output copies.
The tests cover `k = 1, 8, 64`, `B = 5`, `S = 1, 8` and 1/16 groups, at 128 and 1,024
committed rows. At `k = 1` every claim has the same empty column point, so there is only
one derived group.

| `k` | groups | public tree | P4 peak | P4 message remaining |
|---|---:|---:|---:|---:|
| 1 | 1 | 96 | 6,880 | 160 |
| 8 | 1 | 992 | 7,496 | 384 |
| 8 | 16 | 992 | 59,456 | 3,264 |
| 64 | 1 | 8,160 | 12,424 | 2,176 |
| 64 | 16 | 8,160 | 77,824 | 18,496 |

All figures are bytes. At `k = 64`, 16 one-image originals use 4,848 bytes of prepared
group metadata for 128 rows and 5,664 bytes for 1,024 rows. Preparation peaks at 5,104
and 5,920 bytes respectively, including its temporary power vector. The 816-byte increase
comes from three additional row-index coordinates per image and proximity point, rather
than storage per committed row. Both segment counts give the same figures.

The P4 peak is identical for both witness lengths and both segment counts. Only protocol-message allocations survive P4. Small public equality tables may
scale with the row-index bit count; the test separately enforces a bound that accounts
for them and all block scratch.

With two distinct queried columns, P5's symbol payload is 2,048 bytes at 128 rows and
16,384 bytes at 1,024 rows. Its measured peaks are:

| `k` | P5 peak, 128 rows | P5 peak, 1,024 rows |
|---|---:|---:|
| 1 | 4,232 | 32,904 |
| 8 | 4,600 | 33,272 |
| 64 | 5,752 | 34,424 |

These peaks include both the private columns and final protocol copies. P5 output storage
necessarily scales with `N × distinct queries`; it does not retain the `N × k` witness.
The metadata, whole message copy and multiproof overhead are included in the enforced
bound. The recording allocator separately verifies that aborted P4 data/mask blocks,
shared combination scratch, and unsent P5 columns free only zero bytes, with non-clearing
plain-vector controls. That test needs no GPU; unsafe allocator instrumentation is allowed
in `tessera-gpu` only.

These are component heap requests, not RSS, stack frames, device memory, production-size
measurements, throughput claims or parameter-security certification. The CPU path uses
the existing row encoder to evaluate a row before keeping its sampled/queried symbols.
GPU kernels, protocol/2 hiding and full driver integration remain separate work.
