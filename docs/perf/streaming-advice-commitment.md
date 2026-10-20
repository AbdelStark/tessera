# CPU streaming advice commitment memory

`StreamingCommitter` implements [RFC-0011](../rfcs/RFC-0011-advice-commitment.md#streaming-procedure-witness-pass)
without retaining witness rows, masks, codewords from previous blocks or a Merkle tree.
It borrows the canonical layout's range runs, rather than materializing a row-index vector.

The per-worker heap is exactly:

```text
block:   8 * n * min(B, committed rows in this segment)
chains: 32 * n
```

`RowEncoder` writes into the initialized secret block without allocating. Column hashing
feeds the prior chain and each canonical symbol into the hasher without a serialization
allocation. Every full or final partial block is erased with volatile field zeroization.
Dropping a partially filled committer clears the whole allocation through `SecretBuf`.
Finishing retains just the segment's public digests and borrowed layout identity.

## Allocator measurements

Run on a 64-bit ARM CPU test build, Darwin, Rust 1.95.0:

```sh
cargo test -p tessera-pcs --test streaming_memory -- --nocapture
cargo test -p tessera-gpu --test checkpoint_clearing
```

The rows and masks are public synthetic test data; these measurements certify storage, not
zero knowledge or a protocol parameter set. The synchronous allocation counter observes the current thread. Encoder tables, the layout,
input row, mask source and caller-owned slots for returned parts are built before measurement.
The test measures construction, every push, block hash and finish, with no worker threads.
For 36 combinations of `k`, `B` and `S`, it commits 512 and 4,096 rows, an eightfold increase,
with `S = 1, 2, 8`. Every worker has exactly two allocations, whose total bytes equal its peak;
no allocation occurs per row or block. Only the chain allocation remains after finish.

| `k` | `B = 1` peak | `B = 3` peak | `B = 5` peak | `B = 32` peak |
|---|---:|---:|---:|---:|
| 1 | 80 | 112 | 144 | 576 |
| 8 | 640 | 896 | 1,152 | 4,608 |
| 64 | 5,120 | 7,168 | 9,216 | 36,864 |

All figures are bytes and match the formula for each segment count and both witness lengths.
An empty segment at `k = 8` makes one 512-byte allocation for its initial chains and draws
no masks. Every measured merge makes zero heap allocations.

The recording allocator in `tessera-gpu` inspects a fully initialized streamed codeword block
when it is freed after an abort before flushing: every byte is zero. A plain vector containing
the same synthetic codeword is a negative control and frees nonzero bytes. A separate unit
test checks the live block immediately after a full-block flush and after the next row.
The allocator test needs no GPU; it lives there because the recorder uses the repository's
permitted unsafe allocator instrumentation.

## Whole working set

The figures measure component heap requests, not RSS, stack frames or device memory.
The caller also holds its public layout, shared encoder tables, one input row and PRG state.
The encoder's public tables contain `(1.5 + 2^(r-1)) * k` field elements, 80 MiB at the default
advice parameters. The new committer borrows those tables rather than duplicating them.

Sequentially finishing all `S` segments keeps at most one active block plus `32 * S * n`
bytes of retained chain vectors and the caller's public part slots. Parallel workers each
need their own block. Merging borrows those chains and keeps a fixed public frontier of
33 optional digests on the stack; it allocates no leaves or tree nodes. Compiler stack
placement and the hash implementation's stack state are outside the allocation counter.

At default `k = 2^22`, `n = 2^23`, `B = 32`, the formula gives 2 GiB for one block and
256 MiB for one segment's chains. Those production-size figures are analytic, not a
measurement of a 2 GiB block or GPU execution. This report makes no throughput claim.
