# Folded P2b merge allocations

[RFC-0054's merge](../../rfcs/RFC-0054-token-groups.md#the-merge) regenerates each
polynomial once after its table-backed rounds. Before
[#669](https://github.com/AbdelStark/tessera/issues/669), folding shortened its
value and public-weight vectors while retaining their full capacities. Every
completed polynomial therefore kept its original allocations until P2b ended.

The two-pass prover now copies the folded values into a clearing allocation and
the folded public weights into a vector before retaining them. Replacing the old
`SecretBuf` clears its entire allocation, including truncated entries. A polynomial
that has not folded keeps its existing buffers. Messages, challenges, polynomial
order, front padding and final evaluations are unchanged.

For polynomial `i`, let `N_i = 2^n_i` and let `d_i` be the number of its own
variables bound by the first `j*` rounds, after front padding. The retained value
and weight payload is `32 * N_i / 2^d_i` bytes over Goldilocks Fp2, rather than
`32 * N_i`. The regenerated source, permutation, full-size work arrays, compact
copy overlap, kept class tables, metadata and proof outputs remain additional.
This component change supplies no complete grouped admission bound and does not
enable the currently refused grouped physical route.

## Allocation regression

The public fixture uses 14-variable polynomials, two groups and a four-round cut.
The counted interval owns the complete two-pass function and its regeneration
copies; fixture inputs and kept tables are built beforehand and remain additional.
These are requested allocation peaks, not process RSS or a whole-proof bound.

| Polynomials | Original peak, bytes | Compact peak, bytes |
|---|---:|---:|
| 4 | 2,491,408 | 1,016,848 |
| 12 | 6,686,288 | 1,279,568 |
| Additional eight | 4,194,880 | 262,720 |

The regression permits at most 393,216 additional bytes: two folded extension
arrays per extra polynomial plus public descriptor headroom. It fails on the
original implementation. Separate compiled mutations retaining full-size values
or weights each fail this test. Corrupting copied values and omitting the initial
value fold each fail the message oracle; both five-test no-op/restored controls pass.

The false-claim test first verifies honest in-memory and two-pass controls against
the same polynomial tables at every cut. It then changes exactly one claim and
checks that its claimed sum differs. Review found that its old separately seeded
fixture already disagreed with those tables; the added honest control failed
before that fixture was corrected. A fifth compiled mutant omitting the single
perturbation now fails the test, confirming that a negative case actually changes
the honest claim.

## Timing scope

The host reports Apple M1 Max and 64 GiB RAM. Safe compaction itself performs no
field arithmetic. It copies two Fp2 arrays, about `64 * L` bytes of read/write
traffic for folded length `L`, plus `16 * L` bytes to clear the additional secret
allocation when released. The original full secret allocation must still clear.
Using [Apple's advertised 400 GB/s chip memory bandwidth](https://www.apple.com/uk/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/)
gives an ideal traffic floor of 0.2 ns per retained pair, or about 2.46 microseconds
for twelve length-1,024 pairs. That chip-wide ceiling is not measured single-core
bandwidth or a runtime prediction. Full P2b still performs field folds and weights.

Criterion 0.5.1 uses ten samples, one second of warm-up and two seconds of
measurement per case, with ThinLTO and one codegen unit. The first run overlapped
compilation and is retained in the raw record. Two subsequent frozen-binary pairs
alternate baseline/candidate order without active Cargo builds. The original 1B
prover and both model selectors continue as co-runners; the host is not dedicated.

| Case, twelve polynomials | Baseline mean, ms | Compact mean, ms | Baseline / compact CV |
|---|---:|---:|---:|
| Uniform height 14, cut 0 | 27.367 | 27.024 | 10.72% / 8.06% |
| Uniform height 14, cut 4 | 19.129 | 18.629 | 9.64% / 6.98% |
| Heights 14/13/12, cut 4 | 9.812 | 10.022 | 11.25% / 7.23% |

Every case exceeds RFC-0020's 3% variation threshold. Differences range from
-2.6% to +2.1%; no throughput improvement is established. The measured benefit is
the allocation reduction. [Raw observations](../../../crates/tessera-sumcheck/benches/windowed/measurements/merge-compaction.json)
retain the initial runs, both alternating pairs, iteration counts, timing estimates
and frozen binary identities.

Reproduce the component tests and timing with:

```sh
cargo test -p tessera-sumcheck --test merge --locked -- --nocapture
CARGO_PROFILE_BENCH_LTO=thin CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 \
  cargo bench --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml \
  --bench merge --locked -- --save-baseline local --noplot
```

This is public synthetic component evidence. Independent grouped proof tests,
complete physical lifetime bounds, and actual NI-100 real-model time, memory,
acceptance and fidelity qualification remain separate requirements under
[#641](https://github.com/AbdelStark/tessera/issues/641),
[#629](https://github.com/AbdelStark/tessera/issues/629) and
[#617](https://github.com/AbdelStark/tessera/issues/617).
