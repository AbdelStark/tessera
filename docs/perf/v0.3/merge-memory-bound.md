# Two-pass merge memory bound

`MergeMemory::for_two_pass` prices RFC-0054's group-table constructor and two-pass
merge from polynomial shapes, the polynomial index of every group's claims, and
cut `j*`. It allocates nothing and reads no point, challenge, witness or source.
The counts must describe every actual claim, including repeated claims. The
function rejects malformed geometry and checked storage overflow.

This is a component reservation for the remaining grouped physical composition
in [#641](https://github.com/AbdelStark/tessera/issues/641), under
[RFC-0018's memory scheduler](../../rfcs/RFC-0018-gpu-prover-runtime.md#memory-scheduler).
It does not enable physical grouping or establish a real-model resource result.

## Group tables

For polynomial `i`, let `n_i` be its variables, `m_i` its free variables and
`a_i = max(0, j* - (n_max - n_i))` its own rounds answered from tables.
The legal cut ensures `a_i <= m_i`. For every claim, independently, the bound
charges one class with:

- a point of `m_i` field elements;
- `a_i` clearing-table descriptors;
- table payloads of `2, 4, ..., 2^a_i` field elements.

Claims with equal shapes, points and group bits can share a class in the actual
constructor. The bound assumes every claim is a distinct class; a favorable
transcript coincidence cannot reduce a pre-P1 reservation. Cloned polynomial
shapes, their group-variable arrays, per-group completion metadata and the class
vector are additional retained storage. Vector growth includes simultaneous old
and new allocations, following the pinned Rust 1.95 geometric capacity policy.

The constructor first combines every class's whole free-variable table before
contracting any of them. Its scratch therefore includes the sum of `2^m_i`
extension-field entries over all claims of the largest group, plus keys and
combined-buffer descriptors. Truncating a combined buffer retains its capacity
until that class is consumed. All retained class tables are charged beside this
scratch, including those created while a combined buffer is still live.

## Merge proving

The engine owns its label buffer, beta powers, both sum vectors, state descriptors,
round messages, challenge point, final evaluations and evaluation-index metadata.
Those charges include the returned proof. The first table-backed rounds also
create one equality table at a time.

The regeneration pass prices each polynomial in order. Earlier polynomials keep
two compact folded arrays, of `2^(n_i-a_i)` elements each. Beside them, the current
polynomial can hold its base-field permutation, full extension-field values and
weights, an equality table, order metadata and compact replacement copies. These
sets overlap during construction. The bound sums these current-polynomial scratch
terms conservatively; it does not assume all temporaries are freed at their first
possible last use. Later binding preserves the compact arrays' capacities.

[The compaction change](merge-state-memory.md) removes full-capacity retention
between polynomials. It does not remove the current polynomial's construction
peak or the source allocation returned by regeneration.

## Additional owners

The following are outside this component bound and must coexist in its caller's
complete phase model:

- Borrowed `Merge`, polynomial, claim-index and claim-point inputs, including their
  spare capacities and construction work.
- Caller-supplied group blocks and the regeneration callback's returned source
  allocation, including spare capacity, cache storage and construction scratch.
- Transcript/encoding callback allocations and the surrounding group/tree/source,
  native/stage/opening and packet owners.
- Stack, allocator bookkeeping, runtime/process overhead and caller model storage.

`table_bytes` bounds retained group tables. `construction_peak_bytes` includes
those tables and their constructor scratch. `proving_peak_bytes` is additional to
borrowed group tables and merge inputs; it already includes `output_bytes`.
A caller therefore compares table construction against retained tables plus merge
proving, adding its other simultaneous owners to each phase. It must not add only
the larger of table storage and proving storage when both are live.

## Validation

Counting-allocation tests execute the actual constructors and prover at every
legal cut on mixed shapes, shared and distinct classes, repeated claims, empty
and zero groups, front-padded constants, fully fixed polynomials, a full cut with
no group variables, and a larger
12-polynomial fixture. They separately check construction peak, retained table
storage, proving peak and returned output. The source-clone allocation is added
explicitly in the proving test; its allocation-free transcript stub isolates
engine storage. Existing tests separately compare complete messages and verify
proofs through the independent verifier's arithmetic and transcript.

An aggregate-overflow case describes `2^20` distinct classes of `2^40` entries:
`2^20 * 2^40 * 16 = 2^64` bytes. Only its 8 MiB public claim-index list is created.
Planning refuses with `WindowError::Allocation`, without loading a witness or
allocating a table. Invalid shape/cut/index cases assert exact typed errors.
