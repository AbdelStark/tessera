# Constraint-system digest conformance

RFC-0007 requires the compiler and independent verifier to derive the same constraint
system. Its `ConstraintSystemDigest` is a CI diagnostic, outside the proof protocol.
The diagnostic compares public derived metadata; it never reads weights, activations,
tokens, masks or other secret values.

The collectors live in `crates/tessera-e2e/examples/constraint_digest/`, on the
prover side of the DEP-2 test boundary. `verifier.rs` imports only independent
`tv-*` representations and test support. The compiler collector uses the compiler's
representations. Neither collector calls the other's lowering, range analysis,
schedule derivation or canonical encoder. The comparison/reporting code is test
harness code, and is not part of the verifier's acceptance predicate.

## Snapshot version 1

The snapshot includes:

- Every lowered tensor's shape, kind and producer; graph input roles, integer types
  and declared bounds; ordered node operations, all operation parameters and wiring.
- The constants, shift classes, weight references, lowering-derived table definitions,
  checkpoints and residual routes, in creation order, and the sequence length.
- Every independently analyzed tensor interval and the public gadget policy.
- Actual gadget choices, split widths, every table reference (including reused limb
  families), and all independently generated auxiliary table definitions.
- Stage sites, claims, atoms, routes, split parts, relation parameters, variable and
  degree counts, evaluation claims, symbolic points, layer selectors and coefficients.

Template leaf claims are represented at stage 1's layer point, retaining their full
per-layer coefficient vectors. Leaf split counts retain the per-layer wire count.
Symbolic points expand coordinate runs. Coefficient terms combine identical sorted
`LT` factors over Goldilocks, with scalar constants broadcast to the real layers.
This uses exact symbolic data, not randomized evaluation of a polynomial. Integer
interval endpoints are decimal strings to preserve the full signed i128 range.
Softmax references expand the compiler's shared limb family once per full limb,
matching the verifier's per-limb walk; repeated table ids are retained. The verifier's
public `selection::derive` view observes its existing validation pass. Capture is
disabled during normal validation and changes no acceptance checks.
The node operations identify RFC-0009 gadget semantics; witness/PCS placement and
proof acceptance remain the responsibilities of their separate components.
The native OneHot stage records booleanity, vocabulary masking, row count and
token index as four distinct relations. Their public vocabulary/row dimensions,
projected token claim and degree bounds are included in the comparison.

## Canonical diagnostic framing

Objects visit keys in UTF-8 lexical order; arrays visit their positions in ascending
order. Each node emits three UTF-8 strings: its JSON-pointer path, type, and payload.
Each string is framed by its byte length as an unsigned little-endian u64.

| Type | Payload |
|---|---|
| `object`, `array` | Number of children in decimal |
| `null` | Empty |
| `bool` | `0` or `1` |
| `integer` | Exact decimal integer, within signed-i64 or unsigned-u64 range |
| `string` | Raw UTF-8 bytes |

Every node is emitted, including empty containers. Pointer keys escape `~` as `~0`
and `/` as `~1`. Floating point is excluded. Both collectors hash their independently
encoded bytes with BLAKE3 `derive_key` under the CI-only context
`tessera/ci/constraint-system-digest/1`. This adds no protocol hash domain or TCE field.
The independent Python framing vectors also cover type distinctions, empty values,
64-bit integer boundaries, i128 endpoints represented as strings, and Unicode.

The reporter returns the first differing semantic path, traversing sorted object
fields and array positions. It retains both values and distinguishes a missing entry
from a JSON null. A changed scalar constant is located under
`/graph/constants/<index>`; the node's `ConstMul` parameter identifies its reader.

## Validation

The initial independent collector was checked against every field of all 24 Python
lowering/range oracle cases before inspection of the compiler implementation, and
against the existing stage summaries. Both workspaces consume the independent
framing vectors. The joint comparison covers 40 constraint systems: four conformance
models, two oracle variants, and Llama-3.1-8B/405B shapes in both profiles. Tiny cases
use lengths 1, 4, 5 and max_context; production-size cases use 1, 1024, 16384 and
131072. Shape cases allocate metadata only, and contain no model tensor values.
Their public descriptions are inputs to the lowerers, not complete production
claim lists with regenerated lookup-table artifacts.

Run the test or the diagnostic executable:

```sh
cargo test -p tessera-e2e --example constraint-digest --locked
cargo run -p tessera-e2e --example constraint-digest --locked
cargo run -p tessera-e2e --example constraint-digest --locked -- --model tiny-2-servable --n 1
```

The `constraint-digest` CI job runs the test and the single-model executable. A
mismatch exits with status 1 and prints the model, length, first differing pointer
and both values. Invalid command arguments exit with status 2. Argument selection
allows a source fault to be reproduced without materializing a large model.

