# RFC-0029: The Claim List Is the Table Set

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Supersedes, on acceptance:
  - [RFC-0006](RFC-0006-compiler-ir.md#outputs) 'Outputs': `CompileOutput.tables`, a `TableSetManifest`;
  - [03 §5](../spec/03-data-model.md#5-artifact-catalog): the `TableSetManifest` schema, `0x0203`, `.tts`;
  - [02 §2.2](../spec/02-public-api.md#22-tessera-rust-prover-side): the `tables/` that `tessera compile` writes.
- Origin: [#316](https://github.com/AbdelStark/tessera/issues/316)

## Summary

Three documents name a public artifact, `TableSetManifest` (schema `0x0203`, `.tts`), that `tessera compile` writes to `tables/`. None defines it.

Everything such a manifest could hold is already elsewhere:

- each table's `TableSpec` is in the claim list's `tables`, with its kind, parameters, `log2_rows` and digest;
- its rows follow from that specification by RFC-0008's generators;
- for a human reader, its rows and lookups are in the commitment report.

This RFC retires the artifact. The claim list's `tables` is the table set. Every party that needs a table builds it from its `TableSpec` and checks the digest.

## Motivation

`TableSetManifest` appears in three places:

- the artifact catalog (03 §5), which attributes it to RFC-0008, where it never appears;
- `CompileOutput.tables` (RFC-0006);
- the `tables/` directory that `tessera compile` writes (02 §2.2).

`tessera compile` ([#87](https://github.com/AbdelStark/tessera/issues/87)) cannot write an artifact that no document defines. Defining it means deciding what it holds, and each candidate fails:

- **Digests and row counts.** The claim list's `TableSpec`s already carry both. A copy would be a second source of truth that nothing binds: the public statement binds `claim_list_digest` (INV-DM-3), and the verifier regenerates every table from the claim list rather than reading a list of digests ([RFC-0007](RFC-0007-claim-list.md#verifier-validation) step 4).
- **The rows.** At 405B, 4.4 to 7.6 · 10^9 table rows carry a value ([`docs/perf/v0.2/`](../perf/v0.2/README.md)), 33 to 57 GiB as field elements. Each row is either a closed form of its index (`Range`, `Requant`) or a gather from a core of at most `2^24` values (`Silu`, `Exp`): the structure RFC-0008 'Structured evaluation' relies on. Storing them trades a few integer operations per row for a read from disk. RFC-0018 rejects that trade for the witness (its 'Alternatives Considered' 1).
- **The dense cores.** A cache of the generators' cores saves their generation: about 2.7 s on one core at 405B ('Cost'). That is a private optimization of the prover, not a public artifact, and a cached core is only as trustworthy as its re-hash against the table's digest.

The artifact has no content of its own, and nothing reads it: the prover and the verifier both work from the claim list.

## Proposed Design

### The table set

The claim list's `tables: Vec<TableSpec>`, sorted by `TableId` ([RFC-0026](RFC-0026-claim-list-layout.md#schema)), is a compiled model's table set. No other artifact defines, lists or carries tables. Each party that needs a table builds it from its specification:

- **The verifier** regenerates each table from its specification and compares digests (RFC-0007 step 4). `tessera-verify tables` prints them.
- **The prover** builds each table from its specification before the witness pass and checks its digest, as the CPU witness engine does. A table that does not regenerate to its digest is `ERR-INTERNAL`, and the prover sends nothing.
- **The Python reference** (`tessera_quant.tables`) generates the same tables from the same kinds and parameters. The three implementations are checked against pinned digests ([RFC-0008](RFC-0008-logup-and-tables.md#generators-and-digests) 'Generators and digests').
- **The commitment report** lists each table's kind, rows and lookups for a human reader (RFC-0006 'Outputs'). It is a JSON report, never read back into a proof.

The prover reads the claim list its plan was compiled from. The execution plan carries it ([RFC-0041](RFC-0041-execution-plan.md#the-plan)). This RFC adds nothing to the plan.

### Outputs

```rust
pub struct CompileOutput {
    pub claim_list: ClaimList,          // public, RFC-0007; its tables are the table set
    pub plan: ExecutionPlan,            // private
    pub report: CommitmentReport,       // public JSON, tessera.commitment-report/1
}
```

`tessera compile --out-dir DIR` writes `claims.tcl`, `plan.tep` and `commitment-report.json`.

### Schema `0x0203`

`0x0203` and `.tts` leave the artifact catalog. They stay unassigned: an artifact that needs a schema takes a new id through its own RFC. No writer ever emitted them, and no release has written any TCE artifact ([09 §3](../spec/09-release-and-versioning.md#3-protocol-and-schema-versions)), so there is nothing to migrate. Every reader treats an envelope of schema `0x0203` as a schema it does not expect: `ERR-CODEC-MAGIC` ([04 §3](../spec/04-error-model.md#3-error-codes)).

### Cost

Building 405B's tables from their specifications takes 2.66 to 2.68 s on one core of an Apple M4 Max. The figure holds for each of the six configurations of [`docs/perf/v0.2/`](../perf/v0.2/README.md) at `B_max = 32`: 15 to 17 tables, 5.0 to 11.2 · 10^9 padded rows. RoPE's `2^23` pairs take 2.6 s of it. `Range` and `Requant`, which are closed forms, take none. The verifier does the same work at every claim-list validation.

That cost matters beside a proof: a 405B request of 1K tokens projects 0.9 to 1.0 s of proof time. A prover keeps the cost off its critical path without any artifact:

- RoPE's pairs are independent, so the prover spreads them over its host's cores;
- a long-lived prover builds its tables once for every batch it proves.

### Unchanged

`TableSpec`, its encoding and its digest; the table kinds and their generators (RFC-0008); claim-list validation; `tessera-verify tables`; the commitment report.

## Alternatives Considered

1. **A manifest of the tables' digests and row counts**, the catalog's apparent intent. Rejected: it restates the claim list's `tables`, no statement digest binds it, and no verifier may read it in place of regeneration.
2. **Materialized tables in `tables/`**, rows for the prover to load. Rejected: tens of GiB at 405B, each row a closed form or a gather from a core. RFC-0008 'Alternatives Considered' 5 rejects shipping table contents with the claim list, for their size and because the verifier would trust them.
3. **A public cache of the dense cores, checked by digest.** Rejected as a protocol artifact: it saves seconds per prover start, which parallel generation or a long-lived prover saves as well. A prover may still cache cores privately. That is an implementation detail (tier C, [02 §1](../spec/02-public-api.md#1-stability-tiers)), and a cached core must hash to its table's digest before use.
4. **Keeping `0x0203` reserved for a later definition.** Rejected: an id reserved for an undefined artifact is the gap #316 found.

## Drawbacks

- The prover regenerates its tables at each start, at the cost above.
- A model's tables live inside the claim list, a TCE artifact, rather than in a file of their own. `tessera-verify tables --claims` prints their digests, and the commitment report lists them.

## Migration / Rollout

v0.2, before any artifact is released. The following change in place:

- the catalog;
- `CompileOutput`;
- `tessera compile`'s outputs;
- `tessera-codec`'s schema registry, which drops `TABLE_SET_MANIFEST`. Its test reads the catalog from 03 §5, so the two change together.

Nothing in the verifier, the Python package or the conformance vectors reads `0x0203`.

## Testing Strategy

- **The catalog.** The schema registry is 03 §5's catalog, which `tessera-codec`'s test reads from the specification itself. Neither has `0x0203`.
- **Determinism.** RFC-0006's test compares claim-list and plan bytes.
- **Regeneration.** For every conformance vector, `tessera-tables` refuses a specification whose digest, `log2_rows`, id or parameters differ from the regenerated one. The verifier's claim-list validation brings its own negative tests (RFC-0007 'Testing Strategy', [#107](https://github.com/AbdelStark/tessera/issues/107)).

## Open Questions

None.

## References

- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0018](RFC-0018-gpu-prover-runtime.md), [RFC-0026](RFC-0026-claim-list-layout.md).
- [#316](https://github.com/AbdelStark/tessera/issues/316), [#87](https://github.com/AbdelStark/tessera/issues/87), [#101](https://github.com/AbdelStark/tessera/issues/101).
