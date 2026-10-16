# 09 — Release and Versioning

- Status: Accepted for v1 planning
- Governs: version numbers, protocol and schema evolution, deprecation, changelog, contribution workflow, licensing, supported platforms, reproducible builds

## 1. What is versioned

| Thing | Scheme | Owner of the number |
|---|---|---|
| Prover release train: crates under `crates/`, the `tessera` CLI, the `tessera-quant` Python package | One shared SemVer version, tags `vX.Y.Z` | Workspace `Cargo.toml` and `pyproject.toml`, kept equal by a CI check |
| Verifier release train: crates under `verifier/`, the `tessera-verify` CLI | Its own SemVer version, tags `verify-vX.Y.Z` | `verifier/Cargo.toml` |
| Protocol | Integer `N`, string `tessera-protocol/N` | [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#messages) |
| Artifact schemas | `(schema_id, version)` per artifact type | [03 §5](03-data-model.md#5-artifact-catalog) |
| Parameter sets | Name plus version, for example `live-64/1` | RFC-0014 |
| Table definitions | Covered by the protocol version | RFC-0008 |

The two release trains are separate so the verifier can ship on its own schedule and so neither train's version implies anything about the other's code. Compatibility between them is expressed only through protocol versions.

## 2. Semantic versioning

- Before 1.0: a minor bump (`0.Y`) may break tier-B interfaces; patch bumps do not. Each milestone in the roadmap is a minor release (`v0.1.0` … `v0.5.0`).
- From 1.0: standard SemVer for tier B. Tier-A changes follow §3 regardless of SemVer.
- The minimum supported Rust version (MSRV) is 1.88 (edition 2024): the field marks its selects with `core::hint::select_unpredictable`, stable since 1.88, so that they stay conditional moves in the binary (INV-ARCH-2). An MSRV increase is a minor bump and is noted in the changelog.

## 3. Protocol and schema versions

- The first protocol version, `tessera-protocol/1`, is assigned when the first end-to-end proof verifies (milestone v0.3). Before that, artifacts carry protocol version 0, which no release verifier accepts.
- Any change to a tier-A surface (message layout, transcript labels, challenge derivation, table definitions, parameter sets, verdict semantics) increments the protocol version and requires an accepted RFC that supersedes the affected section.
- A verifier release supports exactly the protocol versions listed in its release notes, at most two: the current one and the previous one.
- Schema readers accept versions `v` and `v − 1`; writers emit `v` only. Readers never guess: unknown versions fail with `ERR-CODEC-VERSION`.
- A JSON report version (`tessera.<name>/<v>`, [03 §5](03-data-model.md#5-artifact-catalog)) is frozen once a release writes it, and a later change to its shape is version `v + 1`. Until then its schema under `spec/schemas/` changes in place with its writer, as artifacts carry protocol version 0 until the first protocol version is assigned.
- The protocol version current at v1.0 (expected `tessera-protocol/2`, which adds zero knowledge in v0.4) is frozen there. The v1.0 release notes publish its full transcript-label list, parameter sets and table digests.

## 4. Deprecation policy

- A tier-B item scheduled for removal is marked deprecated (Rust `#[deprecated]`, Python `DeprecationWarning`, CLI warning on stderr) for at least one minor release before removal, after 1.0.
- A protocol version is deprecated when its successor ships and removed from the verifier one minor release later.
- Deprecations are listed in the changelog under "Deprecated" with the replacement and the removal release.

## 5. Changelog discipline

- `CHANGELOG.md` (prover train) and `verifier/CHANGELOG.md` (verifier train) follow the Keep a Changelog 1.1 structure: Added, Changed, Deprecated, Removed, Fixed, Security.
- Every PR that changes a tier-A or tier-B surface adds an entry under "Unreleased". CI fails a PR that touches `crates/*/src`, `verifier/`, `python/tessera_quant/` or `spec/` without a changelog entry, unless the PR carries the `no-changelog` label.
- Security fixes are listed with their advisory identifier.

## 6. Contribution workflow

- Work happens on topic branches and lands on `main` through pull requests. `main` is always releasable: every CI gate in [07 §9](07-testing-strategy.md#9-ci-gates) passes.
- Every PR links the issue it closes, and every issue links the spec section or RFC it implements.
- Reviews: at least one maintainer approval. Changes under `verifier/`, to protocol schemas, transcript labels, parameter sets or soundness arguments additionally require approval from a maintainer who did not author the change, and a linked accepted RFC.
- Commits carry a Developer Certificate of Origin sign-off (`Signed-off-by:`), checked by CI.
- Commit subjects are imperative, at most 72 characters; the body states why.
- **RFC process.** New load-bearing decisions start as a PR adding `docs/rfcs/RFC-NNNN-<slug>.md` in Draft status. Acceptance is a merged PR that sets Accepted. A decision changes only through a new RFC that marks the old one "Superseded by RFC-MMMM". RFC numbers are never reused.

## 7. Licensing

- Tessera is dual-licensed under Apache-2.0 OR MIT, at the user's option. Every source file carries an SPDX header (`// SPDX-License-Identifier: Apache-2.0 OR MIT`, `# SPDX-License-Identifier: Apache-2.0 OR MIT`).
- Allowed third-party licenses (checked by `cargo deny` and `pip-licenses`): MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unicode-3.0, CC0-1.0. Anything else, including MPL-2.0, needs an exception recorded in `deny.toml` with a reason. The verifier admits no exceptions.
- Model weights are never redistributed. Test vectors use randomly generated weights. Users supply checkpoints under their own licenses.
- Contributions are accepted under the same dual license (inbound = outbound), affirmed by the DCO sign-off.

## 8. Supported platforms

| Component | Tier 1 (CI-tested) | Tier 2 (builds) |
|---|---|---|
| Prover, GPU backend | Linux x86-64, CUDA ≥ 12.8, sm_90 (H100/H200) | sm_100 (B200), Linux aarch64 |
| Prover, CPU backend | Linux x86-64 | macOS aarch64, Linux aarch64 |
| Verifier | Linux x86-64, Linux aarch64 | macOS aarch64 |
| Python front-end | Linux x86-64, Python 3.11–3.13 | macOS aarch64 |

## 9. Release process

1. Freeze: the milestone's issues are closed or moved; release notes are drafted from the changelog.
2. Gates: every CI gate passes on the release commit, plus the release-only gates: 24 CPU-hour verifier fuzzing (v1.0), the performance report for the milestone, and the fidelity report when quantization changed.
3. Tag `vX.Y.Z` and/or `verify-vX.Y.Z`; the release workflow builds artifacts.
4. Reproducible verifier binaries: built with `--locked`, a pinned toolchain (`rust-toolchain.toml`), `SOURCE_DATE_EPOCH`, and `--remap-path-prefix`. Two independent builders reproduce the binary hash before the release is published. The SHA-256 of every binary is listed in the release notes.
5. Publication: while the repository is private, releases are GitHub releases with source archives and verifier binaries. Publication to crates.io and PyPI waits until the repository is public; package names are reserved at that time.
