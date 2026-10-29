# Release checklist

The steps for every release of the prover train (`vX.Y.Z`) and the verifier train (`verify-vX.Y.Z`), from [09 §9](../spec/09-release-and-versioning.md#9-release-process) and the security gates of [06 §9](../spec/06-security.md#9-security-gates). Copy this list into the release's tracking issue and tick each item with a link to its evidence: a CI run, a report, a review, a commit. A release ships only when every item that applies is ticked; an item that does not apply says why.

Each milestone is a minor release (`v0.1.0` … `v0.5.0`, then `v1.0.0`, [09 §2](../spec/09-release-and-versioning.md#2-semantic-versioning)). Its exit criteria are in [SPEC.md](../../SPEC.md#release-plan).

The [reproducible-build procedure](reproducible-builds.md) describes the release
workflow, dry runs, second-builder checks and draft artifacts.

## 1. Freeze

- [ ] Every issue in the milestone is closed or moved to a later milestone, with a comment saying why.
- [ ] The milestone's exit criteria ([SPEC.md](../../SPEC.md#release-plan)) and its success criteria (`SC-n` in [00 §7](../spec/00-overview.md#7-success-criteria)) are met; each has a link to its evidence.
- [ ] Release notes are drafted from the `Unreleased` sections of [CHANGELOG.md](../../CHANGELOG.md) and [verifier/CHANGELOG.md](../../verifier/CHANGELOG.md).

## 2. Gates

- [ ] Every CI gate of [07 §9](../spec/07-testing-strategy.md#9-ci-gates) passes on the release commit: the `ci`, `pr-policy` and `dependencies` checks, and the macOS tier-2 build on `main`.
- [ ] Performance report for the milestone, with its run manifests archived ([08 §7](../spec/08-performance-budget.md#7-profiling-plan)). At v0.1 this is the unit-cost measurement and the recorded go/no-go decision under `docs/perf/v0.1/` ([RFC-0020](../rfcs/RFC-0020-cost-model.md#decision-rule)).
- [ ] Fidelity report, if quantization changed since the previous release ([RFC-0005](../rfcs/RFC-0005-integer-model.md#fidelity-evaluation)).
- [ ] v1.0 only: 24 CPU-hours of verifier fuzzing without findings, covering every decoder and check.

## 3. Security gates

- [ ] Threat-model review: [06 Security](../spec/06-security.md) is updated for this milestone, and new trust boundaries are listed.
- [ ] Before any v0.4 zero-knowledge implementation merges: the simulator argument (`docs/zk/simulator.md`) is written and reviewed by a second maintainer ([RFC-0013](../rfcs/RFC-0013-zero-knowledge.md#simulator-argument)).
- [ ] v1.0 only: the protocol and the verifier are reviewed by a party outside the implementation team, and every finding is resolved or recorded with its disposition.
- [ ] Security fixes since the previous release are listed under "Security" in the changelogs, with their advisory identifiers.

## 4. Versions and changelogs

- [ ] The prover version is the same in the workspace `Cargo.toml` and in `python/pyproject.toml` ([09 §1](../spec/09-release-and-versioning.md#1-what-is-versioned)), which `cargo xtask lint` checks; the verifier version is set in `verifier/Cargo.toml`.
- [ ] Each changelog's `Unreleased` section is renamed to the release version and date, and a new empty `Unreleased` section is added.
- [ ] A change of minimum supported Rust version is noted in the changelog ([09 §2](../spec/09-release-and-versioning.md#2-semantic-versioning)).
- [ ] Protocol version: `tessera-protocol/1` is assigned when the first end-to-end proof verifies (v0.3). Any change to a tier-A surface since then has a new protocol version and a superseding RFC. The verifier release notes list the supported protocol versions, at most the current and the previous one ([09 §3](../spec/09-release-and-versioning.md#3-protocol-and-schema-versions)).
- [ ] v1.0 only: the protocol version is frozen, and the release notes publish its full transcript-label list, parameter sets and table digests.
- [ ] Deprecations are listed under "Deprecated" with their replacement and removal release ([09 §4](../spec/09-release-and-versioning.md#4-deprecation-policy)).

## 5. Tag and build

- [ ] Tag the release commit `vX.Y.Z`, `verify-vX.Y.Z`, or both. The release workflow builds the artifacts.
- [ ] Reproducible verifier binaries: built with `--locked`, the pinned toolchain of `rust-toolchain.toml`, `SOURCE_DATE_EPOCH` and `--remap-path-prefix`, in a clean `CARGO_HOME` and environment. `TESSERA_GIT_COMMIT` and `TESSERA_RUSTC_VERSION` are the release commit and `rustc --version`, which the verifier's run manifest records ([05 §6](../spec/05-observability.md#6-run-manifest)).
- [ ] Two independent builders reproduce the verifier binary hashes.
- [ ] The SHA-256 of every binary is listed in the release notes.

## 6. Publish

- [ ] While the repository is private: a GitHub release with source archives and verifier binaries, whose notes link this checklist's completed copy.
- [ ] When the repository becomes public: enable private vulnerability reporting, so that [SECURITY.md](../../SECURITY.md)'s reporting path works for outside reporters; reserve the crates.io and PyPI package names; publish crates and the Python package from the tagged commit.

## 7. After the release

- [ ] Milestone closed; the next milestone's issues are triaged.
- [ ] Lessons that change the process are proposed as changes to this checklist, or as RFCs if they change a specified rule.
