# Contributing to Tessera

Tessera's verifier is the trusted base of audits between parties that do not trust each other, so every change is held to the specification and checked mechanically before it lands. This guide is the practical form of [09 §5–§7](docs/spec/09-release-and-versioning.md#5-changelog-discipline); where they differ, the specification wins.

## Start from the specification

- Read [SPEC.md](SPEC.md), then the RFC for the component you are changing ([reading order](SPEC.md#reading-order)). In Proposed Design sections and requirement tables, "must" is normative.
- Every change starts from an issue that names the RFC or specification section it implements and lists its acceptance criteria. [docs/roadmap/IMPLEMENTATION.md](docs/roadmap/IMPLEMENTATION.md) lists the issues and their dependencies.
- If the specification is wrong or ambiguous, say so in the issue or pull request and fix the document in the same change when the fix is clear. A change to a tier-A surface (protocol messages, transcript labels, table definitions, parameter sets, verdict semantics; [02 §1](docs/spec/02-public-api.md#1-stability-tiers)) needs a superseding RFC.
- Cite stable identifiers (`RFC-0004`, `INV-TCE-2`, `SP-6`, `ERR-CODEC-LIMIT`) in code comments, tests and pull requests.

## Workflow

1. Branch from an up-to-date `main`, one branch per issue.
2. Make the smallest complete change that meets the issue's acceptance criteria.
3. Add the tests the change needs. Every check the verifier performs has a test that violates exactly that check and asserts the exact rejection; shared conformance vectors live in `spec/vectors/` as data, not code ([07](docs/spec/07-testing-strategy.md)).
4. Run the gates locally (below), commit with a sign-off, and open a pull request that closes the issue.

`main` is always releasable: every gate passes on every commit.

## Commits

- **Sign off every commit** with `git commit -s`. The `Signed-off-by: Name <email>` line certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/) for your contribution, and that you contribute it under the project's dual license, Apache-2.0 OR MIT (inbound = outbound). CI rejects a commit whose sign-off does not name its author.
- Write the subject in the imperative mood, at most 72 characters, prefixed with the area: `core:`, `verifier:`, `pcs:`, `infra:`, `ci:`, `docs:`. Use the body to say why the change is needed.

## Changelog

A pull request that changes a release train adds an entry under `## [Unreleased]` in that train's changelog ([Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/) sections: Added, Changed, Deprecated, Removed, Fixed, Security):

| Change touches | Changelog |
|---|---|
| `crates/*/src`, `python/tessera_quant/` | [CHANGELOG.md](CHANGELOG.md) |
| `verifier/` | [verifier/CHANGELOG.md](verifier/CHANGELOG.md) |
| `spec/` | either |

A change that needs no entry carries the `no-changelog` label. Security fixes cite their advisory identifier.

## Local gates

During bootstrap, run the narrow tests affected by the change and the repository/PR policy gates before pushing. The [automatic pipeline](README.md#continuous-integration) runs the fast protocol and tiny-model coverage; use the manual quality workflow for the complete suites, oracle regeneration and release/toolchain checks. The commands below are the full validation inventory, not a requirement to rerun every suite for every PR:

```sh
cargo fmt --all --check && cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
(cd verifier && cargo fmt --all --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked)
cargo xtask lint                 # unsafe code, SPDX headers, licenses, Markdown links, policy files, conflict markers
cargo xtask check-isolation      # the verifier shares no code with the prover
cargo xtask check-test-features  # test-only features fail release builds
cargo xtask check-pr --base origin/main   # changelog entry and sign-off for your branch
cargo xtask check-deps           # needs cargo-deny 0.20.2 and cargo-vet 0.10.2
cargo xtask gen-vectors          # rewrite spec/vectors/: codec vectors and every oracle's; a test fails while they are stale
```

## Reviews

- Every pull request needs at least one maintainer approval.
- Changes under `verifier/`, to protocol schemas, transcript labels, parameter sets or soundness arguments also need approval from a maintainer who did not author the change, and a linked accepted RFC ([09 §6](docs/spec/09-release-and-versioning.md#6-contribution-workflow)).
- A pull request that regenerates conformance vectors must pass the verifier's vector tests unchanged, or change verifier code in the same pull request with that second review ([RFC-0001](docs/rfcs/RFC-0001-workspace-and-verifier-independence.md#shared-artifacts-that-are-not-code)).

## RFCs

Load-bearing decisions are recorded as RFCs in [docs/rfcs/](docs/rfcs/):

1. **Draft.** Open a pull request that adds `docs/rfcs/RFC-NNNN-<slug>.md` with `Status: Draft`, using the next unused number and the sections of the existing RFCs: Summary, Motivation, Goals, Non-Goals, Proposed Design, Alternatives Considered, Drawbacks, Migration / Rollout, Testing Strategy, Open Questions, References.
2. **Accepted.** The decision is accepted when a pull request setting `Status: Accepted` merges.
3. **Superseded.** An accepted RFC changes only through a new RFC that replaces it; the old one is marked `Superseded by RFC-MMMM`.

RFC numbers are never reused. To discuss an idea before writing it up, open an issue with the RFC proposal template.

## Issue labels

| Label | Meaning |
|---|---|
| `type:*` | Exactly one per issue: `feature`, `bug` (behavior differs from the specification), `test`, `docs`, `ci`, `refactor`, `security`, `perf`, `tracking` |
| `area:*` | Exactly one per issue: the component of [01 §2](docs/spec/01-architecture.md#2-components) that owns the work |
| `priority:p0` | Required for its milestone's exit criteria |
| `priority:p1` | Required for v1.0, not for its own milestone's exit |
| `priority:p2` | Optional |
| `effort:s`, `effort:m`, `effort:l` | Up to 2 days, up to a week, 1 to 3 weeks |
| `spec:rfc-NNNN`, `spec:sNN-*` | The RFC or specification section the issue implements |
| `tracking` | A subsystem dashboard listing child issues |
| `blocked` | Waiting on another issue, hardware or a decision; the issue says which |
| `no-changelog` | The pull request needs no changelog entry |
| `good first issue`, `help wanted` | Self-contained work for newcomers; work where outside help is welcome |

## Releases

Every release follows [docs/release/checklist.md](docs/release/checklist.md): freeze, gates, the security gates of [06 §9](docs/spec/06-security.md#9-security-gates), versions and changelogs, reproducible verifier builds and publication ([09 §9](docs/spec/09-release-and-versioning.md#9-release-process)). The release's tracking issue carries a completed copy, and the release notes link it.

## Security and conduct

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md), never in a public issue. Participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
