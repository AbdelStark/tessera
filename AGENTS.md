# AGENTS.md

Context for coding agents working in this repository. Humans start with [README.md](README.md), [SPEC.md](SPEC.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## What Tessera is

Tessera proves that a committed integer language model `D` produced a recorded token stream: a sumcheck/GKR proof over the Goldilocks field, sound against a prover that controls all hardware, zero-knowledge against a malicious verifier, checked by an independent verifier that shares no code with the prover. It serves verification between parties that do not trust each other. **A soundness bug is the worst bug this repository can have: it certifies a cheating prover.** Code that decides whether a proof is accepted, or that touches secrets, gets an adversarial reading before it ships (the `soundness-review` skill).

## Ground truth

- The specification is normative: [SPEC.md](SPEC.md), [docs/spec/](docs/spec/) and the accepted RFCs in [docs/rfcs/](docs/rfcs/). In Proposed Design sections and requirement tables, "must" means must.
- Code follows the specification. When the specification is wrong or inconsistent, say so: fix the document in the same pull request when the fix is unambiguous, otherwise open an issue. A change to a tier-A surface ([02 §1](docs/spec/02-public-api.md#1-stability-tiers)) needs a superseding RFC.
- Stable identifiers name every rule: `RFC-0004`, `INV-TCE-2`, `SP-6`, `DEP-1`, `ERR-CODEC-LIMIT`, `REJ-SUMCHECK`, `RISK-2` ([conventions](SPEC.md#conventions)). Cite them in code comments, tests and pull requests.
- Each issue names the RFC section it implements and its acceptance criteria; [docs/roadmap/IMPLEMENTATION.md](docs/roadmap/IMPLEMENTATION.md) has the dependency graph and the critical path.

## Repository map

| Path | What |
|---|---|
| `crates/` | Prover workspace: one crate per component of [01 §2](docs/spec/01-architecture.md#2-components), dependencies in the DAG order of [01 §3](docs/spec/01-architecture.md#3-dependency-rules) |
| `verifier/` | Independent verifier workspace (`tv-*`): own lockfile, `allowlist.toml`, stricter `deny.toml`, `supply-chain/` (cargo vet) |
| `xtask/` | Repository gates and automation (`cargo xtask help`); `xtask/fixtures/` holds the gates' self-test inputs |
| `python/` | `tessera-quant`, the quantization front-end (RFC-0005): uv project, own lockfile, same version as the crates |
| `spec/vectors/` | Conformance vectors: data shared by both workspaces, produced by independent oracles |
| `docs/` | Specification, RFCs, roadmap, release checklist |
| `research/` | Non-normative knowledge base of the `autoresearch` skill: state-of-the-art surveys, scored ideas, pre-registered proof-of-concept experiments, the research cost model; nothing in either workspace depends on it |
| `.agents/` | Canonical agent skills (`.agents/skills/`) and multi-agent workflows (`.agents/workflows/`); `.claude/` and `.codex/` hold relative symlinks to `.agents/` |

## Commands

```sh
cargo test -p <crate>                                    # narrowest first
cargo run -p tessera-cli --example recorder_integration -- /absolute/path/to/tessera NEW_DIR  # serving integration guide
cargo test -p tessera-cli --features otel --locked        # optional telemetry and CLI coverage
cargo test -p tessera-protocol --features tls --lib transport --locked   # optional mTLS transport
cargo test --workspace --locked                          # prover workspace
(cd verifier && cargo test --locked)                     # verifier workspace
(cd verifier && cargo test -p tv-core --features test-params --locked)   # tv-core's drivers and tamper corpus, which the line above skips
(cd verifier/fuzz && cargo fuzz run decode)              # fuzz targets, own nightly pin: verifier/fuzz/README.md
cargo clippy --workspace --all-targets --locked -- -D warnings   # also inside verifier/
cargo fmt --all --check                                  # also inside verifier/
cargo xtask lint                 # unsafe code, SPDX headers, licenses, Markdown links, policy files, telemetry, report schemas, one version for crates and Python, conflict markers, agent context
cargo xtask check-isolation      # verifier independence (RFC-0001)
cargo xtask check-test-features  # test-only features fail release builds
cargo xtask check-mask-rank      # R10: every mask row's revealed points independent on the compiled schedule, conformance and 405B claim lists (#488)
cargo xtask check-branchless     # the field's operations compile branch-free in x86_64 loops (INV-ARCH-2)
cargo xtask check-pr --base origin/main   # changelog entry and DCO sign-off of the branch
cargo xtask check-deps           # cargo-deny and cargo-vet; needs cargo-deny 0.20.2, cargo-vet 0.10.2
cargo xtask gen-vectors          # rewrite spec/vectors/ after changing codec or field cases or an oracle (generate.py); needs python3
(cd python && uv sync --extra cpu && uv run pytest)      # Python package; also ruff check, ruff format --check, pyright
```

## Invariants a change must not break

Each is a security property or a merge-blocking gate; breaking one is a bug even when every test passes.

1. **Verifier independence** (SP-6, DEP-1..3, [RFC-0001](docs/rfcs/RFC-0001-workspace-and-verifier-independence.md)). Nothing under `verifier/` depends on, includes or copies prover code; verifier code is written from the specification (the `verifier-independence` skill). Every third-party crate the verifier uses, transitive ones included, is in `verifier/allowlist.toml` with a reason. Prover crates use verifier crates as dev-dependencies only.
2. **Unsafe code.** Forbidden everywhere except `tessera-field` (SIMD backends) and `tessera-gpu` (CUDA FFI), which deny it crate-wide and opt in per module, each block with a `// SAFETY:` comment. The verifier has none.
3. **Canonical encoding** ([RFC-0003](docs/rfcs/RFC-0003-canonical-encoding.md), INV-TCE-1..4, INV-DM-1). One encoding per value; decoders return a typed error for anything else and check lengths against limits before allocating. Floating point never enters hashed or tier-A data (INV-DM-6).
4. **Fiat–Shamir** ([RFC-0004](docs/rfcs/RFC-0004-hashing-and-transcript.md), INV-FS-1..4). Every prover message is absorbed, as TCE bytes, before any challenge that depends on it is squeezed; challenges are exactly uniform (rejection sampling); labels follow the schedule of [RFC-0014](docs/rfcs/RFC-0014-interaction-and-soundness.md#transcript-schedule).
5. **True integers and uniqueness** ([RFC-0006](docs/rfcs/RFC-0006-compiler-ir.md#range-analysis), INV-IR-1..2; [RFC-0009](docs/rfcs/RFC-0009-gadgets.md), Lemma 0; SP-3). Every lookup key and gadget input is a true integer inside its table and the centered field range, and every committed value upstream of the logits has exactly one passing witness; the verifier re-derives ranges from public data and never trusts prover-declared ranges (SP-7).
6. **Redaction and clearing** ([05 §3](docs/spec/05-observability.md#3-data-classification-and-redaction), INV-ARCH-6). Secret-class values (weights, activations, tokens, masks, seeds) reach no log, error, metric or report; they live in `Secret<T>`, what the engines derive from them in `SecretBuf`, which clears every allocation it releases (OBS-6), and errors carry `PublicContext` only. Only the readers `secret-access.toml` lists take a value out of `Secret<T>` (`expose_secret`), and its borrowers compute on tables a reader lends them: clippy rejects the call elsewhere, and `cargo xtask lint` rejects any other file that names it, under any `cfg`, or switches clippy's rule off.
7. **Verifier robustness** ([RFC-0019](docs/rfcs/RFC-0019-verifier.md#robustness), SP-8). Every input yields `Accept`, `Reject(reason)` or `Inconclusive(error)`: no panic, no allocation beyond limits, no unbounded loop. `tv-core` and `tv-codec` deny `unwrap`, `expect`, `panic!` and unchecked indexing.
8. **Test-only features** (`test-rng`, `test-params`, `unsafe-debug-dump`; [RFC-0001](docs/rfcs/RFC-0001-workspace-and-verifier-independence.md#cargo-features), OBS-4) fail to compile in release builds.

## How work is done

One issue, one branch, one pull request; the `issue-workflow` skill has the loop, [CONTRIBUTING.md](CONTRIBUTING.md) the rules it follows: DCO sign-off on every commit, a changelog entry per release train, the review rules and the RFC process.

The [documentation build](docs/documentation.md) stages canonical Markdown, builds both Rust API references in fresh target directories, runs tier-B doctests and checks the assembled site's links. CI uploads a private artifact without hosting it.

### Agent skills and workflows

`.agents/` is the single source of truth for skills and workflows (mirrored via relative symlinks into `.claude/` and `.codex/`, checked by `cargo xtask lint`). Load the matching skill before starting work:

| Skill / workflow | When to load |
|---|---|
| [`issue-workflow`](.agents/skills/issue-workflow/SKILL.md) | Implementing any roadmap issue end to end (spec -> tests -> implementation -> soundness review -> gates -> PR -> distill) |
| [`soundness-review`](.agents/skills/soundness-review/SKILL.md) | Diff-level adversarial review of any change touching verification, transcript labels, range bounds, or secrets |
| [`soundness-audit`](.agents/skills/soundness-audit/SKILL.md) | Deep 5-surface mathematical and implementation audit across RFCs, `crates/`, and `verifier/` |
| [`redteam-scheme`](.agents/skills/redteam-scheme/SKILL.md) | Offensive exploit development: forging witnesses, mutating transcripts, grinding challenges, or crafting parser bombs |
| [`verifier-independence`](.agents/skills/verifier-independence/SKILL.md) | Writing or reviewing code and dependencies under `verifier/` (`RFC-0001`, `SP-6`, `DEP-1..3`) |
| [`performance-optimization`](.agents/skills/performance-optimization/SKILL.md) | Optimizing field arithmetic, Poseidon2, sumcheck, LogUp/GKR, Ligero PCS, SIMD, or CUDA kernels (`08-performance-budget.md`) |
| [`harness-engineering`](.agents/skills/harness-engineering/SKILL.md) | Building or updating `xtask` gates, `spec/vectors/` oracles, fuzz targets, or running `scripts/mutate.py` fault injection |
| [`continuous-learning`](.agents/skills/continuous-learning/SKILL.md) | Distilling bugs, review corrections, and spec ambiguities into permanent gates, tests, RFCs, or skills |
| [`autoresearch`](.agents/skills/autoresearch/SKILL.md) | Running structured cryptographic research iterations on open protocol frontiers |
| `autoresearch-sweep.js`, `soundness-audit-sweep.js`, `redteam-campaign.js` | Multi-agent workflows in `.agents/workflows/` for parallel research sweeps, 5-surface audits, and red-team campaigns |

Changes to the design start as research. The `autoresearch` skill surveys the state of the art, scores ideas against the cost model, tests the best with a pre-registered proof of concept under [research/](research/README.md), and promotes a validated result through the RFC process; the research ledger ([research/LEDGER.md](research/LEDGER.md)) says what was tried and what comes next.

## Tests that count

Tests exist to catch soundness and conformance bugs. In order of value:

1. **Adversarial**: for every check, a test that violates exactly that check and asserts the exact rejection (`RejectReason`, error code). A check without one is unproven.
2. **Conformance vectors** in `spec/vectors/`, produced by an independent oracle (for example Python big integers) and checked by both workspaces.
3. **Algebraic properties**: ring laws, round trips, `decode(encode(x)) = x` (`proptest` on the prover side, seeded `tv-testkit` generators in the verifier).
4. **Soundness calibration** over the test field `F65537`, where cheating succeeds often enough to measure against the analytic bound.
5. **End to end** through the command-line tools.

A test that restates the implementation (a getter, a constructor, plumbing) adds maintenance without catching anything.

## Pitfalls

- The verifier is a separate workspace: run its cargo commands inside `verifier/`; run `cargo xtask` from the repository root.
- Edit skills and workflows only under `.agents/`; `.claude/` and `.codex/` are symlink mirrors enforced by `cargo xtask lint`.
- `gh pr edit` fails on this repository (Projects classic deprecation): set PR bodies with `gh api -X PATCH repos/AbdelStark/tessera/pulls/N -F body=@file`, labels with `gh api -X POST repos/AbdelStark/tessera/issues/N/labels -f "labels[]=name"`.
- `rm`, `cp` and `mv` are interactive (`-i`) on the maintainer's machine; scripts move scratch files instead of deleting them, and overwrite with `/bin/cp -f` and `/bin/mv -f`.
- Vendored stand-ins under `xtask/fixtures/*/vendor/` model crates.io packages for offline gate self-tests; they are not the real crates.
