---
name: issue-workflow
description: The one-issue, one-branch, one-PR loop of this repository. Use when picking up, implementing, shipping or closing a GitHub issue, and when working through the backlog.
---

# Issue workflow

One issue, one branch, one pull request, merged before the next issue starts. [CONTRIBUTING.md](../../../CONTRIBUTING.md) holds the rules (sign-off, changelog, reviews); these are the steps. Each step ends on its completion criterion.

1. **Read.** `gh issue view N`. Read every RFC and specification section it cites, and check that each issue on its `Depends on` line is closed. Read the issue's shapes, sizes and costs as evidence of what the RFC means. One that the RFC's formulas cannot express is a gap in the specification, to fix in the same pull request: an "`m×k` by `k×n`" weight matmul against a single-token formula, for instance. Done when you can name every acceptance criterion and the change that meets it. Discover unfamiliar source and RFC filenames with `rg --files` before opening them: modules may use `native.rs` beside `stages/mod.rs`, and RFC filenames include their exact titles (#641).
2. **Branch.** `git fetch origin && git checkout -b <area>/<short-description> --no-track origin/main`. Read the affected normative sections from this worktree before implementation; if the initial reading came from another checkout, re-read them here. A preserved `main` can omit merged security amendments, such as RFC-0009's vocabulary zero-check (#378). Done when `git status` shows the new branch with no upstream.
3. **Implement** the smallest complete change that meets the acceptance criteria, in the crate that [01 §2](../../../docs/spec/01-architecture.md#2-components) assigns the component to. When the issue and the architecture disagree (a validation command naming another crate, for instance), follow the architecture and say so in the pull request. Done when the code compiles without warnings.
4. **Test**, adversarial cases first (`AGENTS.md`, Tests that count). Done when every acceptance criterion maps to a test or a command, and each adversarial test was seen to fail against a broken implementation.
5. **Document.** Rustdoc on every public item (`missing_docs` fails CI). A changelog entry for the release train the change touches. README and `AGENTS.md` when commands, gates or invariants change. Done when `cargo xtask lint` passes.
6. **Validate**, narrow to wide: the crate's tests, then every command in `AGENTS.md`, in both workspaces. Keep source frozen and serialize Cargo checks that share a target directory, including feature builds and Clippy: Cargo releases its build lock during tests, so a sibling rebuild can invalidate rustdoc's dependencies (`E0460`). Use distinct target directories for independent parallel runs. For code that decides acceptance or handles secrets, apply the `soundness-review` checklist; for code under `verifier/`, follow `verifier-independence`. Done when every gate is green and its result is recorded for the pull request, every count and figure from a run you made yourself: a delegated agent's report is a claim to check, not a result.
   Prebuild the [exact-mode CLI pair](../../../docs/testing/exact-cli.md) and set `TESSERA_BIN` / `TESSERA_VERIFY_BIN` before broad tests: a cold fallback Cargo build can consume the CLI watchdog before any command is exercised.
7. **Commit** with `git commit -s`: an imperative subject of at most 72 characters with the issue's area prefix, a body that says why, and `Closes #N`. Done when `cargo xtask check-pr --base origin/main` passes.
8. **Open the pull request**: `gh pr create --base main --milestone <milestone> --label <label>…` with the sections of `.github/pull_request_template.md` (Problem, Solution, Validation, Caveats). `gh pr edit` fails on this repository, so a later change to the description goes through `gh api -X PATCH repos/AbdelStark/tessera/pulls/<PR> -F body=@<file>`. Done when the pull request exists and its `closingIssuesReferences` (GraphQL) name exactly the issues it closes.
9. **Wait for CI**: `gh pr checks <PR> --watch`. On failure, read the failing step's log (`gh run view <run> --log-failed`), fix, push, and wait again.
   - `main` requires branches to be up to date. When it moves first, rebase onto `origin/main`.
   - Resolve every file that `git diff --name-only --diff-filter=U` lists. `CHANGELOG.md`, `SPEC.md`'s RFC index and 03 §5's catalog collect concurrent additions, so keep both sides' entries, in order.
   - `cargo xtask lint` rejects a leftover conflict marker.

   Done when the merge state is `CLEAN`.
10. **Comment** on the issue: what was implemented, the pull-request link, and the validation results. Then merge with `gh pr merge <PR> --merge --match-head-commit <sha> --delete-branch`, `<sha>` being the head whose checks you saw pass, confirm the issue closed, and return to `main` with `git checkout main && git pull --ff-only`.
11. **Distill.** When the issue uncovered a soundness subtlety, spec ambiguity, or recurring pitfall, apply [`continuous-learning`](../continuous-learning/SKILL.md) to codify it into a gate, test, RFC, or skill. Done when the lesson is captured at the highest applicable level of the escalation hierarchy.

## When an issue cannot be completed here

- **Blocked** (hardware, credentials, a decision only a maintainer can make): comment with the blocker and its evidence, add the `blocked` label (`gh api -X POST repos/AbdelStark/tessera/issues/N/labels -f "labels[]=blocked"`), and move to the next issue.
- **Partly verifiable**, for example a criterion that needs the GPU runner or a future release: ship what can be validated with `Refs #N` in place of `Closes #N` (in the commit and the pull request), list the unmet criterion under Caveats, and leave the issue open with a comment naming what remains. Write "#N stays open until …": GitHub closes the issue on a closing verb (close, fix, resolve, in any tense) before `#N` anywhere in the description or a commit message, a sentence in Caveats included. Check the issue's state after the merge.
