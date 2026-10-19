---
name: continuous-learning
description: Distill soundness bugs, spec discrepancies, review corrections, and environment pitfalls into permanent repository gates, conformance vectors, and agent skills. Use after completing an issue, fixing a bug, resolving review feedback, or hitting a recurring pitfall.
---

# Continuous Learning & Self-Improvement

Every bug, review round, spec ambiguity, or tool friction in Tessera has a structural root cause. Fixing the immediate instance is only half the job: if the harness, specification, or agent context is not updated, a future session will repeat the mistake.

## The Escalation Hierarchy

When capturing a lesson, always encode it at the **strongest level** that fits, ordered from strongest (mechanical) to weakest (prose):

| Rank | Surface | When to use | Example |
|---|---|---|---|
| **1 (Strongest)** | **Compiler type or lint** (`Secret<T>`, `#![deny(clippy::...)]`, `secret-access.toml`) | An invariant can be expressed in Rust's type system or Clippy lints so violations fail `cargo check` / `cargo clippy`. | Wrapping witness values in `Secret<T>` and denying `expose_secret` via `clippy.toml`. |
| **2** | **`xtask` repository gate** (`xtask/src/*.rs` + self-test in `xtask/fixtures/`) | A structural, dependency, licensing, or agent-context rule spans files or workspaces. | `xtask/src/isolation.rs` rejecting prover dependencies in `verifier/`; `xtask/src/agent_context.rs` checking `.claude/` and `.codex/` symlinks. |
| **3** | **Adversarial unit test or conformance vector** (`spec/vectors/`, `crates/*/tests/`, `verifier/crates/*/src/`) | A protocol check, codec boundary, or algebraic identity could regress or diverge between prover and verifier. | Adding a tampered-transcript test that asserts `RejectReason::Sumcheck` when a blinding evaluation is omitted. |
| **4** | **Normative specification & RFCs** (`SPEC.md`, `docs/spec/`, `docs/rfcs/`) | An implementation uncovered an underspecified label, off-by-one equation, or missing requirement in an RFC. | Updating `RFC-0008` and `RFC-0014` in the same PR when separating `tree/{l}/lambda` from `tree/{l}/gamma` (`#62`). |
| **5** | **Specialized skill or reference** (`.agents/skills/<skill>/`) | A domain-specific audit check, attack vector, or optimization pattern applies whenever that workflow is triggered. | Adding the blinding-claim absorption check to `soundness-review/SKILL.md` and `redteam-scheme/attack-catalog.md`. |
| **6 (Last resort)** | **Always-loaded `AGENTS.md`** | A repository-wide pitfall or command rule affects every session regardless of task. | Documenting `gh api` workarounds for classic Projects deprecation in `AGENTS.md#pitfalls`. |

Never put domain-specific checklists into `AGENTS.md` when they belong in a specialized skill (`Rank 5`), and never write prose in a skill (`Rank 5`) when an `xtask` gate (`Rank 2`) or test (`Rank 3`) can enforce the invariant mechanically.

## Distillation Loop

Run this four-step loop at the end of every non-trivial issue (`Step 7` of `issue-workflow`), soundness audit, or debugging session.

### Step 1 — Extract the Root-Cause Delta

Ask four concrete questions about the session:
1. **Did a check or test miss a bug initially?** What exact input or mutation slipped through?
2. **Did the specification and code disagree, or was an RFC ambiguous?** Which section, equation, or transcript label needed interpretation?
3. **Did a reviewer or gate reject an earlier iteration?** What assumption caused the false start?
4. **Did a command, build step, or tool fail unexpectedly?** Was it a missing flag, workspace boundary, or environment quirk?

If the answer to all four is "no—everything worked on the first pass with existing rules," stop here. Do not add noise or restate existing rules.

Done when you have a list of 1–3 specific root-cause deltas (or confirmed zero deltas).

### Step 2 — Classify Each Delta in the Escalation Hierarchy

For each delta from Step 1, select the highest applicable rank (1 through 6):

- **Protocol / soundness bug found** -> Rank 3 (adversarial test) **and** Rank 4 (RFC clarification if spec was ambiguous) **and** Rank 5 (add the pattern to `soundness-review`, `soundness-audit`, or `redteam-scheme`).
- **Codec or field edge case found** -> Rank 3 (add case to `spec/vectors/generate.py` and run `cargo xtask gen-vectors`).
- **Repo hygiene or structural drift** -> Rank 2 (add or tighten check in `xtask/src/` with a negative fixture test).
- **Performance pitfall or hardware bottleneck** -> Rank 5 (record in `performance-optimization/kernels-and-roofline.md`).
- **Universal repo command pitfall** -> Rank 6 (add a single concise bullet to `AGENTS.md#pitfalls`).

Done when every delta is mapped to a concrete file and rank.

### Step 3 — Codify Concisely (Context Hygiene Rules)

When editing `.agents/skills/`, `.agents/workflows/`, or `AGENTS.md`:

1. **Co-locate with existing rules.** Update the existing checklist item or table row rather than appending a chronological "Lessons Learned" log at the bottom of a file.
2. **State the positive rule and the exact failure mode.** Write what the code *must* do and cite the issue/PR or invariant ID (`RFC-0008`, `INV-FS-1`, `#62`, `#237`).
3. **Keep `AGENTS.md` lean.** `AGENTS.md` is loaded into every conversation. Every line added to `AGENTS.md` must earn its token cost across all tasks; move detailed procedures into `.agents/skills/<name>/`.
4. **Edit `.agents/` only.** Never edit `.claude/` or `.codex/` directly; they are symlinks to `.agents/` verified by `cargo xtask lint`.

Done when the rule, test, or gate is implemented in its canonical location with zero duplicate prose.

### Step 4 — Verify the New Guardrail

1. If you added a test or vector: mutate the code under test (using `harness-engineering/scripts/mutate.py` or a manual edit) and confirm the new test fails.
2. If you updated a skill, workflow, or `AGENTS.md`: run `cargo xtask lint` to verify all Markdown links, SPDX headers, and `.agents/` symlinks pass.

Done when the new test kills its target mutant and `cargo xtask lint` passes cleanly.
