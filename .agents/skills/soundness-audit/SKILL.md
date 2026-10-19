---
name: soundness-audit
description: Deep, whole-component or cross-RFC audit of cryptographic soundness, witness uniqueness, Fiat-Shamir binding, zero-knowledge simulation and F65537 calibration. Use when asked to audit a crate, protocol stage, RFC or release milestone for soundness or zero-knowledge defects.
---

# Soundness audit

A soundness bug is the worst bug this repository can have: it certifies a cheating prover ([06 §2](../../../docs/spec/06-security.md#2-adversaries)). While `soundness-review` checks a single diff before merge, this skill audits a whole component, protocol stage or RFC boundary from first principles. Every step ends on a checkable criterion; an unverified claim is an open finding.

1. **Map the surface.** List the crates, modules and RFC sections in scope, every transcript label the component absorbs or squeezes ([RFC-0014](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#transcript-schedule)), every row of the soundness budget it contributes to ([RFC-0014](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#soundness-budget)), every `RejectReason` or `ErrorCode` it can emit ([04](../../../docs/spec/04-error-model.md)), and every secret-class type it touches ([05 §3](../../../docs/spec/05-observability.md#3-data-classification-and-redaction)). Done when the inventory table has no unmapped public entry point or protocol message.
2. **Audit Fiat–Shamir causality and spec-to-code equivalence.** Trace the absorb/squeeze DAG in both `crates/` and `verifier/` against the RFC-0014 schedule:
   - Every prover degree of freedom (commitment root, round message, tree evaluation, combined row, blinding evaluation `q_g` or `q_T`, scoring claim `Û`) is absorbed as canonical TCE bytes before any challenge that combines or checks it is squeezed (INV-FS-1).
   - No challenge is reused across distinct algebraic roles: batching challenges (`gamma`), fraction-tree numerator/denominator combiners (`lambda`), masked-sumcheck combiners (`zk-rho`) and opening combiners (`xi`) are independent squeezes.
   - Every value the specification says the verifier derives (table digests, ranges, padding terms, `eq` and `LT` evaluations, root points, query indices,check order) is computed by the verifier from public inputs, never read from the proof.
   - Owed claims returned by an intermediate check (`Reduction`, `Batch`, `LevelClaim`, leaf claims) are `#[must_use]` and consumed by the next stage or opening check.
   Done when every equation and transcript step in the code is matched line for line against the RFC, with `file:line` citations.
3. **Audit integer bounds, Lemma 0 and witness uniqueness.** Apply [uniqueness-and-bounds.md](uniqueness-and-bounds.md) to every integer-to-field conversion, lookup key, gadget relation and padding rule in scope. Done when every modular congruence has a written bound below `(P − 1)/2` (`< 2^62`), every gadget has an alternative-witness test that fails when substituted into the honest prover, and every table or tree padding entry is proven harmless by construction.
4. **Audit zero knowledge and secret isolation.** Apply [zk-and-secrets.md](zk-and-secrets.md) to every revealed field element, polynomial, combined row, opened column and error path in scope. Done when every revealed value is accounted for in the simulator outline of [RFC-0013](../../../docs/rfcs/RFC-0013-zero-knowledge.md#simulator-argument), every evaluation point is checked for Boolean coordinates where `Z(x) = Π x_i(1 − x_i)` vanishes, and every secret path satisfies `OBS-1..4` and `INV-ARCH-2`.
5. **Run differential, mutation and calibration campaigns.** Follow `harness-engineering`:
   - *Differential*: verify that the independent Python oracle (`spec/vectors/*/generate.py`), the prover crate and the verifier crate agree on every golden and rejection vector, and on seeded random instances.
   - *Mutation*: run the mutation harness (`.agents/skills/harness-engineering/scripts/mutate.py`) in a scratch copy over every check in scope, with a no-op control. Every non-equivalent mutant must be caught.
   - *`F65537` calibration*: where the component has an algebraic soundness bound (`d/|F|`, `(N + T)/|F|`, `s/|F|`, `((1 + ρ')/2)^τ`), run a cheating prover over the test field `F65537` (`P_test = 65537`) and verify that empirical false acceptance lies at or below the analytic bound within a 99% binomial confidence interval ([07 §4](../../../docs/spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests)).
   Done when all three campaigns have run and their raw counts are recorded.
6. **Report and lock down.** Classify each finding by [06 §8](../../../docs/spec/06-security.md#8-disclosure-policy): Critical (soundness or ZK break, false `Accept`), High (verifier resource exhaustion or fault-attribution error), Medium (prover crash or completeness gap), Low (spec drift or missing test). For any suspected exploit, run `redteam-scheme` to produce a minimal PoC. Turn every finding into a red-then-green test, a spec amendment or a tracked issue, and apply `continuous-learning` so the class of bug cannot return. Done when the audit report lists every checked invariant with `file:line` and test evidence, and every open finding has a reproducer and an issue or fix.

## Parallel sweep

For a multi-agent audit across all four axes simultaneously (`fiat-shamir-and-spec`, `uniqueness-and-bounds`, `zk-and-secrets`, `oracles-and-mutants`), run the `soundness-audit-sweep` workflow ([../../workflows/soundness-audit-sweep.js](../../workflows/soundness-audit-sweep.js)).
