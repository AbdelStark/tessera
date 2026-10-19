// Multi-agent soundness audit workflow for Tessera.
// Dispatches parallel auditors across all five audit surfaces of .agents/skills/soundness-audit/SKILL.md
// and synthesizes a formal audit report with concrete PoC tests for any finding.

export default {
  name: "soundness-audit-sweep",
  description:
    "Parallel 5-surface mathematical and implementation soundness audit across SPEC, crates/, and verifier/",
  async run(ctx) {
    const scope = ctx.input?.scope || "full repository (docs/rfcs/, crates/, verifier/)";

    const surfaces = [
      {
        name: "Surface 1 — Fiat-Shamir & Transcript Schedule",
        focus:
          "Audit RFC-0004, RFC-0014, crates/tessera-transcript, crates/tessera-pcs, crates/tessera-sumcheck, crates/tessera-logup, and verifier/crates/tv-core for transcript ordering, blinding claim absorption (#237), label separation (#62), and prover/verifier schedule lockstep.",
      },
      {
        name: "Surface 2 — Algebraic Reductions (Sumcheck, GKR, LogUp, PCS)",
        focus:
          "Audit RFC-0008, RFC-0010, RFC-0011, RFC-0012, crates/tessera-sumcheck, crates/tessera-logup, crates/tessera-pcs, and verifier/crates/tv-core for degree bounds, zero-denominator guards, stacked tree root points (#58), and Ligero column opening consistency.",
      },
      {
        name: "Surface 3 — True-Integer Range Analysis, Gadgets & Lemma 0",
        focus:
          "Load .agents/skills/soundness-audit/uniqueness-and-bounds.md. Audit RFC-0006, RFC-0009, crates/tessera-ir, crates/tessera-gadgets, crates/tessera-tables, and verifier/crates/tv-tables for centered range overflows, carry uniqueness, tie-breaking rules, and SP-7 public range re-derivation.",
      },
      {
        name: "Surface 4 — Canonical Encoding & Verifier Robustness",
        focus:
          "Audit RFC-0003, RFC-0019, crates/tessera-codec, verifier/crates/tv-codec, and verifier/crates/tv-core for INV-TCE-1..4, non-canonical field rejection, trailing bytes, pre-allocation Limits enforcement, and panic-freedom (SP-8).",
      },
      {
        name: "Surface 5 — Zero-Knowledge, Blinding & Secret Redaction",
        focus:
          "Load .agents/skills/soundness-audit/zk-and-secrets.md. Audit RFC-0013, 05-observability.md, secret-access.toml, and all crates for b >= t + q_row blinding sufficiency, Secret<T> redaction (INV-ARCH-6), and verifier independence (RFC-0001).",
      },
    ];

    const reports = await Promise.all(
      surfaces.map((s) =>
        ctx.agent({
          role: s.name,
          prompt: `Load and execute .agents/skills/soundness-audit/SKILL.md for scope: ${scope}.
Focus area: ${s.focus}
Verify every claim against actual source lines and test coverage. Report each surface item as VERIFIED, UNTESTED, or VULNERABLE with exact file:line citations.`,
        })
      )
    );

    return await ctx.agent({
      role: "Lead Soundness Auditor",
      prompt: `Combine the 5 surface audit reports below into the formal Audit Report Format defined in .agents/skills/soundness-audit/SKILL.md.
Prioritize any VULNERABLE or UNTESTED findings by severity (Critical / High / Medium / Low) and specify the exact adversarial regression tests required.

Surface reports:
${reports.join("\n\n---\n\n")}`,
    });
  },
};
