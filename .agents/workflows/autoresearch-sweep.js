// Multi-agent autoresearch sweep workflow for Tessera.
// Runs parallel research subagents across open frontier problems (agenda -> survey -> ideate -> experiment),
// then synthesizes findings into ranked recommendations and ledger updates.

export default {
  name: "autoresearch-sweep",
  description:
    "Parallel autonomous cryptographic research sweep across Tessera's frontier agenda (F1-F6)",
  async run(ctx) {
    const target = ctx.input?.frontier || "all open P0/P1 frontiers (F1, F2, F3)";

    const triage = await ctx.agent({
      role: "Research Agenda Coordinator",
      prompt: `Load .agents/skills/autoresearch/SKILL.md and .agents/skills/autoresearch/agenda.md.
Inspect the repository specifications (SPEC.md, docs/rfcs/RFC-0011-advice-commitment.md, docs/rfcs/RFC-0008-logup-and-tables.md, docs/rfcs/RFC-0009-gadgets.md, docs/rfcs/RFC-0014-interaction-and-soundness.md) and research/AGENDA.md if present.
Select up to 3 highest-leverage sub-questions for: ${target}.
Return a JSON array of objects with fields: id, title, rfc, hypothesis, baselineMetric, falsificationTest.`,
    });

    const tracks = ["Track 1 (PCS & Opening Size)", "Track 2 (Lookup & GKR Degree)", "Track 3 (Gadget Decomposition & Range Proof)"];
    const results = await Promise.all(
      tracks.map((track) =>
        ctx.agent({
          role: `Cryptographic Researcher — ${track}`,
          prompt: `Follow .agents/skills/autoresearch/SKILL.md (survey.md, ideate.md, experiment.md).
Assigned track: ${track}.
Context from agenda triage:
${triage}

1. Formulate a concrete mathematical hypothesis with explicit parameter equations over Goldilocks (p = 2^64 - 2^32 + 1, extension Fp4).
2. Check soundness impact against RFC-0014 (epsilon_total <= 2^-124), zero-knowledge impact against RFC-0013, and verifier complexity against RFC-0019.
3. Write and run a self-contained Python 3 verification/cost model script to compute exact proof size (KB), prover field operations, and soundness bits before vs after.
4. Return a structured report with Verdict (KEEP / DISCARD / INCONCLUSIVE), exact metrics table, and required RFC changes if KEEP.`,
        })
      )
    );

    return await ctx.agent({
      role: "Principal Cryptographer & Synthesizer",
      prompt: `Follow .agents/skills/autoresearch/promote.md.
Synthesize the parallel research track findings below into a unified evaluation table.
Rank surviving proposals by proof-size and prover-time reduction per unit of specification complexity, flag any soundness risks, and draft the ledger / RFC promotion plan.

Track findings:
${results.join("\n\n---\n\n")}`,
    });
  },
};
