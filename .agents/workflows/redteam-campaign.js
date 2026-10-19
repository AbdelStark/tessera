// Multi-agent adversarial red-team campaign workflow for Tessera.
// Spawns specialized exploit-development subagents targeting each protocol layer
// to construct forged proofs, mutations, or parser bombs against verifier/.

export default {
  name: "redteam-campaign",
  description:
    "Parallel adversarial exploit campaign attempting to forge accepted proofs or trigger verifier DoS/panics",
  async run(ctx) {
    const target = ctx.input?.target || "all protocol layers (Codec, Transcript, Sumcheck, LogUp, PCS, Gadgets)";

    const cells = [
      {
        role: "Red-Team Cell A — Transcript & Batching Forgery",
        vector:
          "Vector 1 (Fiat-Shamir grinding & unbound claims) and Vector 2 (Sumcheck & batching cancellation) from .agents/skills/redteam-scheme/attack-catalog.md.",
      },
      {
        role: "Red-Team Cell B — LogUp, GKR & Gadget Aliasing",
        vector:
          "Vector 3 (LogUp & Projective GKR fraction attacks) and Vector 5 (Field wrap-around & Lemma 0 non-uniqueness) from .agents/skills/redteam-scheme/attack-catalog.md.",
      },
      {
        role: "Red-Team Cell C — Ligero PCS & Codec/Parser Bombs",
        vector:
          "Vector 4 (Ligero PCS & Merkle commitment attacks) and Vector 6 (TCE canonical encoding & parser bombs) from .agents/skills/redteam-scheme/attack-catalog.md.",
      },
    ];

    const cellReports = await Promise.all(
      cells.map((cell) =>
        ctx.agent({
          role: cell.role,
          prompt: `Load .agents/skills/redteam-scheme/SKILL.md and .agents/skills/redteam-scheme/attack-catalog.md.
Target: ${target}.
Assigned attack vectors: ${cell.vector}

1. Inspect the verifier implementation in verifier/crates/ and existing adversarial tests.
2. Run .agents/skills/harness-engineering/scripts/mutate.py --dry-run on the relevant verifier modules to identify conditional checks, and check whether each check has an explicit adversarial test asserting the exact RejectReason or CodecError.
3. Attempt to construct an exploit input or identify any surviving mutant where a corrupted proof is not rejected with its specific error code.
4. Report every attempted exploit, the verifier line that blocked it (or failed to block it), and any surviving mutants.`,
        })
      )
    );

    return await ctx.agent({
      role: "Red-Team Commander",
      prompt: `Compile the red-team cell reports below into the Campaign Report Format specified in .agents/skills/redteam-scheme/SKILL.md.
List all confirmed vulnerabilities, surviving mutants (missing adversarial tests), and hardened defenses.

Cell reports:
${cellReports.join("\n\n---\n\n")}`,
    });
  },
};
