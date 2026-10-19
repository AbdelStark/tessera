# Ideate and triage

Output: idea notes `research/ideas/IDEA-NNNN-<slug>.md`, shaped by the template (`research/templates/idea.md`), each ending `triaged` or `rejected`, and the ledger's ideas table.

1. **Collect seeds** from: the surveys named (or every `current` survey) and their "Candidate ideas"; the sensitivity tables of baseline.md §2 (`research/baseline.md#2-the-cost-model`), which say which term is worth moving at which context; the risks of [00 §9](../../../docs/spec/00-overview.md#9-risks) and the [open questions](../../../SPEC.md#open-questions); the research bets of source §9; the "Reopens if" column of baseline.md §7 (`research/baseline.md#7-already-rejected`). Done when the seed list holds at least ten entries or every source above is exhausted.
2. **Generate through lenses**, one pass each: commitment (`c_c`), lookup (`c_ℓ`, `K_lin`), max certificate (`c_π`), counts (`C_lin`, `K_lin` through the compiler), attention (`A`), passes (`regen`, `stream`), parameters (`τ`, size, verify), zero knowledge, hardware mapping, and one wild lens: a different paradigm (approximate sumcheck, proof-aware `D` in bounded mode, folding, distributed proving, another commitment family). Done when every lens produced an idea or a sentence saying why it cannot this round.
3. **Write each idea** in the template with every cell filled: a mechanism precise enough to implement; the expected gain as a cost-model scenario with its command and table; the security gate rows (`pass`, `fail` or `unknown`, each with the reason); the prior-rejection check against baseline §7 and the RFC's Alternatives Considered; the kill criterion; the cheapest experiment with the evidence level reachable here. Done when no cell is empty and no number lacks an evidence level.
4. **Refute** every idea through four lenses, each recorded in the note's Refutations table:
   - *soundness*: as ADV-P with a grinding budget and as ADV-V: a second witness, a challenge drawn before the message it depends on is fixed, a leaked bit, a verifier that must trust the prover;
   - *prior art*: already in the specification, already rejected (which item), or excluded by a known result such as the lookup separation principle [Jo26];
   - *cost model*: does the term it moves matter at the context it targets, and is the gain larger than the uncertainty of the inputs;
   - *feasibility*: hardware, dependencies, tier-A surfaces, effort.
   In a sweep the refuters are agents who did not write the idea. Done when every idea has four verdicts.
5. **Score and decide** with the rubric; set `triaged`, or `rejected` with the reason and the item cited. Update the ledger's ideas table; the highest-priority idea whose experiment is feasible here becomes the queue's next `experiment` action. Done when the ledger reflects every idea.

## Rubric

| Factor | How to set it |
|---|---|
| Gain | Δμ in percent at the context the idea targets (16K unless it targets short context), from the cost model; or the scoreboard metric moved (proof size, verification time, soundness bits) as a percentage. Quote the evidence level of the inputs. |
| Confidence | 0.9: mechanism and numbers from primary sources on comparable hardware. 0.6: mechanism sound, numbers from other hardware or another field. 0.3: mechanism plausible, numbers extrapolated. 0.1: speculative. |
| Cost | `s`: days, no RFC (weight 1). `m`: a week, an RFC below tier A (weight 2). `l`: weeks, a tier-A RFC, or an H100 before anything is known (weight 4). |
| Priority | gain × confidence / cost weight. Ties go to the idea testable here now. |

Hard gates override the score: a `fail` in the security gate, an impossibility result, or a prior rejection with nothing new make the idea `rejected`, with the reason in the note.
