# Research knowledge base

Non-normative. The specification ([SPEC.md](../SPEC.md), [docs/spec/](../docs/spec/), [docs/rfcs/](../docs/rfcs/)) says what Tessera is. This directory records what we have learned about how it could be faster, smaller or stronger, and what we tried. A result changes the design only through an RFC ([CONTRIBUTING.md](../CONTRIBUTING.md#rfcs)).

The `autoresearch` skill ([.claude/skills/autoresearch/SKILL.md](../.claude/skills/autoresearch/SKILL.md)) runs the loop that fills this directory: survey the state of the art, generate and triage ideas, test the best with a proof of concept, promote what survives, and record everything here so that the next iteration starts where the last one stopped. The goal it serves: a design whose measured cost, proof size, verification time, soundness, zero knowledge and statement strength beat every known proof-of-inference protocol, with the evidence to show it.

## Layout

| Path | Holds | Identifier |
|---|---|---|
| [LEDGER.md](LEDGER.md) | The state: every survey, idea and experiment with its status, the queue of next actions, the iteration log. Read first, update last. | |
| [baseline.md](baseline.md) | Tessera's design as a cost structure, the scoreboard that defines "beats every known protocol" in numbers, the levers, and the alternatives the RFCs already rejected | |
| [landscape.md](landscape.md) | Known systems and protocols with their measured or claimed numbers | |
| [bibliography.md](bibliography.md) | Every source cited in this directory, one line each | `[Key2026]` |
| [surveys/](surveys/) | State-of-the-art surveys, one per agenda topic | `SUR-NNNN` |
| [ideas/](ideas/) | Candidate improvements: mechanism, expected gain, security gate, kill criterion, refutations, score | `IDEA-NNNN` |
| [experiments/](experiments/) | Proof-of-concept experiments: a pre-registered report and its code, one directory each | `EXP-NNNN` |
| [tools/](tools/) | Shared research tooling: the cost model, the proximity-parameter calculator, the knowledge-base checker (`check_kb.py`: citation keys, ledger rows and status words, counters, pre-registration sections; run with lint at the end of every unit of work, and by CI's `checks` job), and `sampling_streams.py`, the measurements behind RFC-0061's choice of sampling stream | |
| [templates/](templates/) | The shape of a [survey](templates/survey.md), an [idea](templates/idea.md) and an [experiment](templates/experiment.md) | |

Identifiers are sequential, zero-padded and never reused; the ledger's counters assign them.

## Status vocabulary

| Kind | States |
|---|---|
| Survey | `current (YYYY-MM-DD)`, `stale` (a staleness trigger in the survey fired, or older than six months) |
| Idea | `proposed` → `triaged` → `experimenting` → `validated`, `refuted` or `inconclusive` → `promoted` (RFC draft or issue exists), `shelved` (blocked on hardware or a dependency, says which) or `rejected` (says why) |
| Experiment | `planned` (pre-registered) → `running` → `done: validated`, `done: refuted` or `done: inconclusive` |

## Evidence levels

Every number carries one. A number without an evidence level is a defect in the note that holds it.

| Level | Meaning |
|---|---|
| E0 | Analytic: operation counts, bounds, the cost model |
| E1 | Measured on a CPU here (the machine is named in the report) |
| E2 | Measured on the target GPU (H100 SXM) under the RFC-0020 protocol |
| E3 | Measured end to end on a real proof |

Sources carry one of `primary` (paper, code, our own measurement), `secondary` (talk, blog post, documentation) or `claim` (a vendor figure nobody outside has reproduced).

## Rules

1. **Soundness first.** An idea that lowers any soundness bound below G-4, admits a second witness upstream of the logits (SP-3), weakens zero knowledge (G-5) or couples the verifier to prover code (SP-6) is `rejected`, whatever it saves. The security gate of every idea note says so explicitly.
2. **Nothing here is a dependency.** No crate under `crates/` or `verifier/` depends on anything under `research/`. An experiment may depend on prover or verifier crates by path to reuse the field or the hash. A Rust experiment is its own Cargo workspace (an empty `[workspace]` table in its manifest), so the root workspace and its lockfile stay untouched.
3. **Pre-registration.** An experiment's hypothesis and success criterion are written and committed before its code. Amendments are appended with a date, never edited in place.
4. **Same harness.** A comparison measures the baseline and the candidate in the same code, on the same machine, on the same inputs. Numbers quoted from papers on other hardware are context, never the baseline.
5. **Gates still apply.** SPDX headers on every `.rs`, `.py`, `.cu` file; `license = "Apache-2.0 OR MIT"` in every `Cargo.toml` and `pyproject.toml`; every Markdown link resolves (`cargo xtask lint` checks all three).
6. **No secrets.** No model weights, prompts or private data; experiments run on synthetic inputs from seeded generators.
7. **Projections are labeled.** Tessera's own numbers are projections until the RFC-0020 measurements replace them (08 §1).

Historical experiments EXP-0001 through EXP-0011 were imported at commit `2410757`, whose production parent is `1505d63`. Their original outputs and environment lines describe that checkout; use it to reproduce the historical tables. New dated amendments and experiments explicitly compare current production. The current research model follows the compiler reports and may produce different planning totals.

## Running the loop

The slash commands and saved workflow are entry points for the original harness. Other agents can invoke the same skill and ledger directly using their native tools. The workflow's orchestration can be checked without API calls with `node research/tools/check_workflow.mjs`; `python3 -I research/tools/check_loop.py` checks the optional Claude CLI runner's exit status and overlap lock. Neither check spends API budget or installs a schedule. A planned experiment pauses before code until its frozen pre-registration is committed; preparation can run in parallel, CPU measurements run serially.

- `/autoresearch` does one unit of work, chosen from the ledger's queue; `/autoresearch survey <topic>`, `/autoresearch ideate`, `/autoresearch experiment IDEA-NNNN` and `/autoresearch promote IDEA-NNNN` choose it by hand.
- `/autoresearch-sweep` runs the parallel workflow in [.claude/workflows/autoresearch-sweep.js](../.claude/workflows/autoresearch-sweep.js): surveys over several agenda topics at once, ideation from several lenses, adversarial triage, optional experiments, consolidation.
- Continuous improvement is the loop rerun: `/loop 24h /autoresearch` inside a session, or [tools/loop.sh](tools/loop.sh) from a scheduler outside one (cron, launchd, a CI timer; `LOOP_MODE=sweep` runs the parallel sweep, `LOOP_BUDGET_USD` caps a run; [tools/launchd/com.tessera.autoresearch.plist](tools/launchd/com.tessera.autoresearch.plist) is a ready launchd agent for every six hours, installed by the maintainer because it spends API budget unattended). Every run is bounded (one unit of work) and ends with the ledger consistent (`tools/check_kb.py` and `cargo xtask lint` pass), so runs compose. The loop edits files only; the maintainer reviews `git status` and commits through the issue workflow.
