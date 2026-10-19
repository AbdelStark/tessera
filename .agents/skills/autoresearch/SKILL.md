---
name: autoresearch
description: The research loop that improves Tessera's scheme with evidence: survey the state of the art on a topic, ideate and adversarially triage improvements, run a pre-registered proof-of-concept experiment, promote a validated result into an RFC draft, and record everything under research/. Use when asked to research techniques or papers, to speed up or shrink the proof or its verification, to evaluate a protocol or system against Tessera, to run an experiment, or to continue the autoresearch loop.
---

# Autoresearch

One invocation does one unit of work and leaves `research/README.md` consistent, so that invocations compose into continuous improvement. The knowledge base's rules (evidence levels, soundness first, pre-registration, same harness) are in its README; the thing to beat is `research/baseline.md`; the topics are in [agenda.md](agenda.md).

Modes, given as arguments: `survey <topic>`, `ideate [SUR-NNNN ...]`, `experiment IDEA-NNNN`, `promote IDEA-NNNN`. With none, the ledger chooses.

1. **Orient.** Read `research/LEDGER.md`: counters, queue, open items. Read the scoreboard and the sensitivity tables of `baseline.md`, so you know which term is worth moving at which context. With no mode given, take the first queue entry whose mode is feasible on this machine (no CUDA here: work that needs E2 evidence is shelved with that reason, not attempted). Done when one line states the unit of work, the identifier it creates or moves, and why it is the highest-value feasible item.
2. **Work the phase** by its file: [survey.md](survey.md), [ideate.md](ideate.md), [experiment.md](experiment.md), [promote.md](promote.md). Each ends on its own completion criterion; stop there.
3. **Consolidate.** Update the ledger: status lines, counters, the queue reordered by what was learned, an iteration-log row. Add every new source to `bibliography.md`; move every new or changed number into `landscape.md`. Run `python3 -I research/tools/check_kb.py` (citation keys resolve, every note has its ledger row and the same status word, counters ahead of every identifier, experiment reports pre-registered) and `cargo xtask lint` from the repository root (Markdown links, SPDX headers, manifest licenses). Done when both pass and the queue's first entry names the next unit of work with its reason.
4. **Report.** The final message says what was learned (every number with its evidence level), what changed in the knowledge base (paths), and what the next run does. The ledger is the record; the message points at it.

## Standards in every phase

- **Evidence on every number**: E0–E3 and primary, secondary or claim, as the README defines them. Tessera's own figures are projections until the RFC-0020 gate measures them; say so each time.
- **Soundness before gain.** An idea's security gate decides before its Δμ does; a `fail` row rejects the idea whatever it saves (AGENTS.md: a soundness bug is the worst bug this repository can have). Apply the `soundness-review` checklist to any mechanism that changes what the verifier accepts.
- **Hardware honesty.** This machine reaches E0 (analytic) and E1 (CPU). Name what an H100 run would settle; never extrapolate a GPU number from a CPU one.
- **Primary sources, dated.** Follow a claim to the paper or the code; record the URL and the date checked; a vendor figure stays a `claim` until reproduced.
- **The specification is normative, the knowledge base is not.** When evidence contradicts an RFC's decision, record it in the note and in the report; the design changes through an RFC ([promote.md](promote.md)), never by editing the knowledge base alone.
- **Scratch outside the knowledge base.** Downloads and intermediate files go to the session scratchpad; only notes, tools and experiment code enter `research/`. Move files with `/bin/mv -f` and copy with `/bin/cp -f`; delete nothing.
- **No git writes.** The loop edits files. Branching, committing and pull requests belong to the maintainer, through `issue-workflow`.

## Running continuously

The saved workflow uses the original harness's `agent`, `parallel`, `pipeline` and `phase` helpers. When those are unavailable, execute the same phases with the host's native agent tools; the skill and ledger define completion. Serialize CPU measurements even when preparation is parallel. The headless shell runner requires the Claude CLI; its stub checks do not establish a live scheduled run.

`/loop 24h /autoresearch` repeats one bounded unit of work per day. The `autoresearch-sweep` workflow ([.claude/workflows/autoresearch-sweep.js](../../workflows/autoresearch-sweep.js)) runs surveys, ideation, triage and experiments in parallel for a whole pass over the agenda; `/autoresearch-sweep` with `{"topics": ["lookups", "passes"], "experiments": 1}` narrows it. Outside a session, `research/tools/loop.sh` runs one unit headlessly (`LOOP_MODE=sweep` for the whole pass) so that cron, launchd or a CI timer can keep the loop going; it edits files and never touches git. Stopping between runs loses nothing: the ledger holds the state.
