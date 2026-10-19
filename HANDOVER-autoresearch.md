# Handover: the Tessera autoresearch loop

Written on 2026-10-19 by the session that built the loop (session `aed06211-62ae-4636-920a-02942416f73e`), for the agent that continues it. It is committed on its own, as the branch's last commit, so that the next agent can read it from the branch; drop that commit before the research pull request merges.

**Status when written.** The writing session relaunched EXP-0012 on 2026-10-19 at about 09:45 CEST. The user stopped that run a few minutes later, while it was still reading its inputs, and it left nothing on disk. The other two runs of section 4.2 were not relaunched, and the re-sync of section 4.1 was not started. The user then asked for the state to be committed and pushed, and for the work to continue with another agent. No agent, workflow or scheduled job of the writing session is running.

## 1. The request and where it stands

The user's goal, verbatim:

> i want you to create a skills / workflows to basically do a huge sota research etc on techniques that could help us speed up / improve etc our scheme and design for autoresearch. end to end workflow including the research of the state of the art and new techniques, thinking about potential new stuff that could help us, implementing a POC or something to build some confidence level on the potential direction etc, and the flow must consolidate a knowledge base in the repo for the learnings / experiments etc. goal is to have self continuous improvements and be able to iterate on the design and improve it until we get the absolute new state of the art that beat absolutely all known protocols and schemes.

The machine the goal asks for exists and has run. It is a skill, a parallel sweep workflow, a knowledge base with its own consistency checker and research cost model, and a headless runner for an OS scheduler. It has produced three sweeps (13 surveys, 33 ideas), 11 judged experiments, one Draft RFC, and specification findings that the backlog session verified, filed and closed upstream.

The end state of the goal is not reached and must not be claimed. Every number is E0 (analytic) or E1 (CPU wall clock on this Mac). The decisive unit costs need an H100 (E2, the RFC-0020 v0.1 gate), which this machine cannot supply. Under the `counted` cost profile the projected proof multiplier misses the SC-2 ceilings by about 1.8×. The goal is open-ended: take units of work from the queue rather than try to finish it in one session. The original session ran under `/goal` with a Stop hook that blocked stopping until the goal held; a new session has no such hook unless the user sets one.

## 2. Where everything is

| What | Where |
|---|---|
| Worktree | `/Users/abdel/dev/me/tessera-research`, branch `research/autoresearch`, base `1505d63` (2026-10-16) |
| Committed state | Two commits on top of the base, pushed to `origin/research/autoresearch` on 2026-10-19: the research loop (AGENTS.md, `.claude/skills/autoresearch/`, `.claude/workflows/`, RFC-0021, `research/`), then this file. Not rebased onto main; no pull request opened. |
| Upstream | `origin/main` at `11e104c` (pull request #380), 206 commits past the base on 2026-10-19 |
| Main checkout | `/Users/abdel/dev/me/tessera` belongs to the backlog session `tessera-ba`, which switches branches there. Never work in it; never use a bare `git stash` (the stash stack is shared). |

The pieces:

- **Skill.** [.claude/skills/autoresearch/SKILL.md](.claude/skills/autoresearch/SKILL.md) with `survey.md`, `ideate.md`, `experiment.md`, `promote.md` and `agenda.md` (13 topics). `/autoresearch` with no argument takes the first feasible queue item; `survey <topic>`, `ideate`, `experiment IDEA-NNNN` and `promote IDEA-NNNN` choose by hand.
- **Workflow.** [.claude/workflows/autoresearch-sweep.js](.claude/workflows/autoresearch-sweep.js) runs Orient, Survey, Ideate (10 lenses), Write, Refute (4 lenses), Score, Experiment and Consolidate. Arguments: `{"topics": [...], "max_ideas": 10, "experiments": 1}`. One run takes about 2 hours, 33 agents and 6.6 to 6.8 million subagent tokens. The Workflow tool runs only when the user explicitly asks for workflows; the original goal did.
- **Knowledge base.** [research/README.md](research/README.md) gives the layout, the status vocabulary, the evidence levels E0 to E3 and rules 1 to 7. [research/LEDGER.md](research/LEDGER.md) is the state machine: counters, queue, survey, idea and experiment tables, iteration log; read it first and update it last. [research/baseline.md](research/baseline.md) holds the cost model (§2), the parameters (§4), the scoreboard (§5) and the already-rejected list (§7). Also `landscape.md`, `bibliography.md` (365 entries), `templates/`, `surveys/` (SUR-0001 to SUR-0013), `ideas/` (IDEA-0001 to IDEA-0033) and `experiments/` (EXP-0001 to EXP-0011 judged, EXP-0013 drafted).
- **Tools** in `research/tools/`: `cost_model.py`, `proximity.py`, `check_kb.py`, `loop.sh` and `launchd/com.tessera.autoresearch.plist` (section 7 says what each does).
- **Draft RFC.** [docs/rfcs/RFC-0021-johnson-regime-proximity.md](docs/rfcs/RFC-0021-johnson-regime-proximity.md), revision 3: uniform `Fp2` proximity coefficients squeezed right after the seed (new transcript row 3b), a Johnson-regime budget row with `E₀` from ArkLib's `johnsonExceptionCount`, the list-binding argument of EXP-0005 read adversarially twice, and parameter sets `live-64/2` (`τ_A` 130, `τ_W` 68) and `ni-100/2` (`τ_A` 163, `τ_W` 85). OQ-0021-4 is open for the maintainer.
- **AGENTS.md.** One repository-map row for `research/` and one paragraph in "How work is done".

## 3. First steps

1. Read `research/README.md`, `research/LEDGER.md` (the queue, then the last rows of the iteration log), `research/baseline.md` §2 and §5, and the skill's `SKILL.md` and `experiment.md`.
2. Run the gates from the worktree root. All three passed on 2026-10-19 after the ledger corrections of that day.

   ```sh
   python3 -I research/tools/check_kb.py
   cargo xtask lint
   python3 -I research/tools/cost_model.py --check
   ```

3. Look upstream, read-only: `git fetch origin`, `gh issue list --repo AbdelStark/tessera --state all --limit 50`, and `ListAgents` to see whether `tessera-ba` is still running.

## 4. Pending work, in order

### 4.1 Re-sync with main

Facts checked on 2026-10-19:

- Every issue the loop fed upstream is closed as completed (section 5).
- Main runs RFC-0020, then RFC-0022 to RFC-0046. RFC-0021 is unused there, so the Draft keeps its number.
- Main's RFC-0014 proximity row already reads `((1 + ρ')/2)^τ + ϑ·n/|Fp2|`, with "the field term about 100 bits, which no proof of work covers" (#254 and #103).
- Main has no `research/`, no `.claude/workflows/` and no `.claude/skills/autoresearch/`. On the paths the branch touches, main changed only `AGENTS.md` (it added a `python/` row to the repository map), plus two other skills the branch does not touch.

Steps:

1. The maintainer rebases the branch onto main. That is a git write: ask the user first. The expected conflict is the repository-map table of `AGENTS.md` only.
2. After the rebase, `cost_model.py --check` reads `spec/vectors/cost-model/cost-model.json` by itself; until then it uses its embedded table.
3. Reconcile the research cost model with the normative one, `tessera bench model --source-defaults [--json]` (RFC-0020). Issues #248 (the column-answer pass P5), #250 (streamed passes), #251 (weight column answers), #252 (the P4 row combination) and #266 (RISK-2's profile) changed what the normative model prices. `tessera-ba` also reported that the two models' inputs differ: 8.8 in the research model against 9.4 in the normative one at 1K for the `minimal` count. Record the deltas in baseline §2; where the two disagree, the normative model wins, as the tool's docstring says.
4. Read what landed on main, then set the statuses with that evidence: IDEA-0016 (#248), IDEA-0021 (#258; check whether `crates/tessera-sumcheck` now proves tree levels with the split `eq`, and note that #238 refactored `prove_forests` into `prove_trees`) and IDEA-0033 (#266) become `promoted`. Close queue items 1, 3, 4, 5 and 7, finish item 6's remainder (re-express IDEA-0011, IDEA-0031 and IDEA-0032 with IDEA-0028 against the 326,656 count), and reorder the queue.
5. Re-read RFC-0021 against main's RFC-0011 to RFC-0014 and the newer RFCs that touch the opening: RFC-0031 (commitment id), RFC-0036 (protocol messages), RFC-0045 (leaf layout) and RFC-0046 (advice segments). Two passages are known to be stale. The Draft calls today's field term `n/|Fp2|`, while main has `ϑ·n/|Fp2|`. Its layout sentence (the proximity row's blinding rows as the last two rows, non-zero rows in layout order) must match RFC-0046. The maintainer then opens its pull request through `issue-workflow`.

### 4.2 Relaunch the three interrupted experiments

On 2026-10-17 at about 04:25, three delegated runs stopped on an API usage-credit error. Their state:

| Run | Idea | Left on disk | Ledger |
|---|---|---|---|
| EXP-0012 | IDEA-0022, integer round window for the leaf level (priority 3.1) | nothing | row withdrawn; the identifier is reserved for this run, and the counter stays at EXP-0014 |
| EXP-0013 | IDEA-0026, delayed 128-bit reduction in the node sums, with IDEA-0030 folded in (2.3) | a complete draft `README.md` (status `planned`, criterion written); no `results/preregistration.txt`; no code | its `planned` row says the digest is not frozen |
| EXP-0009 amendment | IDEA-0019, the amortized ragged loop (3.7) | nothing | queue item 10 |

Run them before the rebase, or port their harnesses. The harnesses build against the worktree's crates by path (`../../../crates/...`). At `1505d63` those crates are the code EXP-0006 to EXP-0009 measured. Main has refactored the forest prover (#238) and closed #258 as completed, so EXP-0008's harness may not compile there, and its baseline would no longer be production code. Every report states its commit in the environment line.

Each run is one delegated agent with the brief of Appendix A; the three can run in parallel because they write separate directories. After each returns:

- replace its ledger row with the row it returns;
- update the idea rows (priority, status, experiment and decision cells), matching the notes' `| Status |` lines;
- add any new bibliography keys in their topic sections;
- add an iteration-log row;
- rerun the gates.

### 4.3 The rest of the queue

The ledger's queue is the authority; the items feasible here after 4.1 and 4.2 are:

- item 11, the IDEA-0015 calibration (lazy radix-16 encoder NTT);
- items 12 and 13, the lemma exhaustions over `F65537` for IDEA-0029 and IDEA-0020, which settle SP-3 before any cost work;
- item 14, IDEA-0011's experiment; item 15, IDEA-0002, whose remaining unit is a tier-A descriptor RFC; item 16, IDEA-0014; item 18, IDEA-0010's hypothesis (a);
- items 19 to 25, the E0 questions and merges; item 26, the survey refresh triggers; item 27, the cost-model bookkeeping (zero knowledge as its own `≈ 1.001` line, `commit_ps` scenarios at 150, 240 and 360 ps); item 28, the specification notes from SUR-0006 and SUR-0008; item 29, the NG-7 Lean blueprint route;
- items 30 onward, the "ideate: revisit" entries.

Item 17 (IDEA-0025, rate 2/3 as a live parameter) is a maintainer decision.

### 4.4 Maintainer-only

- The rebase onto main and the pull requests through `issue-workflow`, for the research branch and for RFC-0021. Commits are signed off (`git commit -s`) and carry no tool branding. The branch touches no release train (`research/`, `.claude/`, `docs/rfcs/`, `AGENTS.md`), so its pull request needs no changelog entry and carries the `no-changelog` label (CONTRIBUTING 'Changelog'). The user's rules forbid git commit, checkout, stash, reset, push, rebase, merge, clean and any deletion without their confirmation.
- The branch's upstream is `origin/research/autoresearch`; push it by name, never to `main`.
- Installing a scheduler for the headless loop, because it spends API budget unattended.

### 4.5 Keeping the loop running

The session cron job of the first session is gone: such jobs live only as long as their session. Three ways to keep the loop going:

- **Inside a session:** `/loop 24h /autoresearch`, or `CronCreate` with the prompt of Appendix B (every 3 hours at minute 41 was the cadence; recurring jobs expire after 7 days).
- **Headless:** `research/tools/loop.sh` runs one unit with `claude -p` and an allow-list. `LOOP_MODE=sweep` runs the whole sweep, `LOOP_BUDGET_USD` caps a run (default 25) and logs go to `~/.cache/tessera-autoresearch/`.
- **Scheduled:** `research/tools/launchd/com.tessera.autoresearch.plist` runs `loop.sh` every 6 hours. Its header gives the install and removal commands, and `WORKTREE` is a placeholder to replace.

Pause any scheduler while a sweep or a delegated experiment runs, so that two writers never edit the ledger at once.

## 5. What went upstream

All closed as completed by 2026-10-18. The backlog session verified each finding on main before filing it.

| Issue | Finding | Closed |
|---|---|---|
| #237 | RFC-0013's blinding claim and RFC-0012's weight check bound no claim (EXP-0005's first reading) | 2026-10-16, pull request #244 |
| #239 | one `Fp` blinding row cannot mask an `Fp2` combined row, and the other `Fp` masks have the same flaw (EXP-0005's second reading) | 2026-10-16, pull request #245 |
| #103 | grinding coverage of challenges a prover can re-roll (a comment from the same reading) | 2026-10-18 |
| #247 | `SoftmaxRow`'s padded leaves must be the constant 1, otherwise a second witness exists (IDEA-0019's triage) | 2026-10-16 |
| #248 | the regeneration pass that answers column queries is unpriced (IDEA-0016, EXP-0007) | 2026-10-16, pull request #253 |
| #250, #251, #252 | the streamed-pass count, weight column answers and the P4 row combination, filed from the backlog session's own reading after #248 (the first matches SUR-0003's correction) | 2026-10-17 |
| #254 | the tensor generator's log-rows factor `ϑ` in the proximity bound (SUR-0001, EXP-0003) | 2026-10-17 |
| #257 | INV-ARCH-2 against secret-indexed multiplicity counting, raised by the backlog session and priced by EXP-0010 | 2026-10-17 |
| #258 | the split-`eq`, Gruen-corrected tree-level kernel (EXP-0008, reproduced independently before filing) | 2026-10-17 |
| #266 | the `servable` count, RISK-2's profile, `o8` at `|o| ≤ 2^39`, the `minimal` accumulator's one-bit margin (EXP-0011) | 2026-10-17 |

## 6. Results so far

| Experiment | Question | Verdict | Result |
|---|---|---|---|
| EXP-0001 | IDEA-0001: list-decoding `τ` under [BCIKS20] | refuted | the Johnson-regime field term caps at 57.2 bits |
| EXP-0002 | IDEA-0004: calibrate `c_c` | inconclusive | `c_c` counted at 58 to 157 ps against 30 assumed (E0 fitted to E1) |
| EXP-0003 | IDEA-0012: Johnson-regime `τ` under linear-error bounds | validated | `τ_A` 155 to 130, `τ_W` 97 to 68; the basis of RFC-0021 |
| EXP-0004 | IDEA-0013: advice rate 1/4 | refuted | fewer columns, but `μ` +10 to 21% |
| EXP-0005 | IDEA-0012: the list-decoding extractor argument | inconclusive by its own criterion | the argument holds after two adversarial readings, which found #237 and #239 |
| EXP-0006 | IDEA-0008: the product-tree leaf cost | inconclusive (CV clause) | `c_π` 45 to 117 ps against 12 assumed |
| EXP-0007 | IDEA-0016: an output-pruned NTT for the column answers | validated | 3.3× per row in coefficient form, 1.7× in evaluation form; −8 to −10% of `μ` against the honest baseline with P5 priced |
| EXP-0008 | IDEA-0021: the split-`eq` Gruen level kernel | validated | 1.75× per leaf with bit-identical, verifier-accepted messages; `c_π` 73 to 41.8 ps; −8.1% of `μ(16K)` under `counted` |
| EXP-0009 | IDEA-0019: analytic padding blocks | inconclusive | 78 to 99% of the padding's cost recovered, but a per-tree constant fails the 5% per-real-leaf clause |
| EXP-0010 | issue #257: the timing leak of P1 | validated | at most 7.2 bits per proof at a 0.1% jitter floor (2.5 at 3%); envelope 2.7 to 5.5% of `μ`, oblivious counting +44 to +311%; the stated exception was recommended |
| EXP-0011 | IDEA-0033: the `servable` profile's count | validated | 326,656 commitments and lookups per layer per token, not RISK-2's 220,160 |

The research cost model's standing projections on 2026-10-19, with the `servable` count since 2026-10-17. SC-2's ceilings are `μ(1K) ≤ 12` and `μ(16K) ≤ 45`.

| Profile | μ(1K) | μ(4K) | μ(16K) | μ(128K) | SC-2 |
|---|---|---|---|---|---|
| `source` (the design document's model) | 3.2 | 6.5 | 18.7 | 88.7 | holds |
| `corrected` | 10.4 | 15.6 | 36.6 | 172.3 | holds, by 1.15× and 1.23× |
| `counted` (unit costs from EXP-0002, SUR-0002, EXP-0006) | 21.6 | 34.2 | 82.7 | 376.0 | misses, by 1.80× and 1.84× |

Validated levers not yet folded into the baseline: the split-`eq` kernel (`c_π` 73 to 41.8 ps) and the pruned column pass, which only pays off once P5 is priced, as #248 now requires. Idea scores dated before 2026-10-17 were computed against the RISK-2 count (`μ(16K)` 35.1 corrected, 78.6 counted) and say so in their notes.

## 7. Working rules and gotchas

- **Repository rules (AGENTS.md).** The specification is normative, and a soundness bug is the worst bug. Verifier independence (SP-6). SPDX headers on `.rs`, `.py` and `.cu` files; `license = "Apache-2.0 OR MIT"` in every `Cargo.toml`. Markdown links must resolve, and `cargo xtask lint` scans untracked files too, this one included. `rm`, `cp` and `mv` are interactive on this machine: use `/bin/cp -f` and `/bin/mv -f`, and delete nothing.
- **The user's rules.** No git writes and no deletions without their confirmation. No Haiku subagents. Verify a claim about the code by reading the file. No tool branding in repository text.
- **Research rules (`research/README.md`).** Pre-register before any code: the README's SHA-256 and the UTC time go in `results/preregistration.txt`, and amendments are appended with a date, never edited in place. Every number carries an evidence level. One unit of work per run. The soundness gate comes before any gain. Nothing in either workspace depends on `research/`. Synthetic inputs only.
- **The worktree's Bash guard.** It refuses any command containing the word "source": never pass `--profile source`, and say "the design document" in command strings. It also refuses heredocs and complex compound commands (loops with arithmetic, long `&&` chains with `cd`). Write a Python script to your scratchpad and run it with `python3 -I`; pipes and `;` work.
- **Editing the ledger.** The Edit tool refuses files not read in the session or changed on disk since. The first session edited `LEDGER.md` and the notes with small Python scripts that assert each anchor occurs exactly once before replacing it. `check_kb.py` checks citation keys, that each note's status word matches its ledger row, that the counters are ahead, the required report sections, that an experiment names its idea or the issue it answers, and that every identifier in the experiments table has a file on disk.
- **Delegated agents.** Tell them not to edit `LEDGER.md`, `landscape.md` or `bibliography.md`; the parent consolidates from the JSON they return. Give each its own directory. Never edit the ledger inline while a sweep's consolidation agent runs; a sweep edits it itself.
- **The cost model.** `python3 -I research/tools/cost_model.py` takes `--profile {corrected,counted}`, `--set name=value` (for example `commit_ps`, `lookup_ps`, `leaf_ps`, `commits_per_token`, `regen_full_equivalents`), `--contexts`, `--sensitivity`, `--json`, `--check` and `--vector PATH`. The `counted` unit costs (`c_c` 94, `c_ℓ` 101, `c_π` 73 ps) are E0 planning values.
- **The other tools.** `proximity.py` computes the proximity budgets: `--johnson`, `--bchks25`, `--arklib` (the exact ArkLib `johnsonExceptionCount` and the pairwise Johnson list bound) and `--sweep-rates`.
- **Reading papers.** ePrint PDFs return 403 to WebFetch; use the ECCC, arXiv or CiC copies. ArkLib's Lean files come with `gh api -H "Accept: application/vnd.github.raw+json" repos/Verified-zkEVM/ArkLib/contents/<path>`; RFC-0021 pins commit `4dae3acff846125521b7b3998010885071746f16`.
- **Working with `tessera-ba`.** The loop writes a finding up with RFC line references and evidence levels and sends it with `SendMessage`. `tessera-ba` verifies it on main and files it; record the issue number in the ledger. Never ask it to do something this session is not allowed to do.
- **Budget.** The first session ran on Fable 5.1 until its credits ran out on 2026-10-17; the user then switched the default model to Opus 5.5. A delegated experiment cost 0.3 to 0.45 million tokens and 30 to 45 minutes; a sweep about 6.7 million tokens and 2 hours.

## 8. Where to find detail

- This session's transcript: `/Users/abdel/.claude/projects/-Users-abdel-dev-me-tessera-research/aed06211-62ae-4636-920a-02942416f73e.jsonl`.
- The second and third sweeps' journals, one result per agent: `/Users/abdel/.claude/projects/-Users-abdel-dev-me-tessera-research/aed06211-62ae-4636-920a-02942416f73e/subagents/workflows/wf_c9779763-a6f/journal.jsonl` and `.../wf_80b73d0a-086/journal.jsonl`.
- The project memory: `/Users/abdel/.claude/projects/-Users-abdel-dev-me-tessera/memory/autoresearch-knowledge-base.md`.
- The first session's scratchpad, which the system may clear: `/private/tmp/claude-501/-Users-abdel-dev-me-tessera/aed06211-62ae-4636-920a-02942416f73e/scratchpad/`. It holds the update scripts, the downloaded ArkLib files and `cost-model.json`, a copy of main's pinned vector.

## Appendix A: briefs for the three experiments

Each brief is the specific part followed by the common rules. Paste both into the delegated agent's prompt.

### Common rules

- Work only inside `/Users/abdel/dev/me/tessera-research`; never `cd` to `/Users/abdel/dev/me/tessera`.
- Read first, in this order: `research/README.md`, `.claude/skills/autoresearch/experiment.md` (the five steps and the rules of evidence; follow them exactly), `research/templates/experiment.md`, the idea notes, and EXP-0008's report and harness as the worked example. The checker requires the sections `## Hypothesis`, `## Success criterion`, `## Method`, `## Results` and `## Verdict`, and a `| Pre-registered on |` line.
- Pre-register before any harness code: a README with status `planned`, the hypothesis, a success criterion that names the numbers meaning `validated`, `refuted` and `inconclusive`, the method and the reproduction commands. Record its SHA-256 and the UTC time in `results/preregistration.txt`, and keep the frozen text as `results/README.preregistered.txt`, as EXP-0008 did.
- Baseline and candidate run in one program, on the same seeded inputs (SplitMix64, seed stated), on the same machine. The candidate's round messages, final evaluations and claims must be bit-identical to the baseline's and to the production engine's (compare digests), and the `tv-core` verifier driver must accept them, as in EXP-0008. A faster candidate with a different digest is a bug, not a result.
- Measure as RFC-0020 does: 3 warm-ups, 10 measured iterations, median, minimum, maximum and coefficient of variation, every iteration at least 20 ms by repetition. Flag a coefficient above 3% and rerun the arm. Record the environment (CPU, OS, rustc and cargo versions, date, git commit) and save the raw output under `results/`.
- Count operations (E0) in a `count.py`, as EXP-0008 did. Convert to `μ` only with `python3 -I research/tools/cost_model.py --profile counted --set ...`. Do not edit the tool; its `--check` must keep passing.
- The Rust harness is its own Cargo workspace: an empty `[workspace]` table, `license = "Apache-2.0 OR MIT"`, `publish = false`, `#![forbid(unsafe_code)]`, path dependencies on `../../../crates/<name>` and `../../../verifier/crates/<name>`, the SPDX header `// SPDX-License-Identifier: Apache-2.0 OR MIT` on every `.rs`, the `Cargo.lock` kept on disk, `cargo fmt` and clippy clean. Python files carry `# SPDX-License-Identifier: Apache-2.0 OR MIT` and run with `python3 -I`.
- Judge only against the pre-registered criterion. List the threats to validity: CPU against GPU, counts against wall clock, cache effects, input distribution. Give the verdict word, set the README status to `done: <verdict>`, and update the named idea notes' `| Status |` lines, Score tables and triage paragraphs with the result and the reason.
- Edit nothing outside the experiment directory and the named idea notes; never touch `LEDGER.md`, `landscape.md` or `bibliography.md`. No git writes. Never put the word "source" in a Bash command string. No heredocs. Build output goes in the experiment's own `target/`, which `.gitignore` covers. Copy with `/bin/cp -f` and move with `/bin/mv -f`.
- Return a JSON object, as data for the parent, with these keys: `id`, `path`, `status`, `verdict`, `idea_status`, `priority_before`, `priority_after`, `confidence_before`, `confidence_after`, `key_results` (each with its evidence level), `threats`, `follow_ups` (naming what an H100 run would settle), `ledger_row`, `idea_row_summary`, `new_bibliography_keys` (a list of `{key, entry}`, checked against `research/bibliography.md`) and `problems`. The `ledger_row` has the six cells of the existing rows: `| [EXP-NNNN](experiments/EXP-NNNN-<slug>/README.md) | question | idea | status | result summary | evidence levels and reproduction |`.

### A.1 EXP-0012: IDEA-0022, integer round window for the leaf level

- Directory: `research/experiments/EXP-0012-integer-leaf-round-window/`. The identifier is reserved; add the ledger row back when the README exists.
- Idea: [research/ideas/IDEA-0022-integer-leaf-level-round-window.md](research/ideas/IDEA-0022-integer-leaf-level-round-window.md) (`triaged`, 3.1). The leaf level proves its first `v = 4` rounds in one pass from a `3^v`-point grid of per-pair products, accumulated in `u64`/`u128` with one Goldilocks reduction per block, with `eq` decomposed. It hosts IDEA-0017's windowed prover as a case and absorbs IDEA-0010 mechanism 2 (the `v = 1` case).
- Criterion: from the note's Mechanism, Expected gain, Kill criterion and Cheapest experiment sections. Pin the reduction cadence at pre-registration: there is no slack at 2^5 blocks, so state the overflow bound and check it in the harness.
- Baseline: EXP-0008's split-`eq` Gruen kernel (copy its harness); the production `prove_forests` as a context arm. Sizes 2^20 to 2^22 leaves in stacked forests; report nanoseconds per leaf.
- Notes it may edit: IDEA-0022, plus the triage paragraphs of IDEA-0017 and IDEA-0010 for the cases it hosts.

### A.2 EXP-0013: IDEA-0026 with IDEA-0030, delayed reduction

- Directory: [research/experiments/EXP-0013-delayed-reduction-node-sums/README.md](research/experiments/EXP-0013-delayed-reduction-node-sums/README.md) exists as a complete draft. Review it against [IDEA-0026](research/ideas/IDEA-0026-delayed-reduction-fraction-node-sums.md), [IDEA-0030](research/ideas/IDEA-0030-deferred-reduction-level-kernel.md) and EXP-0008. Change it if it needs changing: no code exists, so the pre-registration is still open. Then freeze it, with the digest in `results/preregistration.txt` and the copy in `results/README.preregistered.txt`, before any code.
- The drafted criterion: `validated` iff `σ_leaf ≥ 0.15` and `σ_generic ≥ 0.08` for the scale-first arm against EXP-0008's split kernel at 2^21 entries, with bit-identical messages, verifier acceptance and CVs at most 3%; `refuted` iff `σ_leaf < 0.10`; `inconclusive` otherwise. IDEA-0030's own case is `σ ≥ 0.05` per leaf on three 2^22-leaf forest shapes and `σ ≥ 0.03` for the relation-first arm on both fraction levels.
- Arms: `split` (the baseline), `scale-first`, `relation-first` and `deferred`, with `prove_batch` and `prove_forests` as context, and a dot-product microkernel as an observation. `count.py` counts the reductions separately and gives the int32 translation; the H100 has no native 128-bit accumulate, so the report says how its int32 pipe would emulate the accumulators.
- Notes it may edit: IDEA-0026 and IDEA-0030. Replace the `planned` row when it returns.

### A.3 The EXP-0009 amendment: IDEA-0019, the amortized ragged loop

- Directory: [research/experiments/EXP-0009-analytic-padding-blocks/README.md](research/experiments/EXP-0009-analytic-padding-blocks/README.md). Rule 3 governs: leave the original sections untouched. Append a section `## Amendment <date>: amortized ragged loop` and record its SHA-256 and UTC time in `results/preregistration-amendment.txt` before any code.
- Change: sort each height class's rows by real length so that equal-length trees are adjacent and share the prefix, the `TAIL` term and the boundary column, and vectorize the odd-tail folds. Add the amended candidate as a new arm and keep the original arms, so the original numbers stay reproducible.
- Criterion: unchanged, both clauses on the four judged configurations S1 to S4. `R_same` must be within 10% of the padding factor `f`, and the cost per real leaf within 5% of an unpadded forest's. Same seeds and protocol.
- Original numbers: `a(c)/u` 1.126, 1.095, 1.002 and 1.144; `R_same` 1.832, 1.259, 1.596 and 1.978 at `f` 2.00, 1.33, 1.60 and 2.13. The original report's E0 extrapolation predicts the 5% clause passes at every `T` once the constant is amortized.
- If both clauses pass on all four, the status becomes `done: validated` by the dated amendment and IDEA-0019 becomes `validated`; otherwise it stays `done: inconclusive` and the section says why. Notes it may edit: IDEA-0019. Return the complete replacement row for EXP-0009.

## Appendix B: the loop prompt

The prompt the first session's cron job ran every 3 hours at minute 41:

```text
Run one unit of the Tessera autoresearch loop in this worktree (/Users/abdel/dev/me/tessera-research, branch research/autoresearch): invoke the autoresearch skill with no arguments, read research/LEDGER.md, take the first queue item that is feasible here (skip the maintainer's items), do that unit of work, consolidate, then run python3 -I research/tools/check_kb.py and cargo xtask lint until both pass. Make no git commit, checkout, stash, rebase, merge or push and delete nothing; move with /bin/mv -f and copy with /bin/cp -f; never put the word source in a Bash command string (the worktree guard refuses it). Finish with a short report: what the unit produced with evidence levels, what changed in the ledger, and the queue's first feasible item now.
```
