export const meta = {
  name: 'autoresearch-sweep',
  description: 'Survey the state of the art across Tessera research topics, ideate, adversarially triage, run proofs of concept and consolidate research/',
  whenToUse: 'A parallel pass of the autoresearch loop over several agenda topics. args: { topics?: string[] (agenda keys), max_topics?: number (default 4), max_ideas?: number (default 12), experiments?: number (default 1) }',
  phases: [
    { title: 'Orient', detail: 'ledger and agenda: topics, identifiers, date' },
    { title: 'Survey', detail: 'one agent per topic, primary sources' },
    { title: 'Ideate', detail: 'one agent per lens over the new surveys' },
    { title: 'Write', detail: 'one idea note per distinct seed' },
    { title: 'Refute', detail: 'four refuter lenses over every idea' },
    { title: 'Score', detail: 'rubric, statuses' },
    { title: 'Experiment', detail: 'pre-registered proofs of concept for the top feasible ideas' },
    { title: 'Consolidate', detail: 'ledger, landscape, bibliography, lint' },
  ],
}

// The process each agent follows lives in .agents/skills/autoresearch/*.md and the shapes of the
// notes in research/templates/. This script only decides who runs when, with which identifier,
// and merges what they return. Shared files (ledger, landscape, bibliography) have one writer:
// the consolidation agent.

const cfg = {
  topics: args && Array.isArray(args.topics) ? args.topics : null,
  maxTopics: args && args.max_topics ? args.max_topics : 4,
  maxIdeas: args && args.max_ideas !== undefined ? args.max_ideas : 12,
  experiments: args && args.experiments !== undefined ? args.experiments : 1,
}

const pad = (n) => String(n).padStart(4, '0')

const RULES = `You are one agent of the autoresearch sweep for the Tessera repository; the current directory is the repository root.
Read first: research/README.md (rules, evidence levels, statuses) and the phase file named below. Write only the files named below. Never edit research/LEDGER.md, research/landscape.md or research/bibliography.md: the consolidation agent merges what you return. Derive the field, extension degree and security targets from RFC-0002 and RFC-0014 (currently Goldilocks/Fp2); read current compiler counts and the explicit missing terms in research/baseline.md. Never run a git command that changes state (no add, commit, checkout, stash, branch, reset). Move files with /bin/mv -f and copy with /bin/cp -f (cp and mv are interactive on this machine). Get today's date with \`date +%F\`. Keep scratch files in the session scratchpad, never under research/. Every number you write carries an evidence level (E0-E3) and a source quality (primary, secondary, claim). Your final output is data for the orchestrator: fill the requested JSON exactly.`

const ORIENT = {
  type: 'object',
  required: ['date', 'next_sur', 'next_idea', 'next_exp', 'topics', 'pending_experiments'],
  properties: {
    date: { type: 'string' },
    next_sur: { type: 'integer' },
    next_idea: { type: 'integer' },
    next_exp: { type: 'integer' },
    pending_experiments: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'path', 'experiment_id', 'experiment_path'],
        properties: {
          id: { type: 'string' }, path: { type: 'string' },
          experiment_id: { type: 'string' }, experiment_path: { type: 'string' },
        },
      },
    },
    topics: {
      type: 'array',
      items: {
        type: 'object',
        required: ['key', 'name', 'questions', 'reason'],
        properties: {
          key: { type: 'string' },
          name: { type: 'string' },
          questions: { type: 'array', items: { type: 'string' } },
          reason: { type: 'string' },
        },
      },
    },
    notes: { type: 'string' },
  },
}

const SURVEY = {
  type: 'object',
  required: ['id', 'path', 'summary', 'seeds', 'bibliography', 'landscape'],
  properties: {
    id: { type: 'string' },
    path: { type: 'string' },
    summary: { type: 'string' },
    seeds: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'lever', 'mechanism', 'gain', 'killer'],
        properties: {
          title: { type: 'string' },
          lever: { type: 'string' },
          mechanism: { type: 'string' },
          gain: { type: 'string' },
          killer: { type: 'string' },
          sources: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    bibliography: {
      type: 'array',
      items: { type: 'object', required: ['key', 'entry'], properties: { key: { type: 'string' }, entry: { type: 'string' } } },
    },
    landscape: {
      type: 'array',
      items: { type: 'object', required: ['row', 'change'], properties: { row: { type: 'string' }, change: { type: 'string' } } },
    },
  },
}

const IDEAS = {
  type: 'object',
  required: ['ideas'],
  properties: {
    ideas: {
      type: 'array',
      items: {
        type: 'object',
        required: ['title', 'slug', 'lever', 'origin', 'mechanism', 'why', 'gain_hint', 'security_notes', 'prior_rejection', 'kill', 'experiment', 'feasible_here'],
        properties: {
          title: { type: 'string' },
          slug: { type: 'string' },
          lever: { type: 'string' },
          origin: { type: 'string' },
          mechanism: { type: 'string' },
          why: { type: 'string' },
          gain_hint: { type: 'string' },
          security_notes: { type: 'string' },
          prior_rejection: { type: 'string' },
          kill: { type: 'string' },
          experiment: { type: 'string' },
          feasible_here: { type: 'boolean' },
        },
      },
    },
  },
}

const WRITTEN = {
  type: 'object',
  required: ['id', 'path', 'title', 'gain_16k_pct', 'security_fail', 'feasible_here', 'effort', 'summary'],
  properties: {
    id: { type: 'string' },
    path: { type: 'string' },
    title: { type: 'string' },
    gain_16k_pct: { type: 'number' },
    gain_1k_pct: { type: 'number' },
    security_fail: { type: 'boolean' },
    security_unknowns: { type: 'array', items: { type: 'string' } },
    feasible_here: { type: 'boolean' },
    effort: { type: 'string', enum: ['s', 'm', 'l'] },
    summary: { type: 'string' },
  },
}

const VERDICTS = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'verdict', 'argument'],
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['survives', 'weakened', 'refuted'] },
          argument: { type: 'string' },
        },
      },
    },
  },
}

const SCORED = {
  type: 'object',
  required: ['ideas'],
  properties: {
    ideas: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'path', 'title', 'priority', 'status', 'feasible_here', 'effort', 'reason'],
        properties: {
          id: { type: 'string' },
          path: { type: 'string' },
          title: { type: 'string' },
          priority: { type: 'number' },
          status: { type: 'string' },
          feasible_here: { type: 'boolean' },
          effort: { type: 'string' },
          reason: { type: 'string' },
        },
      },
    },
  },
}

const EXPERIMENT = {
  type: 'object',
  required: ['id', 'path', 'status', 'verdict', 'summary'],
  properties: {
    id: { type: 'string' },
    path: { type: 'string' },
    status: { type: 'string' },
    verdict: { type: 'string' },
    idea_status: { type: 'string' },
    summary: { type: 'string' },
  },
}

const CONSOLIDATED = {
  type: 'object',
  required: ['lint_ok', 'kb_check_ok', 'summary', 'next_actions'],
  properties: {
    lint_ok: { type: 'boolean' },
    kb_check_ok: { type: 'boolean' },
    summary: { type: 'string' },
    next_actions: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'string' } },
  },
}

// ---------------------------------------------------------------- Orient
phase('Orient')
const topicRule = cfg.topics
  ? `exactly these agenda keys: ${JSON.stringify(cfg.topics)} (a key the agenda does not define is dropped and named in notes)`
  : `up to ${cfg.maxTopics} agenda topics needing a new or refreshed survey, in ledger queue order; return no topics when no refresh trigger has fired`
const orient = await agent(`${RULES}
Phase: orient. Read research/LEDGER.md, research/baseline.md (sections 2 and 5) and .agents/skills/autoresearch/agenda.md.
Return: today's date; the next free counters from the ledger's Counters table as integers (SUR-0007 means 7); and the topics this sweep surveys: ${topicRule}. For each topic return its key, its name, its standing questions verbatim from the agenda, and the reason it is worth surveying now (the term it moves and that term's share of μ in the sensitivity tables). Also return pending_experiments in queue order: CPU-feasible planned or interrupted experiments whose frozen plans match HEAD, with the existing experiment_id, experiment_path, idea id and idea path. Reuse their identifiers and plans; do not assign a new experiment to resume one. Write no files.`, { label: 'orient', phase: 'Orient', effort: 'medium', schema: ORIENT })

if (!orient || !orient.topics) return { error: 'invalid orientation', orient }
const pendingExperiments = orient.pending_experiments || []
if (!orient.topics.length && !pendingExperiments.length) return { error: 'orient returned no topics', orient }
log(`sweep over ${orient.topics.map((t) => t.key).join(', ')}; counters SUR ${orient.next_sur}, IDEA ${orient.next_idea}, EXP ${orient.next_exp}`)

// ---------------------------------------------------------------- Survey
phase('Survey')
const surveyResults = await parallel(
  orient.topics.map((t, i) => () => {
    const id = `SUR-${pad(orient.next_sur + i)}`
    return agent(`${RULES}
Phase: survey. Follow .agents/skills/autoresearch/survey.md step by step and the template research/templates/survey.md.
Topic key: ${t.key}. Topic: ${t.name}. Why now: ${t.reason}
Standing questions:
${t.questions.map((q, j) => `${j + 1}. ${q}`).join('\n')}
Identifier: ${id}. Write the survey to research/surveys/${id}-${t.key}.md with status current (${orient.date}). Use WebSearch and WebFetch; primary sources first; record the URL and the date checked for each. Return in the JSON: the identifier and path; a five-sentence summary; the candidate ideas as seeds; one bibliography entry per source (key, and the full line in the style of research/bibliography.md); and the landscape changes (the row, and what changes with its source key).`, { label: `survey:${t.key}`, phase: 'Survey', effort: 'high', schema: SURVEY })
  }),
)
const surveys = surveyResults.filter(Boolean)
log(`${surveys.length}/${orient.topics.length} surveys written`)
if (!surveys.length && !pendingExperiments.length) return { error: 'no survey completed', orient }

// ---------------------------------------------------------------- Ideate
phase('Ideate')
const LENSES = [
  ['commitment', 'the per-element commitment cost c_c: encoding, hashing, rate, small-value commitments'],
  ['lookup', 'the per-lookup cost c_ℓ and the lookup count K_lin: LogUp micro-architecture, alternative lookup arguments, fused or shared lookups'],
  ['certificate', 'the max-certificate cost c_π and other per-score certificates'],
  ['counts', 'the committed-element and lookup counts C_lin and K_lin through the compiler: range analysis, rescale placement, quantization co-design (RISK-2)'],
  ['attention', 'the number of scores proven A: certified block skipping, sparsity certificates, proof-aware D in bounded mode'],
  ['passes', 'the regeneration and streaming terms regen and stream: pass fusion, checkpointing, streaming-friendly sumcheck, memory tiers, distributed proving (RISK-9, RISK-10)'],
  ['parameters', 'τ, proof size and verification time: proximity bounds, rate, grinding, query reduction, recursion'],
  ['zk', 'zero-knowledge overhead: masking cost, hiding budget, simulator composition'],
  ['hardware', 'mapping work onto tensor cores, HBM bandwidth and several GPUs: folds as GEMMs, kernel fusion, memory layouts'],
  ['wild', 'a different paradigm: approximate sumcheck, folding, another commitment family, a different statement decomposition, anything the other lenses would not reach'],
]
const surveyList = surveys.map((s) => `- ${s.id}: ${s.path} (seeds: ${s.seeds.map((x) => x.title).join('; ')})`).join('\n')
const lensResults = await parallel(
  (surveys.length ? LENSES : []).map(([key, desc]) => () =>
    agent(`${RULES}
Phase: ideate, generation only, no files written. Follow steps 1 and 2 of .agents/skills/autoresearch/ideate.md through one lens: ${key}: ${desc}.
Read research/baseline.md (sections 2, 5, 6 and 7) and these new surveys:
${surveyList}
Return at most 3 ideas for this lens. For each: title; slug (lowercase, hyphenated, at most five words); lever (the terms moved); origin (survey identifier and finding, RISK, OQ, or new); mechanism (precise enough to implement); why (evidence with bibliography keys and evidence levels); gain_hint (the cost-model scenario to run, such as "--set lookup_ps=20", and the rough Δμ); security_notes (soundness, uniqueness, zero knowledge, verifier independence, post-quantum); prior_rejection (the baseline §7 or RFC Alternatives Considered item that covers it and what is new, or "none, checked"); kill (the observation that would refute it); experiment (the cheapest test and the evidence level it reaches here, E0 or E1, or "E2 needed"); feasible_here (true when an E0 or E1 experiment is possible on a CPU-only machine).`, { label: `ideate:${key}`, phase: 'Ideate', effort: 'high', schema: IDEAS }),
  ),
)
const lensIdeas = lensResults.filter(Boolean).flatMap((r) => r.ideas || [])

const tokens = (s) => new Set(s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter((w) => w.length > 3))
const similar = (a, b) => {
  const A = tokens(a)
  const B = tokens(b)
  let shared = 0
  for (const w of A) if (B.has(w)) shared++
  return shared / Math.max(1, Math.min(A.size, B.size)) >= 0.6
}
const distinct = []
for (const idea of lensIdeas) {
  const dup = distinct.some((u) => u.slug === idea.slug || similar(`${u.title} ${u.mechanism}`, `${idea.title} ${idea.mechanism}`))
  if (!dup) distinct.push(idea)
}
const chosen = distinct.slice(0, cfg.maxIdeas)
const dropped = distinct.slice(cfg.maxIdeas).map((i) => i.title)
log(`${lensIdeas.length} ideas generated, ${distinct.length} distinct, ${chosen.length} kept${dropped.length ? `, ${dropped.length} dropped by max_ideas (listed in the result and the ledger)` : ''}`)

// ---------------------------------------------------------------- Write
phase('Write')
const writtenResults = await pipeline(chosen, (idea, _item, i) => {
  const id = `IDEA-${pad(orient.next_idea + i)}`
  return agent(`${RULES}
Phase: write one idea note. Follow step 3 of .agents/skills/autoresearch/ideate.md and the template research/templates/idea.md exactly: every cell filled, every number with an evidence level. Identifier ${id}; path research/ideas/${id}-${idea.slug}.md; status proposed, dated ${orient.date}. Leave the Refutations table and the Score table with their headers and one placeholder row "pending triage"; the refute and score agents fill them.
The idea, from the generation pass:
${JSON.stringify(idea, null, 2)}
Run the cost-model scenario with python3 -I research/tools/cost_model.py --set ... and paste its table into the Expected gain section (a scenario the model cannot express is stated in words with its derivation). Return the identifier, the path, the title, Δμ in percent at 16K and at 1K (negative means cheaper), whether any security-gate row is fail, the unknown rows, feasible_here, the effort (s, m or l) and a one-sentence summary.`, { label: `write:${id}`, phase: 'Write', effort: 'medium', schema: WRITTEN })
})
const written = writtenResults.filter(Boolean)
log(`${written.length}/${chosen.length} idea notes written`)

// ---------------------------------------------------------------- Refute
phase('Refute')
const REFUTERS = [
  ['soundness', 'Read as ADV-P with a large grinding budget and as ADV-V: look for a second passing witness upstream of the logits, a challenge drawn before the message it depends on is fixed, a leaked bit about weights or activations, a verifier made to trust prover-declared data, a reduction of any bound in RFC-0014. Apply .agents/skills/soundness-review/SKILL.md to the mechanism.'],
  ['prior-art', 'Check research/baseline.md section 7 and the "Alternatives Considered" section of every RFC under docs/rfcs/: is the idea already in the specification, already rejected (cite the item), or excluded by a known result such as the lookup separation principle [Jo26]? Say what, if anything, is new.'],
  ['cost-model', 'Rerun the idea\'s cost-model scenario with python3 -I research/tools/cost_model.py. Does the term it moves matter at the context it targets (sensitivity tables, baseline.md section 2)? Is the claimed gain larger than the uncertainty of its inputs? Are the inputs\' evidence levels honest?'],
  ['feasibility', 'Hardware needed, dependencies, tier-A surfaces (docs/spec/02-public-api.md section 1) and RFCs touched, implementation effort, and whether the cheapest experiment reaches E0 or E1 on a CPU-only machine as claimed.'],
]
const ideaList = written.map((w) => `- ${w.id}: ${w.path}: ${w.title}`).join('\n')
const refuterResults = written.length
  ? await parallel(
      REFUTERS.map(([lens, instruction]) => () =>
        agent(`${RULES}
Phase: refute (step 4 of .agents/skills/autoresearch/ideate.md), lens: ${lens}. You did not write these ideas; try to kill each one. ${instruction}
Ideas:
${ideaList}
For every idea return a verdict (survives, weakened or refuted) and the argument in two to five sentences with file:line or citation evidence. Write no files.`, { label: `refute:${lens}`, phase: 'Refute', effort: 'high', schema: VERDICTS }),
      ),
    )
  : []
const verdictsById = {}
refuterResults.forEach((set, k) => {
  if (!set || !set.verdicts) return
  for (const v of set.verdicts) {
    if (!verdictsById[v.id]) verdictsById[v.id] = []
    verdictsById[v.id].push({ lens: REFUTERS[k][0], verdict: v.verdict, argument: v.argument })
  }
})

// ---------------------------------------------------------------- Score
phase('Score')
const scored = written.length
  ? await agent(`${RULES}
Phase: score (step 5 of .agents/skills/autoresearch/ideate.md and its rubric). For each idea below, edit its note: fill the Refutations table (one row per verdict) and the Score table, and set the status to triaged, or to rejected when a hard gate fires (a security-gate fail, an impossibility result, a prior rejection with nothing new), with the reason. Do not edit the ledger.
Ideas with their refutations:
${JSON.stringify(written.map((w) => ({ ...w, verdicts: verdictsById[w.id] || [] })), null, 2)}
Return every idea with its priority (gain × confidence / cost weight), status, feasible_here, effort and a one-line reason.`, { label: 'score', phase: 'Score', effort: 'high', schema: SCORED })
  : { ideas: [] }

// ---------------------------------------------------------------- Experiment
phase('Experiment')
const newCandidates = (scored && scored.ideas ? scored.ideas : [])
  .filter((i) => i.status === 'triaged' && i.feasible_here)
  .sort((a, b) => b.priority - a.priority)
const candidates = [...pendingExperiments, ...newCandidates]
  .slice(0, cfg.experiments)
log(cfg.experiments ? `${candidates.length} experiment(s) to run` : 'experiments disabled (args.experiments = 0)')
const experimentResults = []
let nextExperiment = orient.next_exp
// Timing shares one CPU: finish each experiment before starting the next.
for (const idea of candidates) {
  const id = idea.experiment_id || `EXP-${pad(nextExperiment++)}`
  experimentResults.push(await agent(`${RULES}
Phase: experiment. Follow .agents/skills/autoresearch/experiment.md step by step and the template research/templates/experiment.md. Idea: ${idea.id} at ${idea.path}. Identifier ${id}; ${idea.experiment_path ? `resume existing report ${idea.experiment_path} without altering its frozen plan` : `directory research/experiments/${id}-<slug>/`} with README.md (the report), the code and results/. Pre-register the hypothesis and success criterion, freeze the text and SHA-256, and check the frozen file exists in HEAD before any harness code. If it is not committed, return planned and stop before code; the maintainer commits it between runs. For an already committed pre-registration, build baseline and candidate in the same harness with a correctness check; measure as RFC-0020 does; judge against the pre-registered criterion only; then update the idea note's status and score with the verdict. Do not edit the ledger. Return the identifier, the path, the status (done: validated, done: refuted, done: inconclusive, or shelved with the reason), the verdict, the idea's new status, and a summary with the key numbers and their evidence level.`, { label: `experiment:${idea.id}`, phase: 'Experiment', effort: 'xhigh', schema: EXPERIMENT }))
}
const experiments = experimentResults.filter(Boolean)

// ---------------------------------------------------------------- Consolidate
phase('Consolidate')
const consolidated = await agent(`${RULES}
Phase: consolidate (step 3 of .agents/skills/autoresearch/SKILL.md). You are the only agent that edits research/LEDGER.md, research/landscape.md and research/bibliography.md.
1. Bibliography: merge these entries, skipping keys that already exist: ${JSON.stringify(surveys.flatMap((s) => s.bibliography))}
2. Landscape: apply these changes, dated ${orient.date}: ${JSON.stringify(surveys.flatMap((s) => s.landscape))}
3. Ledger: add one row per survey ${JSON.stringify(surveys.map((s) => ({ id: s.id, path: s.path, summary: s.summary, seeds: s.seeds.length })))}, per idea ${JSON.stringify(scored && scored.ideas ? scored.ideas : [])} and per experiment ${JSON.stringify(experiments)}; advance the counters past every identifier used; reorder the queue by what was learned (the highest-priority triaged idea with a feasible experiment first, then the agenda topics still without a current survey); add an iteration-log row dated ${orient.date} naming this sweep and its counts.
4. Ideas dropped by max_ideas, to list in the queue as "ideate: revisit" entries: ${JSON.stringify(dropped)}
5. Run cargo xtask lint from the repository root; fix every broken link or missing header in files under research/ and only there; rerun until it passes or the failure lies outside research/.
6. Run python3 -I research/tools/check_kb.py from the repository root; fix every FAIL it prints (an unresolved citation key, a note without a ledger row or with a status word that differs from its row, a counter behind an identifier in use, an experiment report without its pre-registration sections); rerun until it passes.
Return lint_ok, kb_check_ok, a five-sentence summary of the sweep for the maintainer (numbers with evidence levels), the next actions as the queue now lists them, and any problems.`, { label: 'consolidate', phase: 'Consolidate', effort: 'medium', schema: CONSOLIDATED })

return {
  date: orient.date,
  topics: orient.topics.map((t) => t.key),
  surveys: surveys.map((s) => ({ id: s.id, path: s.path, summary: s.summary, seeds: s.seeds.length })),
  ideas: scored && scored.ideas ? scored.ideas : [],
  dropped,
  experiments,
  consolidated,
}
