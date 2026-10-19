// SPDX-License-Identifier: Apache-2.0 OR MIT
// Check orchestration with stub agents; no research or API budget is spent.
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const code = (await readFile(new URL('../../.agents/workflows/autoresearch-sweep.js', import.meta.url), 'utf8'))
  .replace('export const meta =', 'const meta =');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const sweep = new AsyncFunction('args', 'phase', 'agent', 'parallel', 'pipeline', 'log', code);

for (const mode of ['normal', 'no-topics', 'no-experiments', 'refuted', 'resume']) {
  let active = 0, maxActive = 0, experiments = 0;
  const labels = [];
  const agent = async (prompt, options) => {
    const label = options.label;
    labels.push(label);
    if (label === 'orient') return { date: '2026-10-19', next_sur: 14, next_idea: 34, next_exp: 14,
      topics: ['no-topics', 'resume'].includes(mode) ? [] : [{ key: 'lookups', name: 'Lookups', reason: 'refresh', questions: ['Q'] }],
      pending_experiments: mode === 'resume' ? [{ id: 'IDEA-0026', path: 'idea.md', experiment_id: 'EXP-0013', experiment_path: 'existing/README.md' }] : [] };
    if (label.startsWith('survey:')) return { id: 'SUR-0014', path: 'survey.md', summary: 'stub', seeds: [], bibliography: [], landscape: [] };
    if (label.startsWith('ideate:')) return { ideas: label === 'ideate:lookup' ? [0, 1].map(i => ({ slug: `stub${i}`, title: `Seed ${i}`, mechanism: `${i}` })) : [] };
    if (label.startsWith('write:')) return { id: label.slice(6), path: 'idea.md', title: label, gain_16k_pct: -1, security_fail: false, feasible_here: true, effort: 's' };
    if (label.startsWith('refute:')) return { verdicts: [] };
    if (label === 'score') return { ideas: [34, 35].map(i => ({ id: `IDEA-00${i}`, path: 'idea.md', title: 'stub', priority: i, status: mode === 'refuted' ? 'rejected' : 'triaged', feasible_here: true, effort: 's' })) };
    if (label.startsWith('experiment:')) {
      assert.match(prompt, /committed/);
      if (mode === 'resume') {
        assert.match(prompt, /Identifier EXP-0013/);
        assert.match(prompt, /resume existing report existing\/README.md/);
      }
      active++; maxActive = Math.max(maxActive, active); experiments++;
      await new Promise(resolve => setTimeout(resolve, 5));
      active--;
      return { id: mode === 'resume' ? 'EXP-0013' : 'EXP-0014', path: 'plan.md', status: 'planned', verdict: 'pending', summary: 'awaiting committed registration' };
    }
    if (label === 'consolidate') return { lint_ok: true, kb_check_ok: true, summary: 'stub', next_actions: ['commit plan'] };
    throw new Error(`unexpected agent ${label}`);
  };
  const result = await sweep({ experiments: mode === 'no-experiments' ? 0 : 2 }, () => {}, agent,
    jobs => Promise.all(jobs.map(job => job())), (items, fn) => Promise.all(items.map((item, i) => fn(item, item, i))), () => {});
  if (mode === 'no-topics') assert.equal(result.error, 'orient returned no topics');
  else {
    assert.equal(labels.at(-1), 'consolidate');
    assert.equal(experiments, mode === 'normal' ? 2 : mode === 'resume' ? 1 : 0);
    if (mode === 'resume') assert.equal(result.experiments[0].id, 'EXP-0013');
    assert.ok(maxActive <= 1, 'timed experiments overlapped');
  }
}
console.log('ok: sweep phases, hard gate, serial timing and committed-plan resumption; no API used');
