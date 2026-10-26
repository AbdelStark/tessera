#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The gate of draft RFC-0073's residual rule ('What the prover cannot detect').

It replays the rules of model.py, at B_T = 32 (rows end an epoch first) and B_T = 64 (columns do):

1. The full replay. A history of three epochs and two links, with a backup after every write.
   Each of the eight entry files is then put back at every version it had, or lost, in every
   combination, and an honest operator proves on. Six variants of what is presented: nothing; the
   latest .twr; that and a real ledger (accepted proofs' rows only, the last proof of each epoch
   rejected after its rows were revealed, beside another model's chain); that and every kept
   Round1; every kept Round1 alone; and one entry file present but corrupt, each in turn. The
   residual rule classifies every outcome before the run, and the gate checks it both ways
   against the harm observed (a row reused or a matrix over-opened): 0 harmful outcomes outside
   the rule, 0 in it without harm. Every outcome also stays within the per-epoch bound.
2. Targeted replays of what the full replay cannot build: crashes, double crashes, a crash and
   then a restore, a second process, copies and their bound, aliases, a corrupted refuting file,
   the bound of whole-entry restores, a matrix with B_T = 0, and presented artifacts.
3. Negative controls: each earlier or other reading of a rule must show the harm it predicts, or
   the replay cannot see the bug the rule prevents.

Usage: python3 -I docs/testing/weight-entry-model/gate.py [--jobs N]
Standard library only. Exit status 0 when every check passes, 1 otherwise.
"""

import argparse
import importlib.util
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # leave no __pycache__ in the docs tree
_spec = importlib.util.spec_from_file_location('weight_entry_model', Path(__file__).resolve().with_name('model.py'))
m = importlib.util.module_from_spec(_spec)
sys.modules['weight_entry_model'] = m  # pickle finds the model's classes under this name
_spec.loader.exec_module(m)

from weight_entry_model import (  # noqa: E402
    CORRUPT,
    RECORD,
    Directory,
    LinkingDirectory,
    Prover,
    Refused,
    World,
    link_file,
    marks,
    new_entry,
    operate,
    presented_ledger,
    presented_public,
    presented_round1,
    public,
    root,
    state,
)

ROWS = (32, 64)
FILES = [RECORD, state(0), state(1), state(2), public(1), public(2), link_file(0), link_file(1)]
VARIANTS = ('none', 'twr', 'realistic', 'ground', 'round1s', 'corrupt')
STEPS = 14  # proofs the operator attempts after a restore: past every epoch's budget
OTHER_MODEL = [(0, root('K9', 0)), (1, root('K9', 1))]  # another entry's chain in the same ledger


# ---- The residual rule and its bound ----------------------------------------------------------------


def chain_of(ledger, first_root):
    return next((c for c in ledger['chains'] if c and c[0] == (0, first_root)), [])


def names_past(artifact, roots):
    """'What a presented artifact catches': the artifact names a root of this entry past the
    record (a ledger through this entry's chain only)."""
    pairs = chain_of(artifact, roots[0]) if artifact['kind'] == 'ledger' else artifact['roots']
    return any(e >= len(roots) or roots[e] != rt for e, rt in pairs)


def shown_end(artifact, roots):
    """The first row past every reservation the artifact shows of the record's latest root."""
    if artifact['kind'] == 'ledger' and roots[-1] not in dict(chain_of(artifact, roots[0])).values():
        return 0
    return artifact['used_rows'].get(roots[-1], 0)


def in_residual(choice, truth, artifacts):
    """RFC-0073 'What the prover cannot detect': the residual rule on the entry as restored, before
    the run, with 'What a presented artifact catches'. The truth is the last record the prover
    wrote."""
    # The precondition: every file decodes, and the record is present and behind the truth.
    if any(v == CORRUPT for v in choice.values()):
        return False
    rec, truth_rec = choice[RECORD], truth[RECORD]
    if rec is None or rec == truth_rec:
        return False
    roots = rec['roots']
    latest = len(roots) - 1
    if any(names_past(a, roots) for a in artifacts):
        return False  # a presented root past the record refuses every command
    st = choice[state(latest)]
    if latest < len(truth_rec['roots']) - 1:
        # Across a link: no state or .twr of an epoch past L, no link from L or a later epoch, and
        # epoch-L.twc absent or open.
        later = [f for f in FILES if f != RECORD and (f[1] > latest if f[0] in ('twc', 'twr') else f[1] >= latest)]
        if any(choice[f] is not None for f in later):
            return False
        return st is None or st['next_blinding_row'] <= m.B - m.LINK_ROWS
    # Within the epoch: epoch-L.twc present, and its cursor, raised to the record's marks and to
    # every presented reservation, behind the truth's. A lost latest state is rebuilt: harmless.
    if st is None:
        return False
    shown = max([shown_end(a, roots) for a in artifacts], default=0)
    cursor = max(st['next_blinding_row'], rec['marks']['next_blinding_row'], shown)
    return cursor < truth_rec['marks']['next_blinding_row']


def bound_excess(world, events=1, epochs=4, K='K1'):
    """'The bound', per matrix and per epoch: each event lets the proofs that follow spend that
    epoch's budget once more. Rows revealed again: up to B_T − 4 in epoch 0 and B_T − 8 later;
    distinct columns past P_W: up to P_W − τ_link in epoch 0 and P_W − 2τ_link later. Returns the
    epochs past it, with their figures."""
    excess = []
    for e in range(epochs):
        rt = root(K, e)
        rows = world.extra_reveals(rt)
        cols = max(0, len(world.cols.get(rt, ())) - m.P_W)
        row_bound = events * (m.B - m.LINK_ROWS if e == 0 else m.B - 2 * m.LINK_ROWS)
        col_bound = events * (m.P_W - m.TAU_LINK if e == 0 else m.P_W - 2 * m.TAU_LINK)
        if rows > row_bound or cols > col_bound:
            excess.append(f'epoch {e}: rows {rows}/{row_bound}, columns past P_W {cols}/{col_bound}')
    return excess


def bound_figures(world, events=1, epochs=4, K='K1'):
    out = []
    for e in range(epochs):
        rt = root(K, e)
        row_bound = events * (m.B - m.LINK_ROWS if e == 0 else m.B - 2 * m.LINK_ROWS)
        col_bound = events * (m.P_W - m.TAU_LINK if e == 0 else m.P_W - 2 * m.TAU_LINK)
        cols = max(0, len(world.cols.get(rt, ())) - m.P_W)
        out.append(f'e{e} {world.extra_reveals(rt)}/{row_bound} rows, {cols}/{col_bound} cols')
    return '; '.join(out)


# ---- The rules the full replay runs: RFC-0073's, and the readings the controls run ------------------


class RaisedToState(Prover):
    """RFC-0012's reserve row as read 6 found it worded: 'its cursor and column count start from
    the record's marks, raised to the state's': the cursor to the state's, the count to
    |opened_columns|."""

    def budget_start(self, rec, st, shown):
        cursor = max(rec['marks']['next_blinding_row'], st['next_blinding_row'] if st else 0)
        count = max(rec['marks']['opened_columns'], len(st['opened_columns']) if st else 0)
        return self.raise_by_shown(rec, shown, cursor, count)

    def raise_by_shown(self, rec, shown, cursor, count):
        end = (shown or {}).get(rec['roots'][-1])
        if end is not None and end > cursor:
            count += m.TAU_W * -(-(end - cursor) // m.R)
            cursor = end
        return cursor, count


class LargerOf(RaisedToState):
    """'From the larger of the state's cursor and the record's marks', as RFC-0073's Summary,
    RFC-0064 'Not proofs' and the simulator worded it: the cursor raised, the count the record's."""

    def budget_start(self, rec, st, shown):
        cursor = max(rec['marks']['next_blinding_row'], st['next_blinding_row'] if st else 0)
        return self.raise_by_shown(rec, shown, cursor, rec['marks']['opened_columns'])


RULES = {'rfc': Prover, 'raised-to-state': RaisedToState, 'larger-of': LargerOf}
CONTROL_RUNS = [(64, 'none', 'raised-to-state'), (64, 'ground', 'raised-to-state'), (64, 'none', 'larger-of')]


# ---- 1. The full replay ---------------------------------------------------------------------------


def history(prover_class=Prover):
    """Three epochs and two links: proofs until reserve fails, then weights refresh; two proofs in
    epoch 2. Returns the world, the prover, a backup after every write, and the proofs a verifier
    rejected after their rows were revealed: the last of each epoch."""
    world = World()
    p = prover_class(world, Directory(), {}, 'A')
    backups = []
    p.on_write = lambda: backups.append(p.dir.snapshot())
    p.commit_weights('K1')
    last = {}
    for epoch in range(3):
        proofs = 0
        while not (epoch == 2 and proofs == 2):
            try:
                last[epoch] = p.prove()
                proofs += 1
            except Refused as r:
                assert r.code == 'ERR-WITNESS-BUDGET', r
                p.weights_refresh()
                break
    p.on_write = None
    return world, p, backups, set(last.values())


def prepare(B):
    m.B = B
    world, p, backups, rejected = history()
    versions = {f: [] for f in FILES}
    for backup in backups:
        for f in FILES:
            v = backup.get(f)
            if v is not None and v not in versions[f]:
                versions[f].append(v)
    return {
        'B': B,
        'blob': pickle.dumps((world, p)),
        'truth': p.dir.snapshot(),
        'versions': versions,
        'sizes': [len(versions[f]) + 1 for f in FILES],  # every version, or lost
        'rejected': rejected,
    }


def ledger_rows(world, rejected):
    """root -> the first row past every reservation of the proofs not rejected."""
    rows = {}
    for (rt, row), pids in world.rows.items():
        if pids - rejected:
            rows[rt] = max(rows.get(rt, 0), row + 1)
    return rows


def kept_round1s(world):
    return [presented_round1(e, rt, end) for e, rt, end in world.round1s.values()]


def presenting(variant, world, H):
    """What the operator presents to every command: fixed artifacts, and a function of the world
    for those that grow as proofs are made."""
    roots = H['truth'][RECORD]['roots']
    twr = [presented_public(len(roots) - 1, roots[-1])]
    chain = list(enumerate(roots))  # the verifier accepted every link
    if variant == 'twr':
        return twr, None
    if variant == 'realistic':
        other_rows = {OTHER_MODEL[1][1]: 10}
        return twr, lambda: [presented_ledger([OTHER_MODEL, chain], {**other_rows, **ledger_rows(world, H['rejected'])})]
    if variant == 'ground':
        return twr, lambda: kept_round1s(world)
    if variant == 'round1s':
        return [], lambda: kept_round1s(world)
    return [], None  # none, corrupt


def replay(H, variant, idx, corrupt, rule):
    m.B = H['B']
    choice = {f: None if i == 0 else H['versions'][f][i - 1] for f, i in zip(FILES, idx)}
    if corrupt is not None:
        choice[corrupt] = CORRUPT
    world, p = pickle.loads(H['blob'])
    p.__class__ = RULES[rule]
    for f in FILES:
        if choice[f] is None:
            p.dir.remove(f)
        else:
            p.dir.put(f, choice[f])
    presented, evidence = presenting(variant, world, H)
    predicted = in_residual(choice, H['truth'], [*presented, *(evidence() if evidence else ())])
    outcome = operate(p, STEPS, presented, evidence)
    return world, outcome, predicted, choice


def label(H, choice):
    out = []
    for f in FILES:
        v = choice[f]
        out.append('-' if v is None else 'X' if v == CORRUPT else 'cur' if v == H['truth'].get(f) else f'v{H["versions"][f].index(v)}')
    return '(' + ' '.join(out) + ')'


PREPARED = {}


def run_chunk(job):
    """Replays the combinations start .. stop of one variant (one corrupt file for 'corrupt')."""
    B, variant, k, start, stop, rule = job
    H = PREPARED[B]
    sizes = list(H['sizes'])
    if k is not None:
        sizes[k] = 1  # the corrupt file replaces that file's versions
    agg = dict.fromkeys(('runs', 'harmful', 'rows', 'cols', 'residual', 'outside', 'harmless', 'bound'), 0)
    examples = []
    for flat in range(start, stop):
        idx = []
        for size in reversed(sizes):
            flat, i = divmod(flat, size)
            idx.append(i)
        world, outcome, predicted, choice = replay(H, variant, idx[::-1], None if k is None else FILES[k], rule)
        rows, cols = world.reused_rows(), world.over_opened()
        harm = bool(rows or cols)
        excess = bound_excess(world)
        agg['runs'] += 1
        agg['harmful'] += harm
        agg['rows'] += bool(rows)
        agg['cols'] += bool(cols)
        agg['residual'] += predicted
        agg['bound'] += bool(excess)
        if harm != predicted:
            agg['outside' if harm else 'harmless'] += 1
        if (harm != predicted or excess) and len(examples) < 4:
            kind = 'past the bound' if excess else 'harmful outside the rule' if harm else 'in the rule, harmless'
            examples.append(f'{kind}: {label(H, choice)} rows {len(rows)} over-opened {cols} {excess} {outcome}')
    return B, variant, rule, agg, examples


def full_replay(jobs, report):
    for B in ROWS:
        PREPARED[B] = prepare(B)
    runs = [(B, variant, 'rfc') for B in ROWS for variant in VARIANTS] + CONTROL_RUNS
    work = []
    for B, variant, rule in runs:
        sizes = PREPARED[B]['sizes']
        total = 1
        for s in sizes:
            total *= s
        parts = [(k, total // sizes[k]) for k in range(len(FILES))] if variant == 'corrupt' else [(None, total)]
        for k, n in parts:
            work += [(B, variant, k, a, min(a + 1000, n), rule) for a in range(0, n, 1000)]
    if jobs > 1 and 'fork' in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context('fork').Pool(jobs) as pool:
            done = list(pool.imap_unordered(run_chunk, work, chunksize=4))
    else:
        done = [run_chunk(w) for w in work]
    results = {}
    for B, variant, rule, agg, examples in done:
        r = results.setdefault((B, variant, rule), {'examples': []} | dict.fromkeys(agg, 0))
        for key, v in agg.items():
            r[key] += v
        r['examples'] += examples
    for B in ROWS:
        H = PREPARED[B]
        grid = ' x '.join(str(s) for s in H['sizes'])
        report.line(f'-- B_T = {B}: versions per file (lost included) {grid}')
        report.line(
            f'   {"variant":<10} {"runs":>8} {"harmful":>8} {"reuse":>8} {"over-open":>9} {"in rule":>8} '
            f'{"outside":>8} {"harmless":>8} {"bound":>6}'
        )
        for variant in VARIANTS:
            r = results[(B, variant, 'rfc')]
            report.line(
                f'   {variant:<10} {r["runs"]:>8,} {r["harmful"]:>8,} {r["rows"]:>8,} {r["cols"]:>9,} '
                f'{r["residual"]:>8,} {r["outside"]:>8,} {r["harmless"]:>8,} {r["bound"]:>6,}'
            )
            report.check(
                f'B_T = {B}, {variant}: the rule matches the harm both ways, within the per-epoch bound',
                r['outside'] == 0 and r['harmless'] == 0 and r['bound'] == 0,
                '; '.join(r['examples'][:4]),
                quiet=True,
            )
            if variant in ('ground', 'round1s', 'corrupt'):
                report.check(f'B_T = {B}, {variant}: no harm', r['harmful'] == 0, quiet=True)
    return results


# ---- 2. Targeted replays ----------------------------------------------------------------------------

SCRIPT = ['commit', 'prove', 'prove', 'prove', 'prove', 'link', 'prove', 'prove', 'prove', 'prove', 'link', 'prove']


def run_script(p, crash_at, after, backups=None):
    """SCRIPT, the process dying at command `crash_at` before its write number `after`. Returns
    whether it died."""
    if backups is not None:
        p.on_write = lambda: backups.append(p.dir.snapshot())
    try:
        for i, cmd in enumerate(SCRIPT):
            if i == crash_at:
                p.writes_left = after
            try:
                if cmd == 'commit':
                    p.commit_weights('K1')
                elif cmd == 'prove':
                    try:
                        p.prove()
                    except Refused as r:
                        if r.code != 'ERR-WITNESS-BUDGET':
                            raise
                        p.weights_refresh()
                else:
                    p.link()
            except m.Crash:
                p.writes_left = None
                return True
            p.writes_left = None
        return False
    finally:
        p.on_write = None


def crashes(report):
    """RFC-0073 'Fresh and rebuilt states', 'Crashes fail safe'."""
    for B in ROWS:
        m.B = B
        points = bad = 0
        for ci in range(len(SCRIPT)):
            for k in range(8):
                world = World()
                p = Prover(world, Directory(), {}, 'A')
                if not run_script(p, ci, k):
                    continue
                points += 1
                operate(p, STEPS)
                bad += world.harm()
        report.check(f'B_T = {B}: a crash before every write, then a restart: {points} crash points, {bad} harmful', bad == 0 and points > 0)
        points = bad = 0
        for ci in range(len(SCRIPT)):
            for k in range(8):
                for j in range(12):
                    world = World()
                    p = Prover(world, Directory(), {}, 'A')
                    if not run_script(p, ci, k):
                        continue
                    p.writes_left = j
                    try:
                        operate(p, STEPS)
                        continue
                    except m.Crash:
                        p.writes_left = None
                    points += 1
                    operate(p, STEPS)
                    bad += world.harm()
        report.check(f'B_T = {B}: a second crash at every write of the recovery: {points} runs, {bad} harmful', bad == 0 and points > 0)


def crash_then_restore(report):
    """'What the prover cannot detect': with crashes the rule is conservative. A crash at every
    write of the scripted history, then the record alone, or the whole entry, restored to every
    backup: no harmful outcome lies outside the rule, whose truth is the last record written."""
    for B in ROWS:
        m.B = B
        runs = outside = harmless = 0
        examples = []
        for ci in range(len(SCRIPT)):
            for k in range(8):
                world = World()
                p = Prover(world, Directory(), {}, 'A')
                backups = []
                if not run_script(p, ci, k, backups):
                    continue
                truth = p.dir.snapshot()
                if truth.get(RECORD) is None:
                    continue  # the crash came before the first record
                blob = pickle.dumps((world, p))
                restores = [('record', {RECORD: b[RECORD]}) for b in backups if RECORD in b]
                restores += [('entry', b) for b in backups]
                for kind, files in restores:
                    w, q = pickle.loads(blob)
                    if kind == 'entry':
                        for f in q.dir.paths():
                            if f not in files:
                                q.dir.remove(f)
                    for f, v in files.items():
                        q.dir.put(f, v)
                    choice = {f: q.dir.get(f) for f in FILES}
                    predicted = in_residual(choice, truth, [])
                    operate(q, STEPS)
                    runs += 1
                    if w.harm() and not predicted:
                        outside += 1
                        if len(examples) < 3:
                            examples.append(f'crash at {SCRIPT[ci]}#{ci} write {k}, {kind} restored')
                    harmless += predicted and not w.harm()
        report.check(
            f'B_T = {B}: a crash, then the record or the whole entry restored to every backup: {runs} runs, '
            f'{outside} harmful outside the rule, {harmless} in it harmless (the conservative direction)',
            outside == 0 and runs > 0,
            '; '.join(examples),
        )


def second_process(report):
    """RFC-0073 'One proof at a time per entry: the lock': a second process is refused while the
    first proof holds the lock, the entry intact or any file removed; no harm follows."""
    for B in ROWS:
        m.B = B
        fails = []
        removals = [None, RECORD, state(0), state(1), public(0), public(1), link_file(0)]
        for removed in removals:
            world = World()
            a = new_entry(world)
            operate(a, 7)  # into epoch 1
            b = Prover(world, a.dir, a.keystore, 'A2')
            seen = []

            def second(removed=removed, b=b, a=a, seen=seen):
                if removed is not None:
                    a.dir.remove(removed)
                try:
                    b.prove()
                    seen.append('proved')
                except Refused as r:
                    seen.append(r.code)

            try:
                a.prove(before_queries=second)
            except Refused:
                pass  # the first proof's own file was removed: ERR-RES-IO before record_queries' write
            operate(a, STEPS)
            if seen != ['ERR-RES-IO'] or world.harm():
                fails.append(f'{removed}: second process {seen}, reused {len(world.reused_rows())}')
        report.check(
            f'B_T = {B}: a second process, the entry intact or each of {len(removals) - 1} files removed: ERR-RES-IO, no harm',
            not fails,
            '; '.join(fails),
        )


def copy_of(directory):
    d = Directory()
    d.files = dict(directory.files)
    return d


def copies(report):
    """'What the prover cannot detect': a copy proving beside the entry is the second residual case,
    with a per-epoch bound and no end in time; serialized copies, each presented the other's
    reservations, are caught."""
    for B in ROWS:
        m.B = B
        world = World()
        a = new_entry(world)
        a.prove()
        b = Prover(world, copy_of(a.dir), a.keystore, 'A-copy')
        a.prove(kept_round1s(world), before_reserve=lambda world=world, b=b: b.prove(kept_round1s(world)))
        report.check(
            f'B_T = {B}: two copies proving concurrently reuse rows (the residual, as stated): {len(world.reused_rows())} rows',
            world.harm(),
        )
        world = World()
        a = new_entry(world)
        a.prove()
        b = Prover(world, copy_of(a.dir), a.keystore, 'A-copy')
        for _ in range(4):
            for p in (a, b):
                operate(p, 1, evidence=lambda world=world: kept_round1s(world))
        report.check(
            f"B_T = {B}: serialized copies, each presented the other's reservations: reused {len(world.reused_rows())}, over-opened {world.over_opened()}",
            not world.harm(),
        )
        for n in (2, 3):
            world = World()
            a = new_entry(world)
            provers = [a] + [Prover(world, copy_of(a.dir), a.keystore, f'copy{i}') for i in range(1, n)]
            for _ in range(30):
                for p in provers:
                    operate(p, 1)
            epochs = max(e for e, _rt, _end in world.round1s.values()) + 1
            excess = bound_excess(world, events=n - 1, epochs=epochs)
            every = all(world.extra_reveals(root('K1', e)) > 0 for e in range(epochs - 1))
            report.check(
                f'B_T = {B}: {n} copies proving in turn, {epochs} epochs: rows reused in every epoch, within {n - 1} x the per-epoch bound '
                f'({bound_figures(world, n - 1, min(epochs, 3))}, ...)',
                not excess and every,
                '; '.join(excess),
            )


def alias_runs(prover_class):
    """A hard-link alias proving concurrently, a symbolic-link alias proving in turn, and a
    hard-link alias of the record and the state made while a proof holds the lock. Returns
    (what, rows reused, refusals) for each."""
    out = []
    world = World()
    a = prover_class(world, LinkingDirectory(), {}, 'A')
    a.commit_weights('K1')
    a.prove()
    b = prover_class(world, LinkingDirectory(), a.keystore, 'B')
    b.dir.hard_link(a.dir)
    refusals = []

    def second():
        try:
            b.prove()
        except Refused as r:
            refusals.append(r.code)

    try:
        a.prove(before_reserve=second)
    except Refused as r:
        refusals.append(r.code)
    out.append(('hard-link alias, concurrent', len(world.reused_rows()), refusals))
    world = World()
    a = prover_class(world, LinkingDirectory(), {}, 'A')
    a.commit_weights('K1')
    a.prove()
    b = prover_class(world, LinkingDirectory(), a.keystore, 'B')
    b.dir.symlink(a.dir)
    refusals = []
    for p in (a, b, a):
        try:
            p.prove()
        except Refused as r:
            refusals.append(r.code)
    out.append(('symbolic-link alias, in turn', len(world.reused_rows()), refusals))
    world = World()
    a = prover_class(world, LinkingDirectory(), {}, 'A')
    a.commit_weights('K1')
    a.prove()
    b = prover_class(world, LinkingDirectory(), a.keystore, 'B')
    refusals = []
    try:
        a.prove(before_reserve=lambda: b.dir.hard_link(a.dir, {RECORD, state(0)}))
    except Refused as r:
        refusals.append(r.code)
    try:
        b.prove()
    except Refused as r:
        refusals.append(r.code)
    out.append(('hard-link alias made during a proof', len(world.reused_rows()), refusals))
    return out


def aliases(report):
    """'The model entry', the path rules, at open and before every write."""
    for B in ROWS:
        m.B = B
        for what, reused, refusals in alias_runs(Prover):
            report.check(f'B_T = {B}: {what}: reused {reused}, refused {refusals}', reused == 0 and 'ERR-USAGE' in refusals)


def rolled_back_across_link(B, prover_class=Prover):
    """Record at epoch 0 from a backup, epoch 1's files lost, epoch 0's current (closed) state
    corrupted: the refuting file does not decode. Returns the world, the prover and the truth's
    latest .twr and Round1s, as the operator kept them."""
    m.B = B
    world = World()
    p = prover_class(world, Directory(), {}, 'A')
    p.commit_weights('K1')
    for _ in range(3):
        p.prove()
    backup = p.dir.snapshot()
    operate(p, 6)  # the link, then proofs in epoch 1
    roots = p.dir.get(RECORD)['roots']
    kept = [presented_public(len(roots) - 1, roots[-1]), *kept_round1s(world)]
    p.dir.put(RECORD, backup[RECORD])
    for f in (state(1), public(1), link_file(0)):
        p.dir.remove(f)
    p.dir.put(state(0), CORRUPT)
    return world, p, kept


def corrupted(report):
    """'The entry check' item 2 and 'The operator's duty': a corrupted file is never treated as lost.
    It is moved aside only together with the latest artifacts presented to weights refresh."""
    for B in ROWS:
        world, p, _kept = rolled_back_across_link(B)
        outcome = operate(p, 6)
        report.check(f'B_T = {B}: a corrupted refuting file: {outcome}; reused {len(world.reused_rows())}', not world.harm() and 'ERR-CODEC' in outcome)
        world, p, kept = rolled_back_across_link(B)
        p.dir.remove(state(0))  # moved aside
        try:
            p.weights_refresh(kept)
            refresh = 'ran'
        except Refused as r:
            refresh = str(r)
        outcome = operate(p, 6)
        report.check(
            f'B_T = {B}: moved aside, then weights refresh with the latest .twr and kept Round1s: {refresh[:40]}; then {outcome}',
            'retired' in refresh and not world.harm(),
        )


def restore_whole_entry(p, backup):
    for f in p.dir.paths():
        if f not in backup:
            p.dir.remove(f)
    for f, v in backup.items():
        p.dir.put(f, v)


def bounds(report):
    """The whole entry restored to just after commit-weights, then to just after the first link."""
    for B in (32, 34, 64):
        m.B = B
        for start in ('commit-weights', 'the first link'):
            world = World()
            p = new_entry(world)
            if start == 'the first link':
                while True:
                    try:
                        p.prove()
                    except Refused:
                        p.weights_refresh()
                        break
            backup = p.dir.snapshot()
            operate(p, STEPS)
            restore_whole_entry(p, backup)
            operate(p, STEPS)
            excess = bound_excess(world)
            report.check(f'B_T = {B}: the whole entry restored to just after {start}: {bound_figures(world)}', not excess and world.harm())


def zero_row_matrix(report):
    """'The model entry': closed and open over the matrices with committed rows only."""
    for B in ROWS:
        m.B = B
        world = World()
        p = new_entry(world, zero_row_matrices=1)
        outcome = operate(p, 8)
        report.check(f'B_T = {B}: a layout with a B_T = 0 matrix: {outcome}, {world.proofs} proofs, no harm', outcome == 'ran' and world.proofs > 0 and not world.harm())


def within_epoch_rollback(B, prover_class=Prover):
    """The record and the latest state restored to after the first proof of epoch 0; three proofs
    since. Returns the world, the prover and the Round1s the operator kept."""
    m.B = B
    world = World()
    p = prover_class(world, Directory(), {}, 'A')
    p.commit_weights('K1')
    p.prove()
    backup = p.dir.snapshot()
    for _ in range(2):
        p.prove()
    kept = kept_round1s(world)
    restore_whole_entry(p, backup)
    return world, p, kept


def across_link_rollback(B, prover_class=Prover):
    """The record and epoch 0's state restored to after the first proof, epoch 1's files lost.
    Returns the world, the prover and the Round1s the operator kept."""
    m.B = B
    world = World()
    p = prover_class(world, Directory(), {}, 'A')
    p.commit_weights('K1')
    p.prove()
    backup = p.dir.snapshot()
    operate(p, 8)
    kept = kept_round1s(world)
    for f in (state(1), public(1), link_file(0)):
        p.dir.remove(f)
    p.dir.put(RECORD, backup[RECORD])
    p.dir.put(state(0), backup[state(0)])
    return world, p, kept


def presented_artifacts(report):
    """'The entry check' item 5 and 'Stale persists'."""
    for B in ROWS:
        m.B = B
        world = World()
        p = new_entry(world)
        p.prove()
        forged = [presented_public(1, root('forged', 1))]
        first = operate(p, 1, presented=forged)
        try:
            p.weights_refresh(forged)
            refresh = 'ran'
        except Refused as r:
            refresh = r.code
        after = operate(p, 1)
        retired = p.dir.get(RECORD)['retired']
        report.check(
            f'B_T = {B}: a forged later root refuses prove and weights refresh ({refresh}), retires nothing; then: {after}',
            first.startswith('stopped: ERR-USAGE') and refresh == 'ERR-USAGE' and after == 'ran' and not retired,
        )
        world, p, _kept = across_link_rollback(B)
        true_root = root('K1', 1)
        try:
            p.weights_refresh([presented_public(1, true_root)])
            refresh = 'ran'
        except Refused as r:
            refresh = str(r)
        after = operate(p, 4)
        report.check(
            f'B_T = {B}: an authenticated later root retires the record, which stays retired: {after}',
            'retired' in refresh and 'the record is retired' in after and not world.harm(),
        )
        world, p, kept = across_link_rollback(B)
        first = operate(p, 1, presented=kept)
        try:
            p.weights_refresh(kept)
            refresh = 'ran'
        except Refused as r:
            refresh = str(r)
        report.check(
            f'B_T = {B}: kept Round1s alone, across a link: a Round1 of a later root refuses prove and retires through weights refresh; reused {len(world.reused_rows())}',
            first.startswith('stopped: ERR-USAGE') and 'retired' in refresh and not world.harm(),
        )
        world, p, kept = within_epoch_rollback(B)
        try:
            p.weights_refresh(kept)
            refresh = 'ran'
        except Refused as r:
            refresh = str(r)
        after = operate(p, STEPS)
        report.check(
            f'B_T = {B}: kept Round1s presented once to weights refresh, then nothing: the raise is in the record; {after}, reused {len(world.reused_rows())}',
            refresh == 'ran' and not world.harm(),
        )
        world = World()
        p = new_entry(world)
        retired_chain = [(0, root('K0', 0)), (1, root('K0', 1)), (2, root('K0', 2))]
        ledger = presented_ledger([OTHER_MODEL, retired_chain, [(0, root('K1', 0))]], {OTHER_MODEL[1][1]: 30, root('K0', 2): 30})
        outcome = operate(p, 8, presented=[ledger])
        report.check(
            f"B_T = {B}: a ledger also holding another model's and a retired entry's chains: {outcome}, {world.proofs} proofs",
            outcome == 'ran' and world.proofs > 0 and not world.harm(),
        )
        world = World()
        p = new_entry(world)
        p.prove()
        backup = p.dir.snapshot()
        operate(p, 8)
        p.dir.put(RECORD, backup[RECORD])
        first = operate(p, 1)
        for f in (state(1), public(1), link_file(0)):
            p.dir.remove(f)
        p.dir.put(state(0), backup[state(0)])
        after = operate(p, 4)
        report.check(
            f'B_T = {B}: a stale record is retired; removing the files the check named leaves it retired: {after}',
            'retired' in first and 'the record is retired' in after and not world.harm(),
        )


# ---- 3. Negative controls ---------------------------------------------------------------------------


class Read4Count(Prover):
    """reserve before RFC-0073 counted columns up front: the cursor and the count start from the
    larger of the state's and the record's, a presented reservation raises them, and the record
    write at reserve raises the cursor only. record_queries writes the state only: with nothing
    else, this is RFC-0012's row before RFC-0073."""

    def budget_start(self, rec, st, shown):
        cursor = max(st['next_blinding_row'] if st else 0, rec['marks']['next_blinding_row'])
        count = max(len(st['opened_columns']) if st else 0, rec['marks']['opened_columns'])
        end = (shown or {}).get(rec['roots'][-1])
        if end is not None and end > cursor:
            count += m.TAU_W * -(-(end - cursor) // m.R)
            cursor = end
        return cursor, count

    def write_reservation(self, rec, st, cursor, count):
        self.write(RECORD, {**rec, 'marks': marks(cursor + m.R, count)})
        self.write(state(len(rec['roots']) - 1), {**st, 'next_blinding_row': cursor + m.R})


class Rfc0012Before(Read4Count):
    """RFC-0012's record_queries row before RFC-0073: the record's count only copies the state's,
    at reserve."""


class CountAtRecordQueries(Read4Count):
    """Read 4's rule: record_queries raises the record's count, then writes the state."""

    def record_queries(self, rec, queries):
        path = state(len(rec['roots']) - 1)
        st = self.reread(path)
        now = self.reread(RECORD)
        fresh = len(queries - st['opened_columns'])
        self.write(RECORD, {**now, 'marks': marks(now['marks']['next_blinding_row'], now['marks']['opened_columns'] + fresh)})
        self.write(path, {**st, 'opened_columns': st['opened_columns'] | queries})


class PathRulesAtOpenOnly(Prover):
    """The path rules in the entry check only, not again before each write."""

    def check_path_before_write(self, path):
        pass


class NoPathRules(PathRulesAtOpenOnly):
    """No path rules at all."""

    def check_paths(self):
        pass


class CorruptAsLost(Prover):
    """A corrupted entry file read as lost, as RFC-0012 'Failure modes' read before RFC-0073."""

    def check_decodes(self):
        for path in self.dir.paths():
            if self.dir.get(path) == CORRUPT:
                self.dir.remove(path)
                self.seen.pop(path, None)

    def readable_record(self):
        for path in self.dir.paths():
            if path != RECORD and self.dir.get(path) == CORRUPT:
                self.dir.remove(path)
        return super().readable_record()


class EveryMatrix(Prover):
    """Closed and open read over every matrix, a B_T = 0 one included."""

    def committed(self, st):
        return [(st['next_blinding_row'], m.B)] + [(0, 0)] * self.zero_row_matrices


class UnscopedLedger(Prover):
    """Item 5 reading every (epoch, root) of a ledger, other entries' chains included."""

    def chain_of(self, ledger, roots):
        return [pair for chain in ledger['chains'] for pair in chain]


class RaiseNotPersisted(Prover):
    """weights refresh that does not write a presented raise into the record."""

    def persist_raise(self, rec, shown):
        pass


def controls(report, results):
    for B, variant, rule in CONTROL_RUNS:
        r = results[(B, variant, rule)]
        report.check(
            f"control: B_T = {B}, {variant}, the count read as '{rule}': {r['outside']:,} harmful outside the rule "
            f"({r['rows']:,} reuse, {r['cols']:,} over-open)",
            r['outside'] > 0,
        )
    m.B = 64  # columns end an epoch first
    for cls in (Rfc0012Before, Prover):
        world = World()
        p = cls(world, Directory(), {}, 'A')
        p.commit_weights('K1')
        backup = p.dir.snapshot()
        for _ in range(4):
            p.prove()
        p.dir.put(state(0), backup[state(0)])  # a restored state alone
        operate(p, 8)
        if cls is Prover:
            report.check(f'B_T = 64: a restored latest state alone, under RFC-0073: over-opened {world.over_opened()}', not world.harm())
        else:
            report.check(
                f"control: RFC-0012's record_queries row before RFC-0073, a restored state: over-opened {world.over_opened()}",
                bool(world.over_opened()),
            )
    # Read 4's count at record_queries, the whole entry restored from each backup, the latest .twr
    # and every kept Round1 presented.
    world0, p0, backups, _ = history(CountAtRecordQueries)
    blob = pickle.dumps((world0, p0))
    roots = p0.dir.get(RECORD)['roots']
    harmful = 0
    for backup in backups:
        world, p = pickle.loads(blob)
        restore_whole_entry(p, backup)
        operate(p, STEPS, [presented_public(len(roots) - 1, roots[-1])], lambda world=world: kept_round1s(world))
        harmful += bool(world.over_opened())
    report.check(
        f'control: the count raised at record_queries (read 4), every Round1 presented: {harmful} of {len(backups)} restores over-open',
        harmful > 0,
    )
    for B in ROWS:
        m.B = B
        for what, reused, _refusals in alias_runs(NoPathRules):
            report.check(f'control: B_T = {B}, {what} without the path rules: reused {reused}', reused > 0)
        what, reused, _refusals = alias_runs(PathRulesAtOpenOnly)[2]
        report.check(f'control: B_T = {B}, {what}, path rules at open only: reused {reused}', reused > 0)
        world, p, _kept = rolled_back_across_link(B, CorruptAsLost)
        operate(p, 6)
        report.check(f'control: B_T = {B}, a corrupted refuting file read as lost: reused {len(world.reused_rows())}', world.harm())
        world = World()
        p = EveryMatrix(world, Directory(), {}, 'A', zero_row_matrices=1)
        p.commit_weights('K1')
        outcome = operate(p, 8)
        report.check(f'control: B_T = {B}, closed and open over every matrix, B_T = 0 included: {outcome}', world.proofs == 0)
        world = World()
        p = UnscopedLedger(world, Directory(), {}, 'A')
        p.commit_weights('K1')
        ledger = presented_ledger([OTHER_MODEL, [(0, root('K1', 0))]], {})
        outcome = operate(p, 8, presented=[ledger])
        report.check(f"control: B_T = {B}, a ledger read beyond this entry's chain: {outcome}", world.proofs == 0)
        world, p, kept = within_epoch_rollback(B, RaiseNotPersisted)
        p.weights_refresh(kept)
        operate(p, STEPS)
        report.check(f'control: B_T = {B}, a presented raise not written into the record: reused {len(world.reused_rows())}', world.harm())


# ---- The report -------------------------------------------------------------------------------------


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def line(self, text):
        print(text, flush=True)

    def check(self, what, ok, detail='', quiet=False):
        if ok:
            self.passed += 1
            if not quiet:
                print(f'   PASS {what}', flush=True)
        else:
            self.failed.append(what)
            print(f'   FAIL {what}' + (f': {detail}' if detail else ''), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1, help='worker processes for the full replay')
    args = parser.parse_args()
    started = time.monotonic()
    report = Report()
    report.line("== 1. The full replay: RFC-0073's residual rule against the harm observed")
    results = full_replay(args.jobs, report)
    report.line('== 2. Targeted replays')
    for part in (crashes, crash_then_restore, second_process, copies, aliases, corrupted, bounds, zero_row_matrix, presented_artifacts):
        part(report)
    report.line('== 3. Negative controls: each must show the harm its rule prevents')
    controls(report, results)
    elapsed = time.monotonic() - started
    verdict = 'PASS' if not report.failed else f'FAIL ({len(report.failed)} failed)'
    report.line(f'GATE: {verdict}: {report.passed} checks passed in {elapsed:.0f} s')
    return 0 if not report.failed else 1


if __name__ == '__main__':
    sys.exit(main())
