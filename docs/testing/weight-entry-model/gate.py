#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The gate of draft RFC-0073's residual rule ('What the prover cannot detect').

It replays the rules of model.py, at B_T = 32 (rows end an epoch first) and B_T = 64 (columns do):

1. The full replay. A history of three epochs and two links, with a backup after every write.
   Each of the eight entry files is then put back at every version it had, or lost, in every
   combination, and an honest operator proves on. Five variants: nothing presented; the latest
   .twr; that and a real ledger (accepted proofs' rows only, the last proof of each epoch
   rejected after its rows were revealed); that and every sent reservation; and one entry file
   present but corrupt, each in turn. The residual rule classifies every outcome before the run,
   and the gate checks it both ways against the harm observed (a row reused or a matrix
   over-opened): 0 harmful outcomes outside the rule, 0 in it without harm.
2. Targeted replays of what the full replay cannot build: crashes, double crashes, a second
   process, copies, aliases, a corrupted refuting file, the residual's bound, a matrix with
   B_T = 0, presented roots forged and authenticated, and a retired record.
3. Negative controls: each earlier or naive reading of a rule must show the harm it predicts,
   or the replay cannot see the bug the rule prevents.

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
    public,
    root,
    state,
)

ROWS = (32, 64)
FILES = [RECORD, state(0), state(1), state(2), public(1), public(2), link_file(0), link_file(1)]
VARIANTS = ('none', 'twr', 'realistic', 'ground', 'corrupt')
STEPS = 14  # proofs the operator attempts after a restore: past every epoch's budget


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


def in_residual(choice, truth, variant, shown):
    """RFC-0073 'What the prover cannot detect': the residual rule, on the entry as restored, before
    the run. `shown`: the reservations presented, if any ('What a presented artifact catches')."""
    # The precondition: every file decodes, and the record is present and behind the truth.
    if any(v == CORRUPT for v in choice.values()):
        return False
    rec = choice[RECORD]
    if rec is None or rec == truth[RECORD]:
        return False
    latest = len(rec['roots']) - 1
    st = choice[state(latest)]
    if latest < len(truth[RECORD]['roots']) - 1:
        # Across a link: no state, .twr or link of an epoch past L, and epoch-L.twc absent or open.
        if variant != 'none':
            return False  # the latest .twr is presented, and refuses every command
        later = [
            f for f in FILES if f != RECORD and ((f[0] in ('twc', 'twr') and f[1] > latest) or (f[0] == 'twl' and f[1] >= latest))
        ]
        if any(choice[f] is not None for f in later):
            return False
        return st is None or st['next_blinding_row'] <= m.B - m.LINK_ROWS
    # Within the epoch: epoch-L.twc present, and its cursor, raised to the record's marks and to
    # every presented reservation, behind the truth's. A lost latest state is rebuilt: harmless.
    if st is None:
        return False
    cursor = max(st['next_blinding_row'], rec['marks']['next_blinding_row'], (shown or {}).get(rec['roots'][latest], 0))
    return cursor < truth[state(latest)]['next_blinding_row']


def replay(H, variant, idx, corrupt):
    m.B = H['B']
    choice = {f: None if i == 0 else H['versions'][f][i - 1] for f, i in zip(FILES, idx)}
    if corrupt is not None:
        choice[corrupt] = CORRUPT
    world, p = pickle.loads(H['blob'])
    for f in FILES:
        if choice[f] is None:
            p.dir.remove(f)
        else:
            p.dir.put(f, choice[f])
    roots = H['truth'][RECORD]['roots']
    presented = [] if variant in ('none', 'corrupt') else [(len(roots) - 1, roots[-1])]
    rejected = {'realistic': H['rejected'], 'ground': set()}.get(variant)
    ledger = None if rejected is None else (lambda: ledger_rows(world, rejected))
    shown = ledger() if ledger else None
    predicted = in_residual(choice, H['truth'], variant, shown)
    outcome = operate(p, STEPS, presented, ledger)
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
    B, variant, k, start, stop = job
    H = PREPARED[B]
    sizes = list(H['sizes'])
    if k is not None:
        sizes[k] = 1  # the corrupt file replaces that file's versions
    agg = {'runs': 0, 'harmful': 0, 'rows': 0, 'cols': 0, 'residual': 0, 'outside': 0, 'harmless': 0}
    examples = []
    for flat in range(start, stop):
        idx = []
        for size in reversed(sizes):
            flat, i = divmod(flat, size)
            idx.append(i)
        world, outcome, predicted, choice = replay(H, variant, idx[::-1], None if k is None else FILES[k])
        rows, cols = world.reused_rows(), world.over_opened()
        harm = bool(rows or cols)
        agg['runs'] += 1
        agg['harmful'] += harm
        agg['rows'] += bool(rows)
        agg['cols'] += bool(cols)
        agg['residual'] += predicted
        if harm != predicted:
            agg['outside' if harm else 'harmless'] += 1
            if len(examples) < 4:
                kind = 'harmful outside the rule' if harm else 'in the rule, harmless'
                examples.append(f'{kind}: {label(H, choice)} rows {len(rows)} over-opened {cols} {outcome}')
    return B, variant, agg, examples


def full_replay(jobs, report):
    for B in ROWS:
        PREPARED[B] = prepare(B)
    work = []
    for B in ROWS:
        sizes = PREPARED[B]['sizes']
        total = 1
        for s in sizes:
            total *= s
        for variant in VARIANTS:
            parts = [(None, total)]
            if variant == 'corrupt':
                parts = [(k, total // sizes[k]) for k in range(len(FILES))]
            for k, n in parts:
                step = 1000
                work += [(B, variant, k, a, min(a + step, n)) for a in range(0, n, step)]
    results = {}
    if jobs > 1 and 'fork' in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context('fork').Pool(jobs) as pool:
            done = list(pool.imap_unordered(run_chunk, work, chunksize=4))
    else:
        done = [run_chunk(w) for w in work]
    for B, variant, agg, examples in done:
        r = results.setdefault((B, variant), {'examples': []} | dict.fromkeys(agg, 0))
        for key, v in agg.items():
            r[key] += v
        r['examples'] += examples
    for B in ROWS:
        H = PREPARED[B]
        grid = ' x '.join(str(s) for s in H['sizes'])
        report.line(f'-- B_T = {B}: versions per file (lost included) {grid}')
        report.line(f'   {"variant":<10} {"runs":>8} {"harmful":>8} {"reuse":>8} {"over-open":>9} {"in rule":>8} {"outside":>8} {"harmless":>8}')
        for variant in VARIANTS:
            r = results[(B, variant)]
            report.line(
                f'   {variant:<10} {r["runs"]:>8,} {r["harmful"]:>8,} {r["rows"]:>8,} {r["cols"]:>9,} '
                f'{r["residual"]:>8,} {r["outside"]:>8,} {r["harmless"]:>8,}'
            )
            report.check(
                f'B_T = {B}, {variant}: the rule matches the harm both ways',
                r['outside'] == 0 and r['harmless'] == 0,
                '; '.join(r['examples'][:4]),
                quiet=True,
            )
            if variant in ('ground', 'corrupt'):
                report.check(f'B_T = {B}, {variant}: no harm', r['harmful'] == 0, quiet=True)


# ---- 2. Targeted replays ----------------------------------------------------------------------------

SCRIPT = ['commit', 'prove', 'prove', 'prove', 'prove', 'link', 'prove', 'prove', 'prove', 'prove', 'link', 'prove']


def run_script(p, crash_at, after):
    """SCRIPT, the process dying at command `crash_at` before its write number `after`. Returns
    whether it died."""
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
    """'What the prover cannot detect': a copy proving beside the entry is the second residual case;
    serialized copies, each presented the other's reservations, are caught."""
    for B in ROWS:
        m.B = B
        world = World()
        a = new_entry(world)
        a.prove()
        b = Prover(world, copy_of(a.dir), a.keystore, 'A-copy')
        a.prove(
            presented_rows=ledger_rows(world, set()),
            before_reserve=lambda world=world, b=b: b.prove(presented_rows=ledger_rows(world, set())),
        )
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
                operate(p, 1, ledger=lambda world=world: ledger_rows(world, set()))
        report.check(
            f"B_T = {B}: serialized copies, each presented the other's reservations: reused {len(world.reused_rows())}, over-opened {world.over_opened()}",
            not world.harm(),
        )


def alias_runs(prover_class):
    """A hard-link alias proving concurrently, and a symbolic-link alias proving in turn.
    Returns (rows reused, refusals) for each."""
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
    return out


def aliases(report):
    """'The model entry', the path rules."""
    for B in ROWS:
        m.B = B
        for what, reused, refusals in alias_runs(Prover):
            report.check(f'B_T = {B}: {what}: reused {reused}, refused {refusals}', reused == 0 and 'ERR-USAGE' in refusals)


def rolled_back_across_link(B, prover_class=Prover):
    """Record at epoch 0 from a backup, epoch 1's files lost, epoch 0's current (closed) state
    corrupted: the refuting file does not decode."""
    m.B = B
    world = World()
    p = prover_class(world, Directory(), {}, 'A')
    p.commit_weights('K1')
    for _ in range(3):
        p.prove()
    backup = p.dir.snapshot()
    operate(p, 6)  # the link, then proofs in epoch 1
    p.dir.put(RECORD, backup[RECORD])
    for f in (state(1), public(1), link_file(0)):
        p.dir.remove(f)
    p.dir.put(state(0), CORRUPT)
    return world, operate(p, 6)


def corrupted(report):
    """'The entry check' item 2: a corrupted file is never treated as lost."""
    for B in ROWS:
        world, outcome = rolled_back_across_link(B)
        report.check(f'B_T = {B}: a corrupted refuting file: {outcome}; reused {len(world.reused_rows())}', not world.harm() and 'ERR-CODEC' in outcome)


def restore_whole_entry(p, backup):
    for f in p.dir.paths():
        if f not in backup:
            p.dir.remove(f)
    for f, v in backup.items():
        p.dir.put(f, v)


def bound_check(world, epoch_roots):
    """'The bound': per matrix and epoch, rows reused up to B_T − 4 in epoch 0 and B_T − 8 later;
    distinct columns past P_W up to P_W − τ_link in epoch 0 and P_W − 2τ_link later."""
    worst = []
    ok = True
    for e, rt in epoch_roots:
        rows = sum(1 for (r, _row) in world.reused_rows() if r == rt)
        cols = max(0, len(world.cols.get(rt, ())) - m.P_W)
        row_bound = m.B - m.LINK_ROWS if e == 0 else m.B - 2 * m.LINK_ROWS
        col_bound = m.P_W - m.TAU_LINK if e == 0 else m.P_W - 2 * m.TAU_LINK
        ok &= rows <= row_bound and cols <= col_bound
        worst.append(f'epoch {e}: rows {rows}/{row_bound}, columns past P_W {cols}/{col_bound}')
    return ok, '; '.join(worst)


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
            ok, detail = bound_check(world, [(e, root('K1', e)) for e in range(4)])
            report.check(f'B_T = {B}: the whole entry restored to just after {start}: {detail}', ok and world.harm())


def zero_row_matrix(report):
    """'The model entry': closed and open over the matrices with committed rows only."""
    for B in ROWS:
        m.B = B
        world = World()
        p = new_entry(world, zero_row_matrices=1)
        outcome = operate(p, 8)
        report.check(f'B_T = {B}: a layout with a B_T = 0 matrix: {outcome}, {world.proofs} proofs, no harm', outcome == 'ran' and world.proofs > 0 and not world.harm())


def presented_roots(report):
    """'The entry check' item 5 and 'Stale persists'."""
    for B in ROWS:
        m.B = B
        world = World()
        p = new_entry(world)
        p.prove()
        forged = [(1, root('forged', 1))]
        first = operate(p, 1, presented=forged)
        try:
            p.weights_refresh(presented=forged)
            refresh = 'ran'
        except Refused as r:
            refresh = r.code
        after = operate(p, 1)
        retired = p.dir.get(RECORD)['retired']
        report.check(
            f'B_T = {B}: a forged later root refuses prove and weights refresh ({refresh}), retires nothing; then: {after}',
            first.startswith('stopped: ERR-USAGE') and refresh == 'ERR-USAGE' and after == 'ran' and not retired,
        )
        # The record and epoch 0's state rolled back across a link, epoch 1's files lost: the
        # residual, until weights refresh is presented the true epoch-1 root.
        world = World()
        p = new_entry(world)
        p.prove()
        backup = p.dir.snapshot()
        operate(p, 8)
        true_root = p.dir.get(RECORD)['roots'][1]
        for f in (state(1), public(1), link_file(0)):
            p.dir.remove(f)
        p.dir.put(RECORD, backup[RECORD])
        p.dir.put(state(0), backup[state(0)])
        try:
            p.weights_refresh(presented=[(1, true_root)])
            refresh = 'ran'
        except Refused as r:
            refresh = str(r)
        after = operate(p, 4)
        report.check(
            f'B_T = {B}: an authenticated later root retires the record, which stays retired: {after}',
            'retired' in refresh and 'the record is retired' in after and not world.harm(),
        )
        # A stale record found by a file past it; removing the file does not un-retire it.
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

    def budget_start(self, rec, st, presented_rows):
        cursor = max(st['next_blinding_row'], rec['marks']['next_blinding_row'])
        count = max(len(st['opened_columns']), rec['marks']['opened_columns'])
        shown = (presented_rows or {}).get(rec['roots'][-1])
        if shown is not None and shown > cursor:
            count += m.TAU_W * -(-(shown - cursor) // m.R)
            cursor = shown
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


class NoPathRules(Prover):
    """The entry check without the path rules."""

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


def controls(report):
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
    # and every sent reservation presented.
    world0, p0, backups, _ = history(CountAtRecordQueries)
    blob = pickle.dumps((world0, p0))
    roots = p0.dir.get(RECORD)['roots']
    harmful = 0
    for backup in backups:
        world, p = pickle.loads(blob)
        restore_whole_entry(p, backup)
        operate(p, STEPS, [(len(roots) - 1, roots[-1])], lambda world=world: ledger_rows(world, set()))
        harmful += bool(world.over_opened())
    report.check(
        f"control: the count raised at record_queries (read 4), every reservation presented: {harmful} of {len(backups)} restores over-open",
        harmful > 0,
    )
    for B in ROWS:
        m.B = B
        for what, reused, _refusals in alias_runs(NoPathRules):
            report.check(f'control: B_T = {B}, {what} without the path rules: reused {reused}', reused > 0)
        world, outcome = rolled_back_across_link(B, CorruptAsLost)
        report.check(f'control: B_T = {B}, a corrupted refuting file read as lost: reused {len(world.reused_rows())}', world.harm())
        world = World()
        p = EveryMatrix(world, Directory(), {}, 'A', zero_row_matrices=1)
        p.commit_weights('K1')
        outcome = operate(p, 8)
        report.check(f'control: B_T = {B}, closed and open over every matrix, B_T = 0 included: {outcome}', world.proofs == 0)


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
    full_replay(args.jobs, report)
    report.line('== 2. Targeted replays')
    for part in (crashes, second_process, copies, aliases, corrupted, bounds, zero_row_matrix, presented_roots):
        part(report)
    report.line('== 3. Negative controls: each must show the harm its rule prevents')
    controls(report)
    elapsed = time.monotonic() - started
    verdict = 'PASS' if not report.failed else f'FAIL ({len(report.failed)} failed)'
    report.line(f'GATE: {verdict}: {report.passed} checks passed in {elapsed:.0f} s')
    return 0 if not report.failed else 1


if __name__ == '__main__':
    sys.exit(main())
