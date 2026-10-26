# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The weight model entry of draft RFC-0073, as an executable model.

RFC-0073 'The model entry' through 'What the prover cannot detect' say how a prover keeps one
model's weight commitments, epoch after epoch, so that no blinding row is revealed twice and no
matrix opens more than P_W distinct columns in an epoch (RFC-0012 INV-WC-1). This module states
those rules as code, one section at a time, with RFC-0012 'Budget state machine' and RFC-0064 'The
budget' and 'One link per epoch'. Each class and method cites the section it implements.

gate.py replays these rules against every restore, loss, corruption, crash, copy and alias it
builds, and checks the residual rule of RFC-0073 'What the prover cannot detect' against the harm
it observes. README.md lists what the model abstracts away.

Standard library only; Python 3.11 or later. Stored values are never changed in place: every
transition writes new values, so a backup may share them.
"""

import contextlib
import copy
import random

# ---- Parameters ---------------------------------------------------------------------------------
# One single-claim matrix at test-small figures (README 'What it abstracts'). The gate sets B.

B = 32  # B_T, the matrix's blinding rows: 32 has rows end an epoch first, 64 has columns
R = 6  # r_T = 2·e_T + 4, the rows one reservation burns for one claim (RFC-0012 'Budget state machine')
N_COLS = 512  # n_T, the codeword length the column queries are drawn from
P_W = 24  # the matrix's column budget in an epoch (RFC-0012 INV-WC-1)
TAU_W = 4  # τ_W, the distinct columns one proof opens at most
TAU_LINK = 4  # τ_link, the columns a link opens in each of its two epochs (RFC-0064 'The budget')
LINK_ROWS = 4  # a link's rows: B_T − 4 .. B_T − 1 of epoch e, 0 .. 3 of e + 1 (RFC-0064 'The budget')

# ---- The model entry's files (RFC-0073 'The model entry') ---------------------------------------

RECORD = 'epochs.twe'  # WeightEpochRecord, schema 0x0305


def state(e):
    """epoch-<e>.twc: epoch e's WeightCommitmentState (RFC-0012 'Budget state machine')."""
    return ('twc', e)


def public(e):
    """epoch-<e>.twr: epoch e's WeightCommitmentPublic."""
    return ('twr', e)


def link_file(e):
    """link-<e>.twl: the WeightLinkProof from epoch e to e + 1 (RFC-0064)."""
    return ('twl', e)


def name(path):
    """The file's name in the entry directory."""
    if path == RECORD:
        return RECORD
    kind, e = path
    return f'link-{e}.twl' if kind == 'twl' else f'epoch-{e}.{kind}'


CORRUPT = '<present, does not decode>'  # an entry file that exists but does not decode


def marks(next_blinding_row, opened_columns):
    """BudgetMark: the latest epoch's cursor and column count, in the record."""
    return {'next_blinding_row': next_blinding_row, 'opened_columns': opened_columns}


def root(K, e):
    """R_W of epoch e: a function of the key K, the epoch and the weights (RFC-0073 'The seeds',
    RFC-0064 'One link per epoch'). The model's weights are fixed, so K and e name the root."""
    return f'R[{K}:{e}]'


def link_columns(K, e):
    """The columns the link from epoch e opens: a function of K and e, the same however often the
    link is made (RFC-0064 'One link per epoch')."""
    return frozenset(random.Random(f'link{K}{e}').sample(range(N_COLS), TAU_LINK))


def fresh_state(rt, columns):
    """A fresh state of a linked epoch: from row 4, with the link's columns opened (RFC-0073 'Fresh
    and rebuilt states'; RFC-0064 'The budget')."""
    return {'root': rt, 'next_blinding_row': LINK_ROWS, 'opened_columns': frozenset(columns)}


def rebuilt_state(rt):
    """A rebuilt state: at B_T − 4, where reserve always fails and the link's rows begin, with no
    column opened (RFC-0073 'Fresh and rebuilt states')."""
    return {'root': rt, 'next_blinding_row': B - LINK_ROWS, 'opened_columns': frozenset()}


class Refused(Exception):
    """A command refused with an error code of RFC-0073 'Errors' (04 §5)."""

    def __init__(self, code, why):
        super().__init__(f'{code}: {why}')
        self.code = code
        self.why = why


class Crash(Exception):
    """The process dies before a write: nothing of the write lands, and its lock is released."""


# ---- Ground truth --------------------------------------------------------------------------------


class World:
    """What each root has revealed, to whom, across every node and process: the ground truth.

    A row revealed by two different proofs is reused; a link remade is the same link (RFC-0064
    'One link per epoch'). A root with more than P_W distinct columns opened is over-opened.
    Either breaks RFC-0012 INV-WC-1: that is the harm the gate counts.
    """

    def __init__(self):
        self.rows = {}  # (root, row) -> ids of the proofs and links that revealed it
        self.cols = {}  # root -> the distinct columns opened under it
        self.proofs = 0

    def reveal_rows(self, rt, rows, pid):
        for row in rows:
            self.rows.setdefault((rt, row), set()).add(pid)

    def reveal_cols(self, rt, cols):
        self.cols.setdefault(rt, set()).update(cols)

    def reused_rows(self):
        return sorted(k for k, v in self.rows.items() if len(v) > 1)

    def over_opened(self):
        return {rt: len(c) for rt, c in self.cols.items() if len(c) > P_W}

    def harm(self):
        return bool(self.reused_rows() or self.over_opened())


# ---- The file system ------------------------------------------------------------------------------


class Directory:
    """An entry directory on a local file system (RFC-0073 'The model entry').

    Fixed names map to values. A write is an atomic rename, so it puts a new file under the name.
    The object is the directory's inode: renaming the directory keeps it, and the lock is on it
    (RFC-0073 'One proof at a time per entry: the lock'). A copy of the entry is a new Directory.
    """

    def __init__(self):
        self.files = {}
        self.locked = False

    def get(self, path):
        return self.files.get(path)

    def put(self, path, value):
        self.files[path] = value

    def remove(self, path):
        self.files.pop(path, None)

    def paths(self):
        return list(self.files)

    def snapshot(self):
        return dict(self.files)

    def path_violations(self):
        """Names that are a symbolic link or a file of several hard links: none in a plain entry."""
        return []


class _Inode:
    def __init__(self, value):
        self.value = value
        self.links = 1


class _Symlink:
    def __init__(self, target, path):
        self.target = target
        self.path = path


class LinkingDirectory(Directory):
    """A directory whose names may be hard or symbolic links into another entry: the aliases that
    RFC-0073 'The model entry', the path rules, refuse. Reads follow symbolic links, as open()
    does; a write replaces the name itself, as rename() does."""

    def _resolve(self, path):
        entry = self.files.get(path)
        while isinstance(entry, _Symlink):
            entry = entry.target.files.get(entry.path)
        return entry

    def get(self, path):
        entry = self._resolve(path)
        return None if entry is None else entry.value

    def put(self, path, value):
        self.remove(path)
        self.files[path] = _Inode(value)

    def remove(self, path):
        entry = self.files.pop(path, None)
        if isinstance(entry, _Inode):
            entry.links -= 1

    def paths(self):
        return [p for p in self.files if self.get(p) is not None]

    def snapshot(self):
        return {p: self.get(p) for p in self.paths()}

    def path_violations(self):
        return [
            p
            for p, entry in self.files.items()
            if isinstance(entry, _Symlink) or (isinstance(entry, _Inode) and entry.links > 1)
        ]

    def hard_link(self, other):
        """Give every file of `other` a second name here."""
        for p, entry in other.files.items():
            entry.links += 1
            self.files[p] = entry

    def symlink(self, other):
        """Name every file of `other` here by a symbolic link."""
        for p in other.files:
            self.files[p] = _Symlink(other, p)

    @classmethod
    def holding(cls, values):
        d = cls()
        for p, v in values.items():
            d.put(p, v)
        return d


# ---- The prover -----------------------------------------------------------------------------------


class Prover:
    """One prover process on one node: the entry directory it opens, and the key store.

    Two processes on one entry are two Provers on one Directory. A copy of the entry is a Prover
    on another Directory. `zero_row_matrices` adds matrices with B_T = 0 to the layout, as
    ChannelParams at e_CP = 0 (RFC-0073 'The model entry', the marks).
    """

    def __init__(self, world, directory, keystore, name, zero_row_matrices=0):
        self.world = world
        self.dir = directory
        self.keystore = keystore
        self.name = name
        self.zero_row_matrices = zero_row_matrices
        self.rng = random.Random(name)  # the verifier's column queries, as this process meets them
        self.writes_left = None  # the process dies before its next write once this reaches 0
        self.on_write = None  # called after every entry write: the gate's backups
        self.seen = None  # the files as this transition last read or wrote them

    # -- RFC-0073 'The model entry': closed and open --------------------------------------------

    def committed(self, st):
        """(next_blinding_row, B_T) of each matrix with committed rows. A matrix with B_T = 0 keeps
        marks of 0 and no check compares them."""
        matrices = [(st['next_blinding_row'], B)] + [(0, 0)] * self.zero_row_matrices
        return [(row, bt) for row, bt in matrices if bt > 0]

    def is_closed(self, st):
        """Closed: each matrix's next_blinding_row is B_T, as a link leaves it."""
        return st is not None and all(row == bt for row, bt in self.committed(st))

    def is_open(self, st):
        """Open: each matrix's next_blinding_row is at most B_T − 4."""
        return st is not None and all(row <= bt - LINK_ROWS for row, bt in self.committed(st))

    # -- RFC-0073 'One proof at a time per entry: the lock' -------------------------------------

    @contextlib.contextmanager
    def transition(self):
        """A writer holds the directory's lock for its whole transition. A second writer is refused
        with ERR-RES-IO and does not wait. No file releases the lock; the system releases it when
        the process ends, a crash included."""
        if self.dir.locked:
            raise Refused('ERR-RES-IO', 'the entry is locked by another writer')
        self.dir.locked = True
        self.seen = self.dir.snapshot()
        try:
            yield
        finally:
            self.dir.locked = False
            self.seen = None

    def reread(self, path):
        """A file this transition read or wrote, if it is still the same file (device and inode)."""
        value = self.dir.get(path)
        if value is None or value is not self.seen.get(path):
            raise Refused('ERR-RES-IO', f'{name(path)} changed under the writer')
        return value

    def write(self, path, value):
        """A write by atomic rename. Before it, the file replaced must still be the one this
        transition last read or wrote, and a file created must not exist ('Before each rename')."""
        if self.writes_left is not None:
            if self.writes_left == 0:
                raise Crash(name(path))
            self.writes_left -= 1
        if self.dir.get(path) is not self.seen.get(path):
            raise Refused('ERR-RES-IO', f'{name(path)} changed under the writer')
        value = copy.deepcopy(value)
        self.dir.put(path, value)
        self.seen[path] = value
        if self.on_write:
            self.on_write()

    def write_key(self, value):
        """The key is created once and never replaced: a call that fails if the name exists."""
        if self.writes_left is not None:
            if self.writes_left == 0:
                raise Crash('key')
            self.writes_left -= 1
        if 'key' in self.keystore:
            raise Refused('ERR-RES-IO', 'the key exists')
        self.keystore['key'] = value

    def key(self, rec):
        """K, for commands that read it: weights refresh only (RFC-0073 'The model entry', the key)."""
        if 'key' not in self.keystore:
            raise Refused('ERR-USAGE', 'the key is absent')
        root_0, K = self.keystore['key']
        if root_0 != rec['roots'][0]:
            raise Refused('ERR-USAGE', "the key's root_0 is not the record's first root")
        return K

    # -- RFC-0073 'The entry check' --------------------------------------------------------------

    def check_decodes(self):
        """Item 2: every entry file decodes. A corrupted file is never treated as lost."""
        for path in self.dir.paths():
            if self.dir.get(path) == CORRUPT:
                raise Refused('ERR-CODEC', f'{name(path)} does not decode')

    def check_paths(self):
        """Item 3: the directory holds only its fixed names, under the path rules."""
        for path in self.dir.paths():
            if path != RECORD and not (
                isinstance(path, tuple) and path[0] in ('twc', 'twr', 'twl') and isinstance(path[1], int)
            ):
                raise Refused('ERR-USAGE', f'{path!r}: a file the entry may not hold')
        bad = self.dir.path_violations()
        if bad:
            raise Refused('ERR-USAGE', f'{name(bad[0])}: a symbolic link or a file of several links')

    def retire(self, rec, why):
        """'Stale persists': the record is rewritten with retired set, then the command refuses."""
        self.write(RECORD, {**rec, 'retired': True})
        raise Refused('ERR-USAGE', f'stale record, now retired: {why}')

    def regenerates(self, e, rt):
        """A later root is authenticated when weights refresh regenerates it from K for its epoch."""
        key = self.keystore.get('key')
        return key is not None and rt == root(key[1], e)

    def entry_check(self, presented=(), presented_rows=None, can_authenticate=False):
        """Items 1 to 5, in order, refusing on the first failure. Returns the record.

        `presented`: later-root artifacts, as (epoch, root): a .twr, a .twl or a ledger's
        weight_roots. `presented_rows`: root -> the first row past every reservation a presented
        ledger or kept Round1 shows. `can_authenticate`: the command reads K (weights refresh).
        """
        # 1. The record is present, decodes and is not retired.
        rec = self.dir.get(RECORD)
        if rec is None:
            raise Refused('ERR-USAGE', 'the record is absent: the entry has failed closed')
        if rec == CORRUPT:
            raise Refused('ERR-CODEC', f'{RECORD} does not decode')
        if rec['retired']:
            raise Refused('ERR-USAGE', 'the record is retired')
        # 2. Every entry file decodes.
        self.check_decodes()
        # 3. Fixed names and the path rules.
        self.check_paths()
        # 4. No file of the entry is past the record or differs from it.
        roots = rec['roots']
        latest = len(roots) - 1
        for path in self.dir.paths():
            if path == RECORD:
                continue
            kind, e = path
            v = self.dir.get(path)
            if kind in ('twc', 'twr') and (e > latest or v['root'] != roots[e]):
                self.retire(rec, f'{name(path)} is past the record or differs from it')
            if kind == 'twl' and (e + 1 > latest or v['old'] != roots[e] or v['new'] != roots[e + 1]):
                self.retire(rec, f'{name(path)} is past the record or differs from it')
            if kind == 'twc' and not (self.is_closed(v) or self.is_open(v)):
                raise Refused('ERR-USAGE', f'{name(path)} is neither closed nor open')
        #    The latest epoch's state, if present, is open: a closed one means a lost append.
        st = self.dir.get(state(latest))
        if st is not None and not self.is_open(st):
            self.retire(rec, f'{name(state(latest))} is closed')
        # 5. Presented artifacts. A later root past the record retires the record when it is
        #    authenticated, and otherwise refuses this command only.
        for e, rt in presented:
            if e > latest or roots[e] != rt:
                if can_authenticate and self.regenerates(e, rt):
                    self.retire(rec, f'an authenticated root of epoch {e} past the record')
                raise Refused('ERR-USAGE', f'a presented root of epoch {e} is past the record (this command only)')
        #    A presented reservation raises the marks in reserve; rows past B_T are refused.
        if presented_rows and any(row > B for row in presented_rows.values()):
            raise Refused('ERR-USAGE', 'a presented reservation has rows past B_T')
        return rec

    # -- RFC-0073 'Fresh and rebuilt states', with RFC-0012 'Budget state machine' ---------------

    def commit_weights(self, K):
        """commit-weights, steps 1 to 4: epoch 0 is committed once per key."""
        with self.transition():
            if self.dir.paths() or 'key' in self.keystore:
                raise Refused('ERR-USAGE', 'the entry directory is not empty, or the key path exists')
            r0 = root(K, 0)
            self.write_key((r0, K))
            self.write(RECORD, {'roots': [r0], 'marks': marks(0, 0), 'retired': False})
            self.write(state(0), {'root': r0, 'next_blinding_row': 0, 'opened_columns': frozenset()})
            self.write(public(0), {'root': r0})

    def budget_start(self, rec, st, presented_rows):
        """reserve, 'The cursor and the count': from the record's marks; a record behind the latest
        state is raised to it, and a presented reservation raises them the same way, the count by
        τ_W for every r_T rows added, rounded up. The count is never below the distinct columns
        opened ('The model entry', the marks)."""
        cursor = rec['marks']['next_blinding_row']
        count = rec['marks']['opened_columns']
        if st['next_blinding_row'] > cursor:
            count += TAU_W * -(-(st['next_blinding_row'] - cursor) // R)
            cursor = st['next_blinding_row']
        count = max(count, len(st['opened_columns']))
        shown = (presented_rows or {}).get(rec['roots'][-1])
        if shown is not None and shown > cursor:
            count += TAU_W * -(-(shown - cursor) // R)
            cursor = shown
        return cursor, count

    def reserve_fails(self, cursor, count):
        """RFC-0012's preconditions on the cursor and the count."""
        return cursor + R > B - LINK_ROWS or count + TAU_W > P_W - TAU_LINK

    def reserve(self, rec, presented_rows, hook=None):
        """reserve, only on the state at epoch-<latest>.twc. Returns (root, first row)."""
        latest = len(rec['roots']) - 1
        st = self.dir.get(state(latest))
        if st is None:
            raise Refused('ERR-WITNESS-BUDGET', 'the latest state is lost')
        if latest > 0 and not self.is_closed(self.dir.get(state(latest - 1))):
            raise Refused('ERR-WITNESS-BUDGET', 'the previous epoch is not closed')
        cursor, count = self.budget_start(rec, st, presented_rows)
        if self.reserve_fails(cursor, count):
            raise Refused('ERR-WITNESS-BUDGET', 'too few rows or columns left')
        if hook:
            hook()
        self.write_reservation(rec, st, cursor, count)
        return rec['roots'][latest], cursor

    def write_reservation(self, rec, st, cursor, count):
        """'The order': one write of the record raises the cursor by r_T and the count by the
        session's τ_W; then the state is written; then Round1 is built."""
        latest = len(rec['roots']) - 1
        self.write(RECORD, {**rec, 'marks': marks(cursor + R, count + TAU_W)})
        self.write(state(latest), {**st, 'next_blinding_row': cursor + R})

    def record_queries(self, rec, queries):
        """record_queries writes the state only: the session's distinct queries join
        opened_columns. The record's count was raised at reserve."""
        path = state(len(rec['roots']) - 1)
        st = self.reread(path)
        self.write(path, {**st, 'opened_columns': st['opened_columns'] | queries})

    def prove(self, presented=(), presented_rows=None, before_reserve=None, before_queries=None):
        """One proof, holding the lock from its entry check through record_queries. The hooks let
        another process act at those two points. Returns the proof's id."""
        with self.transition():
            rec = self.entry_check(presented, presented_rows)
            rt, first = self.reserve(rec, presented_rows, before_reserve)
            self.world.proofs += 1
            pid = f'{self.name}:p{self.world.proofs}'
            self.world.reveal_rows(rt, range(first, first + R), pid)  # Round1 .. the combined rows
            if before_queries:
                before_queries()
            queries = frozenset(self.rng.sample(range(N_COLS), TAU_W))
            self.record_queries(rec, queries)
            self.world.reveal_cols(rt, queries)  # Round3Response
            return pid

    def link(self, e=None, presented=()):
        """RFC-0064's link transition from epoch e (default: the latest), under RFC-0073 'Fresh and
        rebuilt states': append and a fresh successor from the latest epoch, else a rebuilt one;
        then epoch e+1's .twr and the link; then close epoch e (the commit point)."""
        with self.transition():
            rec = self.entry_check(presented, can_authenticate=True)
            K = self.key(rec)
            roots = rec['roots']
            latest = len(roots) - 1
            e = latest if e is None else e
            if e > latest or root(K, e) != roots[e]:
                raise Refused('ERR-USAGE', f'the regeneration check of epoch {e}')
            successor = root(K, e + 1)
            columns = link_columns(K, e)
            fresh = e == latest
            if fresh:
                self.write(RECORD, {**rec, 'roots': [*roots, successor], 'marks': marks(LINK_ROWS, len(columns))})
            elif roots[e + 1] != successor:
                raise Refused('ERR-USAGE', f'the regeneration check of epoch {e + 1}')
            # An existing file of epoch e + 1 is kept (RFC-0064 'One link per epoch').
            if self.dir.get(state(e + 1)) is None:
                self.write(state(e + 1), fresh_state(successor, columns) if fresh else rebuilt_state(successor))
            if self.dir.get(public(e + 1)) is None:
                self.write(public(e + 1), {'root': successor})
            if self.dir.get(link_file(e)) is None:
                self.write(link_file(e), {'old': roots[e], 'new': successor})
            old = self.dir.get(state(e)) or rebuilt_state(roots[e])
            self.write(state(e), {**old, 'next_blinding_row': B, 'opened_columns': old['opened_columns'] | columns})
            pid = f'link:{roots[e]}'  # remade, the link is the same bytes: the same reveal
            self.world.reveal_rows(roots[e], range(B - LINK_ROWS, B), pid)
            self.world.reveal_rows(successor, range(LINK_ROWS), pid)
            self.world.reveal_cols(roots[e], columns)
            self.world.reveal_cols(successor, columns)

    def rebuild(self, e, presented=()):
        """A lost state of a recorded epoch, regenerated from K, rebuilt at B_T − 4."""
        with self.transition():
            rec = self.entry_check(presented, can_authenticate=True)
            K = self.key(rec)
            if e >= len(rec['roots']) or root(K, e) != rec['roots'][e]:
                raise Refused('ERR-USAGE', f'the regeneration check of epoch {e}')
            if self.dir.get(state(e)) is None:
                self.write(state(e), rebuilt_state(rec['roots'][e]))

    def readable_record(self):
        """The record, when it and every entry file decode and it is not retired; else None, and
        the next transition's entry check names the failure."""
        rec = self.dir.get(RECORD)
        if rec is None or rec == CORRUPT or rec['retired']:
            return None
        if any(self.dir.get(p) == CORRUPT for p in self.dir.paths()):
            return None
        return rec

    def restart(self):
        """On restart the commit of an unfinished link is completed (RFC-0064 'One link per
        epoch'), as weights refresh's first step does: every epoch below the latest whose link or
        closed state is missing. It needs K."""
        rec = self.readable_record()
        if rec is None or 'key' not in self.keystore:
            return
        for e in range(len(rec['roots']) - 1):
            if self.dir.get(link_file(e)) is None or not self.is_closed(self.dir.get(state(e))):
                self.link(e)

    def weights_refresh(self, presented=(), presented_rows=None):
        """weights refresh, in order: 1. completes every unfinished link below the latest epoch;
        2. rebuilds a lost latest state; 3. links from the latest epoch when its reserve would fail
        for rows or columns. It is ERR-WITNESS-BUDGET's remedy (RFC-0073 'Errors'). Like every
        command that writes, it runs the entry check first, with what is presented to it."""
        with self.transition():
            self.entry_check(presented, presented_rows, can_authenticate=True)
        self.restart()
        rec = self.readable_record()
        if rec is None:
            self.link(presented=presented)  # refused by its entry check
            return
        latest = len(rec['roots']) - 1
        if self.dir.get(state(latest)) is None:
            self.rebuild(latest, presented)
        rec = self.readable_record()
        st = self.dir.get(state(latest))
        if rec is not None and st is not None and self.reserve_fails(*self.budget_start(rec, st, presented_rows)):
            self.link(presented=presented)


# ---- The operator ---------------------------------------------------------------------------------


def operate(prover, steps, presented=(), ledger=None):
    """An honest operator: up to `steps` proofs, each after a restart, with weights refresh on
    ERR-WITNESS-BUDGET (04 §5) and a stop on any other refusal. `ledger`, if given, returns at each
    proof the reservations presented to it. Returns how the run ended."""
    try:
        for _ in range(steps):
            prover.restart()
            rows = ledger() if ledger else None
            try:
                prover.prove(presented, rows)
            except Refused as r:
                if r.code != 'ERR-WITNESS-BUDGET':
                    return f'stopped: {r}'
                prover.weights_refresh(presented, rows)
        return 'ran'
    except Refused as r:
        return f'stopped: {r}'


def new_entry(world, name='A', K='K1', **kw):
    """A committed model entry on a fresh directory and key store."""
    p = Prover(world, Directory(), {}, name, **kw)
    p.commit_weights(K)
    return p
