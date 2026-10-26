# Weight entry model

An executable model of the weight model entry in draft [RFC-0073](../../rfcs/RFC-0073-weight-mask-expansion.md), and the gate that checks its residual rule. It covers the sections from 'The model entry' through 'What the prover cannot detect', with [RFC-0012 'Budget state machine'](../../rfcs/RFC-0012-weight-commitment.md#budget-state-machine) and [RFC-0064 'The budget' and 'One link per epoch'](../../rfcs/RFC-0064-weight-linking-proof.md#one-link-per-epoch). The rules keep a model's weight commitments epoch after epoch so that no blinding row is revealed twice and no matrix opens more than `P_W` distinct columns in an epoch (INV-WC-1).

[RFC-0073 'What the prover cannot detect'](../../rfcs/RFC-0073-weight-mask-expansion.md#what-the-prover-cannot-detect) states which restores and copies the prover cannot catch. That statement rests on this model:
- [`model.py`](model.py) holds the rules. Each method implements one section of the RFCs and cites it.
- [`gate.py`](gate.py) replays the rules against every restore, loss, corruption, crash, copy and alias it builds. It checks the residual rule both ways against the harm it observes.

[#147](https://github.com/AbdelStark/tessera/issues/147)'s slice S4 ports both to `crates/tessera-pcs/tests/weight_entry_model.rs`. That port runs the same state space and pass condition against the code's rules, as RFC-0073's [Testing Strategy](../../rfcs/RFC-0073-weight-mask-expansion.md#testing-strategy) states. Until then this directory is the gate:
- **Who runs it.** No CI job runs it. A change to RFC-0073's entry rules, to RFC-0012's budget rows or to RFC-0064's link rules runs it by hand, and its pull request shows the output.
- **How it changes with the rules.** A rule changes in the RFC and in `model.py` together. The residual rule changes in the RFC and in `gate.py`'s `in_residual` together.

## Run it

From the repository root:

```sh
python3 -I docs/testing/weight-entry-model/gate.py            # every check, B_T = 32 and 64
python3 -I docs/testing/weight-entry-model/gate.py --jobs 1   # one process
```

- **Requirements.** Standard library only, Python 3.11 or later. `-I` keeps the environment and the current directory out of the import path, and `gate.py` loads `model.py` from its own directory.
- **Workers.** The full replay forks one worker per CPU (`--jobs`). Where `fork` is unavailable, it runs in one process.
- **Runtime.** 65 to 85 s with 16 workers on an Apple M4 Max, and about 12 minutes in one process. The targeted replays and the controls take a few seconds.
- **Exit status.** 0 when every check passes. 1 when any outcome is harmful outside the rule, any outcome in the rule is harmless, a targeted replay departs from the RFC, or a negative control fails to show its harm.

## What it checks

1. **The full replay**, at `B_T = 32`, where rows end an epoch first, and at `B_T = 64`, where columns do.
   - **The history.** An honest prover runs three epochs and two links, with a backup after every write. It proves until `reserve` fails, then runs `weights refresh`, and stops after two proofs in epoch 2.
   - **The restores.** Eight entry files: the record, three states, two `.twr` and two links. Each is put back at every version it had, or lost, in every combination.
   - **After a restore,** an honest operator proves 14 more times. It runs `weights refresh` on `ERR-WITNESS-BUDGET` and stops on any other refusal.
   - **Six variants of what the operator presents to every command:**
     - `none`: nothing;
     - `twr`: the latest `.twr`;
     - `realistic`: that and a real ledger, which holds accepted proofs' rows only, beside another model's chain. The last proof of each epoch was rejected after its rows were revealed.
     - `ground`: the latest `.twr` and every kept `Round1`;
     - `round1s`: every kept `Round1` alone;
     - `corrupt`: nothing presented, and one entry file present but undecodable, each file in turn.
   - **The classifier.** `in_residual` classifies every restore before the run, by the RFC's rule and 'What a presented artifact catches'. The truth is the last record the prover wrote.
   - **The pass condition.**
     - No harmful outcome lies outside the rule, and none in the rule is harmless.
     - No outcome passes the per-epoch bound.
     - With every kept `Round1` presented, and with a corrupt file, nothing is harmful.
   - **Its sensitivity.** A test mutation dropped "or open" from the across-a-link case of `in_residual`. The gate then failed, finding 2,025 harmful outcomes outside the rule at `B_T = 32` and 2,406 at `B_T = 64`.
2. **Targeted replays** of what the full replay cannot build:
   - **Crashes:**
     - a crash before every write of a scripted history, then a restart;
     - a second crash at every write of the recovery;
     - a crash at every write, then the record alone or the whole entry restored to every backup. No harmful outcome lies outside the rule. Some in it are harmless, since a crashed transition revealed nothing: the rule is conservative there.
   - **Concurrency:** a second process while a proof holds the lock, with the entry intact and with each file removed.
   - **Copies:**
     - two copies proving concurrently: the residual's second case, which must reuse rows as stated;
     - serialized copies, each presented the other's reservations, which must not;
     - two and three copies proving in turn for six epochs. They reuse rows in every epoch, within the per-epoch bound times the number of other copies.
   - **Aliases:** a hard-link alias, a symbolic-link alias, and a hard-link alias made while a proof holds the lock.
   - **A corrupted refuting file:** first refused; then moved aside and presented with the latest artifacts to `weights refresh`, which retires the record.
   - **The bound:** whole-entry restores to just after `commit-weights` and just after the first link, at `B_T = 32`, 34 and 64, per epoch.
   - **Closed and open:** a layout with a `B_T = 0` matrix.
   - **Presented artifacts:**
     - a forged later root, which refuses one command and retires nothing;
     - an authenticated later root, which retires the record;
     - kept `Round1`s alone across a link, whose later root refuses `prove` and retires the record through `weights refresh`;
     - kept `Round1`s presented once to `weights refresh`, whose raise the record keeps;
     - a ledger that also holds another model's and a retired entry's chains, which proves normally;
     - a retired record, which stays retired after the files the check named are removed.
3. **Negative controls.** An earlier or other reading of a rule must show the harm the rule prevents, or the replay could not see the bug. Each reading:
   - **Two other wordings of 'The cursor and the count',** run over the full replay at `B_T = 64`:
     - the count raised to the state's `|opened_columns|`, as RFC-0012's row once read;
     - the cursor alone raised, as RFC-0073's Summary, RFC-0064 and the simulator once read.
   - **Earlier column counts:** RFC-0012's `record_queries` row before RFC-0073, and read 4's count, raised at `record_queries` rather than at `reserve`.
   - **The path rules:** none at all, and at open only, not before each write.
   - **Corruption:** a corrupted file read as lost.
   - **Closed and open** read over every matrix, a `B_T = 0` matrix included.
   - **Ledgers:** one read beyond this entry's chain.
   - **The raise:** a presented raise that `weights refresh` does not write into the record.

**Harm** is what the ground truth, `World`, records across every node and process:
- **A reused row:** a row revealed by two different proofs. A remade link is the same link.
- **An over-opened matrix:** more than `P_W` distinct columns opened under one root.

**The per-epoch bound** applies to each root `R_W`, for each event (a restore, or each copy past the first):
- **Rows revealed again:** up to `B_T − 4` in epoch 0, and `B_T − 8` later.
- **Distinct columns past `P_W`:** up to `P_W − τ_link` in epoch 0, and `P_W − 2τ_link` later.

## Expected output

Every check passes, so the gate exits 0. The tables are exact: a change to them is a change to the rules or to the history. Only the last line's seconds vary.

```text
== 1. The full replay: RFC-0073's residual rule against the harm observed
-- B_T = 32: versions per file (lost included) 14 x 11 x 11 x 6 x 2 x 2 x 2 x 2
   variant        runs  harmful    reuse over-open  in rule  outside harmless  bound
   none        162,624   13,866   13,866    13,866   13,866        0        0      0
   twr         162,624   11,616   11,616    11,616   11,616        0        0      0
   realistic   162,624   11,616   11,616    11,616   11,616        0        0      0
   ground      162,624        0        0         0        0        0        0      0
   round1s     162,624        0        0         0        0        0        0      0
   corrupt     393,536        0        0         0        0        0        0      0
-- B_T = 64: versions per file (lost included) 15 x 13 x 11 x 6 x 2 x 2 x 2 x 2
   variant        runs  harmful    reuse over-open  in rule  outside harmless  bound
   none        205,920   16,400   16,400    16,400   16,400        0        0      0
   twr         205,920   13,728   13,728    13,728   13,728        0        0      0
   realistic   205,920   13,728   13,728    13,728   13,728        0        0      0
   ground      205,920        0        0         0        0        0        0      0
   round1s     205,920        0        0         0        0        0        0      0
   corrupt     494,448        0        0         0        0        0        0      0
== 2. Targeted replays
   PASS B_T = 32: a crash before every write, then a restart: 41 crash points, 0 harmful
   PASS B_T = 32: a second crash at every write of the recovery: 468 runs, 0 harmful
   PASS B_T = 64: a crash before every write, then a restart: 41 crash points, 0 harmful
   PASS B_T = 64: a second crash at every write of the recovery: 468 runs, 0 harmful
   PASS B_T = 32: a crash, then the record or the whole entry restored to every backup: 1560 runs, 0 harmful outside the rule, 181 in it harmless (the conservative direction)
   PASS B_T = 64: a crash, then the record or the whole entry restored to every backup: 1560 runs, 0 harmful outside the rule, 181 in it harmless (the conservative direction)
   PASS B_T = 32: a second process, the entry intact or each of 6 files removed: ERR-RES-IO, no harm
   PASS B_T = 64: a second process, the entry intact or each of 6 files removed: ERR-RES-IO, no harm
   PASS B_T = 32: two copies proving concurrently reuse rows (the residual, as stated): 6 rows
   PASS B_T = 32: serialized copies, each presented the other's reservations: reused 0, over-opened {}
   PASS B_T = 32: 2 copies proving in turn, 6 epochs: rows reused in every epoch, within 1 x the per-epoch bound (e0 24/28 rows, 10/20 cols; e1 24/24 rows, 15/16 cols; e2 24/24 rows, 15/16 cols, ...)
   PASS B_T = 32: 3 copies proving in turn, 6 epochs: rows reused in every epoch, within 2 x the per-epoch bound (e0 48/56 rows, 25/40 cols; e1 48/48 rows, 31/32 cols; e2 48/48 rows, 31/32 cols, ...)
   PASS B_T = 64: two copies proving concurrently reuse rows (the residual, as stated): 6 rows
   PASS B_T = 64: serialized copies, each presented the other's reservations: reused 0, over-opened {}
   PASS B_T = 64: 2 copies proving in turn, 6 epochs: rows reused in every epoch, within 1 x the per-epoch bound (e0 30/60 rows, 16/20 cols; e1 24/56 rows, 16/16 cols; e2 24/56 rows, 14/16 cols, ...)
   PASS B_T = 64: 3 copies proving in turn, 6 epochs: rows reused in every epoch, within 2 x the per-epoch bound (e0 60/120 rows, 33/40 cols; e1 48/112 rows, 31/32 cols; e2 48/112 rows, 29/32 cols, ...)
   PASS B_T = 32: hard-link alias, concurrent: reused 0, refused ['ERR-USAGE']
   PASS B_T = 32: symbolic-link alias, in turn: reused 0, refused ['ERR-USAGE']
   PASS B_T = 32: hard-link alias made during a proof: reused 0, refused ['ERR-USAGE', 'ERR-USAGE']
   PASS B_T = 64: hard-link alias, concurrent: reused 0, refused ['ERR-USAGE']
   PASS B_T = 64: symbolic-link alias, in turn: reused 0, refused ['ERR-USAGE']
   PASS B_T = 64: hard-link alias made during a proof: reused 0, refused ['ERR-USAGE', 'ERR-USAGE']
   PASS B_T = 32: a corrupted refuting file: stopped: ERR-CODEC: epoch-0.twc does not decode; reused 0
   PASS B_T = 32: moved aside, then weights refresh with the latest .twr and kept Round1s: ERR-USAGE: stale record, now retired: an; then stopped: ERR-USAGE: the record is retired
   PASS B_T = 64: a corrupted refuting file: stopped: ERR-CODEC: epoch-0.twc does not decode; reused 0
   PASS B_T = 64: moved aside, then weights refresh with the latest .twr and kept Round1s: ERR-USAGE: stale record, now retired: an; then stopped: ERR-USAGE: the record is retired
   PASS B_T = 32: the whole entry restored to just after commit-weights: e0 24/28 rows, 9/20 cols; e1 24/24 rows, 15/16 cols; e2 24/24 rows, 10/16 cols; e3 0/24 rows, 0/16 cols
   PASS B_T = 32: the whole entry restored to just after the first link: e0 0/28 rows, 0/20 cols; e1 24/24 rows, 15/16 cols; e2 24/24 rows, 14/16 cols; e3 24/24 rows, 11/16 cols
   PASS B_T = 34: the whole entry restored to just after commit-weights: e0 30/30 rows, 17/20 cols; e1 24/26 rows, 16/16 cols; e2 18/26 rows, 3/16 cols; e3 0/26 rows, 0/16 cols
   PASS B_T = 34: the whole entry restored to just after the first link: e0 0/30 rows, 0/20 cols; e1 24/26 rows, 16/16 cols; e2 24/26 rows, 15/16 cols; e3 24/26 rows, 11/16 cols
   PASS B_T = 64: the whole entry restored to just after commit-weights: e0 30/60 rows, 17/20 cols; e1 24/56 rows, 16/16 cols; e2 18/56 rows, 3/16 cols; e3 0/56 rows, 0/16 cols
   PASS B_T = 64: the whole entry restored to just after the first link: e0 0/60 rows, 0/20 cols; e1 24/56 rows, 16/16 cols; e2 24/56 rows, 15/16 cols; e3 24/56 rows, 11/16 cols
   PASS B_T = 32: a layout with a B_T = 0 matrix: ran, 7 proofs, no harm
   PASS B_T = 64: a layout with a B_T = 0 matrix: ran, 7 proofs, no harm
   PASS B_T = 32: a forged later root refuses prove and weights refresh (ERR-USAGE), retires nothing; then: ran
   PASS B_T = 32: an authenticated later root retires the record, which stays retired: stopped: ERR-USAGE: the record is retired
   PASS B_T = 32: kept Round1s alone, across a link: a Round1 of a later root refuses prove and retires through weights refresh; reused 0
   PASS B_T = 32: kept Round1s presented once to weights refresh, then nothing: the raise is in the record; ran, reused 0
   PASS B_T = 32: a ledger also holding another model's and a retired entry's chains: ran, 7 proofs
   PASS B_T = 32: a stale record is retired; removing the files the check named leaves it retired: stopped: ERR-USAGE: the record is retired
   PASS B_T = 64: a forged later root refuses prove and weights refresh (ERR-USAGE), retires nothing; then: ran
   PASS B_T = 64: an authenticated later root retires the record, which stays retired: stopped: ERR-USAGE: the record is retired
   PASS B_T = 64: kept Round1s alone, across a link: a Round1 of a later root refuses prove and retires through weights refresh; reused 0
   PASS B_T = 64: kept Round1s presented once to weights refresh, then nothing: the raise is in the record; ran, reused 0
   PASS B_T = 64: a ledger also holding another model's and a retired entry's chains: ran, 7 proofs
   PASS B_T = 64: a stale record is retired; removing the files the check named leaves it retired: stopped: ERR-USAGE: the record is retired
== 3. Negative controls: each must show the harm its rule prevents
   PASS control: B_T = 64, none, the count read as 'raised-to-state': 4,576 harmful outside the rule (16,400 reuse, 20,976 over-open)
   PASS control: B_T = 64, ground, the count read as 'raised-to-state': 6,864 harmful outside the rule (0 reuse, 6,864 over-open)
   PASS control: B_T = 64, none, the count read as 'larger-of': 9,152 harmful outside the rule (16,400 reuse, 25,552 over-open)
   PASS control: RFC-0012's record_queries row before RFC-0073, a restored state: over-opened {'R[K1:0]': 38}
   PASS B_T = 64: a restored latest state alone, under RFC-0073: over-opened {}
   PASS control: the count raised at record_queries (read 4), every Round1 presented: 4 of 57 restores over-open
   PASS control: B_T = 32, hard-link alias, concurrent without the path rules: reused 6
   PASS control: B_T = 32, symbolic-link alias, in turn without the path rules: reused 6
   PASS control: B_T = 32, hard-link alias made during a proof without the path rules: reused 6
   PASS control: B_T = 32, hard-link alias made during a proof, path rules at open only: reused 6
   PASS control: B_T = 32, a corrupted refuting file read as lost: reused 24
   PASS control: B_T = 32, closed and open over every matrix, B_T = 0 included: stopped: ERR-USAGE: epoch-0.twc is neither closed nor open
   PASS control: B_T = 32, a ledger read beyond this entry's chain: stopped: ERR-USAGE: a presented root of epoch 0 is past the record (this command only)
   PASS control: B_T = 32, a presented raise not written into the record: reused 12
   PASS control: B_T = 64, hard-link alias, concurrent without the path rules: reused 6
   PASS control: B_T = 64, symbolic-link alias, in turn without the path rules: reused 6
   PASS control: B_T = 64, hard-link alias made during a proof without the path rules: reused 6
   PASS control: B_T = 64, hard-link alias made during a proof, path rules at open only: reused 6
   PASS control: B_T = 64, a corrupted refuting file read as lost: reused 18
   PASS control: B_T = 64, closed and open over every matrix, B_T = 0 included: stopped: ERR-USAGE: epoch-0.twc is neither closed nor open
   PASS control: B_T = 64, a ledger read beyond this entry's chain: stopped: ERR-USAGE: a presented root of epoch 0 is past the record (this command only)
   PASS control: B_T = 64, a presented raise not written into the record: reused 12
GATE: PASS: 86 checks passed in 82 s
```

## What it abstracts

The model is small enough to replay exhaustively. These are its abstractions, and what each leaves unchecked.

- **Files.**
  - **The directory.** Each entry file is one value under its fixed name.
  - **Writes.** A write is an atomic rename, so no file is ever torn or half-written.
  - **Corruption.** A corrupt file is whole but undecodable. The full replay corrupts one file at a time.
  - **Device and inode.** The check before each rename is object identity, against the files as they were when the transition took the lock, as if it had read every one. Under the lock that is equivalent.
  - **Crashes.** A crash is a process death before a write. Every earlier write is durable, and its lock is released.
  - **Not modelled:**
    - the directory being replaced between that check and the rename, RFC-0067's remaining window;
    - temporary files and `run-manifest.json`;
    - non-canonical names;
    - I/O errors, which would be `ERR-RES-IO`.
  - **Aliases** exist only in the alias replays. Their directory follows symbolic links on read and counts hard links. The path rules run in the entry check and again before every write.
- **Versions.**
  - **Their source.** Every version comes from one honest history: a backup after every write, with every file restored independently.
  - **Never restored:** the key store, a file of another model entry, and a record the history never wrote, such as a retired one.
  - **The key.** It is never lost in the full replay, and `.twr` of epoch 0 is never restored or lost.
- **Epochs.**
  - **The full replay.** Three epochs and two links, so a rollback spans at most two links.
  - **The bound replay.** Up to four epochs.
  - **The history's shape is fixed:** proofs until `reserve` fails, then a link.
- **Matrices.**
  - **One matrix.** A single-claim matrix (`r_T = 6`) stands for the ten. One `τ_W` stands for "the largest `τ_W` of the layout's parameter sets".
  - **The `B_T = 0` matrix** appears only in its own replay and control.
  - **Not modelled:** matrices with different claim counts, whose cursors move by different `r_T`.
- **Parameters.**
  - **The values.** `B_T` is 32 or 64, plus 34 in the bound replay. The rest are fixed: `P_W = 24`, `τ_W = 4`, `τ_link = 4`, `n_T = 512`.
  - **What carries over to production** (`B_W = 4,096`, `P_W = 2^16`) is the two regimes: rows ending an epoch first, and columns.
  - **What does not:** the counts, such as the 4,092-row bound at 405B. Those follow from the RFC's formulas, not from this replay.
- **Cryptography.**
  - **Roots and regeneration.** A root is a name of `(K, epoch)`, and the regeneration check compares names.
  - **Columns.** A link's columns are a fixed sample per `(K, epoch)`. A proof's queries are random samples.
  - **Authentication.** A later root is authenticated when `K` regenerates it. RFC-0064's checks of a presented `.twl` are not modelled.
  - **Not modelled:** masks, encodings, TCE beyond "decodes or not", and the transcript.
- **Proofs.**
  - **When rows and columns are revealed.** A proof reveals its rows when `reserve` returns and its columns after `record_queries`.
  - **Not modelled:**
    - `abort` other than by a crash or a refusal;
    - #156's resume;
    - live and non-interactive sessions, which are not told apart;
    - the verifiers' own checks, `Reject(Budget)` and superseded roots.

    The harm counted is what the prover reveals, whoever accepts it.
- **Concurrency.**
  - **Scripted interleavings.** Another process acts at two points of a proof: before `reserve`'s writes, and before `record_queries`.
  - **The lock** is a flag on the directory object, released when its holder ends or crashes.
  - **Copies** share one key store.
- **Presented artifacts.**
  - **What each holds.**
    - A `.twr` is one `(epoch, root)` pair.
    - A ledger holds chains of `(epoch, root)` from epoch 0, and used rows by root.
    - A kept `Round1` holds its epoch, its root and the first row past its reservation.
  - **Reservations** are the first row past the last reservation of a root.
  - **Not modelled:** a `.twl` as an artifact.
- **The operator** is honest:
  - it presents the same artifacts to every command;
  - it runs no restart of its own: `weights refresh` completes unfinished links, in its first step, when `reserve` asks for it;
  - it never moves a corrupted file aside in the full replay;
  - it stops at the first refusal that `weights refresh` does not repair.

## What the model cannot show

Read 6 listed what the RFC relies on beyond these abstractions. Each item is a promise the gate does not check.

- **Ten matrices.**
  - **What the RFC relies on:**
    - per-matrix marks with different `B_T` and `r_T`, such as `B_CP = 12,280` and `r_CP = 18` against `B_W = 4,096` and 6;
    - "the largest `τ_W` of the layout's parameter sets";
    - mixed sessions, at `live-64`'s `τ_W = 97` and `ni-100`'s 122;
    - each opened matrix's cursor, raised in one record write.
  - **Why no failure is expected.** Every rule is per matrix, and a record write is atomic across the ten. The rounding and the largest `τ_W` only raise.
  - **The gap.** A failure across matrices is not excluded by the model. A run with two matrices of different `B_T` and `r_T` would close it.
- **More epochs.** The full replay's rollbacks span at most two links. The across-a-link rule, both ways, is unshown for three or more.
- **Crashes combined with restores.**
  - **What is replayed.** A crash, then the record alone or the whole entry restored to every backup, over one scripted history.
  - **Not replayed:** restores that mix files from several backups after a crash, as the full replay does without one.
- **Real file systems.**
  - **Power loss.** Durability rests on the atomic writer's directory sync (RFC-0040 '`WeightCommitmentState`'), which the model takes for granted.
  - **Not modelled:**
    - the directory being replaced between the device-and-inode check and the rename, the window RFC-0067 states;
    - temporary and non-canonical names;
    - I/O errors.
- **A malicious verifier's artifacts.**
  - **What is modelled.** A forged root refuses one command.
  - **Not modelled,** and harmless by the RFC's own argument:
    - a forged reservation, which burns at most the rest of an epoch and costs a refresh;
    - a forged ledger;
    - a forged `.twl`.
- **Authentication.** Only `weights refresh` authenticates, by regenerating a root from `K`. A `.twl` authenticated by RFC-0064's checks, in `prove` or `weights refresh`, is not modelled. It could only refuse or retire.
- **The key.** It is never lost in the full replay. Losing it stops refreshes: nothing past the budget is proven, but the rule's exactness then fails in the harmless direction.
- **Live and non-interactive sessions, #156's resume, and the verifiers' rejections.** None is modelled. The harm counted is what is revealed, which is conservative.
