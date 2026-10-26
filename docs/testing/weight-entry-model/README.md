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
- **Runtime.** 30 to 50 s with 16 workers on an Apple M4 Max, and about 5.5 minutes in one process. The targeted replays and the controls take under a second.
- **Exit status.** 0 when every check passes. 1 when any outcome is harmful outside the rule, any outcome in the rule is harmless, a targeted replay departs from the RFC, or a negative control fails to show its harm.

## What it checks

1. **The full replay**, at `B_T = 32`, where rows end an epoch first, and at `B_T = 64`, where columns do.
   - **The history.** An honest prover runs three epochs and two links, with a backup after every write. It proves until `reserve` fails, then runs `weights refresh`, and stops after two proofs in epoch 2.
   - **The restores.** Eight entry files: the record, three states, two `.twr` and two links. Each is put back at every version it had, or lost, in every combination.
   - **After a restore,** an honest operator proves 14 more times. It runs `weights refresh` on `ERR-WITNESS-BUDGET` and stops on any other refusal.
   - **Five variants of what the operator presents:**
     - `none`: nothing;
     - `twr`: the latest `.twr`;
     - `realistic`: that and a real ledger, which holds accepted proofs' rows only. The last proof of each epoch was rejected after its rows were revealed.
     - `ground`: the latest `.twr` and every sent reservation;
     - `corrupt`: nothing presented, and one entry file present but undecodable, each file in turn.
   - **The classifier.** `in_residual` classifies every restore before the run, by the RFC's rule and 'What a presented artifact catches'.
   - **The pass condition.** No harmful outcome lies outside the rule, and none in the rule is harmless. With every sent reservation presented, and with a corrupt file, nothing is harmful.
   - **Its sensitivity.** A test mutation dropped "or open" from the across-a-link case of `in_residual`. The gate then failed, finding 2,025 harmful outcomes outside the rule at `B_T = 32` and 2,406 at `B_T = 64`.
2. **Targeted replays** of what the full replay cannot build:
   - a crash before every write of a scripted history, then a restart;
   - a second crash at every write of the recovery;
   - a second process while a proof holds the lock, with the entry intact and with each file removed;
   - two copies proving concurrently: the residual's second case, which must reuse rows as stated;
   - serialized copies, each presented the other's reservations, which must not;
   - a hard-link alias and a symbolic-link alias;
   - a corrupted refuting file;
   - the residual's bound in epoch 0 and in later epochs, at `B_T = 32`, 34 and 64;
   - a layout with a `B_T = 0` matrix;
   - a forged later root, which refuses one command and retires nothing;
   - an authenticated later root, which retires the record;
   - a retired record, which stays retired after the files the check named are removed.
3. **Negative controls.** An earlier or naive reading of a rule must show the harm the rule prevents, or the replay could not see the bug. Each reading:
   - RFC-0012's `record_queries` row before RFC-0073;
   - read 4's column count, raised at `record_queries` rather than at `reserve`;
   - the entry check without the path rules;
   - a corrupted file read as lost;
   - closed and open read over every matrix, a `B_T = 0` matrix included.

**Harm** is what the ground truth, `World`, records across every node and process:
- **A reused row:** a row revealed by two different proofs. A remade link is the same link.
- **An over-opened matrix:** more than `P_W` distinct columns opened under one root.

## Expected output

Every check passes, so the gate exits 0. The tables are exact: a change to them is a change to the rules or to the history. Only the last line's seconds vary.

```text
== 1. The full replay: RFC-0073's residual rule against the harm observed
-- B_T = 32: versions per file (lost included) 14 x 11 x 11 x 6 x 2 x 2 x 2 x 2
   variant        runs  harmful    reuse over-open  in rule  outside harmless
   none        162,624   13,866   13,866    13,866   13,866        0        0
   twr         162,624   11,616   11,616    11,616   11,616        0        0
   realistic   162,624   11,616   11,616    11,616   11,616        0        0
   ground      162,624        0        0         0        0        0        0
   corrupt     393,536        0        0         0        0        0        0
-- B_T = 64: versions per file (lost included) 15 x 13 x 11 x 6 x 2 x 2 x 2 x 2
   variant        runs  harmful    reuse over-open  in rule  outside harmless
   none        205,920   16,400   16,400    16,400   16,400        0        0
   twr         205,920   13,728   13,728    13,728   13,728        0        0
   realistic   205,920   13,728   13,728    13,728   13,728        0        0
   ground      205,920        0        0         0        0        0        0
   corrupt     494,448        0        0         0        0        0        0
== 2. Targeted replays
   PASS B_T = 32: a crash before every write, then a restart: 41 crash points, 0 harmful
   PASS B_T = 32: a second crash at every write of the recovery: 468 runs, 0 harmful
   PASS B_T = 64: a crash before every write, then a restart: 41 crash points, 0 harmful
   PASS B_T = 64: a second crash at every write of the recovery: 468 runs, 0 harmful
   PASS B_T = 32: a second process, the entry intact or each of 6 files removed: ERR-RES-IO, no harm
   PASS B_T = 64: a second process, the entry intact or each of 6 files removed: ERR-RES-IO, no harm
   PASS B_T = 32: two copies proving concurrently reuse rows (the residual, as stated): 6 rows
   PASS B_T = 32: serialized copies, each presented the other's reservations: reused 0, over-opened {}
   PASS B_T = 64: two copies proving concurrently reuse rows (the residual, as stated): 6 rows
   PASS B_T = 64: serialized copies, each presented the other's reservations: reused 0, over-opened {}
   PASS B_T = 32: hard-link alias, concurrent: reused 0, refused ['ERR-USAGE']
   PASS B_T = 32: symbolic-link alias, in turn: reused 0, refused ['ERR-USAGE']
   PASS B_T = 64: hard-link alias, concurrent: reused 0, refused ['ERR-USAGE']
   PASS B_T = 64: symbolic-link alias, in turn: reused 0, refused ['ERR-USAGE']
   PASS B_T = 32: a corrupted refuting file: stopped: ERR-CODEC: epoch-0.twc does not decode; reused 0
   PASS B_T = 64: a corrupted refuting file: stopped: ERR-CODEC: epoch-0.twc does not decode; reused 0
   PASS B_T = 32: the whole entry restored to just after commit-weights: epoch 0: rows 24/28, columns past P_W 9/20; epoch 1: rows 24/24, columns past P_W 15/16; epoch 2: rows 24/24, columns past P_W 10/16; epoch 3: rows 0/24, columns past P_W 0/16
   PASS B_T = 32: the whole entry restored to just after the first link: epoch 0: rows 0/28, columns past P_W 0/20; epoch 1: rows 24/24, columns past P_W 15/16; epoch 2: rows 24/24, columns past P_W 14/16; epoch 3: rows 24/24, columns past P_W 11/16
   PASS B_T = 34: the whole entry restored to just after commit-weights: epoch 0: rows 30/30, columns past P_W 17/20; epoch 1: rows 24/26, columns past P_W 16/16; epoch 2: rows 18/26, columns past P_W 3/16; epoch 3: rows 0/26, columns past P_W 0/16
   PASS B_T = 34: the whole entry restored to just after the first link: epoch 0: rows 0/30, columns past P_W 0/20; epoch 1: rows 24/26, columns past P_W 16/16; epoch 2: rows 24/26, columns past P_W 15/16; epoch 3: rows 24/26, columns past P_W 11/16
   PASS B_T = 64: the whole entry restored to just after commit-weights: epoch 0: rows 30/60, columns past P_W 17/20; epoch 1: rows 24/56, columns past P_W 16/16; epoch 2: rows 18/56, columns past P_W 3/16; epoch 3: rows 0/56, columns past P_W 0/16
   PASS B_T = 64: the whole entry restored to just after the first link: epoch 0: rows 0/60, columns past P_W 0/20; epoch 1: rows 24/56, columns past P_W 16/16; epoch 2: rows 24/56, columns past P_W 15/16; epoch 3: rows 24/56, columns past P_W 11/16
   PASS B_T = 32: a layout with a B_T = 0 matrix: ran, 7 proofs, no harm
   PASS B_T = 64: a layout with a B_T = 0 matrix: ran, 7 proofs, no harm
   PASS B_T = 32: a forged later root refuses prove and weights refresh (ERR-USAGE), retires nothing; then: ran
   PASS B_T = 32: an authenticated later root retires the record, which stays retired: stopped: ERR-USAGE: the record is retired
   PASS B_T = 32: a stale record is retired; removing the files the check named leaves it retired: stopped: ERR-USAGE: the record is retired
   PASS B_T = 64: a forged later root refuses prove and weights refresh (ERR-USAGE), retires nothing; then: ran
   PASS B_T = 64: an authenticated later root retires the record, which stays retired: stopped: ERR-USAGE: the record is retired
   PASS B_T = 64: a stale record is retired; removing the files the check named leaves it retired: stopped: ERR-USAGE: the record is retired
== 3. Negative controls: each must show the harm its rule prevents
   PASS control: RFC-0012's record_queries row before RFC-0073, a restored state: over-opened {'R[K1:0]': 38}
   PASS B_T = 64: a restored latest state alone, under RFC-0073: over-opened {}
   PASS control: the count raised at record_queries (read 4), every reservation presented: 4 of 57 restores over-open
   PASS control: B_T = 32, hard-link alias, concurrent without the path rules: reused 6
   PASS control: B_T = 32, symbolic-link alias, in turn without the path rules: reused 6
   PASS control: B_T = 32, a corrupted refuting file read as lost: reused 24
   PASS control: B_T = 32, closed and open over every matrix, B_T = 0 included: stopped: ERR-USAGE: epoch-0.twc is neither closed nor open
   PASS control: B_T = 64, hard-link alias, concurrent without the path rules: reused 6
   PASS control: B_T = 64, symbolic-link alias, in turn without the path rules: reused 6
   PASS control: B_T = 64, a corrupted refuting file read as lost: reused 18
   PASS control: B_T = 64, closed and open over every matrix, B_T = 0 included: stopped: ERR-USAGE: epoch-0.twc is neither closed nor open
GATE: PASS: 55 checks passed in 32 s
```

## What it abstracts

The model is small enough to replay exhaustively. These are its abstractions, and what each leaves unchecked.

- **Files.**
  - **The directory.** Each entry file is one value under its fixed name.
  - **Writes.** A write is an atomic rename, so no file is ever torn or half-written.
  - **Corruption.** A corrupt file is whole but undecodable. The full replay corrupts one file at a time.
  - **Device and inode.** The check before each rename is object identity.
  - **Not modelled:**
    - the directory being replaced between that check and the rename, RFC-0067's remaining window;
    - temporary files and `run-manifest.json`;
    - non-canonical names;
    - I/O errors, which would be `ERR-RES-IO`.
  - **Aliases** exist only in the alias replays. Their directory follows symbolic links on read and counts hard links.
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
  - **A later root** is an `(epoch, root)` pair.
  - **Reservations** are the first row past the last reservation of a root: a ledger's `used_blinding_rows`, or kept `Round1`s.
- **The operator** is honest:
  - it never moves a corrupted file aside in the full replay;
  - it stops at the first refusal that `weights refresh` does not repair.
