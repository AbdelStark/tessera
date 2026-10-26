# Sumcheck vectors

`sumcheck.json` holds honest proofs of the sumchecks of [RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md): single and batched sumchecks of products of multilinear tables, front-padded as ['Batching within a stage'](../../../docs/rfcs/RFC-0010-proof-schedule.md#batching-within-a-stage) says, with the messages of ['Sumcheck messages'](../../../docs/rfcs/RFC-0010-proof-schedule.md#sumcheck-messages); batches of product trees proven level by level, as ['Tree levels'](../../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels) says; and batches of product forests, trees of one height stacked and entering at a random root point, as the same section says. It also holds leaf stages of LogUp-GKR, the fraction trees of [RFC-0008 'Fraction tree'](../../../docs/rfcs/RFC-0008-logup-and-tables.md#fraction-tree) beside product forests: honest ones, and cheating ones that a verifier must reject at a stated check. Every challenge, claim, point and transcript state of each proof is recorded, so an implementation can locate the first step where it diverges, and the stages on at most 5 variables also record their tables, so a prover can reproduce them whole. Of the sessions, tree batches and forest batches, `verifier/crates/tv-core/tests/` verifies every proof, reproduces every recorded value, and rejects every proof changed in one element; its `forests.rs` drives the forest batches as a verifier's caller would, squeezing each level's root challenges and forming the root claims from the public roots itself, and checks every leaf claim against the stacked leaves. `crates/tessera-sumcheck/tests/vectors.rs` reproduces every stage that records its tables, message by message, and every tree and forest level from the leaves.

[`generate.py`](generate.py) writes it. The script is the independent oracle, an honest prover written from the RFC text alone with Python's arbitrary-precision integers. Its transcript is the oracle of [`../transcript/generate.py`](../transcript/generate.py), imported as that file imports [`../hash/generate.py`](../hash/generate.py); the script reruns the official BLAKE3 test vectors through it before anything is written. Its inputs come from SplitMix64 with the seed the file states, words at or above `P` rejected, the LogUp batches drawing after the forest batches, so `python3 spec/vectors/sumcheck/generate.py` reproduces the file byte for byte. The file is about 430 KB, of which the LogUp batches are about 150 KB.

## Conventions

- `Fp` is Goldilocks, `P = 2^64 − 2^32 + 1`, and an element is written `"0x"` and 16 hex digits. `Fp2 = Fp[X]/(X^2 − 7)`, and an element is `[c0, c1]`.
- A table `f[0], …, f[2^n − 1]` is the multilinear `f̃(x_1, …, x_n)` equal to `f[Σ_i x_i·2^{i−1}]` on `{0, 1}^n`: `x_1` is the least significant bit of the index. Round `j` binds `x_j`, so a point lists its challenges in round order, low variables first. `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))`.
- Each session, tree batch, forest batch and LogUp batch starts a transcript from its `context`, a `SessionContext` as in [`../transcript/`](../transcript/generate.py), and runs its stages or levels on it in order.
- Labels follow [RFC-0014 'Transcript schedule'](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#transcript-schedule), indices from 0: for a level of a forest batch where forests of more than one tree join, first `<prefix>/root/{c}` for each `c`, then `<prefix>/gamma`, then for each round `j` `<prefix>/round/{j}` and `<prefix>/r/{j}`, then `<prefix>/evals`, and for a tree level `<prefix>/line`. A LogUp batch squeezes `logup/alpha` and `logup/beta` and absorbs `logup/roots` before its first level (row 4), squeezes `<prefix>/lambda` right after `<prefix>/gamma` at a level where a fraction tree is present, and absorbs `logup/values` after its last level (row 5). A message is absorbed as TCE `Vec<Fp2>`: the `u64` little-endian length, then `c0 ‖ c1` of each element, 8 little-endian bytes each.

## Sessions

Each session lists its `stages`. A stage is one batched sumcheck:

| Field | Meaning |
|---|---|
| `prefix`, `degree` | `stage/{k}` and the round degree `d` |
| `instances` | in schedule order, each `vars` `n_i`, `factors` `k_i` and `claim` `c_i`: the instance's relation is the product of `k_i` random `Fp2` tables of `2^{n_i}` entries, and `c_i` is that product's sum over the hypercube. When every instance of the stage has at most 5 variables, each also has `tables`: its `k_i` tables in order, each its `2^{n_i}` entries `T[0], …, T[2^{n_i} − 1]` |
| `gamma` | `γ = challenge_fp2(<prefix>/gamma)` |
| `rounds` | the proof's round messages: round `j`'s is `p_j(0), p_j(2), …, p_j(d)` |
| `challenges` | `r_j = challenge_fp2(<prefix>/r/{j})` for each round |
| `claims` | `claim_1 = Σ_i γ^i · 2^{n_max − n_i} · c_i`, then `claim_{j+1} = p_j(r_j)` after each round, the last being `claim_{n_max + 1}` |
| `final_evals` | the proof's final evaluations: each instance's tables at its point, instance by instance, absorbed under `<prefix>/evals` |
| `points`, `values` | each instance's point, the last `n_i` challenges, and its relation's value there, the product of its evaluations; `Σ_i γ^i · values_i = claim_{n_max + 1}` |
| `state`, `counter` | the transcript's state and squeeze count after the stage |

| Session | Stages |
|---|---|
| single instances, products of `d` tables, for `d` from 1 to 4 | 13 stages `stage/0` to `stage/12`, stage `n` one instance on `n` variables, of degree `d` |
| a batch of 3, 7 and 12 variables | degree 3, products of 3, 2 and 3 tables |
| a batch of 12, 3, 7 and 0 variables | degree 4, products of 2, 4, 1 and 3 tables: the largest instance first, and one padded entirely |
| a batch of equal sizes | degree 2, three instances on 5 variables: `γ` without padding |
| batches of `n_max = 0`, then of a single instance | three constants (no round), one instance on 6 variables, and one constant of degree 1, on one transcript |
| degrees 5 to 7 | products of 5, 6 and 7 tables on 3 variables, then a batch of degree 7 of products of 7, 3 and 1 tables on 4, 2 and 0 variables, so that every node of RFC-0014's largest degree in `tessera-protocol/1` is used |

The 31 stages that record their tables are `stage/0` to `stage/5` of each single-instance session, the batch of equal sizes, the two batches of `n_max = 0`, and the four stages of degrees 5 to 7, the last of them front-padded: every degree from 1 to 7 appears, and a padded batch. `tessera-protocol/2`'s reduction stages reach degree 10 (RFC-0013 'Masked extensions' 'Degree'): [#565](https://github.com/AbdelStark/tessera/issues/565)'s code change adds degrees 8 to 10 here, in sessions of `tessera-protocol/2`'s context; the existing sessions are `tessera-protocol/1`'s.

## Tree batches

Each batch lists its `trees` and its `levels`. A tree has a `height` `h`, its `leaves` (`2^h` random `Fp` values) and its `root`, the product of its leaves; level `ℓ` of a tree has `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]`, level 0 being the root and level `h` the leaves. With `H` the tallest tree's height, the batch has levels 0 to `H − 1`, and a tree of height `h` joins at level `H − h` with its root claim, on no variable. The instances present at a level are the trees that have joined, in the order of `trees`: this order is the vectors' choice, where a verifier takes the schedule's.

| Field | Meaning |
|---|---|
| `prefix` | `tree/{ℓ}` |
| `instances` | the claims entering the level, in schedule order: `tree` (its index in `trees`), `point` `r_i` and `value` `v_i`, with `Ṽ(r_i) = v_i` on the tree's current level |
| `gamma`, `rounds`, `challenges`, `claims` | as in a stage, of degree 3, the batched claim `Σ_i γ^i · 2^{n_max − n_i} · v_i` |
| `final_evals` | `u_0, w_0, u_1, w_1, …`: `u_i = Ṽ_{ℓ+1}(0 ‖ r'_i)` and `w_i = Ṽ_{ℓ+1}(1 ‖ r'_i)`, with `r'_i` the last `n_i` challenges; `Σ_i γ^i · eq(r_i, r'_i) · u_i · w_i` is the last claim |
| `rho` | `ρ = challenge_fp2(tree/{ℓ}/line)` |
| `next` | each instance's claim on its next level, in the same order: point `ρ ‖ r'_i`, `ρ` the lowest variable, and value `u_i + ρ·(w_i − u_i)`; after the last level, the leaf claims |
| `state`, `counter` | the transcript after the level |

| Batch | Trees |
|---|---|
| a single product tree of height `h`, for `h` from 1 to 6 | one tree |
| trees of heights 3, 6, 1, 4, 6 and 2 | six trees joining at levels 3, 0, 5, 2, 0 and 4; the tree of height 4 has a zero leaf, so its root is 0, and at its first level one of its two evaluations is 0 |
| a tree of height 3 with a zero leaf, alone | its first level has no round (`n_max = 0`) |
| three trees of height 4 | no tree joins late |
| trees of heights 2 and 5, the shorter first | the shorter tree, with a zero leaf, joins at level 3, front-padded by three rounds |

## Forest batches

Each batch lists its `forests` and its `levels`. A forest stacks `2^s` product trees of one height `h`: its level `ℓ` is a table of `2^{s+ℓ}` entries, whose entry `y·2^ℓ + x` is entry `x` of tree `y`'s level `ℓ`, so the tree index takes the high `s` variables and the pair bit stays the lowest. With `H` the tallest forest's height, a forest joins at level `H − h` and has `s + ℓ − (H − h)` variables at level `ℓ`. At a level where forests of more than one tree join, the root challenges are squeezed before `γ`, `s_max` of them, the largest `s` among the joining forests; each joining forest's root point `r` is the last `s` of them, in the order squeezed, and its root claim is `Σ_y v_y·eq(r, y)` over its public root values `v_y`. A forest of one tree (`s = 0`) takes no challenge and enters with its root value, on no variable. The instances present at a level are the forests that have joined, in the order of `forests`, which is the vectors' choice, as for trees.

| Forest field | Meaning |
|---|---|
| `rows_log`, `height` | `s` and `h` |
| `roots` | the public root values `v_0, …, v_{2^s − 1}`, each an `Fp` value: tree `y`'s product of its leaves, which is the forest's level 0 |
| `leaves` | the `2^{s+h}` stacked leaves, `Fp` values: entry `y·2^h + x` is leaf `x` of tree `y` |

| Level field | Meaning |
|---|---|
| `prefix` | `tree/{ℓ}` |
| `root_challenges` | `challenge_fp2(tree/{ℓ}/root/{c})` for `c = 0, …, s_max − 1`, in the order squeezed; empty where no forest of more than one tree joins |
| `instances` | the claims entering the level, in the order of `forests` among those that have joined: `forest` (its index), `point` and `value`, with `Ṽ(point) = value` on the forest's current level. A forest that joins here enters on its root level with its root claim |
| `gamma`, `rounds`, `challenges`, `claims`, `final_evals`, `rho` | as in a tree batch |
| `next` | each instance's claim on its next level, `forest`, `point` and `value`, as in a tree batch; after the last level, the leaf claims, each on its forest's `2^{s+h}` stacked leaves |
| `state`, `counter` | the transcript after the level |

| Batch | Forests |
|---|---|
| a forest of 2 trees of height 1, alone | `s = 1`: one root challenge `r`, a root claim `v_0·(1 − r) + v_1·r`, one round |
| a SoftmaxRow forest of 8 trees of height 3 | `s = 3`, every tree with a zero leaf at a random position, so every root and the root claim are 0; three root challenges |
| forests of 4 and 2 trees and a single tree, of height 4; a forest of 2 trees of height 2; two single trees of height 1 | at level 0 the forests of `s = 2`, `1` and `0` join under two root challenges: the forest of 2 trees takes the second, and the single tree none. At level 2 the forest of 2 trees of height 2 joins under one root challenge, with 1 variable where the batch has 4, front-padded; its tree 0 has a zero leaf, so its roots are 0 and a non-zero value. At level 3 the two single trees join while forests are active, and no root challenge is squeezed |
| a single tree of height 3; a SoftmaxRow forest of 16 trees whose first leaves are zero and a forest of 4 trees, of height 2 | level 0 has the single tree alone, no root challenge and no round. At level 1 the forests of `s = 4` and `2` join under four root challenges; the second takes the last two, and is front-padded by two rounds. Every zero leaf of the 16 trees is a left one, so their forest's `u` is 0 at level 1 |

The oracle proves forests as it proves trees, with these differences:
- It builds each tree's levels as for a tree batch and concatenates them in the order of `y`, then checks that the stacked tables pair their entries as one tree's do, `F_ℓ[k] = F_{ℓ+1}[2k]·F_{ℓ+1}[2k + 1]`.
- The public roots are each tree's leaves multiplied together, checked against the stacked root level.
- A root claim is `Σ_y v_y·eq(r, y)`, with `eq` from its product formula and `y` read as a Boolean point, lowest bit first, and is checked against the root level's extension at `r`, computed by selecting `r_i` or `1 − r_i` for each bit.
- It proves every tree batch again as forests of one tree, on a fresh transcript of the batch's context, and checks that it gets the tree batch's levels exactly.

## LogUp batches

Each batch is one leaf stage of LogUp-GKR ([RFC-0008 'Fraction tree'](../../../docs/rfcs/RFC-0008-logup-and-tables.md#fraction-tree), [RFC-0010 'Tree levels'](../../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels)): the lookup tree and the table tree of each LogUp instance, proven level by level beside the product forests of the same stage, on one transcript started from the batch's `context`. A cheating batch has the same fields and records what a cheating prover sent, with every value a verifier derives from it.

| Field | Meaning |
|---|---|
| `name` | what the batch exercises |
| `context` | the `SessionContext` the transcript starts from |
| `alpha`, `beta` | `challenge_fp2(logup/alpha)`, then `challenge_fp2(logup/beta)`: the transcript's first two squeezes |
| `instances` | the LogUp instances in schedule order, below |
| `forests` | the product forests, as in a forest batch: `rows_log`, `height`, `roots` and `leaves`; may be empty |
| `trees` | the schedule: every tree as an instance, `{"kind", "index"}`, `kind` being `"lookup"`, `"table"` or `"forest"` and `index` its place in `instances` or `forests` |
| `roots` | the `logup/roots` message, absorbed right after `beta`, even when empty: `n_a, d_a, n_t, d_t` for each instance in `instances` order, the roots of its lookup tree and of its table tree |
| `levels` | the tree levels, below |
| `values` | the `logup/values` message, absorbed after the last level, or after `logup/roots` when no tree has a level, even when empty: `valũe(ρ_a)` for each instance with values, in `instances` order |
| `leaf_checks` | per instance, in `instances` order, the claims its trees end with and the values checked against them, below |
| `state`, `counter` | the transcript after `logup/values` |
| `expected` | `"accept"`, or the first check a verifier fails, `{"reason", "check", "instance"}`, below |

An instance is a table and the lookups into it. With `a = key + β·value` the tuple of a lookup and `t_j = key_j + β·value_j` that of row `j`:

| Instance field | Meaning |
|---|---|
| `rows` | the `T` rows, each `[key, value]` of `Fp` values. A table with values has distinct random keys and random values; a table without, Range-like, has the keys `0` to `T − 1` and every value 0 |
| `has_values` | false for a table without values, whose instance sends no value and takes `v = 0` |
| `lookups` | the `N ≥ 1` lookups, in leaf order, each the index of its row. In a cheating batch, a lookup that is no row of the table is written as its `[key, value]` pair |
| `multiplicities` | `m_j` for each of the `2^{h_t}` leaves of the table tree, `Fp` values: the lookups of row `j`, row 0's also counting the lookup tree's `2^{h_a} − N` padding leaves. An honest prover gives the padding rows 0; a padding row is a copy of row 0, so a multiplicity there counts lookups of row 0 |
| `h_a`, `h_t` | `⌈log2 N⌉` and `⌈log2 T⌉`, the heights of the lookup tree and the table tree |

- The lookup tree has `2^{h_a}` leaves: `(1, α − a_i)` for lookup `i`, then `(1, α − t_0)` for each padding leaf.
- The table tree has `2^{h_t}` leaves: `(−m_j, α − t_j)` for row `j`, then `(−m_j, α − t_0)` for each padding row, a copy of row 0, which an honest prover gives the multiplicity 0.
- Node `x` of level `ℓ` is `(n_L·d_R + n_R·d_L, d_L·d_R)` of entries `2x` (left) and `2x + 1` (right) of level `ℓ + 1`. Level 0 is the root, on no variable.

With `H` the tallest tree's or forest's height, a tree of height `h` joins at level `H − h`: a fraction tree with its root from `roots` as its claims, on no variable, and a forest as in a forest batch. A tree of height 0 never joins a level: its root, its one leaf, is its leaf claim, on no variable, and a batch whose trees all have height 0 has no level, its `logup/values` following `logup/roots`. A batch without LogUp instances absorbs both messages empty. The instances present at a level are the trees of `trees` that have joined, in that order. The order is the vectors' choice and varies: lookup and table trees paired, all lookup trees first, a table tree before its lookup tree, and forests between fraction trees. The roots, the values and the leaf checks follow `instances`.

| Level field | Meaning |
|---|---|
| `prefix`, `root_challenges`, `gamma` | as in a forest batch: `tree/{ℓ}`, the root challenges of the forests of more than one tree that join here, then `challenge_fp2(tree/{ℓ}/gamma)` |
| `lambda` | `challenge_fp2(tree/{ℓ}/lambda)`, squeezed right after `gamma` at a level where a lookup or table tree is present; `null` at a level of forests only |
| `instances` | the claims entering the level, in the order of `trees`: a fraction tree's `{"tree", "point", "numerator", "denominator"}`, with `Ñ(point) = numerator` and `D̃(point) = denominator` on its current level, and a forest's `{"tree", "point", "value"}`; `tree` is the index in `trees` |
| `rounds`, `challenges`, `claims` | as in a stage, of degree 3. The first claim is `Σ_i γ^i · 2^{n_max − n_i} · c_i`, with `c_i` a forest's value and a fraction tree's `numerator + λ·denominator`, and each later one is `p_j(r_j)` with `p_j(1) = claim_j − p_j(0)`, as a verifier derives it |
| `final_evals` | instance by instance: `u, w` for a forest, and `N_0, N_1, D_0, D_1` for a fraction tree, its numerators and denominators on its next level at `0 ‖ r'_i` and `1 ‖ r'_i`. The final check is `Σ_i γ^i · eq(r_i, r'_i) · g_i` against the last claim, with `g_i = u·w` for a forest and `N_0·D_1 + N_1·D_0 + λ·D_0·D_1` for a fraction tree |
| `rho`, `next` | `ρ = challenge_fp2(tree/{ℓ}/line)`, and each instance's claims on its next level at `ρ ‖ r'_i`, in the same form: `u + ρ·(w − u)` for a forest, `N_0 + ρ·(N_1 − N_0)` and `D_0 + ρ·(D_1 − D_0)` for a fraction tree |
| `state`, `counter` | the transcript after the level |

| Leaf check | Meaning |
|---|---|
| `lookup` | the lookup tree's leaf claims, `point` `ρ_a`, `numerator` `p` and `denominator` `q`; `value`, the `v` of `values`, `null` without values; and `key`, the key claim `α − q − β·v` |
| `table` | the table tree's leaf claims, `point` `ρ_t`, `numerator` `p` and `denominator` `q`; `multiplicity`, the multiplicity claim `−p`; and `table_eval`, `t̃(ρ_t)` of the table polynomial `key_t + β·value_t`, its padding rows copies of row 0, which must make `q = α − t̃(ρ_t)` |

For an honest prover, `key`, `value`, `multiplicity` and `table_eval` are the extensions at their points of the lookup side's key and value polynomials, which hold row 0's key and value in their padding slots, of the multiplicities, and of the table polynomial.

A verifier checks in this order and stops at the first failure ([RFC-0019 'Check order'](../../../docs/rfcs/RFC-0019-verifier.md#check-order) step 7), which is `expected`:
1. the roots, instance by instance: `d_a ≠ 0`, `d_t ≠ 0` and `n_a·d_t + n_t·d_a = 0`, else `{"reason": "Lookup", "check": "roots", "instance": k}`, `k` the instance;
2. the levels, each as in a forest batch, else `{"reason": "Sumcheck", "check": "level ℓ", "instance": i}`, `i` the place among the level's `instances` of the tree whose root the message falsified, which the cheating batches make also its LogUp instance and its index in `trees`;
3. the leaf checks, instance by instance, the lookup tree first: its `numerator` is 1, else `"leaf numerator"`, then the table tree's `denominator` is `α − table_eval`, else `"table evaluation"`, both with `"reason": "Lookup"`.

Each cheating batch fails exactly one check, except that a root `(0, 0)` also fails the level where its tree joins, after the roots.

| Batch | Instances and forests |
|---|---|
| one instance alone | `N = 6`, `T = 7`: both trees of height 3, with two padding leaves and one padding row |
| one instance with `N = T = 1` | both trees of height 0: no level, and the values follow the roots |
| three instances | trees of heights 2 and 1, 1 and 2, 0 and 2, joining at levels 0 and 1, 1 and 0, never and 0; the second table has no values |
| four instances padded on both sides | `N` of 3, 4, 5 and 2, `T` of 5, 4, 3 and 1; `trees` lists the lookup trees, then the table trees |
| three instances beside three forests | `trees` interleaves them. Level 0 has a forest of 2 trees of height 4 alone, and no `λ`. Fraction trees join at levels 1, 2 and 3, and a single tree of height 3 at level 1. At level 2, a SoftmaxRow forest of 4 trees of height 2 joins under two root challenges where fraction trees are present. The third instance has `N = T = 1` |
| a lookup tree of height 10 | `N = 513` and `T = 17`: 511 padding leaves, and a table tree of height 5 with 15 padding rows, first in `trees` |
| two instances with `N = T = 1` beside a forest | the forest, of 2 trees of height 2, is alone at both levels, so no `λ` is squeezed |
| a multiplicity on a padding row | `N = T = 3`. The prover counts the lookup tree's padding leaf, a lookup of row 0, on the table's padding row instead of on row 0: `m_0 = 0`, `m_3 = 1`. A padding row is a copy of row 0, so the batch verifies |
| one instance without values | `logup/values` is empty, and absorbed after the last level |
| no LogUp instance | a forest of 2 trees of height 1 alone: `logup/roots` and `logup/values` are empty, and absorbed |

| Cheating batch | What the prover does | `expected` |
|---|---|---|
| a lookup outside the table | instance 1, Range-like with `T = 2`, looks up the key 2, which no multiplicity counts; honest trees | `Lookup`, `roots`, 1 |
| a wrong value | a lookup of a row's key with its value plus 1, counted in that row; honest trees | `Lookup`, `roots`, 0 |
| a multiplicity off by one | `m_1` plus 1; honest trees, beside a forest | `Lookup`, `roots`, 0 |
| a lookup root `(0, 0)` | the message gives instance 1's lookup tree the root `(0, 0)`, which passes `n_a·d_t + n_t·d_a = 0`; honest trees | `Lookup`, `roots`, 1 |
| a table root `(0, 0)` | the same for the table tree | `Lookup`, `roots`, 0 |
| false roots | a multiplicity off by one, hidden by the message's `n_a = −n_t·d_a/d_t`; honest trees | `Sumcheck`, `level 0`, 0 |
| scaled roots | the message gives instance 0's lookup tree the root `(c·n_a, c·d_a)`, the same fraction; honest trees, the tree joining at level 1 | `Sumcheck`, `level 1`, 0 |
| a zeroed leaf numerator | instance 1 looks up the key 2, outside its table, with the leaf numerator 0 and no multiplicity: the roots sum to zero and the levels pass | `Lookup`, `leaf numerator`, 1 |
| a different table | instance 1's trees use its table with one row's value plus 1, a row its lookups use, so those lookups are written as pairs; honest trees over that table | `Lookup`, `table evaluation`, 1 |
| the phantom row | `T = 3`, without the row `(0, 0)`. The prover looks up `(0, 0)` and puts the multiplicity 1 on the padding row; honest trees, whose padding leaf `(−1, α − t_0)` counts a lookup of row 0, so the identity fails | `Lookup`, `roots`, 0 |
| the phantom row, zero padding | the same witness, but the prover pads its table tree with `(−m_j, α)`, as RFC-0008 did before #62: the roots, the levels and the numerator pass | `Lookup`, `table evaluation`, 0 |

The two phantom-row batches guard against the attack that padding the table with a row of key and value 0 allowed: a lookup of `(0, 0)`, which is no row, balanced by a multiplicity on a padding row, which no check constrains. Each guards one side:
- In the first, the prover follows RFC-0008 as fixed. Its padding row is a copy of row 0, so the multiplicity there counts a lookup of row 0, and the roots fail. A prover or a reference that still pads with `(0, α)` computes other trees, and other roots, from the same witness.
- The second is the attack as a verifier sees it. Its tree passes the roots, the levels and the lookup tree's numerator check. Only `t̃` with padding rows copying row 0 rejects it, at the table evaluation: a verifier that kept zero padding rows accepts it. Such a verifier also rejects every honest batch with values and padding rows. A Range-like table cannot tell the two paddings apart, since its row 0 is `(0, 0)`.

The oracle proves LogUp batches as it proves forests, with these differences:
- It builds each tree from its instance as above, and a level proves a fraction tree's relation as three products of three tables, `eq·N_0·D_1`, `eq·N_1·D_0` and `λ·eq·D_0·D_1`, padded literally. A cheating prover's trees are honest trees of its witness; only its roots message may differ from them.
- A separate verifier reads each batch back from the written file and uses only its public data, the tables, the sizes, the forests' roots and `trees`, and its messages. It rederives every challenge, claim, point and state and requires each to be the recorded one. It evaluates the table polynomial and the forests' root claims by folding them variable by variable, where the prover sums over `eq`, and lists every check that fails.

## How the oracle computes

- It pads an instance with fewer variables literally: each table becomes `2^{n_max}` entries, `T'[idx] = T[idx >> (n_max − n_i)]`, constant in the lowest `n_max − n_i` variables, and one plain sumcheck runs over the padded tables of every instance, `γ^i` weighting instance `i`. Rust implementations pad implicitly.
- Each round evaluates the batched polynomial with `x_j` fixed to `0, 1, …, d` and the later variables summed, sends all but `p(1)`, then binds every padded table: `f[2k] + r_j·(f[2k + 1] − f[2k])`.
- Each claim it records is the bound tables' sum, not the round polynomial's value, and each final evaluation is `f̃(point) = Σ_x f[x]·eq(point, x)` from the original table. A LogUp batch records instead the claims a verifier derives, `p_j(r_j)` by Lagrange's formula, which can differ from a cheating prover's; in an honest batch the oracle checks that they are the bound tables' sums.
- A tree level's relation is the product of three tables over the instance's variables: `eq(r_i, x)`, entry by entry from its definition, and the halves `V_{ℓ+1}[2x]` and `V_{ℓ+1}[2x + 1]`.

Before it writes the file, the script checks:
- the official BLAKE3 vectors, in all three modes;
- that the padded batch sums to `Σ_i γ^i · 2^{n_max − n_i} · c_i`;
- for every round, that `p(0) + p(1)` is the claim, and that Lagrange's formula through `p(0), …, p(d)` at the challenge gives the bound tables' sum;
- that each final evaluation, from its definition, is the padded table bound to every challenge, and that the final check holds;
- for trees, that every level's claim is the multilinear extension of that level at its point, that its relation sums to it, that the bound `eq` table is `eq(r_i, r'_i)` from the product formula, that each next claim is the next level's extension at `ρ ‖ r'_i`, and that every leaf claim is the leaves' extension at its point;
- for forests, the same of every level, on the stacked tables; that the stacked levels pair their entries as a tree's; that every root claim is `Σ_y v_y·eq(r, y)` and the root level's extension at `r`; and that every leaf claim is the stacked leaves' extension at its point;
- that exactly the trees with a zero leaf have root 0, alone or in a forest;
- that every tree batch, proven again as forests of one tree, gives the same levels;
- for every fraction tree, that no leaf denominator is 0, that every node's denominator is the product of the leaf denominators under it and its fraction the sum of their fractions, and that the root's `n/d` is `Σ_i n_i/d_i`;
- for the LogUp levels, the checks of the forest levels on numerators and denominators, and that every leaf claim is the extension of the leaves' numerators and denominators at its point;
- that `key`, `value`, `multiplicity` and `table_eval` are the extensions of the key, value, multiplicity and table polynomials at their points, and that the table tree's denominator claim is `α − t̃(ρ_t)` exactly when its trees use the public table, padded with copies of row 0;
- that the identity `Σ_i 1/(α − a_i) = Σ_j m_j/(α − t_j)` holds for every honest witness, the one counting on a padding row included, and fails for each cheat that changes the witness, row 0's multiplicity taking the padding rows' and giving back the lookup tree's padding leaves; which of the root checks each instance fails; and that the padding rows' multiplicities are 0 except where a batch counts there;
- that the verifier finds exactly the checks each LogUp batch is built to fail, in check order, so that every check before `expected` passes;
- that the file it writes parses back to the values it computed, that the forest batches, then the LogUp batches, are each one key appended at its end, and that the file without `logup_batches` has the SHA-256 the script pins, that of the file before them.
