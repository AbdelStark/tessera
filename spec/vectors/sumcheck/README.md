# Sumcheck vectors

`sumcheck.json` holds honest proofs of the sumchecks of [RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md): single and batched sumchecks of products of multilinear tables, front-padded as ['Batching within a stage'](../../../docs/rfcs/RFC-0010-proof-schedule.md#batching-within-a-stage) says, with the messages of ['Sumcheck messages'](../../../docs/rfcs/RFC-0010-proof-schedule.md#sumcheck-messages); and batches of product trees proven level by level, as ['Tree levels'](../../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels) says. Every challenge, claim, point and transcript state of each proof is recorded, so an implementation can locate the first step where it diverges, and the stages on at most 5 variables also record their tables, so a prover can reproduce them whole. `verifier/crates/tv-core/tests/` verifies every proof, reproduces every recorded value, and rejects every proof changed in one element; `crates/tessera-sumcheck/tests/vectors.rs` reproduces every stage that records its tables, message by message, and every tree level from the trees' leaves.

[`generate.py`](generate.py) writes it. The script is the independent oracle, an honest prover written from the RFC text alone with Python's arbitrary-precision integers. Its transcript is the oracle of [`../transcript/generate.py`](../transcript/generate.py), imported as that file imports [`../hash/generate.py`](../hash/generate.py); the script reruns the official BLAKE3 test vectors through it before anything is written. Its inputs come from SplitMix64 with the seed the file states, words at or above `P` rejected, so `python3 spec/vectors/sumcheck/generate.py` reproduces the file byte for byte. The file is about 250 KB.

## Conventions

- `Fp` is Goldilocks, `P = 2^64 − 2^32 + 1`, and an element is written `"0x"` and 16 hex digits. `Fp2 = Fp[X]/(X^2 − 7)`, and an element is `[c0, c1]`.
- A table `f[0], …, f[2^n − 1]` is the multilinear `f̃(x_1, …, x_n)` equal to `f[Σ_i x_i·2^{i−1}]` on `{0, 1}^n`: `x_1` is the least significant bit of the index. Round `j` binds `x_j`, so a point lists its challenges in round order, low variables first. `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))`.
- Each session and tree batch starts a transcript from its `context`, a `SessionContext` as in [`../transcript/`](../transcript/generate.py), and runs its stages or levels on it in order.
- Labels follow [RFC-0014 'Transcript schedule'](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#transcript-schedule), indices from 0: `<prefix>/gamma`, then for each round `j` `<prefix>/round/{j}` and `<prefix>/r/{j}`, then `<prefix>/evals`, and for a tree level `<prefix>/line`. A message is absorbed as TCE `Vec<Fp2>`: the `u64` little-endian length, then `c0 ‖ c1` of each element, 8 little-endian bytes each.

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
| degrees 5 to 7 | products of 5, 6 and 7 tables on 3 variables, then a batch of degree 7 of products of 7, 3 and 1 tables on 4, 2 and 0 variables, so that every node of RFC-0014's largest degree is used |

The 31 stages that record their tables are `stage/0` to `stage/5` of each single-instance session, the batch of equal sizes, the two batches of `n_max = 0`, and the four stages of degrees 5 to 7, the last of them front-padded: every degree from 1 to 7 appears, and a padded batch.

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

## How the oracle computes

- It pads an instance with fewer variables literally: each table becomes `2^{n_max}` entries, `T'[idx] = T[idx >> (n_max − n_i)]`, constant in the lowest `n_max − n_i` variables, and one plain sumcheck runs over the padded tables of every instance, `γ^i` weighting instance `i`. Rust implementations pad implicitly.
- Each round evaluates the batched polynomial with `x_j` fixed to `0, 1, …, d` and the later variables summed, sends all but `p(1)`, then binds every padded table: `f[2k] + r_j·(f[2k + 1] − f[2k])`.
- Each claim it records is the bound tables' sum, not the round polynomial's value, and each final evaluation is `f̃(point) = Σ_x f[x]·eq(point, x)` from the original table.
- A tree level's relation is the product of three tables over the instance's variables: `eq(r_i, x)`, entry by entry from its definition, and the halves `V_{ℓ+1}[2x]` and `V_{ℓ+1}[2x + 1]`.

Before it writes the file, the script checks:
- the official BLAKE3 vectors, in all three modes;
- that the padded batch sums to `Σ_i γ^i · 2^{n_max − n_i} · c_i`;
- for every round, that `p(0) + p(1)` is the claim, and that Lagrange's formula through `p(0), …, p(d)` at the challenge gives the bound tables' sum;
- that each final evaluation, from its definition, is the padded table bound to every challenge, and that the final check holds;
- for trees, that every level's claim is the multilinear extension of that level at its point, that its relation sums to it, that the bound `eq` table is `eq(r_i, r'_i)` from the product formula, that each next claim is the next level's extension at `ρ ‖ r'_i`, and that every leaf claim is the leaves' extension at its point;
- that exactly the trees with a zero leaf have root 0;
- that the file it writes parses back to the values it computed.
