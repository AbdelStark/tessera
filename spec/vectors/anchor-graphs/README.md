# The SHA-256 circuit, node by node

[RFC-0066 'The SHA-256 circuit'](../../../docs/rfcs/RFC-0066-bits-and-sha256.md#the-sha-256-circuit) specifies the anchor's SHA-256 graph to the node. The verifier must rebuild it from public data alone (RFC-0001, SP-7), and its graph must equal the compiler's node for node: the zero-checks' order, the instances and their labels follow node order. These vectors hold both to one reading of the RFC.

- `sha256-graphs.json` is the graph for five parameter sets.
- `sha256-inputs.json` is the circuit's inputs from bytes.

## How they are made

[`generate.py`](generate.py) is the independent oracle.

**Sources.** It was written from:
- RFC-0066;
- [RFC-0072](../../../docs/rfcs/RFC-0072-anchor-relations.md) ('Public tensors' and the sizes table);
- [RFC-0006 'Operation set'](../../../docs/rfcs/RFC-0006-compiler-ir.md#operation-set);
- [RFC-0044](../../../docs/rfcs/RFC-0044-lowering-nodes.md)'s notation;
- FIPS 180-4.

Its author read neither implementation.

**Checks before writing.**
- **SHA-256.** Its own SHA-256 gives `hashlib`'s digests, FIPS 180-4's two examples (`"abc"` and the 56-byte message) and the CAVP digests of [`../sha256/`](../sha256/README.md). Its round constants and initial value come from their definitions, the cube and square roots of the first primes.
- **Shape rules.** Each graph keeps RFC-0006's rules:
  - inputs before their readers, ids dense in creation order;
  - each operation's shapes;
  - `1 ≤ low`, `low + high ≤ 62`.
- **The interpreter.** It runs a graph on each of the 81 input sets of `sha256-inputs.json`, on true integers, its Boolean functions as logic rather than as polynomials.
  - **Which graph.** For 66 sets it is a graph of the file: FIPS 180-4's one-block example, every one-block CAVP short message, three of them at once, the five key commitments of [`../anchor/frames.json`](../anchor/README.md) one by one and three of them at once, and two pairs of its frames of one token. For the other 15 it is a graph of their parameters that the file does not hold.
  - **What it checks.** Every graph gives FIPS 180-4's chaining words after every block. Every `BitDecompose` input lies inside its width, every `BitFn` reads bits, and every `AssertEqual` holds.
- **The inputs.** The public bits and the placed committed bits make up each padded message.
- **RFC-0072's sizes.** The graphs' counts are RFC-0072's sizes table ('Messages, rejections and stages'):
  - 6,032 nodes per public block, 6,038 at `N = 1` with committed bytes, 6,040 at `N > 1` when one block holds all committed bytes, and 6,041 when `0 < K_n < K`;
  - 680 compression zero-checks a block;
  - a first block of public bytes: 2,415 selections, 331 `BitFn` and 79 `Mul` not public;
  - a later block of public bytes (including a trailing padding block of a committed message): 2,441, 335 and 79;
  - a later block with committed bytes (`0 < K_n < K`, `K·M` a power of two): 682, 2,530, 353 and 97 (`+1` input padding zero-check per committed block when `committed [K, M]` has padded positions);
  - the key commitment: 690, 2,523, 349 and 97;
  - three key commitments: the same, and one more zero-check, the committed bytes' input padding check, since `[32, 3]` has padded positions.

  These figures were counted from the prover's builder. A graph that reproduces all of them has, at least, its public tensors where the prover's are.

`python3 -I spec/vectors/anchor-graphs/generate.py` rewrites both files byte for byte in about a second, and `cargo xtask gen-vectors --check` fails when either differs. `sha256-graphs.json` is about 4.2 MB.

## `sha256-graphs.json`

`round_constants` and `initial_value` are FIPS 180-4's `K_t` and `H^{(0)}`, in hex. `cases` holds one graph per parameter set:

| Case | `N` | `M` | Committed offsets | What it covers |
|---|---|---|---|---|
| `key commitment` | 1 | 1 | 21 to 52 | RFC-0066 'For the anchor': `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k`, 53 bytes |
| `three key commitments` | 1 | 3 | 21 to 52 | `M = 3` with committed bytes: `committed` `[32, 3]` has padded positions, and its `Transpose` moves real data |
| `one public block` | 1 | 1 | none | the words read `message` directly |
| `three public messages` | 1 | 3 | none | `M = 3`: three rows of every word, padded to four |
| `two blocks, committed bytes crossing into the second` | 2 | 2 | 41 to 66 | two frames of one token each: a 41-byte header and 26 committed bytes. `committed` `[26, 2]` has padded positions, so it takes an input padding check, and its `Transpose` moves real data |

Each case has `name`, `N`, `M`, `K`, `committed_offsets`, and these fields:

| Field | Meaning |
|---|---|
| `tensors` | the number of tensors: graph inputs and node outputs |
| `inputs[]` | `[id, role, shape, kind, type, lo, hi]`, as in [`../lowering/`](../lowering/README.md). Roles are RFC-0066's input names: `message`, `start`, `masks`, `committed`, `place`, `digest` |
| `nodes[]` | `[id, operation, parameters, inputs, outputs, at]`, as in `../lowering/`: each output `[id, shape, kind]`. `at` is for people |
| `constants` | by index, each a list of integers |
| `chaining` | for each block, the tensor ids of its eight chaining words `a` to `h`, each `[32, M]`: the words RFC-0066's builder returns after every block |
| `public` | the tensors RFC-0072 'Public tensors' makes public, sorted: the public inputs, and the outputs of every node that is not gadget-backed whose inputs are all public |
| `counts` | for each block (`blocks`), for the graph's own nodes outside the blocks (`graph`) and in all (`total`): the zero-checks of stage 1 by kind (`input_padding`, `booleanity`, `bit_padding`, `recomposition`, `assert_equal`, and their sum `zero_checks`), the `selections`, `bitfn` and `mul` nodes whose output is not public, and the `nodes` |

**Kinds** are RFC-0044's: `P` public, `A` committed advice, `v` virtual. A node output's kind is the IR's, which RFC-0006 derives from the operation: every output is `v` but `BitDecompose`'s, which are `A`. Whether a virtual tensor is public is RFC-0072's analysis, which `public` records.

**Operations** carry RFC-0006's names and parameters:

| Operation | Parameters |
|---|---|
| `GatherPublic` | `index`: a constant's index, the constant the list of rows of axis 0 |
| `Slice` | `axis`, `start`, `len` |
| `Reshape`, `Add`, `AssertEqual` | none |
| `Transpose` | `perm`: output axis `i` is input axis `perm[i]` |
| `Mul` | `broadcast`: `Trailing` |
| `ConstMul`, `ConstAdd` | `c`: a constant's index, the constant of one value |
| `BitDecompose` | `low`, `high` |
| `BitFn` | `f`: `Xor`, `Xor3`, `Ch` or `Maj` |

## `sha256-inputs.json`

`cases` holds 81 input sets. Each has:
- `name` and `source`;
- `graph`: the graph of `sha256-graphs.json` with its parameters, or null;
- `N`, `M`, `K` and `committed_offsets`;
- `messages` (hex) and `digests` (hex), and `chaining`: for each message, its eight chaining words after each block, in hex;
- the circuit's inputs, each a tensor written as a string of its entries in row-major order, a character `0` or `1` each:
  - `message` `[512·N, M]`: the padded messages' bits, 0 at every committed byte's bits;
  - `place` `[512·N]`: 1 on the committed bytes' rows, null with no committed byte;
  - `start` `[256, M]`: `H^{(0)}`'s bits;
  - `masks` `[2, 32]`: `SHR^3`'s mask, then `SHR^{10}`'s;
  - `digest` `[256, M]`: the digests' bits;
- `committed` `[K, M]`: the committed bytes, as integers, row `k` the byte at the `k`-th committed offset; null with none.

The input sets are:
- FIPS 180-4's two examples;
- the 65 CAVP short messages, and the two shortest CAVP long messages, of 3 and 5 blocks;
- three short messages at once (`M = 3`);
- the 64-byte CAVP message with its first word and its last three bytes committed;
- the key commitments of `../anchor/frames.json`'s five sessions, and of its sessions 0 to 2 at once (`M = 3`);
- two pairs of its frames of one token;
- its frames of 0 and 40 tokens, of 2 and 4 blocks.

Every digest is checked against its source's and `hashlib`'s.

**Bit order** (RFC-0066 'Layout', 'Byte order', 'The output').
- A word tensor's row `i` is the word's bit of weight `2^i`.
- Message row `32j + i` is bit `i` of word `j`: bit `i mod 8` of byte `4j + 3 − ⌊i/8⌋` of the padded message.
- `start` and `digest` hold word `j` on rows `32j` to `32j + 31`, each word big-endian from the digest's bytes `4j` to `4j + 3`.
- An `SHR^r` mask is 1 on rows `i < 32 − r`.

## Readings

The oracle had to choose these twelve readings of RFC-0066, each a place where two implementations could build different graphs. The compiler's builder made the same choices: `crates/tessera-compiler/tests/anchor_vectors.rs` finds the two graphs equal node for node for all five cases. [RFC-0066 'The SHA-256 circuit'](../../../docs/rfcs/RFC-0066-bits-and-sha256.md#the-sha-256-circuit) now states each reading, so that the verifier's builder (#150 S6) has none left to make; each item below quotes the sentence that fixes it.

1. **`start`'s rows.** `start` holds word `j` of `H^{(0)}` on rows `32j` to `32j + 31`, row `32j + i` the bit of weight `2^i`, and initial word `j` is its `Slice` from row `32j`. RFC-0066 stated this layout for the digest only. 'The output', **Initial values**: "That input, `start`, holds word `j` of `H^{(0)}` on rows `32j` to `32j + 31`, …".
2. **`masks`' rows.** `SHR^3`'s mask is row 0 and `SHR^{10}`'s row 1, and step 1 builds `SHR^3`'s mask first. 'Layout', **`SHR^r`**: "Row 0 of that input is `SHR^3`'s mask and row 1 `SHR^{10}`'s, …".
3. **Per-block message assembly (OQ-0072-2).** With no committed byte (`K = 0`) the graph has no `committed` or `place` input. For each block `n`, when `K_n = 0` its sixteen words are `Slice`s of `message` from row `512n + 32t`; when `K_n > 0` it slices `bytes_n` from `committed` (unless `K_n = K`), decomposes `bytes_n` to `[K_n, M, 8]`, transposes and reshapes to `[8K_n, M]`, gathers 512 rows, multiplies by `place_n` (`Slice` of `place` when `N > 1`, else `place`), `Xor`s `message_n` (`Slice` of `message` when `N > 1`, else `message`) with that product, and slices its sixteen words from row `32t` of that `[512, M]` `Xor`. 'The message', **Each block's bits**.
4. **A function's nodes.** FIPS 180-4's definition order (4.4 to 4.7) orders a function's gathers:
   - `Σ0`: `ROTR^2`, `ROTR^{13}`, `ROTR^{22}`;
   - `Σ1`: `ROTR^6`, `ROTR^{11}`, `ROTR^{25}`;
   - `σ0`: `ROTR^7`, `ROTR^{18}`, `SHR^3`;
   - `σ1`: `ROTR^{17}`, `ROTR^{19}`, `SHR^{10}`.

   An `SHR`'s gather is followed at once by its `Mul`, `Mul(gather, mask)`. The `Xor3` reads the three parts in that order. 'The functions': "A function's parts follow FIPS 180-4's equations (4.4) to (4.7): …".
5. **The message schedule's rows.** `W_t`'s `Add`s are `((σ1(W_{t−2}) + W_{t−7}) + σ0(W_{t−15})) + W_{t−16}`, the order of RFC-0066's table. 'The additions', **Operand order**: "A sum's rows are the left fold of its terms in the table's order, `K_t` aside, each `Add` taking the running sum first: …".
6. **The rounds' rows.**
   - **Shared rows.** `Add(Add(Add(h, Σ1(e)), Ch(e, f, g)), W_t)`.
   - **`e`'s rows.** `Add(shared, d)`.
   - **`a`'s rows.** `Add(Add(shared, Σ0(a)), Maj(a, b, c))`.

   RFC-0066's prose added `d` to the shared rows, while its table listed `e`'s terms with `d` first: a reader who followed the table would have built `Add(d, shared)`, another graph. The table now lists `d` last, `h + Σ1(e) + Ch(e, f, g) + K_t + W_t + d`. 'The additions', **The rounds**: "The shared rows are one tensor, `Add(Add(Add(h, Σ1(e)), Ch(e, f, g)), W_t)`, that both sums read: `e`'s rows are `Add(shared, d)` and `a`'s `Add(Add(shared, Σ0(a)), Maj(a, b, c))`."
7. **The chaining sums' rows.** `Add(H_j, v_j)`, the previous chaining word first. The first block's `H_j` is the `Slice` of `start`. 'The additions', **The chaining**: "Chaining word `j`'s rows are `Add(H_j, v_j)`: …".
8. **The digest's binding.** For each word, the `Slice` of the digest, then `AssertEqual(computed word, digest rows)`. 'The output', **The digest**: "Digest word `j`'s nodes are the `Slice { axis: 0, start: 32j, len: 32 }` of `digest`, then the `AssertEqual` of the last block's chaining word `j` and that `Slice`, in this operand order."
9. **Operand orders that the RFC fixes.** These were not left open; the oracle follows the sentences that fix them:
   - `Add(even, ConstMul(odd))` at each value level: 'Layout', **The integer value of rows**;
   - `Xor(message_n, Mul(gather(c_n), place_n))`: 'The message', **Each block's bits**;
   - `Ch(e, f, g)` and `Maj(a, b, c)`: the inputs of '`BitFn { f }`''s table, and the sums' table;
   - `Mul { Trailing }`, whose broadcast operand comes second: RFC-0006 'Operation set'.
10. **Shapes and permutations.**
    - A block's decomposed bytes are `[K_n, M, 8]`, moved by `Transpose { perm: [0, 2, 1] }` to `[K_n, 8, M]`.
    - A sum's result bits `[M, 32]` are moved by `Transpose { perm: [1, 0] }`.
    - A value level's `ConstMul` constant is `[2^{2^ℓ}]`, a scalar.
    - A round's `ConstAdd` constant is `[K_t]`.
    - The `W_t` and chaining sums have no `ConstAdd`, their constant being 0 ('A sum is': "its `ConstAdd` when the constant is not 0").

    'The graph', after 'A sum is': "A block's bytes' bits `[K_n, M, 8]` move to `[K_n, 8, M]` by `Transpose { perm: [0, 2, 1] }`, a sum's result bits `[M, 32]` to `[32, M]` by `Transpose { perm: [1, 0] }`, and every `ConstMul` and `ConstAdd` constant holds one value, a level's `2^{2^ℓ}` or a round's `K_t`."
11. **The digest input's id.** It follows the last block's last node, the `Transpose` of chaining word `h`, and precedes the first digest `Slice`. The RFC fixed it already, 'The graph': "The digest, public bits `[256, M]`, is declared after the circuit's nodes, before its binding's."
12. **Messages of one circuit.** They share `N`, `K` and the committed offsets, but may differ in length. Their padding is public bits, and the graph does not depend on the messages' bytes. The section's opening: "The `M` messages share `N` and the committed offsets, since `place` and the gather serve them all, and may differ in length, each padded in its own column of `message`, so that the graph depends on `N`, `M` and the offsets alone."

## Who checks them

- **The compiler's builder** (`tessera_compiler::anchor::sha256`) equals each graph node for node: [`crates/tessera-compiler/tests/anchor_vectors.rs`](../../../crates/tessera-compiler/tests/anchor_vectors.rs) compares its inputs, nodes, constants, chaining words, public tensors and counts, the zero-checks from its schedule's stage 1, with each case's, and reports the first differing input and node with its `at`.
- **The compiler's public inputs from bytes** equal `sha256-inputs.json`: the same test computes them with `sha256_pad`, `Sha256Layout::public_bits` and `place`, and `sha256_bits` of `H^{(0)}` and of the digests, then runs its circuit on every set to FIPS 180-4's chaining words.
- **The verifier's builder** (#150, slice S6) must equal each graph node for node, constant for constant, and its public inputs from bytes `sha256-inputs.json`.
