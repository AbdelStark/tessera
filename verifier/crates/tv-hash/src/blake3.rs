// SPDX-License-Identifier: Apache-2.0 OR MIT

//! BLAKE3, written from its specification (O'Connor, Aumasson, Neves and Wilcox-O'Hearn, "BLAKE3:
//! one function, fast everywhere", 2020; the section numbers below are that document's).
//!
//! - The compression function (§2.2) is written once, as a `const fn` over sixteen words, so that
//!   the compiler derives the context keys of the Tessera domains ([`context_key`], §2.3).
//! - An input of at most one chunk, the common case (Merkle nodes, column steps, transcript
//!   operations), takes [`single_chunk`]: one compression per block, the last one as the root.
//! - A longer input goes through [`Hasher`]. It hashes whole subtrees of up to
//!   [`MAX_SUBTREE_CHUNKS`] chunks at a time, [`LANES`] chunks, or [`LANES`] parent nodes, side by
//!   side (§5.3): every word of the state is an array of `LANES` words, one per input, a layout
//!   the compiler turns into vector instructions without `unsafe` code. It keeps the chaining
//!   values of completed subtrees on a stack (§5.1.2) and merges two of them only once it knows
//!   more input follows, so that the root is compressed last, with the `ROOT` flag.
//!
//! The tree (§2.1): the leaves are chunks of 1024 bytes, the last possibly shorter (one empty
//! chunk for an empty input), each hashed block by block; every parent node's left subtree holds
//! the largest power of two of chunks smaller than the parent's total. A subtree of `2^k` chunks
//! starting at a chunk index divisible by `2^k`, and followed by more input, is therefore always
//! a complete node of the tree, whatever the input's length: that is what lets [`Hasher`] hash
//! it before it has seen the rest.

/// Output size, in bytes, of every Tessera hash.
pub(crate) const OUT_LEN: usize = 32;
/// Block size in bytes (§2.1).
pub(crate) const BLOCK_LEN: usize = 64;
/// [`BLOCK_LEN`] as the compression function's `b` input.
const BLOCK_LEN_WORD: u32 = 64;
/// Chunk size in bytes (§2.1).
pub(crate) const CHUNK_LEN: usize = 1024;
/// Blocks per chunk.
const CHUNK_BLOCKS: usize = 16;

// Domain separation flags (§2.1, Table 3).
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
#[cfg(test)]
pub(crate) const KEYED_HASH: u32 = 1 << 4;
pub(crate) const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
pub(crate) const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

/// The initialization vector, the first 32 bits of the fractional parts of the square roots of
/// the first eight primes, as in SHA-256 (§2.2, Table 1).
pub(crate) const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// Inputs hashed side by side: chunks of one subtree, or parent nodes of one level.
const LANES: usize = 16;
/// The most chunks [`Hasher`] hashes as one subtree: a power of two and a multiple of [`LANES`].
const MAX_SUBTREE_CHUNKS: usize = 64;
/// Room for the chaining values of completed subtrees. Before a new one is pushed, the stack is
/// merged down to one entry per 1 bit of the number of chunks before it, so it never holds more
/// than 65 entries: 64 bits of chunk count, plus the one being pushed.
const STACK_CAPACITY: usize = 65;

// ----- The compression function (§2.2) -----------------------------------------------------------

/// The quarter-round `G_i` on four words of the state `v`, mixing in the message words `x`, `y`.
/// Indices are literals, so every access is checked at compile time.
macro_rules! g {
    ($v:ident, $a:literal, $b:literal, $c:literal, $d:literal, $x:expr, $y:expr) => {
        $v[$a] = $v[$a].wrapping_add($v[$b]).wrapping_add($x);
        $v[$d] = ($v[$d] ^ $v[$a]).rotate_right(16);
        $v[$c] = $v[$c].wrapping_add($v[$d]);
        $v[$b] = ($v[$b] ^ $v[$c]).rotate_right(12);
        $v[$a] = $v[$a].wrapping_add($v[$b]).wrapping_add($y);
        $v[$d] = ($v[$d] ^ $v[$a]).rotate_right(8);
        $v[$c] = $v[$c].wrapping_add($v[$d]);
        $v[$b] = ($v[$b] ^ $v[$c]).rotate_right(7);
    };
}

/// One round: `G` on the four columns of the state, then on its four diagonals.
#[inline(always)]
const fn round(mut v: [u32; 16], m: &[u32; 16]) -> [u32; 16] {
    g!(v, 0, 4, 8, 12, m[0], m[1]);
    g!(v, 1, 5, 9, 13, m[2], m[3]);
    g!(v, 2, 6, 10, 14, m[4], m[5]);
    g!(v, 3, 7, 11, 15, m[6], m[7]);
    g!(v, 0, 5, 10, 15, m[8], m[9]);
    g!(v, 1, 6, 11, 12, m[10], m[11]);
    g!(v, 2, 7, 8, 13, m[12], m[13]);
    g!(v, 3, 4, 9, 14, m[14], m[15]);
    v
}

/// The message permutation applied between rounds (§2.2, Table 2).
#[inline(always)]
const fn permute(m: &[u32; 16]) -> [u32; 16] {
    [
        m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14],
        m[15], m[8],
    ]
}

/// The state after the seven rounds, from the chaining value `cv`, the message `block`, the
/// 64-bit `counter`, the number of input bytes `block_len` and the `flags`.
#[inline(always)]
const fn rounds(
    cv: &[u32; 8],
    block: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let (counter_low, counter_high) = split_counter(counter);
    let v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter_low,
        counter_high,
        block_len,
        flags,
    ];
    let m = *block;
    let v = round(v, &m);
    let m = permute(&m);
    let v = round(v, &m);
    let m = permute(&m);
    let v = round(v, &m);
    let m = permute(&m);
    let v = round(v, &m);
    let m = permute(&m);
    let v = round(v, &m);
    let m = permute(&m);
    let v = round(v, &m);
    let m = permute(&m);
    round(v, &m)
}

/// The first eight output words of the compression function, `v[i] ⊕ v[i + 8]`: a node's
/// chaining value, or the first 32 bytes of the root's output.
#[inline(always)]
pub(crate) const fn compress(
    cv: &[u32; 8],
    block: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 8] {
    let v = rounds(cv, block, counter, block_len, flags);
    [
        v[0] ^ v[8],
        v[1] ^ v[9],
        v[2] ^ v[10],
        v[3] ^ v[11],
        v[4] ^ v[12],
        v[5] ^ v[13],
        v[6] ^ v[14],
        v[7] ^ v[15],
    ]
}

/// All sixteen output words, for the extended output of the root (§2.6).
#[cfg(test)]
const fn compress_extended(
    cv: &[u32; 8],
    block: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let v = rounds(cv, block, counter, block_len, flags);
    [
        v[0] ^ v[8],
        v[1] ^ v[9],
        v[2] ^ v[10],
        v[3] ^ v[11],
        v[4] ^ v[12],
        v[5] ^ v[13],
        v[6] ^ v[14],
        v[7] ^ v[15],
        v[8] ^ cv[0],
        v[9] ^ cv[1],
        v[10] ^ cv[2],
        v[11] ^ cv[3],
        v[12] ^ cv[4],
        v[13] ^ cv[5],
        v[14] ^ cv[6],
        v[15] ^ cv[7],
    ]
}

/// The low and high 32-bit words of a 64-bit counter.
#[inline(always)]
const fn split_counter(counter: u64) -> (u32, u32) {
    let [a, b, c, d, e, f, g, h] = counter.to_le_bytes();
    (
        u32::from_le_bytes([a, b, c, d]),
        u32::from_le_bytes([e, f, g, h]),
    )
}

// ----- Bytes and words ---------------------------------------------------------------------------

/// The sixteen little-endian words of a block.
#[inline(always)]
pub(crate) const fn block_words(b: &[u8; BLOCK_LEN]) -> [u32; 16] {
    macro_rules! word {
        ($i:literal) => {
            u32::from_le_bytes([b[4 * $i], b[4 * $i + 1], b[4 * $i + 2], b[4 * $i + 3]])
        };
    }
    [
        word!(0),
        word!(1),
        word!(2),
        word!(3),
        word!(4),
        word!(5),
        word!(6),
        word!(7),
        word!(8),
        word!(9),
        word!(10),
        word!(11),
        word!(12),
        word!(13),
        word!(14),
        word!(15),
    ]
}

/// The eight little-endian words of a 32-byte value: a key, or a chaining value.
#[inline(always)]
pub(crate) const fn digest_words(b: &[u8; OUT_LEN]) -> [u32; 8] {
    macro_rules! word {
        ($i:literal) => {
            u32::from_le_bytes([b[4 * $i], b[4 * $i + 1], b[4 * $i + 2], b[4 * $i + 3]])
        };
    }
    [
        word!(0),
        word!(1),
        word!(2),
        word!(3),
        word!(4),
        word!(5),
        word!(6),
        word!(7),
    ]
}

/// The block that holds `left ‖ right`: a parent node's input, or a Merkle node's.
#[inline(always)]
pub(crate) const fn pair_block(left: &[u32; 8], right: &[u32; 8]) -> [u32; 16] {
    [
        left[0], left[1], left[2], left[3], left[4], left[5], left[6], left[7], right[0], right[1],
        right[2], right[3], right[4], right[5], right[6], right[7],
    ]
}

/// The 32 little-endian bytes of eight words.
#[inline(always)]
pub(crate) const fn words_bytes(w: &[u32; 8]) -> [u8; OUT_LEN] {
    let [a0, a1, a2, a3] = w[0].to_le_bytes();
    let [b0, b1, b2, b3] = w[1].to_le_bytes();
    let [c0, c1, c2, c3] = w[2].to_le_bytes();
    let [d0, d1, d2, d3] = w[3].to_le_bytes();
    let [e0, e1, e2, e3] = w[4].to_le_bytes();
    let [f0, f1, f2, f3] = w[5].to_le_bytes();
    let [g0, g1, g2, g3] = w[6].to_le_bytes();
    let [h0, h1, h2, h3] = w[7].to_le_bytes();
    [
        a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3, e0, e1, e2, e3, f0, f1, f2,
        f3, g0, g1, g2, g3, h0, h1, h2, h3,
    ]
}

/// A last block of at most [`BLOCK_LEN`] bytes, zero-padded, as words, with its length.
#[inline(always)]
fn last_block(bytes: &[u8]) -> ([u32; 16], u32) {
    let mut padded = [0u8; BLOCK_LEN];
    let mut len: u32 = 0;
    for (slot, byte) in padded.iter_mut().zip(bytes) {
        *slot = *byte;
        len = len.wrapping_add(1);
    }
    (block_words(&padded), len)
}

// ----- Context keys (§2.3) -----------------------------------------------------------------------

/// The key that `derive_key` derives from `context` (§2.3): the 32-byte hash of the context
/// string, in the `DERIVE_KEY_CONTEXT` mode, read as eight words.
///
/// It runs at compile time for the Tessera domains, whose contexts are a single block, so it
/// handles contexts of at most [`BLOCK_LEN`] bytes only; a longer one would be truncated. The
/// tests check every domain's key against a derivation through [`Hasher`], which takes any
/// length.
pub(crate) const fn context_key(context: &str) -> [u32; 8] {
    let (block, len) = short_block(context.as_bytes());
    compress(
        &IV,
        &block,
        0,
        len,
        CHUNK_START | CHUNK_END | ROOT | DERIVE_KEY_CONTEXT,
    )
}

/// The zero-padded words of an input of at most one block, and its length, in a form the
/// compiler can evaluate: each word is shifted in at the end of the block, and the words are
/// then shifted into place.
const fn short_block(mut bytes: &[u8]) -> ([u32; 16], u32) {
    let mut words = [0u32; 16];
    let mut shifted: u32 = 0;
    let mut len: u32 = 0;
    loop {
        let word = match bytes {
            [a, b, c, d, rest @ ..] => {
                bytes = rest;
                len = len.saturating_add(4);
                u32::from_le_bytes([*a, *b, *c, *d])
            }
            [a, b, c] => {
                bytes = &[];
                len = len.saturating_add(3);
                u32::from_le_bytes([*a, *b, *c, 0])
            }
            [a, b] => {
                bytes = &[];
                len = len.saturating_add(2);
                u32::from_le_bytes([*a, *b, 0, 0])
            }
            [a] => {
                bytes = &[];
                len = len.saturating_add(1);
                u32::from_le_bytes([*a, 0, 0, 0])
            }
            [] => break,
        };
        words = shift_in(words, word);
        shifted = shifted.saturating_add(1);
    }
    while shifted < 16 {
        words = shift_in(words, 0);
        shifted = shifted.saturating_add(1);
    }
    (words, len)
}

/// `words` moved one place towards the front, with `word` last.
const fn shift_in(w: [u32; 16], word: u32) -> [u32; 16] {
    [
        w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14],
        w[15], word,
    ]
}

// ----- One chunk ---------------------------------------------------------------------------------

/// The 32-byte hash of an input of at most one chunk, which is then the root: its blocks are
/// compressed in order, the first with `CHUNK_START`, the last with `CHUNK_END` and `ROOT`
/// (§2.4, §2.6). An empty input is one empty block.
#[inline]
pub(crate) fn single_chunk(key: &[u32; 8], flags: u32, input: &[u8]) -> [u8; OUT_LEN] {
    let mut cv = *key;
    let mut start = CHUNK_START;
    let mut rest = input;
    while rest.len() > BLOCK_LEN {
        let Some((block, tail)) = rest.split_first_chunk::<BLOCK_LEN>() else {
            break;
        };
        cv = compress(&cv, &block_words(block), 0, BLOCK_LEN_WORD, flags | start);
        start = 0;
        rest = tail;
    }
    let (block, len) = last_block(rest);
    words_bytes(&compress(
        &cv,
        &block,
        0,
        len,
        flags | start | CHUNK_END | ROOT,
    ))
}

/// The 32-byte hash of a single block of exactly 64 bytes given as words, the root of a
/// one-block input: a Merkle node, or anything else of that size.
#[inline]
pub(crate) fn single_block(key: &[u32; 8], flags: u32, block: &[u32; 16]) -> [u8; OUT_LEN] {
    words_bytes(&compress(
        key,
        block,
        0,
        BLOCK_LEN_WORD,
        flags | CHUNK_START | CHUNK_END | ROOT,
    ))
}

/// A compression whose output is not yet used: as a chaining value, or as root output.
#[derive(Clone, Copy)]
struct Output {
    cv: [u32; 8],
    block: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    /// The node's chaining value.
    fn chaining_value(&self) -> [u32; 8] {
        compress(
            &self.cv,
            &self.block,
            self.counter,
            self.block_len,
            self.flags,
        )
    }

    /// The node as the root: its first 32 output bytes, with `ROOT` and counter 0 (§2.6).
    fn root_hash(&self) -> [u8; OUT_LEN] {
        words_bytes(&compress(
            &self.cv,
            &self.block,
            0,
            self.block_len,
            self.flags | ROOT,
        ))
    }

    /// The node as the root, extended to `out.len()` bytes: output block `t` is the
    /// compression with `ROOT` and counter `t` (§2.6).
    #[cfg(test)]
    fn root_extended(&self, out: &mut [u8]) {
        for (counter, piece) in (0u64..).zip(out.chunks_mut(2 * OUT_LEN)) {
            let words = compress_extended(
                &self.cv,
                &self.block,
                counter,
                self.block_len,
                self.flags | ROOT,
            );
            let bytes = words.iter().flat_map(|word| word.to_le_bytes());
            for (slot, byte) in piece.iter_mut().zip(bytes) {
                *slot = byte;
            }
        }
    }
}

/// A parent node over the chaining values of its two children (§2.5).
fn parent_output(left: &[u32; 8], right: &[u32; 8], key: &[u32; 8], flags: u32) -> Output {
    Output {
        cv: *key,
        block: pair_block(left, right),
        counter: 0,
        block_len: BLOCK_LEN_WORD,
        flags: flags | PARENT,
    }
}

/// A non-root parent's chaining value.
fn parent_cv(left: &[u32; 8], right: &[u32; 8], key: &[u32; 8], flags: u32) -> [u32; 8] {
    compress(
        key,
        &pair_block(left, right),
        0,
        BLOCK_LEN_WORD,
        flags | PARENT,
    )
}

/// The chaining value of a whole chunk of exactly [`CHUNK_LEN`] bytes that is not the root.
fn chunk_cv(chunk: &[u8], counter: u64, key: &[u32; 8], flags: u32) -> [u32; 8] {
    let mut cv = *key;
    let mut blocks = chunk.chunks_exact(BLOCK_LEN);
    let mut position: usize = 0;
    for block in &mut blocks {
        let Ok(block) = <&[u8; BLOCK_LEN]>::try_from(block) else {
            break;
        };
        let start = if position == 0 { CHUNK_START } else { 0 };
        let end = if position == CHUNK_BLOCKS.wrapping_sub(1) {
            CHUNK_END
        } else {
            0
        };
        cv = compress(
            &cv,
            &block_words(block),
            counter,
            BLOCK_LEN_WORD,
            flags | start | end,
        );
        position = position.wrapping_add(1);
    }
    cv
}

/// The chunk being filled: its chaining value so far, and its last block, held back until the
/// chunk is known to continue or to end.
#[derive(Clone)]
struct ChunkState {
    cv: [u32; 8],
    /// The chunk's index in the input, its counter `t` (§2.4).
    counter: u64,
    /// The bytes of the block not yet compressed, zero-padded.
    block: [u8; BLOCK_LEN],
    /// How many bytes of `block` are input, 0 to [`BLOCK_LEN`].
    block_len: usize,
    /// Blocks compressed so far, 0 to 15.
    blocks_compressed: usize,
    flags: u32,
}

impl ChunkState {
    fn new(key: &[u32; 8], counter: u64, flags: u32) -> Self {
        Self {
            cv: *key,
            counter,
            block: [0; BLOCK_LEN],
            block_len: 0,
            blocks_compressed: 0,
            flags,
        }
    }

    /// Input bytes held, compressed or not: at most [`CHUNK_LEN`].
    fn len(&self) -> usize {
        self.blocks_compressed
            .saturating_mul(BLOCK_LEN)
            .saturating_add(self.block_len)
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    /// Compresses the held block, which more input follows, so it is not the chunk's last.
    fn compress_held_block(&mut self) {
        self.cv = compress(
            &self.cv,
            &block_words(&self.block),
            self.counter,
            BLOCK_LEN_WORD,
            self.flags | self.start_flag(),
        );
        self.blocks_compressed = self.blocks_compressed.saturating_add(1);
        self.block = [0; BLOCK_LEN];
        self.block_len = 0;
    }

    /// Takes as much of `input` as the chunk has room for and returns the rest. A block is
    /// compressed only when more input follows it, so the last block always waits for
    /// [`output`](Self::output).
    #[must_use = "the bytes the chunk had no room for"]
    fn update<'a>(&mut self, mut input: &'a [u8]) -> &'a [u8] {
        while !input.is_empty() && self.len() < CHUNK_LEN {
            if self.block_len == BLOCK_LEN {
                self.compress_held_block();
            }
            // Whole blocks straight from the input while more input follows them in this chunk.
            if self.block_len == 0 {
                while input.len() > BLOCK_LEN && self.len() < CHUNK_LEN.saturating_sub(BLOCK_LEN) {
                    let Some((block, tail)) = input.split_first_chunk::<BLOCK_LEN>() else {
                        break;
                    };
                    self.cv = compress(
                        &self.cv,
                        &block_words(block),
                        self.counter,
                        BLOCK_LEN_WORD,
                        self.flags | self.start_flag(),
                    );
                    self.blocks_compressed = self.blocks_compressed.saturating_add(1);
                    input = tail;
                }
            }
            let free = self.block.iter_mut().skip(self.block_len);
            let mut taken: usize = 0;
            for (slot, byte) in free.zip(input) {
                *slot = *byte;
                taken = taken.wrapping_add(1);
            }
            self.block_len = self.block_len.saturating_add(taken);
            input = input.get(taken..).unwrap_or_default();
        }
        input
    }

    /// The chunk's last compression, not yet done (§2.4).
    fn output(&self) -> Output {
        let (block, block_len) = last_block(self.block.get(..self.block_len).unwrap_or_default());
        Output {
            cv: self.cv,
            block,
            counter: self.counter,
            block_len,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

// ----- Several inputs side by side (§5.3) --------------------------------------------------------
//
// Each word of the state holds one word for each of `LANES` inputs, and each round is a loop over
// the lanes whose body is the scalar round: a loop the compiler vectorizes, one vector lane per
// input. A whole compression in one loop body is too large for its cost model, so the state goes
// through memory between rounds.

/// One word for each of [`LANES`] inputs.
type Lane = [u32; LANES];

/// Lane `lane` of `word`; every caller iterates `lane` over `0..LANES`.
#[inline(always)]
fn lane_word(word: &Lane, lane: usize) -> u32 {
    word.get(lane).copied().unwrap_or(0)
}

/// Sets lane `lane` of `word`; every caller iterates `lane` over `0..LANES`.
#[inline(always)]
fn set_lane_word(word: &mut Lane, lane: usize, value: u32) {
    if let Some(slot) = word.get_mut(lane) {
        *slot = value;
    }
}

/// The sixteen words of lane `lane` of a state or a message.
#[inline(always)]
fn gather(words: &[Lane; 16], lane: usize) -> [u32; 16] {
    [
        lane_word(&words[0], lane),
        lane_word(&words[1], lane),
        lane_word(&words[2], lane),
        lane_word(&words[3], lane),
        lane_word(&words[4], lane),
        lane_word(&words[5], lane),
        lane_word(&words[6], lane),
        lane_word(&words[7], lane),
        lane_word(&words[8], lane),
        lane_word(&words[9], lane),
        lane_word(&words[10], lane),
        lane_word(&words[11], lane),
        lane_word(&words[12], lane),
        lane_word(&words[13], lane),
        lane_word(&words[14], lane),
        lane_word(&words[15], lane),
    ]
}

/// One round on every lane of the state `v`, with the message words taken in the order of the
/// round's schedule: the identity for the first round, then the message permutation applied once
/// more per round (§2.2). The test `lanes_compress_like_the_scalar_function` checks the lists.
macro_rules! round_lanes {
    ($v:ident, $m:ident, [$($s:literal),*]) => {
        for lane in 0..LANES {
            let message = [$(lane_word(&$m[$s], lane)),*];
            let next = round(gather(&$v, lane), &message);
            for (word, value) in $v.iter_mut().zip(next) {
                set_lane_word(word, lane, value);
            }
        }
    };
}

/// [`compress`] on [`LANES`] inputs at once: each lane has its own chaining value, message and
/// counter, and all share `block_len` and `flags`.
fn compress_lanes(
    cv: &[Lane; 8],
    m: &[Lane; 16],
    counter_low: &Lane,
    counter_high: &Lane,
    block_len: u32,
    flags: u32,
) -> [Lane; 8] {
    let mut v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        [IV[0]; LANES],
        [IV[1]; LANES],
        [IV[2]; LANES],
        [IV[3]; LANES],
        *counter_low,
        *counter_high,
        [block_len; LANES],
        [flags; LANES],
    ];
    round_lanes!(v, m, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    round_lanes!(v, m, [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]);
    round_lanes!(v, m, [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1]);
    round_lanes!(v, m, [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6]);
    round_lanes!(v, m, [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4]);
    round_lanes!(v, m, [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7]);
    round_lanes!(v, m, [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13]);
    let mut out = [[0u32; LANES]; 8];
    let (low, high) = v.split_at(8);
    for (word, (a, b)) in out.iter_mut().zip(low.iter().zip(high)) {
        for (slot, (x, y)) in word.iter_mut().zip(a.iter().zip(b)) {
            *slot = x ^ y;
        }
    }
    out
}

/// The chaining value of each lane, out of word-by-word form.
#[inline(always)]
fn untranspose(cv: &[Lane; 8]) -> [[u32; 8]; LANES] {
    let mut out = [[0u32; 8]; LANES];
    for (lane, words) in out.iter_mut().enumerate() {
        for (word, row) in words.iter_mut().zip(cv) {
            *word = lane_word(row, lane);
        }
    }
    out
}

/// The chaining values of [`LANES`] whole chunks laid end to end in `group`, none of them the
/// root; the first has counter `counter`, the next `counter + 1`, and so on.
fn chunk_cvs_lanes(group: &[u8], counter: u64, key: &[u32; 8], flags: u32) -> [[u32; 8]; LANES] {
    let mut chunks: [&[u8]; LANES] = [&[]; LANES];
    for (slot, chunk) in chunks.iter_mut().zip(group.chunks_exact(CHUNK_LEN)) {
        *slot = chunk;
    }
    let mut counter_low = [0u32; LANES];
    let mut counter_high = [0u32; LANES];
    for (lane, (low, high)) in (0u64..).zip(counter_low.iter_mut().zip(counter_high.iter_mut())) {
        (*low, *high) = split_counter(counter.wrapping_add(lane));
    }
    let mut cv = key.map(|word| [word; LANES]);
    let mut message = [[0u32; LANES]; 16];
    for position in 0..CHUNK_BLOCKS {
        for (lane, rest) in chunks.iter_mut().enumerate() {
            let words = match rest.split_first_chunk::<BLOCK_LEN>() {
                Some((block, tail)) => {
                    *rest = tail;
                    block_words(block)
                }
                // Each lane holds a whole chunk, sixteen whole blocks, so this is never reached.
                None => [0; 16],
            };
            for (row, word) in message.iter_mut().zip(words) {
                set_lane_word(row, lane, word);
            }
        }
        let start = if position == 0 { CHUNK_START } else { 0 };
        let end = if position == CHUNK_BLOCKS.wrapping_sub(1) {
            CHUNK_END
        } else {
            0
        };
        cv = compress_lanes(
            &cv,
            &message,
            &counter_low,
            &counter_high,
            BLOCK_LEN_WORD,
            flags | start | end,
        );
    }
    untranspose(&cv)
}

/// The chaining values of [`LANES`] parent nodes, none of them the root, from the chaining
/// values of their children: `children` holds `2 · LANES` of them, left and right in turn.
fn parent_cvs_lanes(children: &[[u32; 8]], key: &[u32; 8], flags: u32) -> [[u32; 8]; LANES] {
    let mut message = [[0u32; LANES]; 16];
    for (lane, pair) in children.chunks_exact(2).enumerate() {
        if let [left, right] = pair {
            for (row, word) in message.iter_mut().zip(pair_block(left, right)) {
                set_lane_word(row, lane, word);
            }
        }
    }
    let cv = key.map(|word| [word; LANES]);
    let zero = [0u32; LANES];
    untranspose(&compress_lanes(
        &cv,
        &message,
        &zero,
        &zero,
        BLOCK_LEN_WORD,
        flags | PARENT,
    ))
}

/// The chaining value of a complete subtree, not the root: `input` is `2^k` whole chunks, at
/// most [`MAX_SUBTREE_CHUNKS`], and the first has counter `counter`, a multiple of `2^k`.
fn subtree_cv(input: &[u8], counter: u64, key: &[u32; 8], flags: u32) -> [u32; 8] {
    let mut level = [[0u32; 8]; MAX_SUBTREE_CHUNKS];
    let mut above = [[0u32; 8]; MAX_SUBTREE_CHUNKS];

    // The chunks: LANES at a time, then one at a time.
    let mut len: usize = 0;
    let mut slots = level.iter_mut();
    let mut groups = input.chunks_exact(LANES.saturating_mul(CHUNK_LEN));
    let mut next_counter = counter;
    for group in &mut groups {
        for (cv, slot) in chunk_cvs_lanes(group, next_counter, key, flags)
            .into_iter()
            .zip(&mut slots)
        {
            *slot = cv;
            len = len.saturating_add(1);
            next_counter = next_counter.wrapping_add(1);
        }
    }
    for (chunk, slot) in groups.remainder().chunks_exact(CHUNK_LEN).zip(&mut slots) {
        *slot = chunk_cv(chunk, next_counter, key, flags);
        len = len.saturating_add(1);
        next_counter = next_counter.wrapping_add(1);
    }

    // The parents, level by level: LANES at a time, then one at a time.
    while len > 1 {
        let children = level.get(..len).unwrap_or_default();
        let mut parents: usize = 0;
        let mut slots = above.iter_mut();
        let mut groups = children.chunks_exact(LANES.saturating_mul(2));
        for group in &mut groups {
            for (cv, slot) in parent_cvs_lanes(group, key, flags)
                .into_iter()
                .zip(&mut slots)
            {
                *slot = cv;
                parents = parents.saturating_add(1);
            }
        }
        for (pair, slot) in groups.remainder().chunks_exact(2).zip(&mut slots) {
            if let [left, right] = pair {
                *slot = parent_cv(left, right, key, flags);
                parents = parents.saturating_add(1);
            }
        }
        core::mem::swap(&mut level, &mut above);
        len = parents;
    }
    level[0]
}

/// How many chunks of `available` bytes, which start at chunk `counter`, to hash as one subtree:
/// the largest power of two, at most [`MAX_SUBTREE_CHUNKS`], that divides `counter` (so that the
/// subtree is a node of the tree) and leaves at least one byte after it (so that it is not the
/// root, nor holds the last chunk).
fn subtree_chunks(counter: u64, available: usize) -> usize {
    let mut chunks = MAX_SUBTREE_CHUNKS;
    while chunks > 1 {
        let aligned = u64::try_from(chunks).is_ok_and(|size| counter & size.wrapping_sub(1) == 0);
        let leaves_more = chunks
            .checked_mul(CHUNK_LEN)
            .is_some_and(|bytes| bytes < available);
        if aligned && leaves_more {
            break;
        }
        chunks >>= 1;
    }
    chunks
}

// ----- The incremental hasher (§5.1) -------------------------------------------------------------

/// BLAKE3 over an input given in pieces, under a key and mode flags.
#[derive(Clone)]
pub(crate) struct Hasher {
    key: [u32; 8],
    flags: u32,
    /// The last chunk seen, which may be the last of the input.
    chunk: ChunkState,
    /// Chaining values of the completed subtrees before `chunk`, left to right; two at the top
    /// may still have to be merged.
    stack: [[u32; 8]; STACK_CAPACITY],
    stack_len: usize,
}

impl Hasher {
    /// A hasher keyed by `key`, with the mode flags `flags`: none for `hash`, `KEYED_HASH`,
    /// `DERIVE_KEY_CONTEXT` or `DERIVE_KEY_MATERIAL` (§2.3).
    pub(crate) fn new(key: [u32; 8], flags: u32) -> Self {
        Self {
            key,
            flags,
            chunk: ChunkState::new(&key, 0, flags),
            stack: [[0; 8]; STACK_CAPACITY],
            stack_len: 0,
        }
    }

    /// Adds `input` to the bytes hashed so far.
    ///
    /// One pass of the loop takes all of `input`; the loop only makes the hasher correct by
    /// construction should step 3 ever be left with more than a chunk. Each pass takes at least
    /// one byte, in step 3 if not before.
    pub(crate) fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // 1. The chunk being filled. Once it is full and more input follows, it is not the
            //    last chunk: its chaining value moves to the stack.
            if self.chunk.len() > 0 {
                input = self.chunk.update(input);
                if input.is_empty() {
                    return;
                }
                let counter = self.chunk.counter;
                let cv = self.chunk.output().chaining_value();
                self.push(cv, counter);
                self.chunk = ChunkState::new(&self.key, counter.wrapping_add(1), self.flags);
            }
            // 2. Whole subtrees that more input follows.
            while input.len() > CHUNK_LEN {
                let counter = self.chunk.counter;
                let chunks = subtree_chunks(counter, input.len());
                let Some((subtree, rest)) =
                    input.split_at_checked(chunks.saturating_mul(CHUNK_LEN))
                else {
                    break;
                };
                let cv = if chunks == 1 {
                    chunk_cv(subtree, counter, &self.key, self.flags)
                } else {
                    subtree_cv(subtree, counter, &self.key, self.flags)
                };
                self.push(cv, counter);
                let chunks = u64::try_from(chunks).unwrap_or(u64::MAX);
                self.chunk = ChunkState::new(&self.key, counter.wrapping_add(chunks), self.flags);
                input = rest;
            }
            // 3. The rest, at most one chunk, which may be the last: the empty chunk takes it.
            input = self.chunk.update(input);
        }
    }

    /// Pushes the chaining value of a subtree whose first chunk is `first_chunk`, after merging
    /// the subtrees before it down to one per 1 bit of `first_chunk`: those merges are of
    /// subtrees that more input follows, so none of them is the root.
    fn push(&mut self, cv: [u32; 8], first_chunk: u64) {
        merge_down(
            &mut self.stack,
            &mut self.stack_len,
            first_chunk,
            &self.key,
            self.flags,
        );
        if let Some(slot) = self.stack.get_mut(self.stack_len) {
            *slot = cv;
            self.stack_len = self.stack_len.saturating_add(1);
        }
    }

    /// The root node: the stack merged down to the subtrees before the last chunk, then merged
    /// right to left onto the last chunk, the last merge being the root (§5.1.2).
    fn root(&self) -> Output {
        let mut stack = self.stack;
        let mut len = self.stack_len;
        merge_down(
            &mut stack,
            &mut len,
            self.chunk.counter,
            &self.key,
            self.flags,
        );
        let mut output = self.chunk.output();
        for left in stack.iter().take(len).rev() {
            output = parent_output(left, &output.chaining_value(), &self.key, self.flags);
        }
        output
    }

    /// The 32-byte hash of everything added so far.
    pub(crate) fn finalize(&self) -> [u8; OUT_LEN] {
        self.root().root_hash()
    }

    /// The hash of everything added so far, extended to `out.len()` bytes (§2.6).
    #[cfg(test)]
    pub(crate) fn finalize_extended(&self, out: &mut [u8]) {
        self.root().root_extended(out);
    }
}

/// Merges the top two entries of a chaining-value stack until it holds one entry per 1 bit of
/// `chunks`, the number of chunks the stack covers.
fn merge_down(
    stack: &mut [[u32; 8]; STACK_CAPACITY],
    len: &mut usize,
    chunks: u64,
    key: &[u32; 8],
    flags: u32,
) {
    let keep = usize::try_from(chunks.count_ones()).unwrap_or(STACK_CAPACITY);
    while *len > keep && *len >= 2 {
        let right_at = len.wrapping_sub(1);
        let left_at = len.wrapping_sub(2);
        let (Some(left), Some(right)) = (stack.get(left_at), stack.get(right_at)) else {
            return;
        };
        let parent = parent_cv(left, right, key, flags);
        if let Some(slot) = stack.get_mut(left_at) {
            *slot = parent;
        }
        *len = right_at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tv_testkit::{hex, read_vectors};

    /// The context key of `derive_key` derived at run time through the hasher, for contexts of
    /// any length (§2.3).
    fn context_key_by_hasher(context: &[u8]) -> [u32; 8] {
        let mut hasher = Hasher::new(IV, DERIVE_KEY_CONTEXT);
        hasher.update(context);
        digest_words(&hasher.finalize())
    }

    /// The extended output of `input` hashed under `key` and `flags`, `len` bytes long.
    fn extended(key: [u32; 8], flags: u32, input: &[u8], len: usize) -> Vec<u8> {
        let mut hasher = Hasher::new(key, flags);
        hasher.update(input);
        let mut out = vec![0; len];
        hasher.finalize_extended(&mut out);
        assert_eq!(
            out[..OUT_LEN],
            hasher.finalize(),
            "the first 32 bytes of the extended output are the hash"
        );
        out
    }

    /// The official BLAKE3 test vectors (spec/vectors/blake3/README.md), in all three modes and
    /// at their full extended length: the input byte `i` is `i mod 251`.
    #[test]
    fn every_official_vector_in_every_mode_at_full_length() {
        let file = read_vectors("blake3/test_vectors.json");
        let key = file.get("key").and_then(|k| k.as_str()).unwrap();
        let key = digest_words(key.as_bytes().try_into().unwrap());
        let context = file.get("context_string").and_then(|c| c.as_str()).unwrap();
        let derived = context_key_by_hasher(context.as_bytes());
        // The official context is shorter than a block, so the compile-time path applies too.
        assert_eq!(context_key(context), derived);
        let cases = file.get("cases").and_then(|c| c.as_array()).unwrap();
        assert_eq!(cases.len(), 35, "every case of the official file");
        for case in cases {
            let len = case.get("input_len").and_then(|l| l.as_u64()).unwrap();
            let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            for (mode, key, flags) in [
                ("hash", IV, 0),
                ("keyed_hash", key, KEYED_HASH),
                ("derive_key", derived, DERIVE_KEY_MATERIAL),
            ] {
                let expected = hex(case.get(mode).and_then(|d| d.as_str()).unwrap());
                assert_eq!(
                    extended(key, flags, &input, expected.len()),
                    expected,
                    "{mode}, input_len {len}"
                );
                // One update per byte reaches every state boundary the hasher has.
                if len <= 2 * CHUNK_LEN as u64 + 1 {
                    let mut hasher = Hasher::new(key, flags);
                    for byte in &input {
                        hasher.update(core::slice::from_ref(byte));
                    }
                    assert_eq!(hasher.finalize(), expected[..OUT_LEN], "{mode}, bytewise");
                }
                if input.len() <= CHUNK_LEN {
                    assert_eq!(
                        single_chunk(&key, flags, &input),
                        expected[..OUT_LEN],
                        "{mode}, one chunk, input_len {len}"
                    );
                }
            }
        }
    }

    #[test]
    fn lanes_compress_like_the_scalar_function() {
        let mut rng = tv_testkit::SplitMix64::new(0x1a7e5);
        let mut word = || rng.next_u64() as u32;
        for flags in [
            0,
            CHUNK_START,
            CHUNK_END | ROOT,
            PARENT | DERIVE_KEY_MATERIAL,
        ] {
            let cv: [Lane; 8] = core::array::from_fn(|_| core::array::from_fn(|_| word()));
            let m: [Lane; 16] = core::array::from_fn(|_| core::array::from_fn(|_| word()));
            let low: Lane = core::array::from_fn(|_| word());
            let high: Lane = core::array::from_fn(|_| word());
            let block_len = word() % 65;
            let lanes = untranspose(&compress_lanes(&cv, &m, &low, &high, block_len, flags));
            for (lane, out) in lanes.iter().enumerate() {
                let counter = u64::from(low[lane]) | u64::from(high[lane]) << 32;
                let scalar = compress(
                    &core::array::from_fn(|w| cv[w][lane]),
                    &core::array::from_fn(|w| m[w][lane]),
                    counter,
                    block_len,
                    flags,
                );
                assert_eq!(*out, scalar, "lane {lane}, flags {flags}");
            }
        }
    }

    #[test]
    fn short_block_lays_out_words_like_the_general_path() {
        for len in 0..=BLOCK_LEN {
            let input: Vec<u8> = (0..len).map(|i| (i * 7 + 1) as u8).collect();
            let (words, block_len) = short_block(&input);
            assert_eq!((words, block_len), last_block(&input), "length {len}");
        }
    }

    #[test]
    fn subtree_sizes_are_aligned_powers_of_two_that_leave_input() {
        for counter in 0..300u64 {
            for available in [
                1,
                1024,
                1025,
                2048,
                2049,
                9 * 1024,
                64 * 1024,
                64 * 1024 + 1,
            ] {
                let chunks = subtree_chunks(counter, available);
                assert!(chunks.is_power_of_two() && chunks <= MAX_SUBTREE_CHUNKS);
                if chunks > 1 {
                    assert_eq!(counter % chunks as u64, 0, "aligned");
                    assert!(chunks * CHUNK_LEN < available, "leaves input after it");
                }
            }
        }
        assert_eq!(subtree_chunks(0, 64 * 1024 + 1), 64);
        assert_eq!(subtree_chunks(0, 64 * 1024), 32);
        assert_eq!(subtree_chunks(8, 1 << 20), 8);
        assert_eq!(subtree_chunks(3, 1 << 20), 1);
    }
}
