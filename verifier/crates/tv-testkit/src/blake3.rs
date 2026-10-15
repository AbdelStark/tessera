// SPDX-License-Identifier: Apache-2.0 OR MIT

//! BLAKE3 in key-derivation mode, for tests only.
//!
//! Written from the BLAKE3 specification (O'Connor, Aumasson, Neves and Wilcox-O'Hearn, 2020)
//! and the reference algorithm it describes, so that verifier tests can recompute the digests
//! the conformance vectors carry, such as the envelope digest of RFC-0003, without a third-party
//! crate. It follows the specification's structure rather than aiming for speed: one compression
//! function; a chunk state that holds back its last block until it knows whether more input
//! follows; and a stack of subtree chaining values, merged as chunks complete, so that the last
//! chunk or parent node is the root.
//!
//! The verifier's production hashing uses the allowlisted `blake3` crate (RFC-0019); this module
//! is never part of `tessera-verify`.

/// The default output size in bytes.
const OUT_LEN: usize = 32;
/// The bytes of one compression block.
const BLOCK_LEN: usize = 64;
/// The bytes of one chunk: 16 blocks, the leaves of the tree.
const CHUNK_LEN: usize = 1024;

// Domain flags (specification, Table 3).
const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

/// The initialization vector, shared with SHA-256: the first 32 bits of the fractional parts of
/// the square roots of the first eight primes.
const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// The permutation of the message words between rounds (specification, Table 2).
const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// The quarter-round `G`, mixing two message words into four state words.
#[allow(clippy::too_many_arguments)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

/// One round: `G` on the four columns, then on the four diagonals, of the 4×4 state.
fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0; 16];
    for (word, &source) in permuted.iter_mut().zip(MSG_PERMUTATION.iter()) {
        *word = m[source];
    }
    *m = permuted;
}

/// The compression function: seven rounds over the chaining value, the first four IV words, the
/// counter, the block length and the flags, then the feed-forward. Its first eight words are the
/// next chaining value; all sixteen are extended output.
fn compress(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let cv = chaining_value;
    let mut state = [
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
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    let mut block = *block_words;
    for r in 0..7 {
        round(&mut state, &block);
        if r < 6 {
            permute(&mut block);
        }
    }
    for (i, &word) in cv.iter().enumerate() {
        state[i] ^= state[i + 8];
        state[i + 8] ^= word;
    }
    state
}

fn first_8_words(words: [u32; 16]) -> [u32; 8] {
    let mut out = [0; 8];
    out.copy_from_slice(&words[..8]);
    out
}

/// Little-endian 32-bit words of `bytes`, whose length is `4 · N`.
fn words<const N: usize>(bytes: &[u8]) -> [u32; N] {
    let mut out = [0; N];
    for (word, four) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *word = u32::from_le_bytes([four[0], four[1], four[2], four[3]]);
    }
    out
}

/// A node not yet compressed: the last block of a chunk, or a parent. Whether it is the root is
/// decided only when it is compressed.
struct Output {
    input_chaining_value: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        first_8_words(compress(
            &self.input_chaining_value,
            &self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        ))
    }

    /// The root's output: block `t` of the extended output compresses the root node with counter
    /// `t` and the `ROOT` flag.
    fn root_output_bytes(&self, out: &mut [u8]) {
        for (block_counter, out_block) in out.chunks_mut(2 * OUT_LEN).enumerate() {
            let words = compress(
                &self.input_chaining_value,
                &self.block_words,
                block_counter as u64,
                self.block_len,
                self.flags | ROOT,
            );
            for (word, out_word) in words.iter().zip(out_block.chunks_mut(4)) {
                out_word.copy_from_slice(&word.to_le_bytes()[..out_word.len()]);
            }
        }
    }
}

/// The chunk being absorbed.
struct ChunkState {
    chaining_value: [u32; 8],
    chunk_counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: usize,
    blocks_compressed: usize,
    flags: u32,
}

impl ChunkState {
    fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32) -> Self {
        Self {
            chaining_value: key_words,
            chunk_counter,
            block: [0; BLOCK_LEN],
            block_len: 0,
            blocks_compressed: 0,
            flags,
        }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed + self.block_len
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // A full block is compressed only when more input arrives: the chunk's last block
            // is compressed by `output`, with `CHUNK_END` and possibly `ROOT`.
            if self.block_len == BLOCK_LEN {
                self.chaining_value = first_8_words(compress(
                    &self.chaining_value,
                    &words(&self.block),
                    self.chunk_counter,
                    BLOCK_LEN as u32,
                    self.flags | self.start_flag(),
                ));
                self.blocks_compressed += 1;
                self.block = [0; BLOCK_LEN];
                self.block_len = 0;
            }
            let take = (BLOCK_LEN - self.block_len).min(input.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&input[..take]);
            self.block_len += take;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        Output {
            input_chaining_value: self.chaining_value,
            block_words: words(&self.block),
            counter: self.chunk_counter,
            block_len: self.block_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

fn parent_output(left: [u32; 8], right: [u32; 8], key_words: [u32; 8], flags: u32) -> Output {
    let mut block_words = [0; 16];
    block_words[..8].copy_from_slice(&left);
    block_words[8..].copy_from_slice(&right);
    Output {
        input_chaining_value: key_words,
        block_words,
        counter: 0,
        block_len: BLOCK_LEN as u32,
        flags: PARENT | flags,
    }
}

/// An incremental hasher keyed by `key_words`, with the mode's `flags`.
struct Hasher {
    chunk_state: ChunkState,
    key_words: [u32; 8],
    /// Chaining values of completed subtrees, largest first.
    cv_stack: Vec<[u32; 8]>,
    flags: u32,
}

impl Hasher {
    fn new(key_words: [u32; 8], flags: u32) -> Self {
        Self {
            chunk_state: ChunkState::new(key_words, 0, flags),
            key_words,
            cv_stack: Vec::new(),
            flags,
        }
    }

    /// Pushes a completed chunk. Each trailing zero bit of the chunk count completes a subtree,
    /// whose left half is on top of the stack.
    fn add_chunk_chaining_value(&mut self, mut new_cv: [u32; 8], mut total_chunks: u64) {
        while total_chunks & 1 == 0 {
            let left = self.cv_stack.pop().expect("a completed left subtree");
            new_cv = parent_output(left, new_cv, self.key_words, self.flags).chaining_value();
            total_chunks >>= 1;
        }
        self.cv_stack.push(new_cv);
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // A full chunk is finished only when more input arrives, so that the last chunk can
            // be the root.
            if self.chunk_state.len() == CHUNK_LEN {
                let chunk_cv = self.chunk_state.output().chaining_value();
                let total_chunks = self.chunk_state.chunk_counter + 1;
                self.add_chunk_chaining_value(chunk_cv, total_chunks);
                self.chunk_state = ChunkState::new(self.key_words, total_chunks, self.flags);
            }
            let take = (CHUNK_LEN - self.chunk_state.len()).min(input.len());
            self.chunk_state.update(&input[..take]);
            input = &input[take..];
        }
    }

    /// Merges the stack, right to left, onto the last chunk, and writes the root's output.
    fn finalize(&self, out: &mut [u8]) {
        let mut output = self.chunk_state.output();
        for &left in self.cv_stack.iter().rev() {
            output = parent_output(left, output.chaining_value(), self.key_words, self.flags);
        }
        output.root_output_bytes(out);
    }
}

/// Fills `out` with the extended output of `derive_key(context, input)`: the context string is
/// hashed with the IV as key in the `DERIVE_KEY_CONTEXT` mode, and its 32-byte output keys the
/// hash of `input` in the `DERIVE_KEY_MATERIAL` mode.
fn derive_key_into(context: &str, input: &[u8], out: &mut [u8]) {
    let mut context_hasher = Hasher::new(IV, DERIVE_KEY_CONTEXT);
    context_hasher.update(context.as_bytes());
    let mut context_key = [0; OUT_LEN];
    context_hasher.finalize(&mut context_key);
    let mut hasher = Hasher::new(words(&context_key), DERIVE_KEY_MATERIAL);
    hasher.update(input);
    hasher.finalize(out);
}

/// BLAKE3 `derive_key(context, input)` with its default 32-byte output, for inputs of any length.
pub fn derive_key(context: &str, input: &[u8]) -> [u8; 32] {
    let mut out = [0; OUT_LEN];
    derive_key_into(context, input, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hex, read_vectors};

    /// The official vectors (spec/vectors/blake3/README.md): input byte `i` is `i mod 251`, and
    /// each `derive_key` field is an extended output under `context_string`.
    #[test]
    fn derive_key_matches_every_official_vector() {
        let file = read_vectors("blake3/test_vectors.json");
        let context = file
            .get("context_string")
            .and_then(|c| c.as_str())
            .expect("the context string");
        let cases = file.get("cases").and_then(|c| c.as_array()).expect("cases");
        assert_eq!(cases.len(), 35, "every case of the official file");
        let mut multi_chunk = 0;
        for case in cases {
            let len = case
                .get("input_len")
                .and_then(|l| l.as_u64())
                .expect("input_len");
            let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let expected = hex(case
                .get("derive_key")
                .and_then(|d| d.as_str())
                .expect("derive_key"));
            assert_eq!(
                derive_key(context, &input).as_slice(),
                &expected[..32],
                "derive_key, input_len {len}"
            );
            let mut extended = vec![0; expected.len()];
            derive_key_into(context, &input, &mut extended);
            assert_eq!(extended, expected, "extended output, input_len {len}");
            if input.len() > CHUNK_LEN {
                multi_chunk += 1;
            }
        }
        assert_eq!(multi_chunk, 18, "the multi-chunk cases, up to 100 chunks");
    }
}
