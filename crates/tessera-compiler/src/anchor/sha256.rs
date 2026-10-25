// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The SHA-256 circuit of the transcript anchor (RFC-0015 'Circuits' step 3, RFC-0066 'The
//! SHA-256 circuit'): FIPS 180-4 §6.2 over `M` messages of one block count, on committed bits.
//!
//! - **A word** of the `M` messages is a bit tensor `[32, M]`, row `i` its bit of weight `2^i`.
//!   `ROTR^r` gathers row `(i + r) mod 32` into row `i`; `SHR^r` is the same gather times a
//!   public mask that is 0 on its top `r` rows.
//! - **The functions.** `Σ0`, `Σ1`, `σ0` and `σ1` are `Xor3` of three rotations or shifts, and
//!   `Ch` and `Maj` are their `BitFn`s.
//! - **A sum modulo `2^{32}`** adds its terms' bits row by row, takes the integer value of the
//!   rows, adds its public constant, and decomposes the result with `BitDecompose { 32, c }`:
//!   the low bits are the sum's word, transposed back to `[32, M]`, and `c` bits of carry.
//! - **A value** is a tree of five levels, each adding the even rows to `2^{2^ℓ}` times the odd
//!   ones. Range analysis bounds every level exactly: rows in `[0, k]` give a value in
//!   `[0, k·(2^{32} − 1)]`, where `Sum` of the rows times the weights `2^i` would get
//!   `32·k·2^{31}` and fail `BitDecompose`'s width.
//!
//! **The message.** Its bytes, padding included, are public except those the layout commits.
//! A committed byte enters through `BitDecompose { 8, 0 }`, which is also its range check
//! (RFC-0066 Lemma 1), and its bits are gathered into the word positions of its offset and
//! masked to zero elsewhere. The public bits are zero at committed offsets, so the message's
//! bits are the `Xor` of the two, whose supports are disjoint.
//!
//! **Cost.** 6,376 committed bits per block ([`SHA256_BITS_PER_BLOCK`]) and 8 per committed
//! byte, in 6,032 nodes per block: the graph's size grows with the block count, not with `M`.

use tessera_ir::{
    BoolFn, Broadcast, CommitmentClass, GraphBuilder, IntRange, IntType, IrError, NodeKind, Op,
    TensorDecl, TensorId,
};

/// The initial hash value `H^{(0)}` (FIPS 180-4 §5.3.3).
pub const SHA256_IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// The round constants `K_t` (FIPS 180-4 §4.2.2).
pub const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// The carry widths `c` of RFC-0066's sums: the least that bounds each.
const SCHEDULE_CARRY: u8 = 2;
const ROUND_CARRY: u8 = 3;
const CHAINING_CARRY: u8 = 1;

/// The committed bits of one block: 48 schedule words of `32 + 2`, two round words of `32 + 3`
/// in each of 64 rounds, and 8 chaining words of `32 + 1` (RFC-0066 'Committed bits per block').
pub const SHA256_BITS_PER_BLOCK: u64 = 48 * (32 + SCHEDULE_CARRY as u64)
    + 64 * 2 * (32 + ROUND_CARRY as u64)
    + 8 * (32 + CHAINING_CARRY as u64);

/// The rows of the public input `masks`: `SHR^3`'s and `SHR^{10}`'s, each 1 on rows
/// `i < 32 − r` and 0 above (FIPS 180-4 §4.1.2's `σ0` and `σ1`).
pub const SHA256_MASKS: [[i64; 32]; 2] = [shr_mask(3), shr_mask(10)];

const fn shr_mask(r: usize) -> [i64; 32] {
    let mut mask = [0i64; 32];
    let mut i = 0;
    while i < 32 - r {
        mask[i] = 1;
        i += 1;
    }
    mask
}

/// The blocks of a message of `len` bytes once padded (FIPS 180-4 §5.1.1): `⌈(len + 9)/64⌉`.
#[must_use]
pub const fn sha256_blocks(len: usize) -> usize {
    (len + 9).div_ceil(64)
}

/// The message padded to whole blocks (FIPS 180-4 §5.1.1): a 1 bit, zeros, and its length in
/// bits as a big-endian 64-bit integer.
///
/// It copies `message` into a plain buffer, so it takes public bytes only: for a message with
/// committed bytes, pass it with those bytes zeroed, which is all [`Sha256Layout::public_bits`]
/// reads (05 §3: a key or salt lives in `Secret<T>`).
#[must_use]
pub fn sha256_pad(message: &[u8]) -> Vec<u8> {
    let mut padded = Vec::with_capacity(64 * sha256_blocks(message.len()));
    padded.extend_from_slice(message);
    padded.push(0x80);
    padded.resize(64 * sha256_blocks(message.len()) - 8, 0);
    let bits = (message.len() as u64).wrapping_mul(8);
    padded.extend_from_slice(&bits.to_be_bytes());
    padded
}

/// The byte and the bit within it of row `row` of a block-major bit tensor: bit `i` of word `w`
/// is bit `i mod 8` of byte `4w + 3 − ⌊i/8⌋`, since FIPS 180-4's words are big-endian.
const fn byte_of(row: usize) -> (usize, usize) {
    let (w, i) = (row / 32, row % 32);
    (4 * w + 3 - i / 8, i % 8)
}

/// Bytes as bits in the circuit's layout, `[8·len, M]`: row `32w + i` holds bit `i` of word `w`,
/// column `k` the bytes of `rows[k]`, all of one length, a multiple of 4.
///
/// # Panics
///
/// If the rows differ in length or their length is not a multiple of 4.
#[must_use]
pub fn sha256_bits(rows: &[&[u8]]) -> Vec<i64> {
    let len = rows.first().map_or(0, |r| r.len());
    assert!(
        len.is_multiple_of(4) && rows.iter().all(|r| r.len() == len),
        "rows of one length, in whole words"
    );
    let m = rows.len();
    let mut bits = vec![0i64; 8 * len * m];
    for row in 0..8 * len {
        let (byte, bit) = byte_of(row);
        for (k, bytes) in rows.iter().enumerate() {
            bits[row * m + k] = i64::from((bytes[byte] >> bit) & 1);
        }
    }
    bits
}

/// Which bytes of the padded message the prover commits, and the block count: every message of
/// a circuit shares them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sha256Layout {
    /// The padded message's blocks `N`.
    pub blocks: u32,
    /// The messages `M`.
    pub messages: u32,
    /// The committed bytes' offsets in the padded message, ascending and below `64·N`.
    pub committed: Vec<u32>,
}

impl Sha256Layout {
    /// The public value of `place`, `[512·N]`: 1 on the rows of committed bytes' bits.
    #[must_use]
    pub fn place(&self) -> Vec<i64> {
        (0..512 * self.blocks as usize)
            .map(|row| i64::from(self.slot(byte_of(row).0).is_some()))
            .collect()
    }

    /// The public value of `message`: the padded messages' bits, zero at committed bytes.
    ///
    /// It reads only the public bytes of `padded`. A caller holding committed bytes that are
    /// secret, such as `k_s` and `salt_k`, passes the messages with those bytes zeroed, never the
    /// secrets: the `committed` input's values come from the prover's secret-class buffers, which
    /// this crate does not read (05 §3).
    ///
    /// # Panics
    ///
    /// If there are not `M` messages of `64·N` bytes.
    #[must_use]
    pub fn public_bits(&self, padded: &[&[u8]]) -> Vec<i64> {
        assert_eq!(padded.len(), self.messages as usize, "M messages");
        let mut bits = sha256_bits(padded);
        assert_eq!(
            bits.len(),
            512 * self.blocks as usize * padded.len(),
            "N blocks"
        );
        for (row, place) in self.place().into_iter().enumerate() {
            if place == 1 {
                bits[row * padded.len()..][..padded.len()].fill(0);
            }
        }
        bits
    }

    /// The index among committed bytes of the byte at `offset`, if committed.
    fn slot(&self, offset: usize) -> Option<usize> {
        self.committed.iter().position(|&p| p as usize == offset)
    }
}

/// The tensors of a SHA-256 circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sha256 {
    /// The chaining value after each block, as its eight words' bits, each `[32, M]`: the last
    /// is the digest.
    pub chaining: Vec<[TensorId; 8]>,
}

const BIT: IntType = IntType {
    bits: 1,
    signed: false,
};

fn bits(shape: &[u32], kind: NodeKind) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: BIT,
        range: IntRange { lo: 0, hi: 1 },
        kind,
    }
}

/// A virtual tensor, with a placeholder range that [`crate::settle`] replaces.
fn virtual_tensor(shape: &[u32]) -> TensorDecl {
    TensorDecl {
        shape: shape.to_vec(),
        ty: IntType {
            bits: 64,
            signed: true,
        },
        range: IntRange { lo: 0, hi: 0 },
        kind: NodeKind::Virtual,
    }
}

const ADVICE: NodeKind = NodeKind::Committed {
    commitment: CommitmentClass::Advice,
};

/// The declarations of the circuit's inputs, in order:
/// - `message`: public bits `[512·N, M]`, [`Sha256Layout::public_bits`];
/// - `start`: public bits `[256, M]`, the chaining value before the first block, eight words of
///   32 rows: [`SHA256_IV`] through [`sha256_bits`];
/// - `masks`: public `[2, 32]`, [`SHA256_MASKS`];
/// - when bytes are committed, `committed`: advice bytes `[K, M]`, declared `u8` in `[0, 255]`,
///   row `k` the byte at the layout's `k`-th offset, and `place`: public `[512·N]`,
///   [`Sha256Layout::place`].
#[must_use]
pub fn sha256_inputs(layout: &Sha256Layout) -> Vec<TensorDecl> {
    let (n, m) = (layout.blocks, layout.messages);
    let mut inputs = vec![
        bits(&[512 * n, m], NodeKind::Public),
        bits(&[256, m], NodeKind::Public),
        bits(&[2, 32], NodeKind::Public),
    ];
    if !layout.committed.is_empty() {
        let k = u32::try_from(layout.committed.len()).unwrap_or(u32::MAX);
        inputs.push(TensorDecl {
            shape: vec![k, m],
            ty: IntType {
                bits: 8,
                signed: false,
            },
            range: IntRange { lo: 0, hi: 255 },
            kind: ADVICE,
        });
        inputs.push(bits(&[512 * n], NodeKind::Public));
    }
    inputs
}

/// The builder's state: the graph, `M`, and the two `SHR` masks.
struct Circuit<'g> {
    g: &'g mut GraphBuilder,
    m: u32,
    shr3: TensorId,
    shr10: TensorId,
}

impl Circuit<'_> {
    fn one(
        &mut self,
        op: Op,
        inputs: &[TensorId],
        output: TensorDecl,
    ) -> Result<TensorId, IrError> {
        Ok(self.g.node(op, inputs, vec![output])?[0])
    }

    /// The rows of `x` at `index`, each `[M]`.
    fn gather(&mut self, x: TensorId, index: Vec<i64>) -> Result<TensorId, IrError> {
        let rows = u32::try_from(index.len()).unwrap_or(u32::MAX);
        let index = self.g.constant(index)?;
        self.one(
            Op::GatherPublic { index },
            &[x],
            virtual_tensor(&[rows, self.m]),
        )
    }

    /// `ROTR^r` of a word.
    fn rotr(&mut self, x: TensorId, r: i64) -> Result<TensorId, IrError> {
        self.gather(x, (0..32).map(|i| (i + r) % 32).collect())
    }

    /// `SHR^r` of a word, `mask` being `SHR^r`'s.
    fn shr(&mut self, x: TensorId, r: i64, mask: TensorId) -> Result<TensorId, IrError> {
        let rotated = self.rotr(x, r)?;
        self.one(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[rotated, mask],
            virtual_tensor(&[32, self.m]),
        )
    }

    fn bitfn(&mut self, f: BoolFn, inputs: &[TensorId]) -> Result<TensorId, IrError> {
        self.one(
            Op::BitFn { f },
            inputs,
            bits(&[32, self.m], NodeKind::Virtual),
        )
    }

    /// `ROTR^{r0} ⊕ ROTR^{r1} ⊕ ROTR^{r2}`, `Σ0` and `Σ1` (FIPS 180-4 §4.1.2).
    fn big_sigma(&mut self, x: TensorId, [r0, r1, r2]: [i64; 3]) -> Result<TensorId, IrError> {
        let terms = [self.rotr(x, r0)?, self.rotr(x, r1)?, self.rotr(x, r2)?];
        self.bitfn(BoolFn::Xor3, &terms)
    }

    /// `σ0 = ROTR^7 ⊕ ROTR^{18} ⊕ SHR^3`.
    fn sigma0(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let mask = self.shr3;
        let terms = [self.rotr(x, 7)?, self.rotr(x, 18)?, self.shr(x, 3, mask)?];
        self.bitfn(BoolFn::Xor3, &terms)
    }

    /// `σ1 = ROTR^{17} ⊕ ROTR^{19} ⊕ SHR^{10}`.
    fn sigma1(&mut self, x: TensorId) -> Result<TensorId, IrError> {
        let mask = self.shr10;
        let terms = [self.rotr(x, 17)?, self.rotr(x, 19)?, self.shr(x, 10, mask)?];
        self.bitfn(BoolFn::Xor3, &terms)
    }

    /// The words' bits added row by row.
    fn rows(&mut self, words: &[TensorId]) -> Result<TensorId, IrError> {
        let mut sum = words[0];
        for &w in &words[1..] {
            sum = self.one(Op::Add, &[sum, w], virtual_tensor(&[32, self.m]))?;
        }
        Ok(sum)
    }

    /// `Σ_i 2^i·r_i` of a `[32, M]` tensor of rows, `[M]`, in five exact levels.
    fn value(&mut self, mut rows: TensorId) -> Result<TensorId, IrError> {
        let mut count = 32i64;
        let mut weight = 2i64;
        while count > 1 {
            let even = self.gather(rows, (0..count).step_by(2).collect())?;
            let odd = self.gather(rows, (1..count).step_by(2).collect())?;
            count /= 2;
            let shape = [u32::try_from(count).unwrap_or(u32::MAX), self.m];
            let c = self.g.constant(vec![weight])?;
            let scaled = self.one(Op::ConstMul { c }, &[odd], virtual_tensor(&shape))?;
            rows = self.one(Op::Add, &[even, scaled], virtual_tensor(&shape))?;
            weight = weight.saturating_mul(weight);
        }
        self.one(Op::Reshape, &[rows], virtual_tensor(&[self.m]))
    }

    /// `(Σ words + constant) mod 2^{32}`, with `carry` bits for the quotient.
    fn add(&mut self, words: &[TensorId], constant: u32, carry: u8) -> Result<TensorId, IrError> {
        let rows = self.rows(words)?;
        let mut sum = self.value(rows)?;
        if constant != 0 {
            let c = self.g.constant(vec![i64::from(constant)])?;
            sum = self.one(Op::ConstAdd { c }, &[sum], virtual_tensor(&[self.m]))?;
        }
        let outputs = self.g.node(
            Op::BitDecompose {
                low: 32,
                high: carry,
            },
            &[sum],
            vec![
                bits(&[self.m, 32], ADVICE),
                bits(&[self.m, u32::from(carry)], ADVICE),
            ],
        )?;
        self.one(
            Op::Transpose { perm: vec![1, 0] },
            &[outputs[0]],
            bits(&[32, self.m], NodeKind::Virtual),
        )
    }

    /// Word `w` of a bit tensor whose rows are words.
    fn word(&mut self, x: TensorId, w: u32) -> Result<TensorId, IrError> {
        self.one(
            Op::Slice {
                axis: 0,
                start: 32 * w,
                len: 32,
            },
            &[x],
            virtual_tensor(&[32, self.m]),
        )
    }

    /// One block's compression of the chaining value `h` with the 16 words of `block`
    /// (FIPS 180-4 §6.2.2).
    fn compress(
        &mut self,
        h: [TensorId; 8],
        block: [TensorId; 16],
    ) -> Result<[TensorId; 8], IrError> {
        let mut w = block.to_vec();
        for t in 16..64 {
            let s1 = self.sigma1(w[t - 2])?;
            let s0 = self.sigma0(w[t - 15])?;
            let word = self.add(&[s1, w[t - 7], s0, w[t - 16]], 0, SCHEDULE_CARRY)?;
            w.push(word);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for (t, &k) in SHA256_K.iter().enumerate() {
            let s1 = self.big_sigma(e, [6, 11, 25])?;
            let ch = self.bitfn(BoolFn::Ch, &[e, f, g])?;
            let s0 = self.big_sigma(a, [2, 13, 22])?;
            let maj = self.bitfn(BoolFn::Maj, &[a, b, c])?;
            // T1 = h + Σ1(e) + Ch(e, f, g) + K_t + W_t, without K_t, row by row.
            let t1 = self.rows(&[hh, s1, ch, w[t]])?;
            let new_e = self.add(&[t1, d], k, ROUND_CARRY)?;
            let new_a = self.add(&[t1, s0, maj], k, ROUND_CARRY)?;
            (hh, g, f, e, d, c, b, a) = (g, f, e, new_e, c, b, a, new_a);
        }
        let mut out = h;
        for (word, v) in out.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *word = self.add(&[*word, v], 0, CHAINING_CARRY)?;
        }
        Ok(out)
    }
}

/// SHA-256 of `M` padded messages of `N` blocks each (FIPS 180-4 §6.2), the inputs those of
/// [`sha256_inputs`] in its order. It returns the chaining value after every block; bind the
/// last to a public digest with [`sha256_bind`]. The tensors it produces carry placeholder ranges
/// until [`crate::settle`] gives them the range analysis's.
///
/// # Errors
///
/// An [`IrError`] of the builder, for inputs of other shapes or a layout whose offsets are not
/// ascending and below `64·N`.
pub fn sha256(
    g: &mut GraphBuilder,
    layout: &Sha256Layout,
    inputs: &[TensorId],
) -> Result<Sha256, IrError> {
    let bad = |rule| IrError::Declaration { rule };
    let limit = 64 * layout.blocks;
    if !layout.committed.windows(2).all(|p| p[0] < p[1])
        || layout.committed.last().is_some_and(|&p| p >= limit)
    {
        return Err(bad(
            "the committed offsets ascend inside the padded message",
        ));
    }
    let committed = !layout.committed.is_empty();
    let &[message, start, masks, ref rest @ ..] = inputs else {
        return Err(bad("the circuit takes its inputs in sha256_inputs's order"));
    };
    if rest.len() != if committed { 2 } else { 0 } {
        return Err(bad("the circuit takes its inputs in sha256_inputs's order"));
    }
    let m = layout.messages;
    let mut mask = |row: i64| -> Result<TensorId, IrError> {
        let index = g.constant(vec![row])?;
        let picked = g.node(
            Op::GatherPublic { index },
            &[masks],
            vec![bits(&[1, 32], NodeKind::Virtual)],
        )?[0];
        Ok(g.node(Op::Reshape, &[picked], vec![bits(&[32], NodeKind::Virtual)])?[0])
    };
    let (shr3, shr10) = (mask(0)?, mask(1)?);
    let mut c = Circuit { g, m, shr3, shr10 };
    let message = if let [bytes, place] = *rest {
        // Each committed byte's bits, row 8k + j its bit j, gathered into its word positions and
        // masked to zero at public bytes.
        let k = u32::try_from(layout.committed.len()).unwrap_or(u32::MAX);
        let b = c.g.node(
            Op::BitDecompose { low: 8, high: 0 },
            &[bytes],
            vec![bits(&[k, m, 8], ADVICE)],
        )?[0];
        let b = c.one(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            &[b],
            bits(&[k, 8, m], NodeKind::Virtual),
        )?;
        let b = c.one(Op::Reshape, &[b], bits(&[8 * k, m], NodeKind::Virtual))?;
        let index = (0..512 * layout.blocks as usize)
            .map(|row| {
                let (byte, bit) = byte_of(row);
                layout
                    .slot(byte)
                    .map_or(0, |s| i64::try_from(8 * s + bit).unwrap_or(0))
            })
            .collect();
        let gathered = c.gather(b, index)?;
        let placed = c.one(
            Op::Mul {
                broadcast: Broadcast::Trailing,
            },
            &[gathered, place],
            virtual_tensor(&[512 * layout.blocks, m]),
        )?;
        c.one(
            Op::BitFn { f: BoolFn::Xor },
            &[message, placed],
            bits(&[512 * layout.blocks, m], NodeKind::Virtual),
        )?
    } else {
        message
    };
    let mut h = [start; 8];
    for (j, word) in h.iter_mut().enumerate() {
        *word = c.word(start, u32::try_from(j).unwrap_or(0))?;
    }
    let mut chaining = Vec::with_capacity(layout.blocks as usize);
    for n in 0..layout.blocks {
        let mut block = [message; 16];
        for (j, word) in block.iter_mut().enumerate() {
            *word = c.word(message, 16 * n + u32::try_from(j).unwrap_or(0))?;
        }
        h = c.compress(h, block)?;
        chaining.push(h);
    }
    Ok(Sha256 { chaining })
}

/// Binds the chaining words `words` to `digest`, public bits `[256, M]` in the layout of
/// [`sha256_bits`]: one `AssertEqual` per word (RFC-0066 'The output').
///
/// # Errors
///
/// An [`IrError`] of the builder, for a digest of another shape.
pub fn sha256_bind(
    g: &mut GraphBuilder,
    words: &[TensorId; 8],
    digest: TensorId,
    messages: u32,
) -> Result<(), IrError> {
    for (j, &word) in (0u32..).zip(words) {
        let expected = g.node(
            Op::Slice {
                axis: 0,
                start: 32 * j,
                len: 32,
            },
            &[digest],
            vec![bits(&[32, messages], NodeKind::Virtual)],
        )?[0];
        g.node(Op::AssertEqual, &[word, expected], Vec::new())?;
    }
    Ok(())
}

/// The declaration of a digest input for [`sha256_bind`]: public bits `[256, M]`.
#[must_use]
pub fn sha256_digest_input(messages: u32) -> TensorDecl {
    bits(&[256, messages], NodeKind::Public)
}
