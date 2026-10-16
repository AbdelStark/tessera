// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rate-1/2 Reed–Solomon encoding of rows with vanishing masks (RFC-0011 'Encoding').
//!
//! A row of `k` values is `f(ω_k^0), …, f(ω_k^{k−1})` for the polynomial `f` of degree below `k`.
//! The encoding adds `Z_H·m`, with `Z_H = X^k − 1` vanishing on the subgroup `H` of order `k` and
//! `m = Σ_{i < P} m_i·X^i` drawn from a [`MaskSource`], and evaluates `g = f + Z_H·m` on the coset
//! `7·L` of the subgroup `L` of order `n = 2k`: symbol `j` is `g(7·ω_n^j)`.
//!
//! # Algorithm
//!
//! `g = f − m + X^k·m`, so its coefficients are `f_i − m_i` for `i < P`, `f_i` for `P ≤ i < k`,
//! `m_{i−k}` for `k ≤ i < k + P` and zero above. The encoder computes them with an inverse NTT of
//! size `k` and evaluates them with a coset NTT of size `n`, chaining the two kernels of
//! [`ntt`](crate::ntt) so that no bit-reversal permutation is needed:
//!
//! 1. The Cooley–Tukey kernel with the twiddles of `ω_k^{−1}` turns the row, in natural order,
//!    into `k·f_{rev_k(t)}` in position `t`, where `rev_k` reverses `log2 k` bits.
//! 2. The coset NTT of `g` is the NTT of `g(7X)`, whose coefficients are `g_i·7^i`. In the
//!    bit-reversed order of size `n`, coefficient `i < k` sits in position `2·rev_k(i)` and
//!    coefficient `k + i` in position `2·rev_k(i) + 1`. So position `2t` receives the entry `t`
//!    of step 1 times `7^{rev_k(t)}/k` (one precomputed table, which also absorbs the inverse
//!    NTT's `1/k`), and position `2t + 1` receives zero.
//! 3. For each `i < P`, the mask coefficient `m_i` is drawn and `m_i·7^i` subtracted in position
//!    `2·rev_k(i)`, and `m_i·7^{k+i}` written in position `2·rev_k(i) + 1`.
//! 4. The Gentleman–Sande kernel with the twiddles of `ω_n` evaluates, from bit-reversed
//!    coefficients, in natural order.
//!
//! Every step runs in the codeword buffer, so encoding allocates nothing.

use core::fmt;

use tessera_field::{Field, Goldilocks};

use crate::PcsError;
use crate::ntt::{
    Subgroup, TWO_ADICITY, bit_reversed_powers, cooley_tukey, gentleman_sande,
    inverse_of_power_of_two, reverse_bits, twiddles,
};

type Fp = Goldilocks;

/// The shift of the coset every codeword is evaluated on: symbol `j` of a codeword is
/// `g(7·ω_n^j)` (RFC-0011 'Encoding'). 7 generates `Fp^*`, so it lies outside the subgroup `L`,
/// and `7·L` is disjoint from `L` and from its subgroup `H`.
pub const COSET_SHIFT: Goldilocks = Goldilocks::MULTIPLICATIVE_GENERATOR;

/// A source of mask coefficients, one element of `Fp` per call.
///
/// The contract, which the encoder relies on for hiding and cannot check:
///
/// - each call returns a fresh element, uniformly distributed on `Fp` and independent of every
///   other call and of the rows: any `P` codeword symbols outside `H` are then uniform and
///   independent of the row (RFC-0013 'Hiding openings');
/// - the values are secret-class masking randomness (05 §3): an implementation never logs or
///   prints them, and no stream is reused across commitments (INV-PCS-4).
///
/// For production, the source is the per-proof mask stream, a PRG keyed by the per-proof secret
/// seed, so that a restart reproduces it (RFC-0011 'Streaming procedure'). Any closure
/// `FnMut() -> Goldilocks` is a source, which is how tests supply seeded ones.
///
/// [`RowEncoder`] draws exactly `P` elements per row, `m_0` first, and only once a row has passed
/// every check; it keeps none of them.
pub trait MaskSource {
    /// The next mask coefficient.
    fn draw(&mut self) -> Goldilocks;
}

impl<F: FnMut() -> Goldilocks> MaskSource for F {
    fn draw(&mut self) -> Goldilocks {
        self()
    }
}

/// The encoder of rows of `k` values into codewords of `n = 2k` symbols with `P` mask
/// coefficients, `0 ≤ P ≤ k`, for `k` a power of two from `2^0` to `2^31`.
///
/// It precomputes three tables, `2.5·k` elements (80 MiB at `k = 2^22`), once; each row then costs
/// an inverse NTT of size `k`, a coset NTT of size `n` and `P` draws, and allocates nothing.
/// Encoding is deterministic given the mask coefficients: the same row and mask give the same
/// codeword.
#[derive(Clone)]
pub struct RowEncoder {
    log_k: u32,
    mask_coeffs: usize,
    /// `ω_k^{−rev(t)}` for `t < k/2`: the inverse NTT of the row.
    inverse: Vec<Fp>,
    /// `ω_n^{rev(t)}` for `t < k = n/2`: the NTT of the masked coefficients.
    forward: Vec<Fp>,
    /// `7^{rev_k(t)}/k` for `t < k`.
    spread: Vec<Fp>,
    /// `7^k`.
    shift_pow_k: Fp,
}

/// Shows the code's parameters; the tables are public constants and long.
impl fmt::Debug for RowEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowEncoder")
            .field("message_len", &self.message_len())
            .field("codeword_len", &self.codeword_len())
            .field("mask_coeffs", &self.mask_coeffs)
            .finish_non_exhaustive()
    }
}

impl RowEncoder {
    /// The encoder of rows of `message_len = k` values with `mask_coeffs = P` mask coefficients.
    ///
    /// # Errors
    ///
    /// In this order: [`PcsError::NotPowerOfTwo`] if `k` is not a power of two;
    /// [`PcsError::TooLong`] if the codeword's `2k` points exceed `2^32`; and
    /// [`PcsError::MaskTooLarge`] if `P > k`.
    pub fn new(message_len: usize, mask_coeffs: usize) -> Result<Self, PcsError> {
        if !message_len.is_power_of_two() {
            return Err(PcsError::NotPowerOfTwo { len: message_len });
        }
        let log_k = message_len.trailing_zeros();
        let log_n = log_k + 1;
        if log_n > TWO_ADICITY {
            return Err(PcsError::TooLong { log_len: log_n });
        }
        if mask_coeffs > message_len {
            return Err(PcsError::MaskTooLarge {
                mask_coeffs,
                message_len,
            });
        }
        // The table of the largest transform first: were the check on log_n ever lost,
        // two_adic_generator would refuse the size before anything large is allocated.
        let forward = twiddles(log_n, false);
        Ok(Self {
            log_k,
            mask_coeffs,
            inverse: twiddles(log_k, true),
            forward,
            spread: bit_reversed_powers(COSET_SHIFT, log_k, inverse_of_power_of_two(log_k)),
            shift_pow_k: COSET_SHIFT.pow(message_len as u64),
        })
    }

    /// The number of values `k` in a row.
    pub fn message_len(&self) -> usize {
        1 << self.log_k
    }

    /// The number of symbols `n = 2k` in a codeword.
    pub fn codeword_len(&self) -> usize {
        2 << self.log_k
    }

    /// The number of mask coefficients `P` drawn per row.
    pub fn mask_coeffs(&self) -> usize {
        self.mask_coeffs
    }

    /// Encodes `row` into `codeword`: draws `m_0, …, m_{P−1}` from `mask` and writes
    /// `g(7·ω_n^j)` in position `j`, where `g = f + (X^k − 1)·m`, `f` is the polynomial of degree
    /// below `k` with `f(ω_k^i) = row[i]`, and `m = Σ_{i < P} m_i·X^i`.
    ///
    /// `codeword` is overwritten whatever it held, so a block buffer can be reused row after row.
    ///
    /// # Errors
    ///
    /// [`PcsError::RowLength`] unless `row` holds `k` values, then [`PcsError::CodewordLength`]
    /// unless `codeword` holds `n` symbols. Nothing is drawn or written on error.
    pub fn encode_row<M: MaskSource + ?Sized>(
        &self,
        row: &[Fp],
        mask: &mut M,
        codeword: &mut [Fp],
    ) -> Result<(), PcsError> {
        if row.len() != self.message_len() {
            return Err(PcsError::RowLength {
                expected: self.message_len(),
                found: row.len(),
            });
        }
        if codeword.len() != self.codeword_len() {
            return Err(PcsError::CodewordLength {
                expected: self.codeword_len(),
                found: codeword.len(),
            });
        }
        self.encode(row, mask, codeword);
        Ok(())
    }

    /// Encodes several rows with the same tables: `rows` holds them one after another, `k` values
    /// each, and `codewords` receives their codewords in the same order, `n` symbols each. Row `r`
    /// draws its `P` coefficients after row `r − 1`, so the result equals [`encode_row`] on each
    /// row in turn with the same source.
    ///
    /// [`encode_row`]: Self::encode_row
    ///
    /// # Errors
    ///
    /// [`PcsError::PartialRow`] unless `rows` is a whole number of rows, then
    /// [`PcsError::CodewordLength`] unless `codewords` holds `n` symbols per row. Nothing is drawn
    /// or written on error.
    pub fn encode_rows<M: MaskSource + ?Sized>(
        &self,
        rows: &[Fp],
        mask: &mut M,
        codewords: &mut [Fp],
    ) -> Result<(), PcsError> {
        let (k, n) = (self.message_len(), self.codeword_len());
        if rows.len() % k != 0 {
            return Err(PcsError::PartialRow {
                len: rows.len(),
                row_len: k,
            });
        }
        // rows.len() ≤ isize::MAX / 8, so twice it fits.
        let expected = rows.len() * 2;
        if codewords.len() != expected {
            return Err(PcsError::CodewordLength {
                expected,
                found: codewords.len(),
            });
        }
        for (row, codeword) in rows.chunks_exact(k).zip(codewords.chunks_exact_mut(n)) {
            self.encode(row, mask, codeword);
        }
        Ok(())
    }

    /// The four steps of the module's algorithm, on checked lengths.
    fn encode<M: MaskSource + ?Sized>(&self, row: &[Fp], mask: &mut M, codeword: &mut [Fp]) {
        let k = row.len();
        // 1. k·f in bit-reversed order, in the low half.
        let low = &mut codeword[..k];
        low.copy_from_slice(row);
        cooley_tukey(low, &self.inverse, &Subgroup);
        // 2. f_i·7^i in position 2·rev_k(i), zero in the odd positions.
        spread(codeword, &self.spread);
        // 3. The mask: −m_i·7^i in position 2·rev_k(i), m_i·7^{k+i} in the next one.
        let mut power = Fp::ONE; // 7^i
        for i in 0..self.mask_coeffs {
            let scaled = mask.draw() * power;
            let even = 2 * reverse_bits(i, self.log_k);
            codeword[even] -= scaled;
            codeword[even + 1] = scaled * self.shift_pow_k;
            power *= COSET_SHIFT;
        }
        // 4. The values of g at 7·ω_n^j, in natural order.
        gentleman_sande(codeword, &self.forward, &Subgroup);
    }
}

/// Moves entry `t` of the low half, times `scale[t]`, to position `2t`, and writes zero in
/// position `2t + 1`, for every `t < k = codeword.len()/2`, in place.
///
/// Entries `t ∈ [m/2, m)` move to `[m, 2m)`, which holds nothing left to move: the buffer's old
/// contents for `m = k`, entries already moved below. So the moves run from the top down, each
/// block between disjoint slices, and entry 0 stays where it is.
fn spread(codeword: &mut [Fp], scale: &[Fp]) {
    let mut m = codeword.len() / 2;
    while m >= 2 {
        let (from, to) = codeword[..2 * m].split_at_mut(m);
        for ((pair, &y), &s) in to
            .chunks_exact_mut(2)
            .zip(&from[m / 2..])
            .zip(&scale[m / 2..m])
        {
            pair[0] = y * s;
            pair[1] = Fp::ZERO;
        }
        m /= 2;
    }
    codeword[0] *= scale[0];
    codeword[1] = Fp::ZERO;
}
