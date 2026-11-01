// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reed–Solomon encoding of rows with vanishing masks (RFC-0011 'Encoding'), at rate `1/2^r`:
//! rate 1/2 for the advice (RFC-0011), rate 1/4 for the weights (RFC-0012 'Layout', RFC-0040
//! 'Encoding, leaves and roots').
//!
//! A row of `k` values is `f(ω_k^0), …, f(ω_k^{k−1})` for the polynomial `f` of degree below `k`.
//! The encoding adds `Z_H·m`, with `Z_H = X^k − 1` vanishing on the subgroup `H` of order `k` and
//! `m = Σ_{i < P} m_i·X^i` drawn from a [`MaskSource`], and evaluates `g = f + Z_H·m` on the coset
//! `7·L` of the subgroup `L` of order `n = 2^r·k`: symbol `j` is `g(7·ω_n^j)`. With `P = 0`, as
//! for a weight row in `tessera-protocol/1`, `g = f`.
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
//!    bit-reversed order of size `n`, coefficient `i < k` sits in position `2^r·rev_k(i)` and
//!    coefficient `k + i` in position `2^r·rev_k(i) + 2^{r−1}`: reversing `log2 n` bits sends the
//!    low `log2 k` bits of an index to the top and bit `log2 k` to bit `r − 1`. So position `2^r·t`
//!    receives the entry `t` of step 1 times `7^{rev_k(t)}/k` (one precomputed table, which also
//!    absorbs the inverse NTT's `1/k`), and the `2^r − 1` positions after it receive zero.
//! 3. For each `i < P`, the mask coefficient `m_i` is drawn and `m_i·7^i` subtracted in position
//!    `2^r·rev_k(i)`, and `m_i·7^{k+i}` written in position `2^r·rev_k(i) + 2^{r−1}`.
//! 4. The Gentleman–Sande kernel with the twiddles of `ω_n` evaluates, from bit-reversed
//!    coefficients, in natural order.
//!
//! Every field step runs in the codeword buffer. The default scalar execution allocates
//! nothing per row; opt-in threads allocate only their bounded control state.

use core::fmt;
use core::num::NonZeroUsize;

use tessera_field::{Field, Goldilocks};

use crate::PcsError;
use crate::ntt::{
    Subgroup, TWO_ADICITY, bit_reversed_powers, inverse_of_power_of_two, reverse_bits, twiddles,
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

/// The encoder of rows of `k` values into codewords of `n = 2^r·k` symbols with `P` mask
/// coefficients, `0 ≤ P ≤ k`, at rate `1/2^r`, `r ≥ 1`, for `k` a power of two and `n ≤ 2^32`.
///
/// It precomputes three tables, `(1.5 + 2^{r−1})·k` elements (80 MiB at `k = 2^22` and rate 1/2),
/// once; each row then costs an inverse NTT of size `k`, a coset NTT of size `n` and `P` draws,
/// and allocates no field buffer. Scalar execution allocates nothing per row; opt-in workers
/// require the separate [`Self::thread_memory`] reservation. Encoding is deterministic given the mask coefficients: the same row and
/// mask give the same codeword.
#[derive(Clone)]
pub struct RowEncoder {
    log_k: u32,
    /// `r`: `n = 2^r·k`.
    log_inv_rate: u32,
    mask_coeffs: usize,
    threads: NonZeroUsize,
    /// `ω_k^{−rev(t)}` for `t < k/2`: the inverse NTT of the row.
    inverse: Vec<Fp>,
    /// `ω_n^{rev(t)}` for `t < n/2`: the NTT of the masked coefficients.
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
    /// The encoder of rows of `message_len = k` values into codewords of `n = 2^r·k` symbols,
    /// `r = log2_inv_rate`, with `mask_coeffs = P` mask coefficients: `r = 1` for the advice,
    /// `r = 2` for the weights.
    ///
    /// # Errors
    ///
    /// In this order: [`PcsError::NotPowerOfTwo`] if `k` is not a power of two;
    /// [`PcsError::InverseRate`] if `r = 0`, a code of rate 1; [`PcsError::TooLong`] if the
    /// codeword's `n` points exceed `2^32`; and [`PcsError::MaskTooLarge`] if `P > k`.
    pub fn new(
        message_len: usize,
        log2_inv_rate: u32,
        mask_coeffs: usize,
    ) -> Result<Self, PcsError> {
        if !message_len.is_power_of_two() {
            return Err(PcsError::NotPowerOfTwo { len: message_len });
        }
        if log2_inv_rate == 0 {
            return Err(PcsError::InverseRate { log2_inv_rate });
        }
        let log_k = message_len.trailing_zeros();
        let log_n = log_k.saturating_add(log2_inv_rate);
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
            log_inv_rate: log2_inv_rate,
            mask_coeffs,
            threads: NonZeroUsize::MIN,
            inverse: twiddles(log_k, true),
            forward,
            spread: bit_reversed_powers(COSET_SHIFT, log_k, inverse_of_power_of_two(log_k)),
            shift_pow_k: COSET_SHIFT.pow(message_len as u64),
        })
    }

    /// Uses at most `threads` participants for public-size row transforms.
    ///
    /// Defaults to one. Workers share the public tables and borrow disjoint codeword
    /// slices; masks are still drawn serially in their canonical order. Failed OS
    /// thread creation falls back to scalar work. Small transforms remain scalar.
    /// Callers must reserve [`Self::thread_memory`] beside their existing buffers.
    pub fn threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }

    /// Additional conservative thread stack/control storage during an encoding.
    ///
    /// `codeword_len` is the largest transform length. Returns `None` for an
    /// unsupported size or sizing overflow. This excludes external allocator/OS
    /// runtime allocations, guard pages and caches, which require a separate whole-process measurement.
    pub fn thread_memory(codeword_len: usize, threads: NonZeroUsize) -> Option<u64> {
        crate::ntt::parallel::memory(codeword_len, threads)
    }

    /// Successful workers for `transform_len` on this caller thread, for test controls.
    ///
    /// This method and the reduced test scheduling threshold exist only with `test-rng`,
    /// which fails release builds. It does not report field, row or mask values.
    #[cfg(feature = "test-rng")]
    #[doc(hidden)]
    pub fn test_worker_spawns(transform_len: usize) -> usize {
        crate::ntt::parallel::test_spawns(transform_len)
    }

    /// The number of values `k` in a row.
    pub fn message_len(&self) -> usize {
        1 << self.log_k
    }

    /// The number of symbols `n = 2^r·k` in a codeword.
    pub fn codeword_len(&self) -> usize {
        1 << (self.log_k + self.log_inv_rate)
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
        if !rows.len().is_multiple_of(k) {
            return Err(PcsError::PartialRow {
                len: rows.len(),
                row_len: k,
            });
        }
        // rows.len() ≤ isize::MAX / 8, so 2^r times it fits for r ≤ 3; past that, a length that
        // overflows is one no buffer has.
        let expected = rows
            .len()
            .checked_mul(n / k)
            .ok_or(PcsError::CodewordLength {
                expected: usize::MAX,
                found: codewords.len(),
            })?;
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
        let stride = 1 << self.log_inv_rate; // 2^r ≥ 2
        // 1. k·f in bit-reversed order, in the first k positions.
        let low = &mut codeword[..k];
        low.copy_from_slice(row);
        crate::ntt::parallel::ct_parallel(low, &self.inverse, &Subgroup, self.threads.get());
        // 2. f_i·7^i in position 2^r·rev_k(i), zero in the 2^r − 1 positions after it.
        spread(codeword, &self.spread);
        // 3. The mask: −m_i·7^i in position 2^r·rev_k(i), m_i·7^{k+i} in position
        // 2^r·rev_k(i) + 2^{r−1}.
        let mut power = Fp::ONE; // 7^i
        for i in 0..self.mask_coeffs {
            let scaled = mask.draw() * power;
            let low = stride * reverse_bits(i, self.log_k);
            codeword[low] -= scaled;
            codeword[low + stride / 2] = scaled * self.shift_pow_k;
            power *= COSET_SHIFT;
        }
        // 4. The values of g at 7·ω_n^j, in natural order.
        crate::ntt::parallel::gs_parallel(codeword, &self.forward, &Subgroup, self.threads.get());
    }
}

/// Moves entry `t` of the first `k = scale.len()` entries, times `scale[t]`, to position `s·t`,
/// and writes zero in positions `s·t + 1` to `s·t + s − 1`, for every `t < k`, in place, where
/// `s = codeword.len()/k ≥ 2`.
///
/// Entries `t ∈ [m/2, m)` move to `[s·m/2, s·m)`, which holds nothing left to move, since
/// `s·m/2 ≥ m`: the buffer's old contents for `m = k`, entries already moved below. So the moves
/// run from the top down, each block between disjoint slices, and entry 0 stays where it is.
fn spread(codeword: &mut [Fp], scale: &[Fp]) {
    let stride = codeword.len() / scale.len();
    let mut m = scale.len();
    while m >= 2 {
        let (from, to) = codeword[..stride * m].split_at_mut(stride * m / 2);
        for ((run, &y), &s) in to
            .chunks_exact_mut(stride)
            .zip(&from[m / 2..m])
            .zip(&scale[m / 2..m])
        {
            run[0] = y * s;
            run[1..].fill(Fp::ZERO);
        }
        m /= 2;
    }
    codeword[0] *= scale[0];
    codeword[1..stride].fill(Fp::ZERO);
}
