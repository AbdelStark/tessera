// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A table as the claim list names it (RFC-0008 'Table specifications'). The TCE layout of these
//! types is tier A: an enum's tag is its variant's declaration index (RFC-0003), so variants are
//! appended, never reordered.

use tessera_codec::{Decode, Encode, Rational};
use tessera_hash::Digest;

/// A table's index in the claim list, assigned by the compiler (03 §3, INV-DM-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encode, Decode)]
pub struct TableId(pub u32);

/// The operation of a [`TableKind::Bitwise8`] table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum BitOp {
    /// `a ⊕ b`.
    Xor,
    /// `a ∧ b`.
    And,
}

/// A table's kind and parameters. Row `j` of each kind, and the parameters it allows, are those
/// of RFC-0008 'Table specifications'; every key is the row index `j` itself.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub enum TableKind {
    /// Keys `0 ≤ x < 2^bits` with no value: range checks.
    Range {
        /// Key bits, 1 to 32.
        bits: u8,
    },
    /// `rq(k, r_c, lo, hi)` of RFC-0005 for the class `c` and the key `k ∈ [−2^{b−1}, 2^{b−1})`,
    /// looked up as `pack(c, k) = (k + 2^{b−1}) + 2^b · c` (RFC-0009 `Requant1`).
    Requant {
        /// Key bits, at least 1, with `b + ⌈log2 n⌉ ≤ 32` for `n` classes.
        b: u8,
        /// The shift `r_c` of each class `c`, each 1 to 63; one to four classes.
        r_by_class: Vec<u8>,
        /// The saturation floor.
        lo: i32,
        /// The saturation ceiling, at least `lo`.
        hi: i32,
    },
    /// `i ↦ 2^i` for `0 ≤ i < max_exp`.
    Pow2 {
        /// The number of rows, 1 to 63, so that every value is in the centered range.
        max_exp: u8,
    },
    /// `(a, b) ↦ a op b` for bytes `a` and `b`, looked up as `a + 2^8 · b`.
    Bitwise8 {
        /// `⊕` or `∧`.
        op: BitOp,
    },
    /// The AES S-box (FIPS-197 §5.1.1).
    AesSbox,
    /// `x ↦ c • x` in GF(2^8) modulo `x^8 + x^4 + x^3 + x + 1` (FIPS-197 §4.2).
    Gf256Mul {
        /// The constant factor.
        c: u8,
    },
    /// `k ↦ round(2^σ · e^{−k · scale})` for `0 ≤ k < 2^key_bits`, in the fixed point of RFC-0008
    /// 'Fixed-point generators': the softmax exponentials (RFC-0009 `SoftmaxRow`).
    Exp {
        /// Key bits, 1 to 32.
        key_bits: u8,
        /// The output precision `σ`, at most 32: the value at key 0 is `2^σ`.
        sigma: u8,
        /// `σ_q σ_k / √d_head` as the claim list states it, positive.
        scale: Rational,
    },
    /// `f ↦ ⌊2^{out_bits} · 2^{f / 2^{frac_bits}}⌋` for `0 ≤ f < 2^{frac_bits}`, never above the
    /// true value: the scoring bound's powers of two (RFC-0017).
    Exp2Frac {
        /// Fraction bits, 1 to 16.
        frac_bits: u8,
        /// Output bits, 1 to 56, so that the generator's rounding stays below one output unit.
        out_bits: u8,
    },
    /// `(class, k) ↦ clamp(round(silu(h · 2^{in_exp}) / 2^{out_exp}), −128, 127)` with
    /// `h = k >> r_c`, for `k ∈ [−2^{b−1}, 2^{b−1})`, looked up as `pack(c, k)` like `Requant`: the
    /// MLP activation, absorbing the gate's rescale (RFC-0005 'Integer semantics').
    Silu {
        /// Key bits, at least 1, with `b + ⌈log2 n⌉ ≤ 32` for `n` classes.
        b: u8,
        /// The shift `r_c` of each class `c`, each below `b`; one to four classes, with
        /// `Σ_c 2^{b − r_c} ≤ 2^24` values of `h` in all.
        r_by_class: Vec<u8>,
        /// The exponent of `h`'s scale, from −64 to 0.
        in_exp: i8,
        /// The exponent of the output's scale, with `1 ≤ 63 − in_exp + out_exp ≤ 126`.
        out_exp: i8,
    },
    /// The RoPE constants `(C, S) = (round(2^bits · cos(p·ω_i)), round(2^bits · sin(p·ω_i)))` of
    /// each position `p` and frequency `i < d_head / 2`, with Llama-3's scaled frequencies `ω_i`
    /// (RFC-0005): the public constants of `Rotate`, not a lookup table.
    Rope {
        /// The base `θ`, at least 2.
        theta: u32,
        /// Llama-3's scaling `factor`, at least 1.
        factor: u16,
        /// `low_freq_factor`, at least 1 and below `high_freq_factor`.
        low_freq_factor: u16,
        /// `high_freq_factor`.
        high_freq_factor: u16,
        /// `original_max_position_embeddings`, at least 1.
        original_max: u32,
        /// The constants' precision, 1 to 30 (14 in `D`).
        bits: u8,
        /// The positions, with `max_context · d_head / 2 ≤ 2^24` pairs in all.
        max_context: u32,
        /// The head dimension, even and at least 2.
        d_head: u16,
    },
}

/// A table as the claim list names it (RFC-0007), sorted there by [`TableId`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct TableSpec {
    /// The table's index.
    pub id: TableId,
    /// The kind, with its parameters.
    pub kind: TableKind,
    /// `⌈log2 T⌉` for the kind's `T` rows: the table tree has `2^log2_rows` leaves.
    pub log2_rows: u8,
    /// `H("table", TCE(id, kind, log2_rows) ‖ TCE(pieces) ‖ TCE(core))` (RFC-0008 'Generators
    /// and digests'), the pieces a `Vec<u64>` and the core a `Vec<Fp>`.
    pub digest: Digest,
}
