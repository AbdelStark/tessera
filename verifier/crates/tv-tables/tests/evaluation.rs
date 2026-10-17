// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Structured evaluation against the definition (RFC-0008 'Testing Strategy', 'Structured
//! evaluation'): for every lookup kind at small parameters, `eval_mle` equals the brute-force sum
//! `Σ_j eq(ρ, j)·(key_j + β·value_j)` over the padded rows at 100 seeded random points per
//! table, whose coordinates are sometimes 0 or 1, and equals the row itself at every Boolean
//! point, or at 324 of them for each `Bitwise8` table of `2^16` rows. Parameters cover `T` a
//! power of two and not, one to four classes, clamps that bind at either end or not at all,
//! shifts at and beyond the key width, `Exp` tables with and without a zero tail inside them,
//! and `Silu` classes of several shifts.

mod support;

use support::{Rng, boolean_point, brute_force};
use tv_codec::Rational;
use tv_field::Fp2;
use tv_tables::{BitOp, Table, TableError, TableId, TableKind};

/// Random points checked per table, as RFC-0008 'Testing Strategy' asks.
const POINTS: usize = 100;

fn scale(num: i64, den: u64) -> Rational {
    Rational::new(num, den).unwrap()
}

/// Small tables of every lookup kind but `Bitwise8`, whose `2^16` rows have their own test.
fn small_kinds() -> Vec<TableKind> {
    let mut kinds = Vec::new();
    for bits in 1..=6 {
        kinds.push(TableKind::Range { bits });
    }
    let requant = |b: u8, r_by_class: &[u8], lo: i32, hi: i32| TableKind::Requant {
        b,
        r_by_class: r_by_class.to_vec(),
        lo,
        hi,
    };
    kinds.extend([
        // No clamp binds.
        requant(4, &[1], -128, 127),
        requant(6, &[1, 2, 3], i32::MIN, i32::MAX),
        // Both ends bind, three classes: T = 3·2^5 is not a power of two.
        requant(5, &[1, 2, 3], -2, 3),
        // Only one end binds.
        requant(5, &[2, 1], -3, 100),
        requant(5, &[2, 1], -100, 2),
        // Shifts at and beyond the key width: the class is the constant 0, clamped.
        requant(3, &[3, 4, 63], -1, 1),
        requant(3, &[63], 1, 4),
        requant(3, &[63], -4, -1),
        // lo = hi: every value is lo.
        requant(4, &[2], 5, 5),
        // Bounds beyond every value: all lo, all hi.
        requant(4, &[1, 2], 50, 60),
        requant(4, &[1, 2], -60, -50),
        // One-bit keys, four classes.
        requant(1, &[1, 2, 1, 1], -1, 0),
        requant(2, &[1, 2, 1], -1, 1),
    ]);
    for max_exp in [1, 2, 3, 5, 8, 33, 63] {
        kinds.push(TableKind::Pow2 { max_exp });
    }
    kinds.push(TableKind::AesSbox);
    for c in [0, 1, 2, 3, 0x57, 0xFF] {
        kinds.push(TableKind::Gf256Mul { c });
    }
    let exp = |key_bits: u8, sigma: u8, num: i64, den: u64| TableKind::Exp {
        key_bits,
        sigma,
        scale: scale(num, den),
    };
    kinds.extend([
        // A zero tail inside the table, and one that starts at key 1.
        exp(6, 8, 1, 4),
        exp(10, 0, 3, 7),
        exp(5, 16, 1, 1),
        exp(4, 3, 50, 1),
        exp(8, 32, 1, 3),
        // No tail inside the table: z = T.
        exp(4, 16, 1, 1000),
        exp(1, 0, 1, 2),
        exp(7, 32, 1, 1 << 40),
    ]);
    for (frac_bits, out_bits) in [(1, 1), (2, 7), (5, 16), (8, 16), (8, 56)] {
        kinds.push(TableKind::Exp2Frac {
            frac_bits,
            out_bits,
        });
    }
    let silu = |b: u8, r_by_class: &[u8], in_exp: i8, out_exp: i8| TableKind::Silu {
        b,
        r_by_class: r_by_class.to_vec(),
        in_exp,
        out_exp,
    };
    kinds.extend([
        silu(6, &[0, 2], -3, -4),
        silu(8, &[1], 0, -2),
        silu(4, &[0, 1, 2, 3], -64, -62),
        silu(5, &[4, 0, 2], -2, -5),
        silu(1, &[0, 0, 0], 0, 0),
        silu(3, &[2, 1, 0, 2], -1, -3),
        silu(7, &[6], 0, 63),
    ]);
    kinds
}

fn check_points(table: &Table, rng: &mut Rng, points: usize) {
    let n = usize::from(table.log2_rows());
    for _ in 0..points {
        let rho = rng.point(n);
        let beta = rng.fp2();
        assert_eq!(
            table.eval_mle(&rho, beta).unwrap(),
            brute_force(table, &rho, beta),
            "{:?} at {rho:?}",
            table.spec().kind
        );
    }
}

#[test]
fn eval_mle_equals_the_sum_over_the_padded_rows() {
    let mut rng = Rng::new(0x7ab1e5);
    for (i, kind) in small_kinds().into_iter().enumerate() {
        let table = Table::generate(TableId(i as u32), kind).unwrap();
        check_points(&table, &mut rng, POINTS);
    }
}

/// At a Boolean point the extension is the row there, padded rows included.
#[test]
fn eval_mle_at_boolean_points_is_the_row() {
    let mut rng = Rng::new(0xb001);
    for (i, kind) in small_kinds().into_iter().enumerate() {
        let table = Table::generate(TableId(i as u32), kind).unwrap();
        let n = usize::from(table.log2_rows());
        let beta = rng.fp2();
        for j in 0..1u64 << n {
            let (key, value) = table.row(j).unwrap();
            assert_eq!(
                table.eval_mle(&boolean_point(j, n), beta).unwrap(),
                Fp2::from(key) + beta * value,
                "{:?}, row {j}",
                table.spec().kind
            );
        }
    }
}

/// `Bitwise8`, of `2^16` rows, for each operation: the sum over the rows at the 100 random points
/// RFC-0008 'Testing Strategy' asks for, and the row itself at Boolean points. Each evaluation
/// folds every row, so the Boolean points are not all `2^16` of them but 324: the rows `a + 2^8·b`
/// with `b = 167·a + 13 mod 2^8`, which give each operand every byte, the four corners, and 64
/// seeded rows.
#[test]
fn eval_mle_of_bitwise_tables() {
    let mut rng = Rng::new(0xb17);
    for op in [BitOp::Xor, BitOp::And] {
        let table = Table::generate(TableId(1), TableKind::Bitwise8 { op }).unwrap();
        assert_eq!(table.log2_rows(), 16);
        check_points(&table, &mut rng, POINTS);
        let mut rows: Vec<u64> = (0..256)
            .map(|a| a + (((167 * a + 13) % 256) << 8))
            .collect();
        rows.extend([0, 0xFF, 0xFF00, 0xFFFF]);
        rows.extend((0..64).map(|_| rng.below(1 << 16)));
        assert_eq!(rows.len(), 324);
        let beta = rng.fp2();
        for j in rows {
            let (key, value) = table.row(j).unwrap();
            assert_eq!(
                table.eval_mle(&boolean_point(j, 16), beta).unwrap(),
                Fp2::from(key) + beta * value,
                "{op:?}, row {j}"
            );
        }
    }
}

/// Large tables whose rows are never enumerated, against the brute force over a restricted
/// point: with the high coordinates Boolean, the extension is that of one block of rows, which
/// the oracle sums.
#[test]
fn eval_mle_of_tables_too_large_to_enumerate() {
    let mut rng = Rng::new(0x1a46e);
    let kinds = [
        TableKind::Range { bits: 32 },
        TableKind::Requant {
            b: 31,
            r_by_class: vec![1, 40],
            lo: i32::MIN,
            hi: i32::MAX,
        },
        TableKind::Requant {
            b: 30,
            r_by_class: vec![1, 2, 3],
            lo: -524_288,
            hi: 524_287,
        },
        TableKind::Silu {
            b: 22,
            r_by_class: vec![10, 11, 12],
            in_exp: -6,
            out_exp: -4,
        },
        TableKind::Exp {
            key_bits: 24,
            sigma: 16,
            scale: scale(1, 2896),
        },
    ];
    for kind in kinds {
        let table = Table::generate(TableId(0), kind).unwrap();
        let n = usize::from(table.log2_rows());
        let free = 10;
        for _ in 0..4 {
            // The low `free` coordinates random, the others the bits of a random block.
            let block = rng.below(1 << (n - free));
            let mut rho = rng.point(free);
            rho.extend(boolean_point(block, n - free));
            let beta = rng.fp2();
            let base = block << free;
            let expected = (0..1u64 << free).fold(Fp2::ZERO, |sum, u| {
                let (key, value) = table.row(base + u).unwrap();
                sum + support::eq(&rho[..free], u) * (Fp2::from(key) + beta * value)
            });
            assert_eq!(
                table.eval_mle(&rho, beta).unwrap(),
                expected,
                "{:?}, block {block}",
                table.spec().kind
            );
        }
    }
}

#[test]
fn rope_tables_have_no_extension() {
    let kind = TableKind::Rope {
        theta: 10_000,
        factor: 4,
        low_freq_factor: 1,
        high_freq_factor: 2,
        original_max: 32,
        bits: 14,
        max_context: 4,
        d_head: 4,
    };
    let table = Table::generate(TableId(9), kind).unwrap();
    assert_eq!(table.log2_rows(), 3);
    assert_eq!(
        table.eval_mle(&[Fp2::ZERO; 3], Fp2::ONE),
        Err(TableError::NotLookupTable)
    );
}

#[test]
fn a_point_of_another_length_is_refused() {
    let table = Table::generate(TableId(0), TableKind::Range { bits: 4 }).unwrap();
    for found in [0, 3, 5, 40] {
        assert_eq!(
            table.eval_mle(&vec![Fp2::ONE; found], Fp2::ONE),
            Err(TableError::PointLength { expected: 4, found })
        );
    }
}
