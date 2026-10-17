// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0008 'Structured evaluation': for every kind of lookup table, at small parameters, the
//! structured evaluation equals the dense one, `Σ_x eq(ρ, x)·(key_x + β·value_x)` over the
//! `2^log2_rows` rows that [`Table::row`] gives, padding included. The points are 100 per table:
//! random ones, and points with coordinates 0 and 1, where a division by `ρ_i` or `1 − ρ_i`
//! would fail. Row counts are powers of two and not, and `Requant`'s parameters put the clamp
//! on either side, on both, on neither and everywhere. At `2^32` rows, where the dense sum is out
//! of reach, the evaluation at a row's own bits is that row.

use tessera_codec::Rational;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_poly::{PolyError, evaluate};
use tessera_tables::{BitOp, GeneratedTable, Table, TableError, TableId, TableKind};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

/// SplitMix64: reproducible points.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fp2(&mut self) -> Fp2 {
        Fp2::new(
            Fp::from_u64_reduced(self.next_u64()),
            Fp::from_u64_reduced(self.next_u64()),
        )
    }
}

fn requant(b: u8, r_by_class: &[u8], lo: i32, hi: i32) -> TableKind {
    TableKind::Requant {
        b,
        r_by_class: r_by_class.to_vec(),
        lo,
        hi,
    }
}

fn silu(b: u8, r_by_class: &[u8], in_exp: i8, out_exp: i8) -> TableKind {
    TableKind::Silu {
        b,
        r_by_class: r_by_class.to_vec(),
        in_exp,
        out_exp,
    }
}

fn exp(key_bits: u8, sigma: u8, num: i64, den: u64) -> TableKind {
    TableKind::Exp {
        key_bits,
        sigma,
        scale: Rational::new(num, den).expect("a scale"),
    }
}

/// The kind's position in `TableKind`, so that a new kind without a case here fails to compile
/// or the coverage check.
fn kind_index(kind: &TableKind) -> usize {
    match kind {
        TableKind::Range { .. } => 0,
        TableKind::Requant { .. } => 1,
        TableKind::Pow2 { .. } => 2,
        TableKind::Bitwise8 { .. } => 3,
        TableKind::AesSbox => 4,
        TableKind::Gf256Mul { .. } => 5,
        TableKind::Exp { .. } => 6,
        TableKind::Exp2Frac { .. } => 7,
        TableKind::Silu { .. } => 8,
        // Not a lookup table: `GeneratedTable` refuses it.
        TableKind::Rope { .. } => 9,
    }
}

fn kinds() -> Vec<TableKind> {
    vec![
        TableKind::Range { bits: 1 },
        TableKind::Range { bits: 13 },
        // INT8 outputs, one class: 2^10 rows.
        requant(10, &[3], -128, 127),
        // Three classes, so 3 · 2^9 rows and padding; clamped on both sides.
        requant(9, &[1, 4, 6], -5, 7),
        // Four classes, unclamped: every value is the rounded shift.
        requant(12, &[2, 2, 5, 11], i32::MIN, i32::MAX),
        // Shifts of b and more: every value 0 before the clamp.
        requant(6, &[6, 7, 63], -3, 3),
        // Clamped below only, then above only, then everywhere to one value.
        requant(11, &[2], 0, i32::MAX),
        requant(11, &[3, 1], i32::MIN, 0),
        requant(8, &[2], 4, 4),
        // Every value below lo, then every value above hi.
        requant(8, &[1], 1000, 2000),
        requant(8, &[1], -2000, -1000),
        // b = 1, the smallest; and b = 16 with two classes.
        requant(1, &[1], -1, 1),
        requant(16, &[4, 9], -128, 127),
        silu(10, &[2], -4, -3),
        // Three classes, of 2^{b − r_c} core values each.
        silu(9, &[1, 3, 8], -6, -4),
        silu(12, &[4, 0, 6, 11], -8, -2),
        // Saturated at 0 below and at 127 above: long runs at both ends of the core.
        silu(11, &[0], 0, -2),
        // A zero tail inside the table, and none.
        exp(10, 16, 7, 3),
        exp(9, 12, 1, 1000),
        // A tail from key 1: only key 0's 2^σ survives.
        exp(8, 0, 2, 1),
        TableKind::Exp2Frac {
            frac_bits: 8,
            out_bits: 20,
        },
        TableKind::Exp2Frac {
            frac_bits: 3,
            out_bits: 56,
        },
        // One row, five rows and 63: no variable, padding, and the largest.
        TableKind::Pow2 { max_exp: 1 },
        TableKind::Pow2 { max_exp: 5 },
        TableKind::Pow2 { max_exp: 63 },
        TableKind::Bitwise8 { op: BitOp::Xor },
        TableKind::Bitwise8 { op: BitOp::And },
        TableKind::AesSbox,
        TableKind::Gf256Mul { c: 0x57 },
        TableKind::Gf256Mul { c: 1 },
    ]
}

/// 100 points of `n` coordinates: the bits of the last row and of the first padding row,
/// coordinates of 0 and 1 among random ones, and random points.
fn points(rng: &mut SplitMix64, n: usize, rows: u64) -> Vec<Vec<Fp2>> {
    let bits = |j: u64| -> Vec<Fp2> {
        (0..n)
            .map(|i| {
                if (j >> i) & 1 == 1 {
                    Fp2::ONE
                } else {
                    Fp2::ZERO
                }
            })
            .collect()
    };
    let mut points = vec![bits(rows - 1), bits(rows % (1 << n))];
    for trial in 0..98 {
        let point = (0..n)
            .map(|i| match (trial + i) % 7 {
                _ if trial >= 20 => rng.fp2(),
                0 => Fp2::ZERO,
                1 => Fp2::ONE,
                _ => rng.fp2(),
            })
            .collect();
        points.push(point);
    }
    points
}

#[test]
fn the_structured_evaluation_is_the_dense_one_for_every_kind() {
    let mut rng = SplitMix64(0x7AB1E);
    let mut covered = [false; 9];
    for (index, kind) in kinds().into_iter().enumerate() {
        covered[kind_index(&kind)] = true;
        let table = GeneratedTable::new(TableId(0), kind).expect("valid parameters");
        let n = usize::from(table.log2_rows());
        let beta = rng.fp2();
        let column: Vec<Fp2> = (0..1u64 << n)
            .map(|x| {
                let (key, value) = table.row(x).expect("a row of the tree");
                Fp2::from(key) + beta * Fp2::from(value)
            })
            .collect();
        for (trial, rho) in points(&mut rng, n, table.row_count()).iter().enumerate() {
            let dense = evaluate(&column, rho).expect("one coordinate per variable");
            let structured = table
                .eval_mle(rho, beta)
                .expect("one coordinate per variable");
            assert_eq!(
                structured,
                dense,
                "table {index} ({:?}), point {trial}",
                table.spec().kind
            );
        }
    }
    assert_eq!(covered, [true; 9], "every kind of lookup table");
}

#[test]
fn a_point_needs_one_coordinate_per_variable() {
    let table = GeneratedTable::new(TableId(0), requant(9, &[1, 4, 6], -5, 7)).expect("valid");
    for found in [10, 12] {
        let rho = vec![Fp2::ONE; found];
        assert_eq!(
            table.eval_mle(&rho, Fp2::ONE),
            Err(TableError::Point(PolyError::PointLength {
                expected: 11,
                found,
            })),
        );
    }
}

#[test]
fn at_two_to_the_32_rows_a_rows_bits_give_that_row() {
    // 2^32 rows of four classes, and 3 · 2^30 rows padded to 2^32.
    for kind in [
        requant(30, &[1, 7, 30, 40], -128, 127),
        requant(30, &[2, 1, 15], -(1 << 19), (1 << 19) - 1),
        TableKind::Range { bits: 32 },
    ] {
        let table = GeneratedTable::new(TableId(0), kind).expect("valid parameters");
        assert_eq!(table.log2_rows(), 32);
        let beta = Fp2::new(Fp::from_u64_reduced(3), Fp::from_u64_reduced(5));
        for j in [
            0,
            1,
            0x1234_5678,
            (1 << 31) - 3,
            0xBFFF_FFFF,
            0xC000_0000,
            u32::MAX.into(),
        ] {
            let rho: Vec<Fp2> = (0..32)
                .map(|i| {
                    if (j >> i) & 1 == 1 {
                        Fp2::ONE
                    } else {
                        Fp2::ZERO
                    }
                })
                .collect();
            let (key, value) = table.row(j).expect("a row");
            let row = Fp2::from(key) + beta * Fp2::from(value);
            assert_eq!(
                table.eval_mle(&rho, beta),
                Ok(row),
                "{:?}, row {j}",
                table.spec().kind
            );
        }
    }
}
