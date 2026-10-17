// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's tables against the verifier's, `tv-tables`, written separately from RFC-0008
//! (SP-6), on the same TCE bytes: encoded here, decoded by the verifier's codec. For every
//! lookup kind, at the vectors' parameters and at seeded ones, both regenerate the same
//! specification and digest and the same rows, and their structured evaluations agree at 1,000
//! seeded points per kind, some coordinates 0 or 1. Both implementations compute RoPE's
//! frequencies and pairs alike, and accept and refuse the same seeded parameters. The vectors
//! pin chosen cases; this covers the shapes between them. Prover crates may use verifier crates
//! as dev-dependencies only (DEP-2).

use tessera_codec::{Rational, encode};
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_tables::{
    BitOp, GeneratedTable, RopeConstants, RopeTable, Table, TableId, TableKind, TableSpec,
};
use tv_codec::{DecodeLimits, decode};
use tv_field::{Fp, Fp2};

/// Evaluation points per lookup kind, across its parameter sets (issue #100).
const POINTS_PER_KIND: usize = 1_000;

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A canonical field element, as a `u64` below `P`.
    fn canonical(&mut self) -> u64 {
        self.next_u64() % Goldilocks::MODULUS
    }
}

/// The verifier's reading of a prover specification: the prover's TCE bytes, decoded by
/// `tv-codec`.
fn verifier_spec(spec: &TableSpec) -> tv_tables::TableSpec {
    let bytes = encode(spec);
    let decoded = decode(&bytes, &DecodeLimits::MESSAGE, tv_tables::TableSpec::decode)
        .unwrap_or_else(|e| panic!("{spec:?}: {e}"));
    assert_eq!(decoded.to_tce(), bytes, "{spec:?}");
    decoded
}

/// The verifier's kind for a prover kind, through a specification of no particular rows.
fn verifier_kind(kind: &TableKind) -> tv_tables::TableKind {
    verifier_spec(&TableSpec {
        id: TableId(0),
        kind: kind.clone(),
        log2_rows: 0,
        digest: [0; 32],
    })
    .kind
}

fn ext(c0: u64, c1: u64) -> (GoldilocksExt2, Fp2) {
    let prover = GoldilocksExt2::new(
        Goldilocks::from_canonical_u64(c0).unwrap(),
        Goldilocks::from_canonical_u64(c1).unwrap(),
    );
    let verifier = Fp2::new(
        Fp::from_canonical(c0).unwrap(),
        Fp::from_canonical(c1).unwrap(),
    );
    (prover, verifier)
}

fn same(prover: GoldilocksExt2, verifier: Fp2) -> bool {
    let [c0, c1] = prover.coeffs();
    c0.to_canonical_u64() == verifier.c0.to_canonical()
        && c1.to_canonical_u64() == verifier.c1.to_canonical()
}

/// A point of `n` coordinates on both sides, one in eight 0 and one in eight 1.
fn point(rng: &mut SplitMix64, n: usize) -> (Vec<GoldilocksExt2>, Vec<Fp2>) {
    (0..n)
        .map(|_| match rng.below(8) {
            0 => ext(0, 0),
            1 => ext(1, 0),
            _ => ext(rng.canonical(), rng.canonical()),
        })
        .unzip()
}

fn rational(num: i64, den: u64) -> Rational {
    Rational::new(num, den).unwrap()
}

/// Both tables of `kind` under `id`, after checking that they regenerate the same
/// specification, digest included.
fn both(id: u32, kind: &TableKind) -> (GeneratedTable, tv_tables::Table) {
    let prover = GeneratedTable::new(TableId(id), kind.clone())
        .unwrap_or_else(|e| panic!("prover, {kind:?}: {e}"));
    let spec = verifier_spec(prover.spec());
    let verifier =
        tv_tables::Table::from_spec(&spec).unwrap_or_else(|e| panic!("verifier, {kind:?}: {e}"));
    assert_eq!(verifier.spec().to_tce(), encode(prover.spec()), "{kind:?}");
    assert_eq!(verifier.rows(), prover.row_count(), "{kind:?}");
    assert_eq!(verifier.log2_rows(), prover.log2_rows(), "{kind:?}");
    (prover, verifier)
}

/// The rows at the edges of the padded table and at seeded indices agree.
fn compare_rows(rng: &mut SplitMix64, prover: &GeneratedTable, verifier: &tv_tables::Table) {
    let padded = 1u64 << prover.log2_rows();
    let rows = prover.row_count();
    let mut indices: Vec<u64> = (0..padded.min(64)).collect();
    indices.extend([rows.saturating_sub(1), rows, padded - 1, padded]);
    indices.extend((0..64).map(|_| rng.below(padded)));
    for index in indices {
        let expected = prover
            .row(index)
            .map(|(k, v)| (k.to_canonical_u64(), v.to_canonical_u64()));
        let found = verifier
            .row(index)
            .map(|(k, v)| (k.to_canonical(), v.to_canonical()));
        assert_eq!(found, expected, "{:?}, row {index}", prover.spec().kind);
    }
}

/// Seeded parameter sets of one lookup kind, small enough to evaluate quickly.
fn seeded(rng: &mut SplitMix64, family: usize) -> TableKind {
    // One to four classes, each shift drawn from `low..=high`.
    let shifts = |rng: &mut SplitMix64, low: u64, high: u64| -> Vec<u8> {
        let classes = 1 + rng.below(4);
        (0..classes)
            .map(|_| (low + rng.below(high - low + 1)) as u8)
            .collect()
    };
    match family {
        0 => TableKind::Range {
            bits: 1 + rng.below(32) as u8,
        },
        1 => {
            let b = 1 + rng.below(30) as u8;
            let r_by_class = shifts(rng, 1, 63);
            let class_bits = [0, 0, 1, 2, 2][r_by_class.len()];
            let b = b.min(32 - class_bits);
            let (x, y) = (
                rng.next_u64() as i32 >> rng.below(31),
                rng.next_u64() as i32 >> rng.below(31),
            );
            TableKind::Requant {
                b,
                r_by_class,
                lo: x.min(y),
                hi: x.max(y),
            }
        }
        2 => TableKind::Pow2 {
            max_exp: 1 + rng.below(63) as u8,
        },
        3 => TableKind::Bitwise8 {
            op: if rng.below(2) == 0 {
                BitOp::Xor
            } else {
                BitOp::And
            },
        },
        4 => TableKind::AesSbox,
        5 => TableKind::Gf256Mul {
            c: rng.below(256) as u8,
        },
        6 => {
            let key_bits = 1 + rng.below(16) as u8;
            let sigma = rng.below(33) as u8;
            // A tail anywhere from key 1 to past the table: scale ≈ (σ + 2) ln 2 / target.
            let target = 1 + rng.below(1 << (key_bits + 1));
            let den = 1 + rng.below(1 << 30);
            let num = ((u128::from(sigma) + 2) * 0xB172_17F7_D1CF_79AB * u128::from(den)
                / (u128::from(target) << 64))
                .max(1)
                + u128::from(rng.below(2));
            let gcd = {
                let (mut a, mut b) = (num, u128::from(den));
                while b != 0 {
                    (a, b) = (b, a % b);
                }
                a
            };
            TableKind::Exp {
                key_bits,
                sigma,
                scale: rational((num / gcd) as i64, (u128::from(den) / gcd) as u64),
            }
        }
        7 => TableKind::Exp2Frac {
            frac_bits: 1 + rng.below(12) as u8,
            out_bits: 1 + rng.below(56) as u8,
        },
        _ => {
            let b = 1 + rng.below(14) as u8;
            let r_by_class = shifts(rng, 0, u64::from(b) - 1);
            let in_exp = -(rng.below(65) as i8);
            // s = 63 − in_exp + out_exp in 1..=126.
            let s = 1 + rng.below(126) as i16;
            let out_exp = (s - 63 + i16::from(in_exp)) as i8;
            TableKind::Silu {
                b,
                r_by_class,
                in_exp,
                out_exp,
            }
        }
    }
}

/// The vectors' parameters and production-sized tables of each lookup kind.
fn pinned(family: usize) -> Vec<TableKind> {
    let requant = |b: u8, r: &[u8], lo: i32, hi: i32| TableKind::Requant {
        b,
        r_by_class: r.to_vec(),
        lo,
        hi,
    };
    let silu = |b: u8, r: &[u8], in_exp: i8, out_exp: i8| TableKind::Silu {
        b,
        r_by_class: r.to_vec(),
        in_exp,
        out_exp,
    };
    let exp = |key_bits: u8, sigma: u8, num: i64, den: u64| TableKind::Exp {
        key_bits,
        sigma,
        scale: rational(num, den),
    };
    match family {
        0 => vec![
            TableKind::Range { bits: 1 },
            TableKind::Range { bits: 8 },
            TableKind::Range { bits: 32 },
        ],
        1 => vec![
            requant(4, &[1], -128, 127),
            requant(5, &[1, 2, 3], -2, 3),
            requant(8, &[1, 2, 7, 8], -128, 127),
            requant(3, &[63], -1, 1),
            requant(4, &[2], 5, 5),
            requant(31, &[1, 40], i32::MIN, i32::MAX),
            requant(30, &[1, 2, 3], -524_288, 524_287),
        ],
        2 => vec![
            TableKind::Pow2 { max_exp: 1 },
            TableKind::Pow2 { max_exp: 5 },
            TableKind::Pow2 { max_exp: 63 },
        ],
        3 => vec![
            TableKind::Bitwise8 { op: BitOp::Xor },
            TableKind::Bitwise8 { op: BitOp::And },
        ],
        4 => vec![TableKind::AesSbox],
        5 => [0, 2, 3, 0x57].map(|c| TableKind::Gf256Mul { c }).to_vec(),
        6 => vec![
            exp(6, 8, 1, 4),
            exp(4, 16, 1, 1000),
            exp(8, 16, 1, 1),
            exp(10, 0, 3, 7),
            exp(22, 16, 1, 2896),
            exp(22, 16, 1, 11585),
        ],
        7 => vec![
            TableKind::Exp2Frac {
                frac_bits: 8,
                out_bits: 16,
            },
            TableKind::Exp2Frac {
                frac_bits: 1,
                out_bits: 1,
            },
            TableKind::Exp2Frac {
                frac_bits: 16,
                out_bits: 56,
            },
        ],
        _ => vec![
            silu(6, &[0, 2], -3, -4),
            silu(8, &[1], 0, -2),
            silu(4, &[0, 1, 2, 3], -64, -62),
            silu(30, &[14, 15, 16, 17], -10, -4),
            // The vectors' `silu-b8-wide-arguments`: `h = ±89` has `n = 128`, where `E` is 0 by its rule.
            silu(8, &[0], 0, 0),
        ],
    }
}

/// Families of lookup kinds, in declaration order without `Rope`.
const FAMILIES: [&str; 9] = [
    "Range", "Requant", "Pow2", "Bitwise8", "AesSbox", "Gf256Mul", "Exp", "Exp2Frac", "Silu",
];

#[test]
fn both_implementations_agree_on_every_lookup_kind() {
    let mut rng = SplitMix64(0x7ab1_e5d1_ff00);
    for (family, name) in FAMILIES.iter().enumerate() {
        let mut kinds = pinned(family);
        kinds.extend((0..6).map(|_| seeded(&mut rng, family)));
        let tables: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(i, kind)| both(i as u32, kind))
            .collect();
        for (prover, verifier) in &tables {
            compare_rows(&mut rng, prover, verifier);
        }
        for i in 0..POINTS_PER_KIND {
            let (prover, verifier) = &tables[i % tables.len()];
            let (rho_p, rho_v) = point(&mut rng, usize::from(prover.log2_rows()));
            let (beta_p, beta_v) = ext(rng.canonical(), rng.canonical());
            let expected = prover.eval_mle(&rho_p, beta_p).unwrap();
            let found = verifier.eval_mle(&rho_v, beta_v).unwrap();
            assert!(
                same(expected, found),
                "{name}, {:?}, point {i}: prover {expected:?}, verifier {found:?}",
                prover.spec().kind
            );
        }
    }
}

#[test]
fn both_implementations_agree_on_rope_constants() {
    let mut rng = SplitMix64(0x2093);
    let rope =
        |theta, factor, low, high, original_max, bits, max_context, d_head| TableKind::Rope {
            theta,
            factor,
            low_freq_factor: low,
            high_freq_factor: high,
            original_max,
            bits,
            max_context,
            d_head,
        };
    let kinds = [
        rope(10_000, 4, 1, 2, 32, 14, 16, 16),
        rope(10_000, 1, 1, 4, 64, 14, 12, 6),
        rope(500_000, 8, 1, 4, 8192, 14, 8192, 128),
        rope(2, 1, 1, 2, 1, 1, 3, 2),
        rope(u32::MAX, u16::MAX, 1, u16::MAX, u32::MAX, 30, 1 << 10, 64),
        rope(1_000_000, 32, 3, 9, 4096, 20, 256, 96),
    ];
    for (i, kind) in kinds.iter().enumerate() {
        let prover = RopeTable::new(TableId(i as u32), kind.clone())
            .unwrap_or_else(|e| panic!("prover, {kind:?}: {e}"));
        let verifier = tv_tables::Table::from_spec(&verifier_spec(prover.spec()))
            .unwrap_or_else(|e| panic!("verifier, {kind:?}: {e}"));
        assert_eq!(verifier.spec().to_tce(), encode(prover.spec()), "{kind:?}");
        let constants = prover.constants();
        let ours = verifier.rope().unwrap();
        assert_eq!(ours.frequencies(), constants.frequencies(), "{kind:?}");
        let half = constants.frequencies().len() as u64;
        for _ in 0..1_000 {
            let position = rng.below(u64::from(ours.positions()) + 1) as u32;
            let frequency = rng.below(half + 1) as u32;
            assert_eq!(
                ours.pair(position, frequency as usize),
                constants.at(position, frequency),
                "{kind:?}, position {position}, frequency {frequency}"
            );
        }
        // The constants without the digest are the table's.
        let free = RopeConstants::new(kind).unwrap();
        assert_eq!(free.frequencies(), ours.frequencies());
    }
}

/// Seeded parameters around every rule of RFC-0008 'Table specifications', each accepted by
/// both implementations or refused by both. The draws keep accepted tables small, so that both
/// generate them quickly: a large `b`, a slow `Exp` scale or many RoPE positions come with
/// parameters that a rule refuses before any core is built.
#[test]
fn both_implementations_accept_the_same_parameters() {
    let mut rng = SplitMix64(0xacce_97ed);
    let pick =
        |rng: &mut SplitMix64, values: &[i64]| values[rng.below(values.len() as u64) as usize];
    let mut accepted = 0;
    let mut refused = 0;
    for _ in 0..3_000 {
        let kind = match rng.below(7) {
            0 => TableKind::Range {
                bits: pick(&mut rng, &[0, 1, 2, 17, 32, 33, 64, 255]) as u8,
            },
            1 => {
                let classes = pick(&mut rng, &[0, 1, 2, 3, 4, 5]);
                TableKind::Requant {
                    b: pick(&mut rng, &[0, 1, 2, 9, 29, 30, 31, 32, 33]) as u8,
                    r_by_class: (0..classes)
                        .map(|_| pick(&mut rng, &[0, 1, 2, 31, 62, 63, 64, 255]) as u8)
                        .collect(),
                    lo: pick(&mut rng, &[-2, -1, 3]) as i32,
                    hi: pick(&mut rng, &[-2, -1, 3]) as i32,
                }
            }
            2 => TableKind::Pow2 {
                max_exp: pick(&mut rng, &[0, 1, 2, 62, 63, 64, 255]) as u8,
            },
            3 => {
                let (num, den, widths): (i64, u64, &[i64]) = match rng.below(4) {
                    0 => (0, 1, &[0, 1, 8, 32, 33]),
                    1 => (-3, 7, &[0, 1, 8, 32, 33]),
                    // A tail at key 1: accepted at any width.
                    2 => (5, 1, &[0, 1, 8, 16, 24, 25, 32, 33]),
                    // No tail before 2^25 keys: refused from 25 key bits on.
                    _ => (1, 1 << 40, &[0, 1, 8, 25, 32, 33]),
                };
                TableKind::Exp {
                    key_bits: pick(&mut rng, widths) as u8,
                    sigma: pick(&mut rng, &[0, 1, 16, 32, 33, 255]) as u8,
                    scale: rational(num, den),
                }
            }
            4 => TableKind::Exp2Frac {
                frac_bits: pick(&mut rng, &[0, 1, 8, 12, 17]) as u8,
                out_bits: pick(&mut rng, &[0, 1, 56, 57]) as u8,
            },
            5 => {
                // A large `b` meets shifts that leave a core above 2^24, or no core at all.
                let (widths, shifts): (&[i64], &[i64]) = if rng.below(2) == 0 {
                    (&[0, 1, 4, 12], &[0, 1, 3, 11, 12, 40])
                } else {
                    (&[26, 31, 32, 33], &[0, 1, 31, 32, 40])
                };
                let classes = pick(&mut rng, &[0, 1, 2, 3, 4, 5]);
                TableKind::Silu {
                    b: pick(&mut rng, widths) as u8,
                    r_by_class: (0..classes).map(|_| pick(&mut rng, shifts) as u8).collect(),
                    in_exp: pick(&mut rng, &[1, 0, -1, -63, -64, -65]) as i8,
                    out_exp: pick(&mut rng, &[-63, -62, -2, 0, 1, 62, 63, 65]) as i8,
                }
            }
            _ => {
                let (max_context, d_head) = match rng.below(3) {
                    0 => (
                        pick(&mut rng, &[0, 1, 64]),
                        pick(&mut rng, &[0, 1, 2, 4, 7, 128]),
                    ),
                    // More than 2^24 pairs.
                    1 => (1 << 23, 6),
                    _ => (i64::from(u32::MAX), pick(&mut rng, &[2, 4])),
                };
                TableKind::Rope {
                    theta: pick(&mut rng, &[0, 1, 2, 10_000]) as u32,
                    factor: pick(&mut rng, &[0, 1, 8]) as u16,
                    low_freq_factor: pick(&mut rng, &[0, 1, 4]) as u16,
                    high_freq_factor: pick(&mut rng, &[0, 1, 4, 5]) as u16,
                    original_max: pick(&mut rng, &[0, 1, 8192]) as u32,
                    bits: pick(&mut rng, &[0, 1, 14, 30, 31]) as u8,
                    max_context: max_context as u32,
                    d_head: d_head as u16,
                }
            }
        };
        let prover = match &kind {
            TableKind::Rope { .. } => {
                RopeTable::new(TableId(1), kind.clone()).map(|t| t.spec().clone())
            }
            _ => GeneratedTable::new(TableId(1), kind.clone()).map(|t| t.spec().clone()),
        };
        let verifier = tv_tables::Table::generate(tv_tables::TableId(1), verifier_kind(&kind));
        match (&prover, &verifier) {
            (Ok(spec), Ok(table)) => {
                assert_eq!(table.spec().to_tce(), encode(spec), "{kind:?}");
                accepted += 1;
            }
            (Err(_), Err(error)) => {
                assert!(error.is_rejection(), "{kind:?}: {error}");
                refused += 1;
            }
            _ => panic!(
                "{kind:?}: the prover {} and the verifier {}",
                if prover.is_ok() { "accepts" } else { "refuses" },
                if verifier.is_ok() {
                    "accepts"
                } else {
                    "refuses"
                },
            ),
        }
    }
    assert!(
        accepted > 300 && refused > 300,
        "{accepted} accepted, {refused} refused"
    );
}
