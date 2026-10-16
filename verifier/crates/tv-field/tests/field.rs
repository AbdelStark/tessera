// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Fp` and `Fp2` against references that share no code with them: the arbitrary-precision
//! vectors of `spec/vectors/field/` (RFC-0002 'Testing Strategy'), which the prover checks too,
//! and 128-bit integer arithmetic.

use tv_field::{Fp, Fp2, TWO_ADICITY, root_of_unity};
use tv_testkit::{Json, SplitMix64, read_vectors};

const P: u64 = Fp::MODULUS;

/// Values next to the reduction boundaries, over-represented in the randomized tests.
const EDGES: [u64; 13] = [
    0,
    1,
    2,
    7,
    (1 << 32) - 1,
    1 << 32,
    (1 << 32) + 1,
    1 << 63,
    (P - 1) / 2,
    P.div_ceil(2),
    P - (1 << 32),
    P - 2,
    P - 1,
];

/// A canonical value: an edge value one time in eight, otherwise uniform in `[0, P)`.
fn sample(rng: &mut SplitMix64) -> u64 {
    let r = rng.next_u64();
    if r.is_multiple_of(8) {
        return EDGES[(r >> 3) as usize % EDGES.len()];
    }
    loop {
        let x = rng.next_u64();
        if x < P {
            return x;
        }
    }
}

fn element(x: u64) -> Fp {
    Fp::from_canonical(x).expect("a canonical value")
}

fn hex(value: &Json) -> Fp {
    element(value.as_hex_u64().expect("a hex value"))
}

#[test]
fn constants_match_their_definitions() {
    let file = read_vectors("field/constants.json");
    let constants = file.get("goldilocks").expect("the Goldilocks constants");
    let field = |key: &str| {
        constants
            .get(key)
            .unwrap_or_else(|| panic!("missing {key}"))
    };
    assert_eq!(field("modulus").as_hex_u64(), Some(P));
    assert_eq!(field("two_adicity").as_u64(), Some(u64::from(TWO_ADICITY)));

    // The listed factors are prime and, each taken to its full power, multiply to P − 1, so the
    // generator check below covers every maximal subgroup of Fp^*.
    let factors: Vec<u64> = field("p_minus_1_prime_factors")
        .as_array()
        .expect("a list")
        .iter()
        .map(|q| q.as_u64().expect("an integer"))
        .collect();
    let mut rest = P - 1;
    for &q in &factors {
        assert!(
            (2..q).take_while(|d| d * d <= q).all(|d| q % d != 0),
            "{q} is not prime"
        );
        assert_eq!(rest % q, 0, "{q} does not divide P − 1");
        while rest.is_multiple_of(q) {
            rest /= q;
        }
    }
    assert_eq!(rest, 1, "P − 1 has a prime factor missing from the list");
    assert_eq!(1u64 << TWO_ADICITY, (P - 1) & (P - 1).wrapping_neg());

    // 7 generates Fp^*, and is a non-residue, so X^2 − 7 is irreducible.
    assert_eq!(field("multiplicative_generator").as_u64(), Some(7));
    for &q in &factors {
        assert_ne!(
            Fp::SEVEN.pow((P - 1) / q),
            Fp::ONE,
            "7 lies in the subgroup of index {q}"
        );
    }
    assert_eq!(field("extension_non_residue").as_u64(), Some(7));
    assert_eq!(Fp::SEVEN.pow((P - 1) / 2), -Fp::ONE);

    // ROOT_2_K for 0 ≤ k ≤ 32, each of order exactly 2^k.
    let roots = field("root_of_unity_2_k").as_array().expect("a list");
    assert_eq!(roots.len(), TWO_ADICITY as usize + 1);
    for (k, expected) in (0..).zip(roots) {
        let root = root_of_unity(k).expect("k ≤ TWO_ADICITY");
        assert_eq!(root, hex(expected), "ROOT_2_{k}");
        let mut power = root;
        for _ in 1..k {
            power *= power;
        }
        if k > 0 {
            assert_eq!(power, -Fp::ONE, "ROOT_2_{k}^(2^(k−1))");
            power *= power;
        }
        assert_eq!(power, Fp::ONE, "ROOT_2_{k}^(2^k)");
    }
    assert_eq!(root_of_unity(TWO_ADICITY + 1), None);
}

/// The seed a vector file records.
fn seed(file: &Json) -> SplitMix64 {
    SplitMix64::new(file.get("seed").and_then(Json::as_hex_u64).expect("a seed"))
}

/// Every pair `(a, b)` of `values`, `a` in the outer loop.
fn all_pairs<T: Copy>(values: &[T]) -> impl Iterator<Item = (T, T)> + '_ {
    values
        .iter()
        .flat_map(move |&a| values.iter().map(move |&b| (a, b)))
}

/// The vector inputs are pinned, not just the outputs: `generate.py` takes all pairs of the edge
/// values of RFC-0002 'Testing Strategy', then 10,000 pairs drawn from SplitMix64 at the file's
/// seed, redrawing a pair unless both values are below `P`. A file whose inputs were thinned out
/// or replaced fails here even when its outputs are right.
#[test]
fn fp_operations_match_the_vectors() {
    let file = read_vectors("field/goldilocks.json");
    assert_eq!(file.get("modulus").and_then(Json::as_hex_u64), Some(P));
    let edges: Vec<u64> = file
        .get("edge_values")
        .and_then(Json::as_array)
        .expect("edge values")
        .iter()
        .map(|x| x.as_hex_u64().expect("a hex value"))
        .collect();
    assert_eq!(
        edges,
        [0, 1, P - 1, (1 << 32) - 1, 1 << 32, 1 << 63, P - (1 << 32)]
    );
    let mut rng = seed(&file);
    let random = std::iter::repeat_with(|| (rng.next_u64(), rng.next_u64()))
        .filter(|&(a, b)| a < P && b < P)
        .take(10_000);
    let inputs: Vec<(u64, u64)> = all_pairs(&edges).chain(random).collect();

    let cases = file.get("cases").and_then(Json::as_array).expect("cases");
    assert_eq!(cases.len(), inputs.len());
    for (case, &(x, y)) in cases.iter().zip(&inputs) {
        let Some([a, b, sum, difference, product, inverse]) = case.as_array() else {
            panic!("a case is [a, b, a+b, a−b, a·b, a^−1]: {case:?}");
        };
        let (a, b) = (hex(a), hex(b));
        assert_eq!(
            (a, b),
            (element(x), element(y)),
            "inputs differ from the seed's"
        );
        assert_eq!(a + b, hex(sum), "{a:?} + {b:?}");
        assert_eq!(a - b, hex(difference), "{a:?} − {b:?}");
        assert_eq!(a * b, hex(product), "{a:?} · {b:?}");
        let expected = if inverse.is_null() {
            None
        } else {
            Some(hex(inverse))
        };
        assert_eq!(a.inverse(), expected, "{a:?}^−1");
    }
}

/// Pinned inputs again: all pairs of seven edge elements, then 4,096 pairs whose four
/// coefficients are drawn in order from SplitMix64 at the file's seed, each redrawn until it is
/// below `P`.
#[test]
fn fp2_operations_match_the_vectors() {
    let file = read_vectors("field/goldilocks_ext2.json");
    assert_eq!(file.get("modulus").and_then(Json::as_hex_u64), Some(P));
    assert_eq!(file.get("non_residue").and_then(Json::as_u64), Some(7));
    let edges = [
        (0, 0),
        (1, 0),
        (0, 1),
        (P - 1, 0),
        (0, P - 1),
        (P - 1, P - 1),
        (1 << 32, 1 << 63),
    ];
    let mut rng = seed(&file);
    let mut canonical = || loop {
        let x = rng.next_u64();
        if x < P {
            return x;
        }
    };
    let random =
        std::iter::repeat_with(|| ((canonical(), canonical()), (canonical(), canonical())))
            .take(4096);
    let inputs: Vec<((u64, u64), (u64, u64))> = all_pairs(&edges).chain(random).collect();

    let cases = file.get("cases").and_then(Json::as_array).expect("cases");
    assert_eq!(cases.len(), inputs.len());
    for (case, &((x0, x1), (y0, y1))) in cases.iter().zip(&inputs) {
        let Some([a0, a1, b0, b1, p0, p1, i0, i1, norm]) = case.as_array() else {
            panic!("a case is [a0, a1, b0, b1, (a·b)0, (a·b)1, (a^−1)0, (a^−1)1, N(a)]: {case:?}");
        };
        let (a, b) = (Fp2::new(hex(a0), hex(a1)), Fp2::new(hex(b0), hex(b1)));
        let pinned = |c0, c1| Fp2::new(element(c0), element(c1));
        assert_eq!(
            (a, b),
            (pinned(x0, x1), pinned(y0, y1)),
            "inputs differ from the seed's"
        );
        assert_eq!(a * b, Fp2::new(hex(p0), hex(p1)), "{a:?} · {b:?}");
        assert_eq!(a.norm(), hex(norm), "N({a:?})");
        let expected = if i0.is_null() && i1.is_null() {
            None
        } else {
            Some(Fp2::new(hex(i0), hex(i1)))
        };
        assert_eq!(a.inverse(), expected, "{a:?}^−1");
    }
}

/// Every pair of edge values and 2^16 random pairs, against `u128` arithmetic modulo `P`.
#[test]
fn fp_matches_integer_arithmetic() {
    let mut rng = SplitMix64::new(0x7466_6965_6c64_0001);
    let random = (0..1 << 16).map(|_| (sample(&mut rng), sample(&mut rng)));
    let edge_pairs = EDGES
        .iter()
        .flat_map(|&x| EDGES.iter().map(move |&y| (x, y)));
    let reduce = |x: u128| (x % u128::from(P)) as u64;
    for (x, y) in edge_pairs.chain(random) {
        let (a, b) = (element(x), element(y));
        let (wide_x, wide_y) = (u128::from(x), u128::from(y));
        assert_eq!((a + b).to_canonical(), reduce(wide_x + wide_y), "{x} + {y}");
        assert_eq!(
            (a - b).to_canonical(),
            reduce(wide_x + u128::from(P) - wide_y),
            "{x} − {y}"
        );
        assert_eq!((a * b).to_canonical(), reduce(wide_x * wide_y), "{x} · {y}");
        assert_eq!((-a).to_canonical(), reduce(u128::from(P) - wide_x), "−{x}");
        match a.inverse() {
            Some(inverse) => assert_eq!(a * inverse, Fp::ONE, "{x}^−1"),
            None => assert_eq!(x, 0),
        }
        let mut c = a;
        c += b;
        c -= b;
        c *= b;
        assert_eq!(c, a * b, "compound assignment on {x}, {y}");
    }
}

/// The field laws in `Fp2` that the vectors do not pin directly, over 2^16 random triples.
#[test]
fn fp2_is_a_field() {
    let mut rng = SplitMix64::new(0x7466_6965_6c64_0002);
    let mut random = || Fp2::new(element(sample(&mut rng)), element(sample(&mut rng)));
    for _ in 0..1 << 16 {
        let (a, b, c) = (random(), random(), random());
        assert_eq!(a * b, b * a);
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!((a - b) + b, a);
        assert_eq!(a + (-a), Fp2::ZERO);
        assert_eq!(a * Fp2::ONE, a);
        assert_eq!(a * a.conjugate(), Fp2::from(a.norm()));
        assert_eq!((a * b).norm(), a.norm() * b.norm());
        assert_eq!(a * b.c0, a * Fp2::from(b.c0));
        match a.inverse() {
            Some(inverse) => assert_eq!(a * inverse, Fp2::ONE),
            None => assert_eq!(a, Fp2::ZERO),
        }
        let mut d = a;
        d += b;
        d -= c;
        d *= c;
        assert_eq!(d, (a + b - c) * c);
    }
}

/// RFC-0002 'Canonical encoding': decoders reject every value in `[P, 2^64)`, here the endpoints
/// and 10,000 random points, and round-trip every canonical value.
#[test]
fn decoding_accepts_exactly_the_canonical_encodings() {
    let mut rng = SplitMix64::new(0x7466_6965_6c64_0003);
    let random_above = (0..10_000).map(|_| P + rng.next_u64() % (u64::MAX - P));
    for x in [P, P + 1, u64::MAX].into_iter().chain(random_above) {
        assert_eq!(Fp::from_canonical(x), None, "{x:#x}");
        assert_eq!(Fp::from_le_bytes(x.to_le_bytes()), None, "{x:#x}");
        assert_eq!(Fp::reduce(x), element(x - P), "{x:#x} mod P");
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&x.to_le_bytes());
        assert_eq!(Fp2::from_le_bytes(bytes), None, "c0 = {x:#x}");
        bytes = [0u8; 16];
        bytes[8..].copy_from_slice(&x.to_le_bytes());
        assert_eq!(Fp2::from_le_bytes(bytes), None, "c1 = {x:#x}");
    }
    let mut rng = SplitMix64::new(0x7466_6965_6c64_0004);
    for _ in 0..10_000 {
        let (x, y) = (sample(&mut rng), sample(&mut rng));
        let a = Fp::from_le_bytes(x.to_le_bytes()).expect("canonical");
        assert_eq!(a.to_canonical(), x);
        assert_eq!(a.to_le_bytes(), x.to_le_bytes());
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&x.to_le_bytes());
        bytes[8..].copy_from_slice(&y.to_le_bytes());
        let pair = Fp2::from_le_bytes(bytes).expect("canonical");
        assert_eq!(pair, Fp2::new(a, element(y)));
        assert_eq!(pair.to_le_bytes(), bytes);
    }
}
