// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0013 'Testing Strategy' 'The per-row degree test' (#565): the independent check of
//! RFC-0013 'Masked extensions' 'Degree''s table, `tessera_compiler::schedule::degree_row`.
//!
//! For each row of the table, each `relation` it names, each public pattern it admits and each
//! constant class, the test builds the relation on the row's tiny masked instance: every
//! committed or virtual read is its masked extension `V' = Ṽ + Z·R̃_V`, with
//! `Z = Π_{i ≥ 2} y_i(1 − y_i)` over the read's own variables `y` and a random mask over all of
//! them; each public table, constant and kernel is random and multilinear over its relation
//! variables, or the RFC's fixed selector or weight. It evaluates every round polynomial of the instance's
//! sumcheck, at random earlier challenges, at 14 nodes, more than the `d + 2` a degree of `d`
//! needs, and reads its degree from their finite differences. It asserts that every round's degree
//! is at most the row's `d`, and that the largest is exactly `d` on a row the table marks tight
//! and exactly the degree its "Tight" cell gives on another.
//!
//! The summands are written here from the RFCs' formulas, never from the prover's relations, so a
//! table row below the true degree fails, which a vector both makers derive from one table
//! cannot catch. The mutation control lowers each tight row's `d` by 1 and asserts that the check
//! fails.

use tessera_compiler::schedule::{
    Constant, DegreeKey, DegreeRelation, DegreeRow, EqualityOp, NativeKind, degree_row,
};
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_ir::BoolFn;

type F = GoldilocksExt2;

/// The nodes each round polynomial is evaluated at, `0..NODES`: their finite differences give
/// any degree up to `NODES − 2 = 12`, beyond the largest row's 10.
const NODES: usize = 14;

/// SplitMix64 (Steele, Lea and Flood, 2014): reproducible inputs, nothing secret.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn element(&mut self) -> F {
        let c0 = Goldilocks::from_u64_reduced(self.next_u64());
        F::new(c0, Goldilocks::from_u64_reduced(self.next_u64()))
    }

    fn elements(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.element()).collect()
    }

    /// A random table on `vars` variables.
    fn table(&mut self, vars: usize) -> Vec<F> {
        self.elements(1 << vars)
    }
}

/// The multilinear extension of `table` at `point`, its lowest variable the index's least
/// significant bit (RFC-0010 'Multilinear extensions'), by folding from the lowest variable.
fn mle(table: &[F], point: &[F]) -> F {
    assert_eq!(
        table.len(),
        1 << point.len(),
        "a table of 2^n entries at n coordinates"
    );
    let mut values = table.to_vec();
    for &x in point {
        values = values
            .chunks_exact(2)
            .map(|pair| pair[0] + x * (pair[1] - pair[0]))
            .collect();
    }
    values[0]
}

/// `eq(a, b) = Π_i (a_i·b_i + (1 − a_i)·(1 − b_i))`.
fn eq(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(F::ONE, |product, (&x, &y)| {
        product * (x * y + (F::ONE - x) * (F::ONE - y))
    })
}

/// A committed or virtual tensor read through its masked extension (RFC-0013 'Masked
/// extensions'): `V'(y) = Ṽ(y) + Z(y)·R̃(y)`, `Z(y) = Π_{i ≥ 2} y_i(1 − y_i)` sparing the read's
/// lowest variable, and the mask over all of its variables.
struct Masked {
    values: Vec<F>,
    mask: Vec<F>,
}

impl Masked {
    fn new(rng: &mut Rng, vars: usize) -> Self {
        Self {
            values: rng.table(vars),
            mask: rng.table(vars),
        }
    }

    fn at(&self, y: &[F]) -> F {
        let z = y[1..]
            .iter()
            .fold(F::ONE, |product, &v| product * v * (F::ONE - v));
        mle(&self.values, y) + z * mle(&self.mask, y)
    }
}

/// An operand read either way: masked, or public, its multilinear extension.
enum Operand {
    Masked(Masked),
    Public(Vec<F>),
}

impl Operand {
    fn new(rng: &mut Rng, vars: usize, public: bool) -> Self {
        if public {
            Self::Public(rng.table(vars))
        } else {
            Self::Masked(Masked::new(rng, vars))
        }
    }

    fn at(&self, y: &[F]) -> F {
        match self {
            Self::Masked(m) => m.at(y),
            Self::Public(t) => mle(t, y),
        }
    }
}

/// A selector's table: 1 at the real entries, 0 at the padded ones.
fn selector(entries: usize, real: impl Fn(usize) -> bool) -> Vec<F> {
    (0..entries)
        .map(|i| if real(i) { F::ONE } else { F::ZERO })
        .collect()
}

/// `x` with the point `tail` appended.
fn with(x: &[F], tail: &[F]) -> Vec<F> {
    [x, tail].concat()
}

/// A summand: the relation's polynomial at a point of the instance's variables.
type Summand = Box<dyn Fn(&[F]) -> F>;

/// A row's tiny instance: its key, its variables and its summand.
struct Case {
    label: String,
    key: DegreeKey,
    vars: usize,
    summand: Summand,
}

impl Case {
    fn new(
        label: impl Into<String>,
        key: DegreeKey,
        vars: usize,
        summand: impl Fn(&[F]) -> F + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            key,
            vars,
            summand: Box::new(summand),
        }
    }

    /// The degree of each round polynomial of the instance's sumcheck, at the earlier challenges
    /// `r`: `g_j(t) = Σ_{b ∈ {0,1}^{n−j−1}} f(r_1, …, r_{j−1}, t, b)`, evaluated at the nodes
    /// `0..NODES` and read off their finite differences.
    fn round_degrees(&self, r: &[F]) -> Vec<usize> {
        let n = self.vars;
        (0..n)
            .map(|j| {
                let values: Vec<F> = (0..NODES as u64)
                    .map(|t| {
                        let node = F::from(Goldilocks::from_u64_reduced(t));
                        (0..1usize << (n - j - 1)).fold(F::ZERO, |sum, b| {
                            let mut point = r[..j].to_vec();
                            point.push(node);
                            point.extend(
                                (0..n - j - 1)
                                    .map(|i| if b >> i & 1 == 1 { F::ONE } else { F::ZERO }),
                            );
                            sum + (self.summand)(&point)
                        })
                    })
                    .collect();
                degree(&values, &self.label, j)
            })
            .collect()
    }
}

/// The degree of a polynomial from its values at `0, 1, …, NODES − 1`: the order of its last
/// non-zero finite difference, 0 for the zero polynomial. The difference of order `NODES − 1`
/// must vanish, so that the degree is certain.
fn degree(values: &[F], label: &str, round: usize) -> usize {
    let mut row = values.to_vec();
    let mut order = 0;
    let mut last = 0;
    loop {
        if row[0] != F::ZERO {
            last = order;
        }
        if row.len() == 1 {
            break;
        }
        row = row.windows(2).map(|w| w[1] - w[0]).collect();
        order += 1;
    }
    assert!(
        last < NODES - 1,
        "{label}, round {round}: a degree beyond the nodes"
    );
    last
}

/// The check RFC-0013 states: every round at most `row.d`, the largest exactly `row.reach`.
fn check(row: DegreeRow, degrees: &[usize]) -> Result<(), String> {
    let largest = degrees.iter().copied().max().unwrap_or(0);
    if degrees.iter().any(|&d| d > row.d) {
        return Err(format!("a round of degree {largest} above d = {}", row.d));
    }
    if largest != row.reach {
        return Err(format!(
            "the largest degree is {largest}, not {}",
            row.reach
        ));
    }
    Ok(())
}

fn key(relation: DegreeRelation, public: &[bool]) -> DegreeKey {
    DegreeKey {
        relation,
        stacked: false,
        public: public.to_vec(),
        constant: None,
    }
}

fn stacked(key: DegreeKey) -> DegreeKey {
    DegreeKey {
        stacked: true,
        ..key
    }
}

/// Every public pattern of `n` operands, none first.
fn patterns(n: usize) -> impl Iterator<Item = Vec<bool>> {
    (0..1u32 << n).map(move |m| (0..n).map(|i| m >> i & 1 == 1).collect())
}

/// RFC-0066's multilinear polynomial of `f` at its inputs' values.
fn bit_fn(f: BoolFn, v: &[F]) -> F {
    let two = F::ONE + F::ONE;
    let four = two * two;
    match f {
        BoolFn::Xor => v[0] + v[1] - two * v[0] * v[1],
        BoolFn::And => v[0] * v[1],
        BoolFn::Ch => v[0] * v[1] + v[2] - v[0] * v[2],
        BoolFn::Xor3 => {
            v[0] + v[1] + v[2] - two * (v[0] * v[1] + v[0] * v[2] + v[1] * v[2])
                + four * v[0] * v[1] * v[2]
        }
        BoolFn::Maj => v[0] * v[1] + v[0] * v[2] + v[1] * v[2] - two * v[0] * v[1] * v[2],
    }
}

/// The selector of 'Testing Strategy''s shape for a selector over several axes: a top axis of 3
/// entries, padded to 4, over a low axis of 2, on 3 variables.
fn padded_rows() -> Vec<F> {
    selector(8, |i| i >> 1 < 3)
}

/// A selector on the last axis alone, of 8 entries, 5 of them real.
fn padded_channels() -> Vec<F> {
    selector(8, |i| i < 5)
}

/// Every row's tiny instances, as 'Testing Strategy' specifies them.
fn cases() -> Vec<Case> {
    use DegreeRelation as R;
    let mut rng = Rng(0x0565_0013);
    let rng = &mut rng;
    let mut cases = Vec::new();

    // The matmuls: each operand has one coordinate more, above the summed axis, a fixed
    // challenge; built unstacked, stacked over one layer variable, and a MatMulAct over two heads
    // with no layer, its head variable stacked as a layer's.
    for relation in [R::MatMulWeight, R::MatMulAct] {
        let (a, b) = (Masked::new(rng, 4), Masked::new(rng, 4));
        let (ca, cb) = (rng.element(), rng.element());
        cases.push(Case::new(
            format!("{relation}, unstacked"),
            key(relation, &[false, false]),
            3,
            move |x| a.at(&with(x, &[ca])) * b.at(&with(x, &[cb])),
        ));
        let stacks: &[&str] = match relation {
            R::MatMulAct => &["a layer", "two heads"],
            _ => &["a layer"],
        };
        for over in stacks {
            let (a, b) = (Masked::new(rng, 5), Masked::new(rng, 5));
            let (ca, cb, rho) = (rng.element(), rng.element(), rng.element());
            cases.push(Case::new(
                format!("{relation}, stacked over {over}"),
                stacked(key(relation, &[false, false])),
                4,
                move |x| {
                    let (i, l) = x.split_at(3);
                    eq(&[rho], l) * a.at(&[i, &[ca], l].concat()) * b.at(&[i, &[cb], l].concat())
                },
            ));
        }
    }

    // `Mul` and `Equality.Mul`: no broadcast operand.
    for public in patterns(2).filter(|p| !(p[0] && p[1])) {
        let (a, b) = (
            Operand::new(rng, 3, public[0]),
            Operand::new(rng, 3, public[1]),
        );
        let z = rng.elements(3);
        cases.push(Case::new(
            format!("Mul, public {public:?}"),
            key(R::Mul, &public),
            3,
            move |x| eq(&z, x) * a.at(x) * b.at(x),
        ));
    }

    // `SumSquares`: `eq` over the top two variables.
    let (x2, z) = (Masked::new(rng, 3), rng.elements(2));
    cases.push(Case::new(
        "SumSquares",
        key(R::SumSquares, &[false]),
        3,
        move |x| eq(&z, &x[1..]) * x2.at(x) * x2.at(x),
    ));

    // `Rotate`: the kernel as `eq`, random over the instance's variables.
    let (v, kernel) = (Masked::new(rng, 3), rng.table(3));
    cases.push(Case::new("Rotate", key(R::Rotate, &[false]), 3, move |x| {
        mle(&kernel, x) * v.at(x)
    }));

    // `Selection`: its input has a fixed challenge below the selected axis and one above.
    let (v, kernel, lo, hi) = (
        Masked::new(rng, 5),
        rng.table(3),
        rng.element(),
        rng.element(),
    );
    cases.push(Case::new(
        "Selection",
        key(R::Selection, &[false]),
        3,
        move |x| mle(&kernel, x) * v.at(&[&[lo], x, &[hi]].concat()),
    ));

    // `ScaleChannels`: a public factor on the last axis alone, of 8 entries.
    let (v, c, z) = (Masked::new(rng, 3), rng.table(3), rng.elements(3));
    cases.push(Case::new(
        "ScaleChannels",
        key(R::ScaleChannels, &[false]),
        3,
        move |x| eq(&z, x) * mle(&c, x) * v.at(x),
    ));

    // `BitFn`: every public pattern with a row.
    for f in [
        BoolFn::Xor,
        BoolFn::And,
        BoolFn::Ch,
        BoolFn::Xor3,
        BoolFn::Maj,
    ] {
        for public in patterns(f.arity()) {
            let k = key(R::BitFn(f), &public);
            if degree_row(&k).is_err() {
                continue;
            }
            let inputs: Vec<Operand> = public.iter().map(|&p| Operand::new(rng, 3, p)).collect();
            let z = rng.elements(3);
            cases.push(Case::new(
                format!("{f:?}, public {public:?}"),
                k,
                3,
                move |x| {
                    let values: Vec<F> = inputs.iter().map(|i| i.at(x)).collect();
                    eq(&z, x) * bit_fn(f, &values)
                },
            ));
        }
    }

    // The equality zero-checks `eq·(Y' − f)` of a committed output `Y`.
    let equality = |op, public: &[bool], constant| DegreeKey {
        constant,
        ..key(R::Equality(op), &[[false].as_slice(), public].concat())
    };
    for x_public in [false, true] {
        // `ConstMul` by a scalar.
        let (y, v, c, z) = (
            Masked::new(rng, 3),
            Operand::new(rng, 3, x_public),
            rng.element(),
            rng.elements(3),
        );
        cases.push(Case::new(
            format!("Equality.ConstMul, scalar, x public {x_public}"),
            equality(EqualityOp::ConstMul, &[x_public], Some(Constant::Scalar)),
            3,
            move |x| eq(&z, x) * (y.at(x) - c * v.at(x)),
        ));
        // `ConstMul` per channel: the constant on the last axis alone, of 8 entries.
        let (y, v, c, z) = (
            Masked::new(rng, 3),
            Operand::new(rng, 3, x_public),
            rng.table(3),
            rng.elements(3),
        );
        cases.push(Case::new(
            format!("Equality.ConstMul, per channel, x public {x_public}"),
            equality(EqualityOp::ConstMul, &[x_public], Some(Constant::Channel)),
            3,
            move |x| eq(&z, x) * (y.at(x) - mle(&c, x) * v.at(x)),
        ));
        // `ConstMul` per layer: two layers, whose constants differ, with layer weights.
        let (y, v, z) = (
            Masked::new(rng, 4),
            Operand::new(rng, 4, x_public),
            rng.elements(4),
        );
        let c = [rng.element(), rng.element()];
        assert!(c[0] != c[1]);
        cases.push(Case::new(
            format!("Equality.ConstMul, per layer, x public {x_public}"),
            stacked(equality(
                EqualityOp::ConstMul,
                &[x_public],
                Some(Constant::Layer),
            )),
            4,
            move |x| eq(&z, x) * (y.at(x) - mle(&c, &x[3..]) * v.at(x)),
        ));
        // `ConstAdd`: its shift `ĉ = c·ψ`, a selector over several axes.
        let (y, v, c, z, psi) = (
            Masked::new(rng, 3),
            Operand::new(rng, 3, x_public),
            rng.element(),
            rng.elements(3),
            padded_rows(),
        );
        cases.push(Case::new(
            format!("Equality.ConstAdd, x public {x_public}"),
            equality(EqualityOp::ConstAdd, &[x_public], None),
            3,
            move |x| eq(&z, x) * (y.at(x) - v.at(x) - c * mle(&psi, x)),
        ));
    }
    for public in patterns(2) {
        for op in [EqualityOp::Add, EqualityOp::Sub, EqualityOp::Mul] {
            let (y, a, b, z) = (
                Masked::new(rng, 3),
                Operand::new(rng, 3, public[0]),
                Operand::new(rng, 3, public[1]),
                rng.elements(3),
            );
            cases.push(Case::new(
                format!("Equality.{op:?}, public {public:?}"),
                equality(op, &public, None),
                3,
                move |x| {
                    let f = match op {
                        EqualityOp::Add => a.at(x) + b.at(x),
                        EqualityOp::Sub => a.at(x) - b.at(x),
                        _ => a.at(x) * b.at(x),
                    };
                    eq(&z, x) * (y.at(x) - f)
                },
            ));
        }
    }

    // `Booleanity`: `eq·H'·(1 − H')`.
    let (h, z) = (Masked::new(rng, 3), rng.elements(3));
    cases.push(Case::new(
        "Booleanity",
        key(R::Booleanity, &[false]),
        3,
        move |x| eq(&z, x) * h.at(x) * (F::ONE - h.at(x)),
    ));

    // The padding checks `eq·(1 − ψ)·V'`: on the last axis alone, or over several axes.
    for (relation, psi) in [
        (R::OneHotVocabulary, padded_channels()),
        (R::BitPadding, padded_channels()),
        (R::InputPadding, padded_rows()),
        (R::Native(NativeKind::RootPad), padded_rows()),
        (R::Native(NativeKind::ResidualPad), padded_rows()),
        (R::Native(NativeKind::MaxPad), padded_rows()),
        (R::Native(NativeKind::ReciprocalPad), padded_rows()),
    ] {
        let public: &[bool] = if matches!(relation, R::Native(_)) {
            &[]
        } else {
            &[false]
        };
        let (v, z) = (Masked::new(rng, 3), rng.elements(3));
        cases.push(Case::new(
            relation.to_string(),
            key(relation, public),
            3,
            move |x| eq(&z, x) * (F::ONE - mle(&psi, x)) * v.at(x),
        ));
    }

    // `OneHotIndex`, `OneHotCount` and RFC-0053's relations: a vocabulary axis of two variables
    // under a token or time axis of two; their `eq` on the token axis, their weights on the
    // vocabulary axis.
    for token_public in [false, true] {
        let (h, token, v, w, z) = (
            Masked::new(rng, 4),
            Operand::new(rng, 2, token_public),
            rng.table(2),
            rng.element(),
            rng.elements(2),
        );
        cases.push(Case::new(
            format!("OneHotIndex, token public {token_public}"),
            key(R::OneHotIndex, &[false, token_public]),
            4,
            move |x| eq(&z, &x[2..]) * (mle(&v, &x[..2]) * h.at(x) - w * token.at(&x[2..])),
        ));
    }
    let (h, rows, w, z) = (
        Masked::new(rng, 4),
        rng.table(2),
        rng.element(),
        rng.elements(2),
    );
    cases.push(Case::new(
        "OneHotCount",
        key(R::OneHotCount, &[false]),
        4,
        move |x| eq(&z, &x[2..]) * (h.at(x) - w * mle(&rows, &x[2..])),
    ));
    let (h, v, z) = (Masked::new(rng, 4), rng.table(2), rng.elements(2));
    cases.push(Case::new(
        "Tokens",
        key(R::Native(NativeKind::Tokens), &[]),
        4,
        move |x| eq(&z, &x[2..]) * mle(&v, &x[..2]) * h.at(x),
    ));
    let (l, time, vocabulary) = (Masked::new(rng, 4), rng.table(2), rng.table(2));
    cases.push(Case::new(
        "GreedyKey",
        key(R::Native(NativeKind::GreedyKey), &[]),
        4,
        move |x| mle(&time, &x[2..]) * mle(&vocabulary, &x[..2]) * l.at(x),
    ));
    // `L·Hplus` with its coefficients: `Hplus = Σ_j c_j(t)·H'(v, j)` reads `H` at Boolean time
    // coordinates `j`, its coefficients on the time variables it does not read.
    let (l, h, a) = (Masked::new(rng, 4), Masked::new(rng, 4), rng.table(4));
    let c: Vec<Vec<F>> = (0..4).map(|_| rng.table(2)).collect();
    cases.push(Case::new(
        "GreedySelected",
        key(R::Native(NativeKind::GreedySelected), &[]),
        4,
        move |x| {
            let hplus = c.iter().enumerate().fold(F::ZERO, |sum, (j, cj)| {
                let bit = |i: usize| if j >> i & 1 == 1 { F::ONE } else { F::ZERO };
                sum + mle(cj, &x[2..]) * h.at(&[x[0], x[1], bit(0), bit(1)])
            });
            mle(&a, x) * l.at(x) * hplus
        },
    ));
    // An observed request's logits: a zero summand.
    cases.push(Case::new(
        "Observe",
        key(R::Native(NativeKind::Observe), &[]),
        4,
        |_| F::ZERO,
    ));

    // `AssertEqual`: `eq·ψ·(a' − b')`, a selector over several axes.
    for public in patterns(2) {
        let (a, b, z, psi) = (
            Operand::new(rng, 3, public[0]),
            Operand::new(rng, 3, public[1]),
            rng.elements(3),
            padded_rows(),
        );
        cases.push(Case::new(
            format!("AssertEqual, public {public:?}"),
            key(R::AssertEqual, &public),
            3,
            move |x| eq(&z, x) * mle(&psi, x) * (a.at(x) - b.at(x)),
        ));
    }

    // `Recomposition`: `low = 2`, `high = 4`, so `k_b = 1`, `k_h = 2` and `k = 2` bit variables,
    // the lowest, under `n_y = 2`; RFC-0072's exact weights and zero-extensions. `b` is read
    // at the high `k_b` bit coordinates, `h` at both, `x` at `y` alone.
    for x_public in [false, true] {
        let (v, b, h, wb, wh, z, s) = (
            Operand::new(rng, 2, x_public),
            Masked::new(rng, 3),
            Masked::new(rng, 4),
            [1, 0, 2, 0]
                .map(|v| F::from(Goldilocks::from_u64_reduced(v)))
                .to_vec(),
            [1, 2, 4, 8]
                .map(|v| F::from(Goldilocks::from_u64_reduced(v)))
                .to_vec(),
            rng.elements(2),
            F::from(Goldilocks::from_u64_reduced(4)).inverse().unwrap(),
        );
        cases.push(Case::new(
            format!("Recomposition, x public {x_public}"),
            key(R::Recomposition, &[x_public, false, false]),
            4,
            move |x| {
                let (i, y) = x.split_at(2);
                eq(&z, y)
                    * (s * v.at(y)
                        - mle(&wb, i) * (F::ONE - i[0]) * b.at(&with(&i[1..], y))
                        - F::from(Goldilocks::from_u64_reduced(4)) * mle(&wh, i) * h.at(x))
            },
        ));
    }

    // RFC-0050's root coefficient product, `eq` times three committed digits.
    let (a, b, c, z) = (
        Masked::new(rng, 3),
        Masked::new(rng, 3),
        Masked::new(rng, 3),
        rng.elements(3),
    );
    cases.push(Case::new(
        "Product",
        key(R::Native(NativeKind::Product), &[]),
        3,
        move |x| eq(&z, x) * a.at(x) * b.at(x) * c.at(x),
    ));

    // `SoftmaxRow`'s key and forest relations: the causal selector over one key variable under two
    // query variables, `[key ≤ query]`.
    for kind in [NativeKind::Key, NativeKind::Forest] {
        let (v, z) = (Masked::new(rng, 3), rng.elements(3));
        let causal = selector(8, |i| (i & 1) <= i >> 1);
        cases.push(Case::new(
            format!("{kind:?}"),
            key(R::Native(kind), &[]),
            3,
            move |x| eq(&z, x) * mle(&causal, x) * v.at(x),
        ));
    }
    let (a, b, z) = (Masked::new(rng, 3), Masked::new(rng, 3), rng.elements(3));
    cases.push(Case::new(
        "Reciprocal",
        key(R::Native(NativeKind::Reciprocal), &[]),
        3,
        move |x| eq(&z, x) * a.at(x) * b.at(x),
    ));

    // A fold: `Z(p)`, a constant, times `eq(p_col ‖ 0, w)` times the mask row, read unmasked.
    let (row, q, zp) = (rng.table(3), rng.elements(3), rng.element());
    cases.push(Case::new("Fold", key(R::Fold, &[]), 3, move |x| {
        zp * eq(&q, x) * mle(&row, x)
    }));
    cases
}

/// Each case's row and round degrees, at one draw of earlier challenges.
fn measured() -> Vec<(Case, DegreeRow, Vec<usize>)> {
    let mut rng = Rng(0x0565_0002);
    cases()
        .into_iter()
        .map(|case| {
            let row = degree_row(&case.key)
                .unwrap_or_else(|e| panic!("{}: the table has a row: {e}", case.label));
            let r = rng.elements(case.vars);
            let degrees = case.round_degrees(&r);
            (case, row, degrees)
        })
        .collect()
}

/// Every round within the row's `d`; the largest exactly `d` on a tight row, and the "Tight"
/// cell's degree on another.
#[test]
fn every_row_bounds_its_tiny_instance_and_tight_rows_reach_it() {
    let measured = measured();
    for (case, row, degrees) in &measured {
        check(*row, degrees).unwrap_or_else(|e| {
            panic!(
                "{} (d_1 {}, s {}, d {}): {e}; rounds {degrees:?}",
                case.label, row.d_1, row.s, row.d
            )
        });
    }
    // Every relation string of the table is built, each Bit function and equality operation,
    // and the matmul row three ways.
    let names: std::collections::BTreeSet<&str> = measured
        .iter()
        .map(|(c, _, _)| c.key.relation.name())
        .collect();
    assert_eq!(names.len(), 38, "{names:?}");
    // The two rows that reach 10 are the largest.
    let ten: Vec<&str> = measured
        .iter()
        .filter(|(_, row, _)| row.d == 10)
        .map(|(c, _, _)| c.key.relation.name())
        .collect();
    assert_eq!(ten, ["Xor3", "Maj", "Product"]);
}

/// The mutation control: lowering any tight row's `d` by 1 fails the check. A row that is not
/// tight keeps the rule's count, which the test bounds but cannot pin.
#[test]
fn a_tight_row_lowered_by_one_fails() {
    let mut tight = 0;
    for (case, row, degrees) in measured() {
        if !row.is_tight() {
            continue;
        }
        let lowered = DegreeRow {
            d: row.d - 1,
            reach: row.reach - 1,
            ..row
        };
        assert!(
            check(lowered, &degrees).is_err(),
            "{}: a table lowered by 1 passes",
            case.label
        );
        tight += 1;
    }
    assert!(tight > 50, "{tight} tight cases");
}

/// The test's own degree reader: a polynomial of degree `e` reads as `e`, at every `e` it
/// certifies, and the zero polynomial as 0.
#[test]
fn the_degree_reader_reads_every_degree() {
    let mut rng = Rng(0x0565_0003);
    for e in 0..NODES - 1 {
        let mut coefficients = rng.elements(e + 1);
        while coefficients[e] == F::ZERO {
            coefficients[e] = rng.element();
        }
        let values: Vec<F> = (0..NODES as u64)
            .map(|t| {
                let x = F::from(Goldilocks::from_u64_reduced(t));
                coefficients
                    .iter()
                    .rev()
                    .fold(F::ZERO, |acc, &c| acc * x + c)
            })
            .collect();
        assert_eq!(degree(&values, "a polynomial", 0), e);
    }
    assert_eq!(degree(&[F::ZERO; NODES], "zero", 0), 0);
}
