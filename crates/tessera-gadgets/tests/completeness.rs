// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Completeness (RFC-0009 'Testing Strategy', RFC-0024 'Testing Strategy'): on 10^4 seeded keys
//! per gadget, both ends of the key's range among them, at small widths and at production ones:
//!
//! - the honest witness returns RFC-0005's `rq(z, r_c, lo, hi)` (Lemmas 1, 2 and 11), and
//!   `NonNeg`'s limbs are its value's digits (Lemma 3);
//! - every lookup's tuple, computed from the constraints, is a row of its table;
//! - every virtual value is the integer its RFC defines;
//! - at small widths, where the tables fit in memory, the LogUp instances built from the witness
//!   verify through the independent verifier, whose leaf claims are the prover's.
//!
//! Production widths run the gadgets the compiler selects at `B_max = 32`: a key of 32 bits, the
//! splits of 48 and 62 bits, and `RequantWide` for `minimal`'s 8B residual (RFC-0024 'At Llama's
//! shapes'), with one class and with four. At 62 bits, three classes split and four take
//! `RequantWide`, since `4·2^62 > P` (RFC-0009 Lemma 2).

mod common;

use common::{
    Fp, Messages, Session, Shape, SplitMix64, in_table, keys, prove, rq, select_requant, spec_of,
    table_columns, tv_fp2, verify,
};
use tessera_compiler::Gadget;
use tessera_field::PrimeField;
use tessera_gadgets::{Committed, Constraints, Lookups, NonNeg, Requant, Term, Virtual, Witness};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind};

/// Keys per gadget (RFC-0009 'Testing Strategy').
const KEYS: usize = 10_000;

/// The honest assignment at element `i`.
fn honest<'a>(
    keys: &'a [i64],
    classes: Option<&'a [u8]>,
    witness: &'a Witness,
    i: usize,
) -> impl FnMut(Term) -> Fp + 'a {
    move |term| match term {
        Term::Input => Fp::from_i64(keys[i]),
        Term::Class => Fp::from_u64_reduced(u64::from(classes.map_or(0, |c| c[i]))),
        Term::Committed(tensor) => Fp::from_i64(witness.tensor(tensor).expect("a tensor")[i]),
    }
}

/// Every lookup's tuple, as `Lookups` computes it, is a row of its table.
fn tuples_are_rows(lookups: &Lookups<Fp>, tables: &[&GeneratedTable]) {
    for group in lookups.groups() {
        let table = tables
            .iter()
            .find(|t| t.spec().id == group.table())
            .expect("a table of the gadget");
        for (j, &key) in group.keys().iter().enumerate() {
            let value = group.values().map(|v| v[j]);
            assert!(
                in_table(table, key, value),
                "{:?}, entry {j}: no row of table {}",
                group.lookups(),
                group.table().0
            );
        }
    }
}

/// The LogUp instances of `lookups` verify, and the verifier's leaf claims are the prover's.
fn logup_verifies(rng: &mut SplitMix64, lookups: &Lookups<Fp>, tables: &[&GeneratedTable]) {
    let session = Session::random(rng);
    let columns = table_columns(tables);
    let multiplicities = lookups.multiplicities(&columns).expect("honest rows");
    let instances = lookups
        .instances(&columns, &multiplicities)
        .expect("instances");
    let shapes: Vec<Shape> = instances
        .iter()
        .enumerate()
        .map(|(k, instance)| Shape::of(instance, spec_of(tables, lookups, k)))
        .collect();
    let proven = prove(&session, &instances);
    let verified = verify(&session, &shapes, &Messages::of(&proven))
        .unwrap_or_else(|rejected| panic!("an honest witness is rejected: {rejected:?}"));
    for (mine, theirs) in proven.claims().iter().zip(&verified.instances) {
        assert_eq!(tv_fp2(mine.key.value), theirs.key.value, "the key claim");
        assert_eq!(
            mine.value.as_ref().map(|v| tv_fp2(v.value)),
            theirs.value.as_ref().map(|v| v.value),
            "the value claim"
        );
        assert_eq!(
            tv_fp2(mine.multiplicity.value),
            theirs.multiplicity.value,
            "the multiplicity claim"
        );
    }
}

/// The widths of a requantization gadget: `(s, t)`, 0 for a part it does not have.
fn parts(gadget: &Gadget) -> (u32, u32) {
    match *gadget {
        Gadget::Requant1 { .. } => (0, 0),
        Gadget::RequantSplit { split, .. } => (u32::from(split), 0),
        Gadget::RequantWide { split, middle, .. } => (u32::from(split), u32::from(middle)),
        _ => panic!("a requantization gadget"),
    }
}

/// Each virtual value at the honest witness is the integer RFC-0009 and RFC-0024 define, from
/// the key alone.
fn virtuals_hold(
    constraints: &Constraints<Fp>,
    (b, s, t): (u32, u32, u32),
    keys: &[i64],
    classes: Option<&[u8]>,
    witness: &Witness,
) {
    for (virtual_, form) in &constraints.virtuals {
        for (i, &z) in keys.iter().enumerate() {
            let z_off = i128::from(z) + (1i128 << (b - 1));
            let w = i128::from(z) >> s;
            let (h0, h1) = (w.rem_euclid(1 << t), w >> t);
            let expected = match virtual_ {
                Virtual::ZOff => z_off,
                Virtual::ZHi | Virtual::WOff => z_off >> s,
                Virtual::H1Off => z_off >> (s + t),
                Virtual::H1 => h1,
                // `w' = h0 + 2^t·clamp(h1, −2, 1)`: `w` itself inside the saturation's range.
                Virtual::WPrime => h0 + (1i128 << t) * h1.clamp(-2, 1),
                Virtual::TopLimb => panic!("no requantization has a top limb"),
            };
            let found = form.evaluate(honest(keys, classes, witness, i));
            assert_eq!(
                found.to_centered_i128(),
                expected,
                "{virtual_:?} at z = {z}"
            );
        }
    }
}

/// One requantization site: the gadget the compiler selects, checked on `KEYS` keys.
fn requant(rng: &mut SplitMix64, b: u32, shifts: &[u8], (lo, hi): (i64, i64), b_max: u8) {
    let (gadget, specs) = select_requant(b, shifts, lo, hi, b_max);
    let requant = Requant::new(&gadget, &specs).expect("the selection's gadget");
    assert_eq!(requant.key_bits(), b);
    let keys = keys(rng, b, KEYS);
    let classes = (shifts.len() > 1).then(|| common::classes(rng, shifts.len(), KEYS));
    let classes = classes.as_deref();
    let witness = requant.witness(&keys, classes).expect("an honest witness");
    let y = witness.tensor(Committed::Y).expect("y");
    for (i, &z) in keys.iter().enumerate() {
        let c = classes.map_or(0, |c| c[i]);
        assert_eq!(
            y[i],
            rq(z, shifts[usize::from(c)], lo, hi),
            "z = {z}, class {c}"
        );
    }
    if let Some(g) = witness.tensor(Committed::G) {
        // RFC-0024 lookup 4: `g = clamp(h1, −2, 1)`, `h1 = ⌊z / 2^{s+t}⌋`.
        let (s, t) = parts(&gadget);
        for (&z, &g) in keys.iter().zip(g) {
            assert_eq!(g, (z >> (s + t)).clamp(-2, 1), "z = {z}");
        }
    }
    let constraints = requant.constraints::<Fp>();
    let (s, t) = parts(&gadget);
    virtuals_hold(&constraints, (b, s, t), &keys, classes, &witness);
    let lookups = Lookups::new(&constraints, &keys, classes, &witness).expect("columns");
    let tables = requant.tables();
    tuples_are_rows(&lookups, &tables);
    if b_max <= 16 {
        logup_verifies(rng, &lookups, &tables);
    }
}

#[test]
fn requant1_is_complete() {
    let mut rng = SplitMix64::new(0x118_0001);
    // Small, through LogUp: one class, then four.
    requant(&mut rng, 6, &[3], (-4, 3), 8);
    requant(&mut rng, 6, &[1, 2, 3, 4], (-20, 20), 8);
    // Production: a key of B_max bits, and one of 30 bits with four classes.
    requant(&mut rng, 32, &[8], (-128, 127), 32);
    requant(&mut rng, 30, &[8, 9, 10, 11], (-128, 127), 32);
}

#[test]
fn requant_split_is_complete() {
    let mut rng = SplitMix64::new(0x118_0002);
    // Small, through LogUp: s = 6 with two classes (RFC-0028, clamped by the rounding bit), and
    // the balanced s = 6 of one class.
    requant(&mut rng, 12, &[7, 9], (-50, 77), 8);
    requant(&mut rng, 11, &[9], (-128, 127), 8);
    // Production: the balanced splits of 48 and 62 bits (RFC-0028), and one stopped below the
    // rounding bit of a shift of 8. Three classes at 62 bits, 3·2^62 < P, still split at s = 32
    // (RFC-0009 Lemma 2).
    requant(&mut rng, 48, &[40], (-128, 127), 32);
    requant(&mut rng, 62, &[31], (-128, 127), 32);
    requant(&mut rng, 33, &[8, 9], (-128, 127), 32);
    requant(&mut rng, 62, &[33, 34, 35], (-128, 127), 32);
}

#[test]
fn requant_wide_is_complete() {
    let mut rng = SplitMix64::new(0x118_0003);
    // Small, through LogUp: one class, two classes with asymmetric bounds, four classes whose
    // middle and top parts share one Range table, and s = 0.
    requant(&mut rng, 14, &[3], (-3, 3), 10);
    requant(&mut rng, 14, &[3, 4], (-5, 2), 10);
    requant(&mut rng, 13, &[2, 3, 4, 5], (-4, 4), 10);
    requant(&mut rng, 12, &[1, 2], (-2, 3), 9);
    // Production: a 40-bit key with a shift of 8, and `minimal`'s 8B residual, b = 55 at the
    // test shapes' shift of 17, saturating at INT20 with one class and at ±(2^20 − 1) with four.
    requant(&mut rng, 40, &[8], (-128, 127), 32);
    requant(&mut rng, 55, &[17], (-(1 << 19), (1 << 19) - 1), 32);
    requant(
        &mut rng,
        55,
        &[17, 18, 19, 20],
        (-(1 << 20) + 1, (1 << 20) - 1),
        32,
    );
    // Four classes at 62 bits, which n·2^b ≥ P takes from the split (RFC-0009 Lemma 2): s = 32,
    // t = 11, k = 19; and, saturating at ±2^24, the widest middle part, t = 4 + 24 = 28, which
    // leaves the least top part, k = 2 (RFC-0024).
    requant(&mut rng, 62, &[33, 34, 35, 36], (-128, 127), 32);
    let (widest, _) = select_requant(62, &[33, 34, 35, 36], -(1 << 24), 1 << 24, 32);
    assert!(
        matches!(
            widest,
            Gadget::RequantWide {
                split: 32,
                middle: 28,
                ..
            }
        ),
        "{widest:?}"
    );
    requant(&mut rng, 62, &[33, 34, 35, 36], (-(1 << 24), 1 << 24), 32);
}

/// The `Range` tables of `NonNeg { k, L }`, numbered from 0.
fn ranges(bits: &[u8]) -> Vec<tessera_tables::TableSpec> {
    bits.iter()
        .zip(0..)
        .map(|(&bits, id)| {
            GeneratedTable::new(TableId(id), TableKind::Range { bits })
                .expect("a Range table")
                .spec()
                .clone()
        })
        .collect()
}

fn non_neg(rng: &mut SplitMix64, k: u8, l: u8, logup: bool) {
    let n = u32::from(k).div_ceil(u32::from(l));
    let top = u32::from(k) - (n - 1) * u32::from(l);
    let gadget = NonNeg::new(k, l, &ranges(&[l, top as u8])).expect("NonNeg");
    let bound = 1i64 << k;
    let mut values = vec![0, 1, bound - 2, bound - 1];
    while values.len() < KEYS {
        values.push((rng.next_u64() >> (64 - u32::from(k))) as i64);
    }
    let witness = gadget.witness(&values).expect("an honest witness");
    let constraints = gadget.constraints::<Fp>();
    assert_eq!(constraints.committed.len() as u32, n - 1);
    // Lemma 3: the limbs are the value's digits, and the top limb what is left above them.
    for (i, &v) in values.iter().enumerate() {
        for j in 0..n - 1 {
            let limb = witness.tensor(Committed::Limb(j as u8)).expect("a limb")[i];
            assert_eq!(limb, (v >> (j * u32::from(l))) & ((1 << l) - 1), "v = {v}");
        }
        let (_, top_limb) = &constraints.virtuals[0];
        let found = top_limb.evaluate(honest(&values, None, &witness, i));
        assert_eq!(
            found.to_centered_i128(),
            i128::from(v >> ((n - 1) * u32::from(l))),
            "v = {v}"
        );
    }
    let lookups = Lookups::new(&constraints, &values, None, &witness).expect("columns");
    let tables = gadget.tables();
    tuples_are_rows(&lookups, &tables);
    if logup {
        logup_verifies(rng, &lookups, &tables);
    }
}

#[test]
fn non_neg_is_complete() {
    let mut rng = SplitMix64::new(0x118_0004);
    // Small, through LogUp: three limbs into one table and a top of 2 bits; a top of L bits,
    // which shares the limbs' table; one limb alone.
    non_neg(&mut rng, 10, 4, true);
    non_neg(&mut rng, 12, 4, true);
    non_neg(&mut rng, 9, 12, true);
    // Production: SoftmaxRow's brackets at σ = 16, `NonNeg(·, σ + 18)`, and the widest, k = 62.
    non_neg(&mut rng, 34, 16, false);
    non_neg(&mut rng, 62, 16, false);
}
