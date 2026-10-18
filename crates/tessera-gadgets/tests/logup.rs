// SPDX-License-Identifier: Apache-2.0 OR MIT

//! End to end through LogUp: a gadget's witness, its instances, `tessera-logup`'s prover, and
//! the independent verifier's roots, levels and leaf checks (DEP-2), with every table evaluated
//! by `tv-tables` from its specification.
//!
//! - **Honest.** `Requant1`, `RequantSplit`, `RequantWide` (four classes, two of its lookups in
//!   one `Range` table) and `NonNeg` (limbs sharing a table) verify. Each leaf claim the verifier
//!   leaves on a key or value column is the constraints' form evaluated at its point from the
//!   extensions of `z`, `c` and the committed tensors, lookup by lookup in the instance's blocks,
//!   the LogUp padding holding row 0: the reduction the leaf stage makes (RFC-0037). Each
//!   multiplicity claim is the multiplicities' extension.
//! - **Tampered.** A cheating prover commits another witness and the multiplicities that balance
//!   its lookups wherever they are rows. Each is rejected at the roots, `Unbalanced`, naming the
//!   one instance whose check it breaks: a changed `y` or `g` (the output's or the saturation's
//!   table), `RequantWide`'s `z_lo + 2^s` with `h0 − 1` (only `Range { s }`), `RequantSplit`'s
//!   `z_lo + 1` (only the high part's table, whose key is no integer), and `NonNeg`'s limb plus
//!   `2^L` with the next limb carrying it (only `Range { L }`).

mod common;

use common::{
    Fp, Messages, Rejected, Session, Shape, SplitMix64, TvFp2, eq_weights, fp2, height, keys, mle,
    prove, select_requant, spec_of, table_columns, tv_fp, tv_fp2, verify,
};
use tessera_codec::SecretBuf;
use tessera_compiler::Gadget;
use tessera_field::{GoldilocksExt2, PrimeField};
use tessera_gadgets::{
    Committed, Constraints, Lookups, NonNeg, Requant, TableColumns, Term, Witness,
};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};
use tv_core::logup::LookupError;

/// Elements per gadget: a power of two, so that each lookup's block is a subcube.
const N: usize = 1 << 10;

/// A gadget's inputs and constraints, and the tables its lookups read.
struct Case<'a> {
    constraints: Constraints<Fp>,
    tables: Vec<&'a GeneratedTable>,
    input: Vec<i64>,
    classes: Option<Vec<u8>>,
}

impl Case<'_> {
    fn lookups(&self, witness: &Witness) -> Lookups<Fp> {
        Lookups::new(
            &self.constraints,
            &self.input,
            self.classes.as_deref(),
            witness,
        )
        .expect("the columns")
    }

    /// The column a term stands for, as integers.
    fn column(&self, witness: &Witness, term: Term) -> Vec<i64> {
        match term {
            Term::Input => self.input.clone(),
            Term::Class => self
                .classes
                .as_ref()
                .expect("classes")
                .iter()
                .map(|&c| i64::from(c))
                .collect(),
            Term::Committed(tensor) => witness.tensor(tensor).expect("a tensor").to_vec(),
        }
    }
}

/// Proves the instances of `lookups` with `multiplicities` and runs the verifier.
fn run(
    rng: &mut SplitMix64,
    case: &Case<'_>,
    lookups: &Lookups<Fp>,
    columns: &[TableColumns<Fp>],
    multiplicities: &[SecretBuf<Fp>],
) -> (
    Result<tv_core::logup::LeafClaims, Rejected>,
    tessera_logup::ProvenLogUp<GoldilocksExt2>,
) {
    let session = Session::random(rng);
    let instances = lookups
        .instances(columns, multiplicities)
        .expect("instances");
    let shapes: Vec<Shape> = instances
        .iter()
        .enumerate()
        .map(|(k, instance)| Shape::of(instance, spec_of(&case.tables, lookups, k)))
        .collect();
    let proven = prove(&session, &instances);
    (verify(&session, &shapes, &Messages::of(&proven)), proven)
}

/// The honest witness verifies, and each leaf claim follows from the constraints.
fn honest(rng: &mut SplitMix64, case: &Case<'_>, witness: &Witness) {
    let lookups = case.lookups(witness);
    let columns = table_columns(&case.tables);
    let multiplicities = lookups.multiplicities(&columns).expect("honest rows");
    let (verified, proven) = run(rng, case, &lookups, &columns, &multiplicities);
    let verified = verified.unwrap_or_else(|rejected| panic!("rejected: {rejected:?}"));
    assert_eq!(verified.instances.len(), lookups.groups().len());
    for (k, (claims, group)) in verified.instances.iter().zip(lookups.groups()).enumerate() {
        // The prover leaves the claims the verifier derives.
        let mine = &proven.claims()[k];
        assert_eq!(
            tv_fp2(mine.key.value),
            claims.key.value,
            "instance {k}: key"
        );
        // Row 0 of every table these gadgets read has key 0; its value pads the value column.
        let table = case
            .tables
            .iter()
            .find(|t| t.spec().id == group.table())
            .expect("a table");
        let (_, row_0) = table.row(0).expect("row 0");
        let point = &claims.key.point;
        assert_eq!(
            point.len(),
            height(group.lookups().len() * N),
            "instance {k}"
        );
        let (low, high) = point.split_at(height(N));
        let blocks = eq_weights(high);
        // The extension of each term's column at the low point.
        let at = |term: Term| fp2(mle(&case.column(witness, term), low));
        let mut key = TvFp2::ZERO;
        let mut value = TvFp2::ZERO;
        for (j, &index) in group.lookups().iter().enumerate() {
            let lookup = &case.constraints.lookups[index];
            key += blocks[j] * tv_fp2(lookup.key.evaluate::<GoldilocksExt2>(at));
            if let Some(form) = &lookup.value {
                value += blocks[j] * tv_fp2(form.evaluate::<GoldilocksExt2>(at));
            }
        }
        let padding: TvFp2 = blocks[group.lookups().len()..]
            .iter()
            .fold(TvFp2::ZERO, |sum, &w| sum + w);
        assert_eq!(
            claims.key.value, key,
            "instance {k}: kẽy(ρ) from the constraints"
        );
        match &claims.value {
            Some(claim) => {
                assert_eq!(claim.point, *point);
                let padded = value + padding * TvFp2::from(tv_fp(row_0));
                assert_eq!(
                    claim.value, padded,
                    "instance {k}: valũe(ρ) from the constraints"
                );
            }
            None => assert!(group.values().is_none(), "instance {k}: a Range table"),
        }
        let m: Vec<Fp> = multiplicities[k].to_vec();
        assert_eq!(
            claims.multiplicity.value,
            common::mle_fp(&m, &claims.multiplicity.point),
            "instance {k}: m̃(ρ_t)"
        );
    }
}

/// A witness of `witness`'s tensors with `change` applied to `tensor` at element `i`.
fn tampered(
    witness: &Witness,
    committed: &[Committed],
    changes: &[(Committed, i64)],
    i: usize,
) -> Witness {
    Witness::from_tensors(
        committed
            .iter()
            .map(|&tensor| {
                let mut values: SecretBuf<i64> = witness
                    .tensor(tensor)
                    .expect("a tensor")
                    .iter()
                    .copied()
                    .collect();
                for &(changed, delta) in changes {
                    if changed == tensor {
                        values[i] += delta;
                    }
                }
                (tensor, values)
            })
            .collect(),
    )
}

/// The cheating prover's multiplicities: for each group whose keys are all rows, their count,
/// row 0 also counting the lookup tree's padding, which balances any value that is its key's;
/// for a group with a key that is no row, which nothing balances, the honest ones.
fn cheating_multiplicities(
    lookups: &Lookups<Fp>,
    columns: &[TableColumns<Fp>],
    honest: &[SecretBuf<Fp>],
) -> Vec<SecretBuf<Fp>> {
    lookups
        .groups()
        .iter()
        .zip(honest)
        .map(|(group, honest)| {
            let table = columns
                .iter()
                .find(|c| c.table() == group.table())
                .expect("columns");
            let rows = table.tuples().len() as u64;
            let mut counts = vec![0u64; honest.len()];
            let mut named = true;
            for key in group.keys() {
                match key.to_canonical_u64() {
                    row if row < rows => counts[row as usize] += 1,
                    _ => named = false,
                }
            }
            let lookups = group.keys().len();
            counts[0] += ((1usize << height(lookups)) - lookups) as u64;
            match named {
                true => counts.into_iter().map(Fp::from_u64_reduced).collect(),
                false => honest.iter().copied().collect(),
            }
        })
        .collect()
}

/// A cheating prover with `witness`, rejected at the roots, naming `instance`.
fn rejected_at(
    rng: &mut SplitMix64,
    case: &Case<'_>,
    honest: &Witness,
    cheat: &Witness,
    instance: usize,
) {
    let columns = table_columns(&case.tables);
    let honest_m = case
        .lookups(honest)
        .multiplicities(&columns)
        .expect("honest rows");
    let lookups = case.lookups(cheat);
    let multiplicities = cheating_multiplicities(&lookups, &columns, &honest_m);
    let (verdict, _) = run(rng, case, &lookups, &columns, &multiplicities);
    assert_eq!(
        verdict.err(),
        Some(Rejected::Lookup(LookupError::Unbalanced { instance })),
        "expected instance {instance}"
    );
}

/// The instance that holds the lookup `index` of a case.
fn instance_of(lookups: &Lookups<Fp>, index: usize) -> usize {
    lookups
        .groups()
        .iter()
        .position(|g| g.lookups().contains(&index))
        .expect("a group")
}

/// A requantization case from the compiler's selection.
fn requant_case<'a>(
    rng: &mut SplitMix64,
    gadget: &'a Requant,
    shifts: &[u8],
) -> (Case<'a>, Witness) {
    let input = keys(rng, gadget.key_bits(), N);
    let classes = (shifts.len() > 1).then(|| common::classes(rng, shifts.len(), N));
    let witness = gadget
        .witness(&input, classes.as_deref())
        .expect("an honest witness");
    let case = Case {
        constraints: gadget.constraints(),
        tables: gadget.tables(),
        input,
        classes,
    };
    (case, witness)
}

#[test]
fn requant_wide_verifies_and_rejects_each_tampered_part_at_its_instance() {
    let mut rng = SplitMix64::new(0x118_0401);
    // Four classes: s = 1, t = 6, k = 6, so the middle and top parts read one Range { 6 }.
    let shifts = [2u8, 3, 4, 5];
    let (selected, specs) = select_requant(13, &shifts, -4, 4, 10);
    assert!(matches!(
        selected,
        Gadget::RequantWide {
            split: 1,
            middle: 6,
            ..
        }
    ));
    let gadget = Requant::new(&selected, &specs).expect("RequantWide");
    let (case, witness) = requant_case(&mut rng, &gadget, &shifts);
    let lookups = case.lookups(&witness);
    assert_eq!(lookups.groups().len(), 4, "five lookups into four tables");
    assert_eq!(
        lookups.groups()[1].lookups(),
        [1, 2],
        "h0 and h1_off share Range {{ 6 }}"
    );
    honest(&mut rng, &case, &witness);

    let committed = case.constraints.committed.clone();
    let h0 = witness.tensor(Committed::H0).expect("h0");
    let i = (0..N).find(|&i| h0[i] >= 1).expect("an h0 of at least 1");
    // `y + 1`: the output's table alone.
    let cheat = tampered(&witness, &committed, &[(Committed::Y, 1)], i);
    rejected_at(&mut rng, &case, &witness, &cheat, instance_of(&lookups, 4));
    // `g + 1`: the saturation's table, the first of the two whose tuples it changes.
    let g = witness.tensor(Committed::G).expect("g");
    let j = (0..N).find(|&j| g[j] < 1).expect("a g below 1");
    let cheat = tampered(&witness, &committed, &[(Committed::G, 1)], j);
    rejected_at(&mut rng, &case, &witness, &cheat, instance_of(&lookups, 3));
    // `z_lo + 2^s` with `h0 − 1`: every virtual value an integer, every tuple but the low part's
    // a row, which the cheater's multiplicities balance; only Range { s } rejects it.
    let cheat = tampered(
        &witness,
        &committed,
        &[(Committed::ZLo, 2), (Committed::H0, -1)],
        i,
    );
    let cheat = with_best_output(&gadget, &case, cheat, i);
    rejected_at(&mut rng, &case, &witness, &cheat, instance_of(&lookups, 0));
}

/// `cheat` with its `y` at element `i` set to the output table's value at its lookup's key: the
/// cheater's best output for its digits.
fn with_best_output(gadget: &Requant, case: &Case<'_>, cheat: Witness, i: usize) -> Witness {
    let constraints = gadget.constraints::<Fp>();
    let output = constraints.lookups.len() - 1;
    let key = constraints.lookups[output]
        .key
        .evaluate(|term| Fp::from_i64(case.column(&cheat, term)[i]));
    let table = case
        .tables
        .iter()
        .find(|t| t.spec().id == constraints.lookups[output].table)
        .expect("the output table");
    let (_, value) = table.row(key.to_canonical_u64()).expect("a row");
    let y = witness_value(value);
    let current = cheat.tensor(Committed::Y).expect("y")[i];
    tampered(
        &cheat,
        &constraints.committed,
        &[(Committed::Y, y - current)],
        i,
    )
}

fn witness_value(value: Fp) -> i64 {
    i64::try_from(value.to_centered_i128()).expect("a true integer")
}

#[test]
fn requant_split_and_requant1_verify_and_reject_tampered_parts() {
    let mut rng = SplitMix64::new(0x118_0402);
    // RequantSplit at B_max = 8: 12 bits, two classes, s = 6.
    let shifts = [7u8, 9];
    let (selected, specs) = select_requant(12, &shifts, -50, 77, 8);
    assert!(matches!(selected, Gadget::RequantSplit { split: 6, .. }));
    let gadget = Requant::new(&selected, &specs).expect("RequantSplit");
    let (case, witness) = requant_case(&mut rng, &gadget, &shifts);
    honest(&mut rng, &case, &witness);
    let committed = case.constraints.committed.clone();
    let z_lo = witness.tensor(Committed::ZLo).expect("z_lo");
    let i = (0..N)
        .find(|&i| z_lo[i] < 63)
        .expect("a z_lo below 2^s − 1");
    // `z_lo + 1` stays in Range { s }, and its high key is no integer: the high table alone.
    let cheat = tampered(&witness, &committed, &[(Committed::ZLo, 1)], i);
    rejected_at(&mut rng, &case, &witness, &cheat, 1);
    let cheat = tampered(&witness, &committed, &[(Committed::Y, -1)], i);
    rejected_at(&mut rng, &case, &witness, &cheat, 1);

    // Requant1 at B_max = 8: 6 bits and four classes, 8 key bits in all.
    let shifts = [1u8, 2, 3, 4];
    let (selected, specs) = select_requant(6, &shifts, -20, 20, 8);
    assert!(matches!(selected, Gadget::Requant1 { .. }));
    let gadget = Requant::new(&selected, &specs).expect("Requant1");
    let (case, witness) = requant_case(&mut rng, &gadget, &shifts);
    honest(&mut rng, &case, &witness);
    let cheat = tampered(&witness, &[Committed::Y], &[(Committed::Y, 1)], N - 1);
    rejected_at(&mut rng, &case, &witness, &cheat, 0);
}

#[test]
fn non_neg_verifies_and_rejects_a_carried_limb_at_its_range() {
    let mut rng = SplitMix64::new(0x118_0403);
    // k = 10, L = 4: limbs v_0, v_1 in Range { 4 }, one instance; the top limb in Range { 2 }.
    let specs: Vec<TableSpec> = [4u8, 2]
        .iter()
        .zip(0..)
        .map(|(&bits, id)| {
            GeneratedTable::new(TableId(id), TableKind::Range { bits })
                .expect("a Range table")
                .spec()
                .clone()
        })
        .collect();
    let gadget = NonNeg::new(10, 4, &specs).expect("NonNeg");
    let mut input: Vec<i64> = vec![0, 1023];
    while input.len() < N {
        input.push(rng.below(1 << 10) as i64);
    }
    let witness = gadget.witness(&input).expect("an honest witness");
    let case = Case {
        constraints: gadget.constraints(),
        tables: gadget.tables(),
        input,
        classes: None,
    };
    let lookups = case.lookups(&witness);
    assert_eq!(
        lookups.groups()[0].lookups(),
        [0, 1],
        "two limbs in one Range {{ 4 }}"
    );
    honest(&mut rng, &case, &witness);
    let v_1 = witness.tensor(Committed::Limb(1)).expect("v_1");
    let i = (0..N).find(|&i| v_1[i] >= 1).expect("a v_1 of at least 1");
    let committed = case.constraints.committed.clone();
    // `v_0 + 2^4` with `v_1 − 1`: the top limb unchanged, only Range { 4 } rejects it.
    let cheat = tampered(
        &witness,
        &committed,
        &[(Committed::Limb(0), 16), (Committed::Limb(1), -1)],
        i,
    );
    rejected_at(&mut rng, &case, &witness, &cheat, 0);
    // `v_1 + 1`, still a limb: the top limb `(v − v_0 − 2^4·v_1)·2^{−8}` is no integer, and
    // Range { 2 } alone rejects it.
    let j = (0..N).find(|&j| v_1[j] < 15).expect("a v_1 below 15");
    let cheat = tampered(&witness, &committed, &[(Committed::Limb(1), 1)], j);
    rejected_at(&mut rng, &case, &witness, &cheat, 1);
}
