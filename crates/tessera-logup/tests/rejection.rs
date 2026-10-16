// SPDX-License-Identifier: Apache-2.0 OR MIT

//! False instances and changed proofs are rejected (RFC-0008 'Testing Strategy', 'Failure
//! modes'), each at the exact check where the independent verifier stops.
//!
//! - A lookup outside the table, a lookup with a wrong value and a multiplicity off by one are
//!   witnesses the prover takes as they are: it builds honest trees over their leaves, and
//!   `tv_core::logup::check_roots` rejects the roots with `LookupError::Unbalanced` for that
//!   instance. So is the phantom row: a lookup of `(0, 0)`, which is no row, balanced by a
//!   multiplicity on a padding row, which counts lookups of row 0.
//! - Every element of every message, changed in turn, is rejected at the roots, at a level or at
//!   the leaf checks; a changed value of the values message passes the leaf checks, which never
//!   read it (`tv-core`'s 'LogUp leaf checks'), and leaves key and value claims that are not the
//!   polynomials' extensions, which the stages that take them reject.

mod common;

use common::{
    Fp, Messages, Rejected, Session, SplitMix64, TestForest, TestInstance, TestTable, TvFp2, mle,
    padded, prove, verify,
};
use tessera_field::{Field, PrimeField};
use tessera_logup::TreeId;
use tv_core::logup::LookupError;

fn unbalanced(instance: usize) -> Result<(), Rejected> {
    Err(Rejected::Lookup(LookupError::Unbalanced { instance }))
}

/// The verifier's verdict on the honest proof of the stage.
fn verdict(session: &Session, instances: &[TestInstance]) -> Result<(), Rejected> {
    let schedule = TreeId::paired(instances.len());
    let (proven, _) = prove(session, instances, &[], &schedule);
    let verdict = verify(session, instances, &[], &schedule, &Messages::of(&proven)).map(|_| ());
    if let Err(Rejected::Lookup(error)) = verdict {
        assert!(error.is_rejection(), "a fault of the proof");
    }
    verdict
}

/// An honest instance beside the instance under test, so that the rejection names it.
fn beside(rng: &mut SplitMix64, bad: TestInstance) -> Vec<TestInstance> {
    let table = TestTable::random(rng, 7, true);
    vec![TestInstance::random(rng, table, 9), bad]
}

/// Multiplicities that count the lookups of `rows` and, on row 0, the lookup tree's padding
/// leaves for `lookups` lookups, of which those of `rows` are part.
fn counted(table: &TestTable, rows: &[usize], lookups: usize) -> Vec<Fp> {
    let mut m = vec![Fp::ZERO; 1 << common::height(table.rows())];
    for &r in rows {
        m[r] += Fp::ONE;
    }
    m[0] += Fp::from_u64_reduced(((1usize << common::height(lookups)) - lookups) as u64);
    m
}

/// `count` random rows of `table`.
fn random_rows(rng: &mut SplitMix64, table: &TestTable, count: usize) -> Vec<usize> {
    (0..count)
        .map(|_| rng.below(table.rows() as u64) as usize)
        .collect()
}

#[test]
fn a_lookup_outside_the_table_is_rejected_at_the_roots() {
    let mut rng = SplitMix64::new(0x60_2001);
    let session = Session::random(&mut rng);
    for (rows, lookups) in [(2usize, 2usize), (5, 8), (16, 16), (100, 37)] {
        // Without values: the key `rows`, one past the last row, which no multiplicity counts.
        let table = TestTable::random(&mut rng, rows, false);
        let inside = random_rows(&mut rng, &table, lookups - 1);
        let mut bad = TestInstance::of_rows(table.clone(), &inside);
        bad.keys.push(Fp::from_u64_reduced(rows as u64));
        bad.multiplicities = counted(&table, &inside, lookups);
        assert_eq!(verdict(&session, &beside(&mut rng, bad)), unbalanced(1));

        // With values: a pair that is no row, a row's key with a random value, counted nowhere.
        let table = TestTable::random(&mut rng, rows, true);
        let all = random_rows(&mut rng, &table, lookups);
        let mut bad = TestInstance::of_rows(table.clone(), &all);
        if let Some(values) = &mut bad.values {
            values[0] = rng.fp();
        }
        bad.multiplicities = counted(&table, &all[1..], lookups);
        assert_eq!(verdict(&session, &[bad]), unbalanced(0));
    }
}

#[test]
fn a_wrong_value_is_rejected_at_the_roots() {
    let mut rng = SplitMix64::new(0x60_2002);
    let session = Session::random(&mut rng);
    for (rows, lookups) in [(2usize, 2usize), (7, 5), (16, 33), (256, 64)] {
        let table = TestTable::random(&mut rng, rows, true);
        let mut bad = TestInstance::random(&mut rng, table, lookups);
        // A lookup of its row's key with the row's value plus 1, still counted in that row.
        let i = rng.below(lookups as u64) as usize;
        if let Some(values) = &mut bad.values {
            values[i] += Fp::ONE;
        }
        assert_eq!(verdict(&session, &beside(&mut rng, bad)), unbalanced(1));
    }
}

#[test]
fn a_multiplicity_off_by_one_is_rejected_at_the_roots() {
    let mut rng = SplitMix64::new(0x60_2003);
    let session = Session::random(&mut rng);
    for (rows, lookups) in [(1usize, 1usize), (3, 4), (16, 16), (100, 300)] {
        for with_values in [true, false] {
            let table = TestTable::random(&mut rng, rows, with_values);
            let honest = TestInstance::random(&mut rng, table, lookups);
            // One more, one less, on a row; and one more on a padding row, if any.
            let j = rng.below(rows as u64) as usize;
            let leaves = honest.multiplicities.len();
            let mut changes = vec![(j, Fp::ONE), (j, -Fp::ONE)];
            if leaves > rows {
                changes.push((rows + rng.below((leaves - rows) as u64) as usize, Fp::ONE));
            }
            for (j, change) in changes {
                let mut bad = honest.clone();
                bad.multiplicities[j] += change;
                assert_eq!(verdict(&session, &beside(&mut rng, bad)), unbalanced(1));
            }
        }
    }
}

/// The attack that padding the table with a row `(0, 0)` would admit (RFC-0008 'Fraction tree'):
/// a lookup of `(0, 0)`, which is no row, and the multiplicity 1 on a padding row. The padding
/// row is a copy of row 0, so the multiplicity counts a lookup of row 0, and the roots fail.
#[test]
fn the_phantom_row_is_rejected_at_the_roots() {
    let mut rng = SplitMix64::new(0x60_2004);
    let session = Session::random(&mut rng);
    for rows in [3usize, 5, 6, 7] {
        // Random keys and values: no row is (0, 0).
        let table = TestTable::random(&mut rng, rows, true);
        let inside = random_rows(&mut rng, &table, 2);
        let mut bad = TestInstance::of_rows(table.clone(), &inside);
        bad.keys.push(Fp::ZERO);
        if let Some(values) = &mut bad.values {
            values.push(Fp::ZERO);
        }
        bad.multiplicities = counted(&table, &inside, 3);
        bad.multiplicities[rows] += Fp::ONE;
        assert_eq!(verdict(&session, &[bad]), unbalanced(0));
    }
}

/// Every element of every message changed in turn: a root, a round value or an evaluation of a
/// level is rejected, at the roots, at its level or at a later one, or at the leaf checks; a value
/// of the values message passes the verifier and leaves false key and value claims.
#[test]
fn every_changed_message_element_is_caught() {
    let mut rng = SplitMix64::new(0x60_2005);
    let session = Session::random(&mut rng);
    let instances = vec![
        {
            let table = TestTable::random(&mut rng, 5, true);
            TestInstance::random(&mut rng, table, 11)
        },
        {
            let table = TestTable::random(&mut rng, 9, false);
            TestInstance::random(&mut rng, table, 3)
        },
    ];
    let forests = vec![TestForest::random(&mut rng, 1, 3)];
    let schedule = [
        TreeId::Lookup(0),
        TreeId::Forest(0),
        TreeId::Table(1),
        TreeId::Table(0),
        TreeId::Lookup(1),
    ];
    let (proven, _) = prove(&session, &instances, &forests, &schedule);
    let honest = Messages::of(&proven);
    assert!(verify(&session, &instances, &forests, &schedule, &honest).is_ok());
    let delta = TvFp2::ONE;
    let mut changed = 0;
    for k in 0..honest.roots.len() {
        let mut m = Messages::of(&proven);
        m.roots[k] += delta;
        let verdict = verify(&session, &instances, &forests, &schedule, &m);
        assert!(verdict.is_err(), "root {k} changed is rejected");
        changed += 1;
    }
    for level in 0..honest.levels.len() {
        for j in 0..honest.levels[level].rounds.len() {
            for e in 0..honest.levels[level].rounds[j].len() {
                let mut m = Messages::of(&proven);
                m.levels[level].rounds[j][e] += delta;
                let verdict = verify(&session, &instances, &forests, &schedule, &m);
                assert!(
                    matches!(verdict, Err(Rejected::Level(l, _)) if l == level),
                    "level {level}, round {j}, value {e}: {:?}",
                    verdict.err()
                );
                changed += 1;
            }
        }
        for e in 0..honest.levels[level].final_evals.len() {
            let mut m = Messages::of(&proven);
            m.levels[level].final_evals[e] += delta;
            let verdict = verify(&session, &instances, &forests, &schedule, &m);
            match verdict {
                Err(Rejected::Level(l, _)) => assert!(l >= level, "level {level}, eval {e}"),
                Err(Rejected::Lookup(error)) => assert!(error.is_rejection()),
                // A forest's evaluation at the last level moves its leaf claim, which the stage
                // that consumes it checks: it must no longer be the leaves' extension.
                Ok(verified) => {
                    let p = schedule.iter().position(|&id| id == TreeId::Forest(0));
                    let tv_core::tree::LevelClaim::Product(claim) =
                        &verified.leaf_claims[p.expect("scheduled")]
                    else {
                        panic!("a product claim");
                    };
                    let leaves: Vec<TvFp2> = forests[0]
                        .leaves
                        .iter()
                        .map(|&x| TvFp2::from(common::tv_fp(x)))
                        .collect();
                    assert_ne!(
                        claim.value,
                        mle(&leaves, &claim.point),
                        "level {level}, {e}"
                    );
                    assert_eq!(level, honest.levels.len() - 1, "only at the last level");
                }
            }
            changed += 1;
        }
    }
    for k in 0..honest.values.len() {
        let mut m = Messages::of(&proven);
        m.values[k] += delta;
        let verified = verify(&session, &instances, &forests, &schedule, &m)
            .expect("the leaf checks never read v");
        let claims = &verified.claims.instances[0];
        let instance = &instances[0];
        let h_a = instance.lookup_height();
        let keys = padded(&instance.keys, instance.table.keys[0], h_a);
        let values = padded(
            instance.values.as_ref().expect("values"),
            instance.table.value(0),
            h_a,
        );
        assert_ne!(
            claims.key.value,
            mle(&keys, &claims.key.point),
            "a false key claim"
        );
        let value = claims.value.as_ref().expect("a value claim");
        assert_ne!(
            value.value,
            mle(&values, &value.point),
            "a false value claim"
        );
        changed += 1;
    }
    assert!(changed > 50, "{changed} elements changed");
}
