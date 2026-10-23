// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A leaf stage in token groups (RFC-0054 'The leaf stage with groups', 'The table batch';
//! RFC-0014 'Transcript schedule', rows 4 and 5 with groups).
//!
//! - **The schedule.** Every label in order: `logup/alpha`, `logup/beta`, `logup/roots` and
//!   `logup/values` empty, then each group's roots, levels and values under `group/{h}/…`, then
//!   the table batch under `tables/…`.
//! - **The identity.** Each instance's group fractions and table root sum to 0, with `m_0`
//!   counting every group's padding (RFC-0008's identity over every group's lookups).
//! - **The claims.** Every key, value, multiplicity and forest leaf claim is its column's or
//!   leaves' evaluation at its point.
//! - **Only labels change.** A group's levels are the ungrouped levels' messages, the labels
//!   renamed.
//! - **Refusals.** A group's schedule names each lookup tree and forest once, and no table tree.

mod common;

use common::{
    Fp, Fp2, Session, SplitMix64, TestForest, TestTable, height, mle, padded, tv_fp, tv_fp2,
};
use tessera_field::{Field, PrimeField};
use tessera_hash::Transcript;
use tessera_logup::{
    LogUpError, LogUpInstance, TreeId, Tuples, begin_groups, prove_group, prove_table_batch,
};
use tessera_sumcheck::{
    FractionTree, LeafNumerators, SumcheckTranscript, Tree, TreeScope, prove_trees, prove_trees_in,
};

/// A transcript that records every label and can rename a prefix before passing a call on.
struct Recording<'a> {
    inner: &'a mut Transcript,
    labels: Vec<String>,
    rename: Option<(&'static str, &'static str)>,
}

impl Recording<'_> {
    fn name(&mut self, label: &str) -> String {
        self.labels.push(label.to_string());
        match self.rename {
            Some((from, to)) if label.starts_with(from) => format!("{to}{}", &label[from.len()..]),
            _ => label.to_string(),
        }
    }
}

impl SumcheckTranscript<Fp2> for Recording<'_> {
    fn absorb_message(&mut self, label: &str, message: &[Fp2]) {
        let label = self.name(label);
        self.inner.absorb_message(&label, message);
    }

    fn challenge(&mut self, label: &str) -> Fp2 {
        let label = self.name(label);
        SumcheckTranscript::challenge(self.inner, &label)
    }
}

/// Three tables, two with values; three groups, the last holding no lookup of the second table.
struct Fixture {
    tables: Vec<TestTable>,
    /// `rows[h][k]`: group `h`'s lookups of table `k`, as rows; empty when it holds none.
    rows: Vec<Vec<Vec<usize>>>,
    multiplicities: Vec<Vec<Fp>>,
    forest: TestForest,
}

impl Fixture {
    fn new(rng: &mut SplitMix64) -> Self {
        let tables = vec![
            TestTable::random(rng, 6, true),
            TestTable::random(rng, 16, false),
            TestTable::random(rng, 4, true),
        ];
        let mut pick = |table: usize, n: usize| -> Vec<usize> {
            (0..n)
                .map(|_| rng.below(tables[table].rows() as u64) as usize)
                .collect()
        };
        let rows = vec![
            vec![pick(0, 5), pick(1, 8), pick(2, 3)],
            vec![pick(0, 3), pick(1, 2), pick(2, 1)],
            vec![pick(0, 7), vec![], pick(2, 6)],
        ];
        // Each lookup counts its row; row 0 also counts every group's padding.
        let multiplicities = (0..tables.len())
            .map(|k| {
                let mut counts = vec![0u64; 1 << height(tables[k].rows())];
                for group in &rows {
                    let lookups = &group[k];
                    if lookups.is_empty() {
                        continue;
                    }
                    for &r in lookups {
                        counts[r] += 1;
                    }
                    counts[0] += ((1usize << height(lookups.len())) - lookups.len()) as u64;
                }
                counts.into_iter().map(Fp::from_u64_reduced).collect()
            })
            .collect();
        Self {
            tables,
            rows,
            multiplicities,
            forest: TestForest::random(rng, 1, 2),
        }
    }

    /// Group `h`'s lookups of table `k`: its keys and values.
    fn columns(&self, h: usize, k: usize) -> (Vec<Fp>, Option<Vec<Fp>>) {
        let table = &self.tables[k];
        let rows = &self.rows[h][k];
        let keys = rows.iter().map(|&r| table.keys[r]).collect();
        let values = table
            .values
            .as_ref()
            .map(|v| rows.iter().map(|&r| v[r]).collect());
        (keys, values)
    }
}

/// `n/d` summed.
fn add(acc: (Fp2, Fp2), n: Fp2, d: Fp2) -> (Fp2, Fp2) {
    (acc.0 * d + n * acc.1, acc.1 * d)
}

#[test]
fn a_grouped_stage_runs_rfc_0014s_schedule_and_its_claims_hold() {
    for seed in 0..6 {
        let mut rng = SplitMix64::new(0x5450 + seed);
        let session = Session::random(&mut rng);
        let f = Fixture::new(&mut rng);
        let mut inner = session.transcript();
        let mut t = Recording {
            inner: &mut inner,
            labels: Vec::new(),
            rename: None,
        };
        let (alpha, beta) = begin_groups::<Fp2, _>(&mut t);
        let mut sums: Vec<(Fp2, Fp2)> = vec![(Fp2::ZERO, Fp2::ONE); f.tables.len()];
        for h in 0..f.rows.len() {
            let held: Vec<usize> = (0..f.tables.len())
                .filter(|&k| !f.rows[h][k].is_empty())
                .collect();
            let columns: Vec<(Vec<Fp>, Option<Vec<Fp>>)> =
                held.iter().map(|&k| f.columns(h, k)).collect();
            let instances: Vec<LogUpInstance<'_, Fp>> = held
                .iter()
                .zip(&columns)
                .map(|(&k, (keys, values))| {
                    LogUpInstance::new(
                        f.tables[k].tuples(),
                        Tuples::new(keys, values.as_deref()),
                        &f.multiplicities[k],
                    )
                })
                .collect();
            // Group 0 also holds the forest, its lookups scheduled in reverse.
            let (forests, schedule) = match h {
                0 => (
                    vec![f.forest.forest()],
                    vec![
                        TreeId::Lookup(2),
                        TreeId::Forest(0),
                        TreeId::Lookup(1),
                        TreeId::Lookup(0),
                    ],
                ),
                _ => (vec![], (0..held.len()).map(TreeId::Lookup).collect()),
            };
            let proven = prove_group(
                &mut t, h as u32, alpha, beta, &instances, &forests, &schedule,
            )
            .expect("a group");
            // Roots in schedule order, two per lookup tree.
            let lookups: Vec<usize> = schedule
                .iter()
                .filter_map(|id| match id {
                    TreeId::Lookup(k) => Some(*k),
                    _ => None,
                })
                .collect();
            assert_eq!(proven.roots().len(), 2 * lookups.len());
            for (j, &k) in lookups.iter().enumerate() {
                let (n, d) = (proven.roots()[2 * j], proven.roots()[2 * j + 1]);
                sums[held[k]] = add(sums[held[k]], n, d);
            }
            // The values message: the value claims of the trees with values, in schedule order.
            let sent: Vec<Fp2> = lookups
                .iter()
                .filter_map(|&k| proven.claims()[k].value.as_ref().map(|c| c.value))
                .collect();
            assert_eq!(proven.values(), &sent[..], "group {h}'s values");
            // Each instance's claims, in instance order.
            for (k, claims) in proven.claims().iter().enumerate() {
                let table = &f.tables[held[k]];
                let (keys, values) = &columns[k];
                let h_a = height(keys.len());
                let point: Vec<_> = claims.key.point.iter().map(|&x| tv_fp2(x)).collect();
                assert_eq!(point.len(), h_a);
                let key = mle(&padded(keys, table.keys[0], h_a), &point);
                assert_eq!(tv_fp2(claims.key.value), key, "group {h}, key {k}");
                match (values, &claims.value) {
                    (Some(values), Some(claim)) => {
                        let value = mle(&padded(values, table.value(0), h_a), &point);
                        assert_eq!(tv_fp2(claim.value), value, "group {h}, value {k}");
                    }
                    (None, None) => {}
                    _ => panic!("a value claim exactly for a table with values"),
                }
            }
            if h == 0 {
                let claim = &proven.forest_claims()[0];
                let point: Vec<_> = claim.point.iter().map(|&x| tv_fp2(x)).collect();
                let leaves: Vec<_> = f.forest.leaves.iter().map(|&x| tv_fp(x).into()).collect();
                assert_eq!(
                    tv_fp2(claim.value),
                    mle(&leaves, &point),
                    "the forest's leaves"
                );
            }
        }
        let columns: Vec<_> = (0..f.tables.len()).map(|k| f.columns(0, k)).collect();
        let all: Vec<LogUpInstance<'_, Fp>> = columns
            .iter()
            .enumerate()
            .map(|(k, (keys, values))| {
                LogUpInstance::new(
                    f.tables[k].tuples(),
                    Tuples::new(keys, values.as_deref()),
                    &f.multiplicities[k],
                )
            })
            .collect();
        let tables = prove_table_batch(&mut t, alpha, beta, &all).expect("the table batch");
        for (k, sum) in sums.iter_mut().enumerate() {
            let (n, d) = (tables.roots()[2 * k], tables.roots()[2 * k + 1]);
            *sum = add(*sum, n, d);
            assert_ne!(sum.1, Fp2::ZERO);
            assert_eq!(sum.0, Fp2::ZERO, "seed {seed}: instance {k}'s identity");
            let claim = &tables.multiplicities()[k];
            let point: Vec<_> = claim.point.iter().map(|&x| tv_fp2(x)).collect();
            let m: Vec<_> = f.multiplicities[k]
                .iter()
                .map(|&x| tv_fp(x).into())
                .collect();
            assert_eq!(
                tv_fp2(claim.value),
                mle(&m, &point),
                "instance {k}'s multiplicities"
            );
        }
        // RFC-0014 rows 4 and 5 with groups, label by label.
        let labels = &t.labels;
        assert_eq!(
            labels[..4],
            ["logup/alpha", "logup/beta", "logup/roots", "logup/values"]
        );
        let mut at = 4;
        for h in 0..f.rows.len() {
            assert_eq!(labels[at], format!("group/{h}/logup/roots"), "seed {seed}");
            at += 1;
            let tree = format!("group/{h}/tree/");
            let start = at;
            while labels[at].starts_with(&tree) {
                at += 1;
            }
            assert!(at > start, "group {h} has levels");
            assert_eq!(labels[at], format!("group/{h}/logup/values"));
            at += 1;
        }
        assert_eq!(labels[at], "tables/logup/roots");
        assert!(
            labels[at + 1..]
                .iter()
                .all(|l| l.starts_with("tables/tree/"))
        );
        assert!(labels.len() > at + 1, "the table batch has levels");
    }
}

#[test]
fn a_groups_levels_are_the_ungrouped_levels_under_their_own_labels() {
    let mut rng = SplitMix64::new(0x5460);
    let session = Session::random(&mut rng);
    let leaves: Vec<Fp2> = (0..32).map(|_| Fp2::from(rng.fp())).collect();
    let numerators = rng.fps(16);
    let build = || {
        vec![
            Tree::Fraction(
                FractionTree::new(5, LeafNumerators::Unit, |x| leaves[x]).expect("a tree"),
            ),
            Tree::Fraction(
                FractionTree::new(4, LeafNumerators::Base(&numerators), |x| leaves[2 * x])
                    .expect("a tree"),
            ),
        ]
    };
    let mut plain = session.transcript();
    let expected = prove_trees(&mut plain, build()).expect("levels");
    for (scope, from) in [
        (TreeScope::Group(3), "group/3/tree/"),
        (TreeScope::Tables, "tables/tree/"),
    ] {
        let mut inner = session.transcript();
        let mut renamed = Recording {
            inner: &mut inner,
            labels: Vec::new(),
            rename: Some((from, "tree/")),
        };
        let proven = prove_trees_in(&mut renamed, scope, build()).expect("levels");
        assert!(
            renamed.labels.iter().all(|l| l.starts_with(from)),
            "{scope:?}"
        );
        assert_eq!(proven.levels().len(), expected.levels().len());
        for (a, b) in proven.levels().iter().zip(expected.levels()) {
            assert_eq!(a.proof(), b.proof(), "{scope:?}");
        }
    }
}

#[test]
fn a_groups_schedule_names_its_lookup_trees_and_forests_once_and_no_table_tree() {
    let mut rng = SplitMix64::new(0x5470);
    let session = Session::random(&mut rng);
    let f = Fixture::new(&mut rng);
    let (keys, values) = f.columns(0, 0);
    let instance = LogUpInstance::new(
        f.tables[0].tuples(),
        Tuples::new(&keys, values.as_deref()),
        &f.multiplicities[0],
    );
    let forest = f.forest.forest();
    let run = |schedule: &[TreeId]| {
        let mut t = session.transcript();
        let (alpha, beta) = begin_groups::<Fp2, _>(&mut t);
        prove_group(&mut t, 0, alpha, beta, &[instance], &[forest], schedule).map(drop)
    };
    assert_eq!(
        run(&[TreeId::Lookup(0)]),
        Err(LogUpError::ScheduleLength {
            len: 1,
            expected: 2
        })
    );
    assert_eq!(
        run(&[TreeId::Lookup(0), TreeId::Table(0)]),
        Err(LogUpError::ScheduleTree { position: 1 })
    );
    assert_eq!(
        run(&[TreeId::Forest(0), TreeId::Forest(0)]),
        Err(LogUpError::ScheduleTree { position: 1 })
    );
    assert_eq!(
        run(&[TreeId::Lookup(1), TreeId::Forest(0)]),
        Err(LogUpError::ScheduleTree { position: 0 })
    );
    assert_eq!(run(&[TreeId::Forest(0), TreeId::Lookup(0)]), Ok(()));
}
