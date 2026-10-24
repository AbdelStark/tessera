// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An honest prover of a grouped leaf stage through the merge, for tests, and the public data a
//! verifier reads beside it. It is written from RFC-0054 ('The leaf stage with groups', 'The table
//! batch', 'The merge'), RFC-0008 ('Fraction tree'), RFC-0010 ('Tree levels') and RFC-0045 ('The
//! column', 'The leaves', 'The leaf point'), on the test kit's field, transcript and batched
//! sumcheck prover, and shares no code with `tv-core`'s checks. It takes the group layouts of
//! `tv_core::leaf::groups`, which `leaf_groups.rs` checks against oracles of their own.
//!
//! - **The witness.** Every merged polynomial is a table in its natural variable order: a tensor
//!   of a site at one layer, or a forest's stacked leaves (`[keys | queries | heads | layers]`).
//! - **The tables** are what the lookups take: their distinct pairs, row 0 first, padded with
//!   copies of row 0, and multiplicities counting every column entry of every group, each
//!   group's column padding in row 0's.
//! - **The prover** builds each group's columns entry by entry from its blocks' forms, its lookup
//!   trees over the leaf order `σ_h`, its forest shares from the forests' leaves, and proves its
//!   levels by the test kit's batched prover from the tables of the level below; then the table
//!   trees; then the merge, a batched sumcheck of `W_i·P_i` in each polynomial's merge order, with
//!   `W_i` built entry by entry on the cube from the groups' claims.
//!
//! Every extension the verifier's callbacks evaluate is a sum against `eq` from its definition.

#![allow(dead_code)]

use std::collections::HashMap;

use tv_core::leaf::grouped::TableTree;
use tv_core::leaf::groups::{Form as Affine, Group, Grouped, Polynomial, Value};
use tv_core::protocol::{GroupProof, LogUpProof, TreeProof};
use tv_core::sumcheck::SumcheckProof;
use tv_field::{Fp, Fp2};
use tv_tables::TableId;
use tv_testkit::SplitMix64;
use tv_testkit::logup::{Forest, FractionTree};
use tv_testkit::sumcheck::{Fe, Form, Product, Transcript, eq_table, mle, prove_after_gamma, tce};

/// Fp2 to the test kit's field.
pub fn to_fe(x: Fp2) -> Fe {
    Fe(x.c0.to_canonical(), x.c1.to_canonical())
}
/// The test kit's field to Fp2.
pub fn from_fe(x: Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(x.0).unwrap(),
        Fp::from_canonical(x.1).unwrap(),
    )
}
fn base(x: Fp) -> Fe {
    Fe::base(x.to_canonical())
}

/// `f̃(point)` for a table of `2^{point.len()}` values, summed against `eq` (RFC-0010
/// 'Multilinear extensions'), on the verifier's field.
pub fn extension(values: &[Fp2], point: &[Fp2]) -> Fp2 {
    assert_eq!(values.len(), 1 << point.len());
    values.iter().enumerate().fold(Fp2::ZERO, |sum, (x, &v)| {
        sum + v * point.iter().enumerate().fold(Fp2::ONE, |w, (bit, &r)| {
            w * if x >> bit & 1 == 1 { r } else { Fp2::ONE - r }
        })
    })
}

// ----- The witness ------------------------------------------------------------------------------

/// Every merged polynomial's table, in [`Grouped::polynomials`]' order and in its natural variable
/// order: base-field values, canonical.
#[derive(Clone, Debug)]
pub struct Witness {
    pub tables: Vec<Vec<u64>>,
}

impl Witness {
    /// Tensors uniform in `0..range`, forest leaves uniform and nonzero.
    pub fn random(grouped: &Grouped, rng: &mut SplitMix64, range: u64) -> Self {
        let tables = grouped
            .polynomials()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (0..1u64 << p.variables)
                    .map(|_| {
                        if i < grouped.values() {
                            rng.next_u64() % range
                        } else {
                            1 + rng.next_u64() % 1000
                        }
                    })
                    .collect()
            })
            .collect();
        Self { tables }
    }

    fn at(&self, polynomial: usize, index: u64) -> Fe {
        Fe::base(self.tables[polynomial][index as usize])
    }

    /// The table on the verifier's field.
    pub fn fp2(&self, polynomial: usize) -> Vec<Fp2> {
        self.tables[polynomial]
            .iter()
            .map(|&x| Fp2::from(Fp::reduce(x)))
            .collect()
    }
}

/// The natural index of a polynomial's entry: its variables outside `S` take `outside`'s bits in
/// order, those in `S` take `bits`'.
fn natural(polynomial: &Polynomial, outside: impl Fn(usize) -> u64, bits: u32) -> u64 {
    let (mut index, mut free, mut fixed) = (0u64, 0usize, 0u32);
    for v in 0..polynomial.variables {
        let bit = if polynomial.group_variables.contains(&v) {
            fixed += 1;
            u64::from(bits >> (fixed - 1) & 1)
        } else {
            free += 1;
            outside(free - 1)
        };
        index |= bit << v;
    }
    index
}

/// A split value's tensor entry at its block's Boolean entry `entry`: variable `j` outside `S`
/// takes block variable `coordinates[j]`, which lies at `P_h[coordinates[j]]` (RFC-0045 'The leaf
/// point'; RFC-0054 'The group's claim').
fn tensor_entry(witness: &Witness, polynomial: &Polynomial, value: &Value, entry: u64) -> Fe {
    let index = natural(
        polynomial,
        |j| entry >> value.coordinates[j] & 1,
        value.bits,
    );
    witness.at(value.polynomial, index)
}

fn form(witness: &Witness, polynomials: &[Polynomial], group: &Group, f: &Affine, e: u64) -> Fe {
    f.terms.iter().fold(base(f.constant), |sum, &(a, at)| {
        let value = &group.split[at];
        sum + base(a) * tensor_entry(witness, &polynomials[value.polynomial], value, e)
    })
}

/// One instance's column in a group: its blocks' entries in column order, `m` of them, before
/// the row-0 padding (RFC-0045 'The column').
#[derive(Clone, Debug)]
pub struct Column {
    pub table: TableId,
    pub has_values: bool,
    pub height: u32,
    pub entries: Vec<(Fe, Fe)>,
}

pub fn columns(grouped: &Grouped, group: &Group, witness: &Witness) -> Vec<Column> {
    let polynomials = grouped.polynomials();
    group
        .instances
        .iter()
        .map(|instance| {
            let mut entries = vec![(Fe::ZERO, Fe::ZERO); instance.entries as usize];
            for block in &instance.blocks {
                for e in 0..1u64 << block.variables {
                    let key = form(witness, polynomials, group, &block.key, e);
                    let value = block
                        .value
                        .as_ref()
                        .map_or(Fe::ZERO, |v| form(witness, polynomials, group, v, e));
                    entries[(block.offset + e) as usize] = (key, value);
                }
            }
            Column {
                table: instance.table,
                has_values: instance.has_values,
                height: instance.height,
                entries,
            }
        })
        .collect()
}

// ----- The tables -------------------------------------------------------------------------------

/// A table as the test defines it: its rows padded with copies of row 0 to `2^{h_t}`, and the
/// multiplicities of the lookups of every group.
#[derive(Clone, Debug)]
pub struct Table {
    pub id: TableId,
    pub has_values: bool,
    pub rows: Vec<(Fe, Fe)>,
    pub multiplicities: Vec<u64>,
}

impl Table {
    pub fn height(&self) -> u32 {
        self.rows.len().trailing_zeros()
    }
}

type Found = (TableId, bool, Vec<(Fe, Fe)>);

/// Tables of exactly the pairs the lookups take, row 0 the first one met, then
/// [`recount`]'s multiplicities.
pub fn cover(grouped: &Grouped, witness: &Witness) -> Vec<Table> {
    // Each table met: its id, whether it has values, and its distinct pairs.
    let mut found: Vec<Found> = Vec::new();
    for h in 0..grouped.count() {
        let group = grouped.group(h).unwrap();
        for column in columns(grouped, &group, witness) {
            let at = match found.iter().position(|(id, ..)| *id == column.table) {
                Some(at) => at,
                None => {
                    found.push((column.table, column.has_values, Vec::new()));
                    found.len() - 1
                }
            };
            for pair in column.entries {
                if !found[at].2.contains(&pair) {
                    found[at].2.push(pair);
                }
            }
        }
    }
    found.sort_by_key(|(id, ..)| *id);
    let mut tables: Vec<Table> = found
        .into_iter()
        .map(|(id, has_values, mut rows)| {
            let row0 = rows[0];
            rows.resize(rows.len().next_power_of_two(), row0);
            Table {
                id,
                has_values,
                multiplicities: vec![0; rows.len()],
                rows,
            }
        })
        .collect();
    recount(grouped, witness, &mut tables);
    tables
}

/// Every group's column entries counted at their first matching row, each column's padding to
/// `2^h` at row 0; an entry that is no row is counted nowhere.
pub fn recount(grouped: &Grouped, witness: &Witness, tables: &mut [Table]) {
    for table in tables.iter_mut() {
        table.multiplicities = vec![0; table.rows.len()];
    }
    for h in 0..grouped.count() {
        let group = grouped.group(h).unwrap();
        for column in columns(grouped, &group, witness) {
            let table = tables.iter_mut().find(|t| t.id == column.table).unwrap();
            for pair in &column.entries {
                if let Some(j) = table.rows.iter().position(|row| row == pair) {
                    table.multiplicities[j] += 1;
                }
            }
            table.multiplicities[0] += (1u64 << column.height) - column.entries.len() as u64;
        }
    }
}

/// The verifier's table trees.
pub fn table_trees(tables: &[Table]) -> Vec<TableTree> {
    tables
        .iter()
        .map(|t| TableTree {
            table: t.id,
            height: t.height(),
        })
        .collect()
}

/// `t̃(point) = kẽy_t + β·valũe_t` over the padded rows.
pub fn table_eval(tables: &[Table], id: TableId, beta: Fp2, point: &[Fp2]) -> Option<Fp2> {
    let table = tables.iter().find(|t| t.id == id)?;
    let rows: Vec<Fp2> = table
        .rows
        .iter()
        .map(|&(k, v)| from_fe(k) + beta * from_fe(v))
        .collect();
    (rows.len() == 1 << point.len()).then(|| extension(&rows, point))
}

/// Row 0's key, and its value for a table with values.
pub fn row_zero(tables: &[Table], id: TableId) -> Option<(Fp, Option<Fp>)> {
    let table = tables.iter().find(|t| t.id == id)?;
    let (k, v) = table.rows[0];
    let base = |x: Fe| Fp::from_canonical(x.0).unwrap();
    Some((base(k), table.has_values.then(|| base(v))))
}

/// A forest's root table from its stacked leaves: tree `y`'s root is the product of leaves
/// `y·2^h .. (y + 1)·2^h` (RFC-0010 'Tree levels', 'Layout').
pub fn forest_roots(witness: &Witness, polynomial: usize, height: u32) -> Vec<Fp2> {
    witness.tables[polynomial]
        .chunks(1 << height)
        .map(|tree| {
            tree.iter()
                .fold(Fp2::ONE, |p, &x| p * Fp2::from(Fp::reduce(x)))
        })
        .collect()
}

// ----- The levels -------------------------------------------------------------------------------

/// A tree of a batch, in schedule order.
#[derive(Clone, Debug)]
pub enum Tree {
    /// A fraction tree, and the root the prover sends for it.
    Fraction { tree: FractionTree, sent: [Fe; 2] },
    /// A forest of `2^s` product trees.
    Forest(Forest),
}

impl Tree {
    fn height(&self) -> usize {
        match self {
            Tree::Fraction { tree, .. } => tree.height(),
            Tree::Forest(forest) => forest.height(),
        }
    }
}

/// A tree's claims after its last level: `Ṽ` for a forest, `Ñ, D̃` for a fraction tree.
#[derive(Clone, Debug)]
pub struct Leaf {
    pub point: Vec<Fe>,
    pub values: Vec<Fe>,
}

/// A batch's levels under `{prefix}/{ℓ}/…` (RFC-0010 'Tree levels'), its leaf claims and its leaf
/// point `P`: the last line challenge, then the last level's round challenges from the last.
pub fn prove_levels(
    t: &mut Transcript,
    prefix: &str,
    trees: &[Tree],
) -> (Vec<SumcheckProof>, Vec<Leaf>, Vec<Fe>) {
    let tallest = trees.iter().map(Tree::height).max().unwrap_or(0);
    let mut current: Vec<Option<Leaf>> = vec![None; trees.len()];
    let mut proofs = Vec::new();
    let mut leaf_point = Vec::new();
    for level in 0..tallest {
        let joins = |tree: &Tree| tree.height() > 0 && tallest - tree.height() == level;
        let s_max = trees
            .iter()
            .filter_map(|tree| match tree {
                Tree::Forest(f) if joins(tree) => Some(f.rows_log),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let roots: Vec<Fe> = (0..s_max)
            .map(|c| t.challenge_fp2(&format!("{prefix}/{level}/root/{c}")))
            .collect();
        for (i, tree) in trees.iter().enumerate() {
            if joins(tree) {
                current[i] = Some(match tree {
                    Tree::Forest(f) => {
                        let point = roots[s_max - f.rows_log..].to_vec();
                        let value = mle(f.roots(), &point);
                        Leaf {
                            point,
                            values: vec![value],
                        }
                    }
                    Tree::Fraction { sent, .. } => Leaf {
                        point: vec![],
                        values: sent.to_vec(),
                    },
                });
            }
        }
        let entering: Vec<(usize, Leaf)> = current
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.clone().map(|c| (i, c)))
            .collect();
        let gamma = t.challenge_fp2(&format!("{prefix}/{level}/gamma"));
        let lambda = entering
            .iter()
            .any(|(i, _)| matches!(trees[*i], Tree::Fraction { .. }))
            .then(|| t.challenge_fp2(&format!("{prefix}/{level}/lambda")));
        let evens = |x: &[Fe]| -> Vec<Fe> { x.iter().step_by(2).copied().collect() };
        let odds = |x: &[Fe]| -> Vec<Fe> { x.iter().skip(1).step_by(2).copied().collect() };
        let instances: Vec<Product> = entering
            .iter()
            .map(|(i, claim)| {
                let own = claim.point.len();
                let public = vec![eq_table(&claim.point)];
                match &trees[*i] {
                    Tree::Forest(f) => {
                        let below = &f.levels[own - f.rows_log + 1];
                        Product {
                            tables: vec![evens(below), odds(below)],
                            public,
                            lambda: None,
                            form: Form::Product,
                        }
                    }
                    Tree::Fraction { tree, .. } => {
                        let below = &tree.levels[own + 1];
                        let n: Vec<Fe> = below.iter().map(|&(n, _)| n).collect();
                        let d: Vec<Fe> = below.iter().map(|&(_, d)| d).collect();
                        Product {
                            tables: vec![evens(&n), odds(&n), evens(&d), odds(&d)],
                            public,
                            lambda,
                            form: Form::Product,
                        }
                    }
                }
            })
            .collect();
        let level_prefix = format!("{prefix}/{level}");
        let proof = prove_after_gamma(t, &level_prefix, 3, gamma, &instances, None);
        let rho = t.challenge_fp2(&format!("{level_prefix}/line"));
        let mut evals = proof.evals.iter().copied();
        for ((i, claim), instance) in entering.iter().zip(&instances) {
            let own = &proof.challenges[proof.challenges.len() - claim.point.len()..];
            let mine: Vec<Fe> = evals.by_ref().take(instance.tables.len()).collect();
            current[*i] = Some(Leaf {
                point: [&[rho][..], own].concat(),
                values: mine
                    .chunks_exact(2)
                    .map(|pair| pair[0] + rho * (pair[1] - pair[0]))
                    .collect(),
            });
        }
        if level + 1 == tallest {
            leaf_point = std::iter::once(rho)
                .chain(proof.challenges.iter().rev().copied())
                .collect();
        }
        proofs.push(proof_of(&proof.rounds, &proof.evals));
    }
    let leaves = trees
        .iter()
        .zip(current)
        .map(|(tree, claim)| match (tree, claim) {
            (_, Some(claim)) => claim,
            (Tree::Fraction { sent, .. }, None) => Leaf {
                point: vec![],
                values: sent.to_vec(),
            },
            (Tree::Forest(_), None) => panic!("a forest joins a level"),
        })
        .collect();
    (proofs, leaves, leaf_point)
}

fn proof_of(rounds: &[Vec<Fe>], evals: &[Fe]) -> SumcheckProof {
    SumcheckProof {
        rounds: rounds
            .iter()
            .map(|r| r.iter().copied().map(from_fe).collect())
            .collect(),
        final_evals: evals.iter().copied().map(from_fe).collect(),
    }
}
fn fp2s(values: &[Fe]) -> Vec<Fp2> {
    values.iter().copied().map(from_fe).collect()
}

/// `σ_h(i) = (i mod 2) + 2·rev_{h−1}(⌊i / 2⌋)` (RFC-0045 'The leaves').
pub fn sigma(height: u32, i: u64) -> u64 {
    if height <= 1 {
        return i;
    }
    let high = i >> 1;
    let mut reversed = 0;
    for b in 0..height - 1 {
        reversed |= (high >> b & 1) << (height - 2 - b);
    }
    (i & 1) | reversed << 1
}

// ----- The prover -------------------------------------------------------------------------------

/// What a cheating prover changes before it absorbs it; the levels still prove the true trees.
pub enum Tamper<'a> {
    /// Group `h`'s lookup tree of instance `i`, its leaves before the tree is built.
    LookupLeaves(u32, usize, &'a mut Vec<(Fe, Fe)>),
    /// Group `h`'s roots message.
    GroupRoots(u32, &'a mut Vec<Fe>),
    /// Group `h`'s values message.
    Values(u32, &'a mut Vec<Fe>),
    /// Group `h`'s split values.
    Splits(u32, &'a mut Vec<Fe>),
    /// The table roots message.
    TableRoots(&'a mut Vec<Fe>),
}

/// A merge claim: group `h`'s `k`-th, on polynomial `polynomial`, at `z ‖ h_S`.
#[derive(Clone, Debug)]
struct MergeClaim {
    polynomial: usize,
    z: Vec<Fe>,
    bits: u32,
    weight: Fe,
}

/// A proof and what its prover saw.
pub struct Proved {
    pub logup: LogUpProof,
    pub trees: TreeProof,
    /// Each group's leaf point `P_h`.
    pub points: Vec<Vec<Fp2>>,
    /// The merge's challenges.
    pub merge_point: Vec<Fp2>,
}

fn eq_boolean(z: &[Fe], x: u64) -> Fe {
    z.iter().enumerate().fold(Fe::ONE, |w, (j, &r)| {
        w * if x >> j & 1 == 1 { r } else { Fe::ONE - r }
    })
}

/// The grouped stage of `witness` over `tables`, RFC-0014 rows 4 and 5 with groups, in order.
pub fn prove(
    t: &mut Transcript,
    grouped: &Grouped,
    witness: &Witness,
    tables: &[Table],
    mut tamper: impl FnMut(Tamper<'_>),
) -> Proved {
    let polynomials = grouped.polynomials();
    let compress = |(k, v): (Fe, Fe), beta: Fe| k + beta * v;
    // Step 1.
    let alpha = t.challenge_fp2("logup/alpha");
    let beta = t.challenge_fp2("logup/beta");
    t.absorb("logup/roots", &tce(&[]));
    t.absorb("logup/values", &tce(&[]));
    let row0: HashMap<TableId, (Fe, Fe)> = tables.iter().map(|t| (t.id, t.rows[0])).collect();
    // Step 2.
    let mut groups = Vec::new();
    let mut claims: Vec<(u32, Fe, Vec<MergeClaim>)> = Vec::new();
    let mut points = Vec::new();
    for h in 0..grouped.count() {
        let group = grouped.group(h).unwrap();
        let columns = columns(grouped, &group, witness);
        let mut trees = Vec::new();
        let mut value_leaves = Vec::new();
        for (i, column) in columns.iter().enumerate() {
            let size = 1u64 << column.height;
            let entry = |q: u64| {
                column
                    .entries
                    .get(q as usize)
                    .copied()
                    .unwrap_or(row0[&column.table])
            };
            let mut leaves: Vec<(Fe, Fe)> = (0..size)
                .map(|i| {
                    (
                        Fe::ONE,
                        alpha - compress(entry(sigma(column.height, i)), beta),
                    )
                })
                .collect();
            tamper(Tamper::LookupLeaves(h, i, &mut leaves));
            value_leaves.push(
                (0..size)
                    .map(|i| entry(sigma(column.height, i)).1)
                    .collect::<Vec<Fe>>(),
            );
            trees.push(FractionTree::new(leaves));
        }
        let mut roots: Vec<Fe> = trees
            .iter()
            .flat_map(|tree| {
                let (n, d) = tree.root();
                [n, d]
            })
            .collect();
        tamper(Tamper::GroupRoots(h, &mut roots));
        t.absorb(&format!("group/{h}/logup/roots"), &tce(&roots));
        let mut batch: Vec<Tree> = trees
            .into_iter()
            .enumerate()
            .map(|(i, tree)| Tree::Fraction {
                tree,
                sent: [roots[2 * i], roots[2 * i + 1]],
            })
            .collect();
        for share in &group.forests {
            let polynomial = &polynomials[share.polynomial];
            let size = 1u64 << share.variables().unwrap();
            let leaves = (0..size)
                .map(|x| {
                    let index = natural(polynomial, |j| x >> j & 1, share.bits);
                    witness.at(share.polynomial, index)
                })
                .collect();
            batch.push(Tree::Forest(Forest::new(share.trees_log as usize, leaves)));
        }
        let (levels, leaves, point) = prove_levels(t, &format!("group/{h}/tree"), &batch);
        // The values: each lookup tree's value polynomial, in leaf order, at its leaf point.
        let mut values: Vec<Fe> = columns
            .iter()
            .zip(&leaves)
            .zip(&value_leaves)
            .filter(|((column, _), _)| column.has_values)
            .map(|((_, leaf), table)| mle(table, &leaf.point))
            .collect();
        tamper(Tamper::Values(h, &mut values));
        t.absorb(&format!("group/{h}/logup/values"), &tce(&values));
        // The split: each tensor at `P_h` outside `S`, at `h`'s bits on `S`.
        let mut splits: Vec<Fe> = group
            .split
            .iter()
            .map(|value| {
                let polynomial = &polynomials[value.polynomial];
                let table: Vec<Fe> = witness.tables[value.polynomial]
                    .iter()
                    .map(|&x| Fe::base(x))
                    .collect();
                let mut at = Vec::new();
                let (mut free, mut fixed) = (0usize, 0u32);
                for v in 0..polynomial.variables {
                    if polynomial.group_variables.contains(&v) {
                        at.push(Fe::base(u64::from(value.bits >> fixed & 1)));
                        fixed += 1;
                    } else {
                        at.push(point[value.coordinates[free] as usize]);
                        free += 1;
                    }
                }
                mle(&table, &at)
            })
            .collect();
        tamper(Tamper::Splits(h, &mut splits));
        t.absorb(&format!("group/{h}/split"), &tce(&splits));
        let gamma = t.challenge_fp2(&format!("group/{h}/merge/gamma"));
        // Its claims, in order: the split values, then the shares' leaf claims.
        let mut mine = Vec::new();
        let mut weight = Fe::ONE;
        for value in &group.split {
            mine.push(MergeClaim {
                polynomial: value.polynomial,
                z: value
                    .coordinates
                    .iter()
                    .map(|&c| point[c as usize])
                    .collect(),
                bits: value.bits,
                weight,
            });
            weight = weight * gamma;
        }
        for (share, leaf) in group.forests.iter().zip(&leaves[columns.len()..]) {
            mine.push(MergeClaim {
                polynomial: share.polynomial,
                z: leaf.point.clone(),
                bits: share.bits,
                weight,
            });
            weight = weight * gamma;
        }
        claims.push((h, gamma, mine));
        points.push(fp2s(&point));
        groups.push(GroupProof {
            roots: fp2s(&roots),
            levels,
            values: fp2s(&values),
            splits: fp2s(&splits),
        });
    }
    // Step 3. The table trees, leaves `(−m_j, α − t_j)` over the padded rows.
    let trees: Vec<FractionTree> = tables
        .iter()
        .map(|table| {
            FractionTree::new(
                table
                    .rows
                    .iter()
                    .zip(&table.multiplicities)
                    .map(|(&row, &m)| (-Fe::base(m), alpha - compress(row, beta)))
                    .collect(),
            )
        })
        .collect();
    let mut table_roots: Vec<Fe> = trees
        .iter()
        .flat_map(|tree| {
            let (n, d) = tree.root();
            [n, d]
        })
        .collect();
    tamper(Tamper::TableRoots(&mut table_roots));
    t.absorb("tables/logup/roots", &tce(&table_roots));
    let batch: Vec<Tree> = trees
        .into_iter()
        .enumerate()
        .map(|(i, tree)| Tree::Fraction {
            tree,
            sent: [table_roots[2 * i], table_roots[2 * i + 1]],
        })
        .collect();
    let (table_levels, _, _) = prove_levels(t, "tables/tree", &batch);
    // Step 4. The merge: per polynomial, `P_i` and `W_i` in its merge order (outside `S` first,
    // then `S`, each lowest first), `W_i` built on the cube from every group's claims.
    let merge_beta = t.challenge_fp2("merge/beta");
    let mut by_polynomial: Vec<Vec<MergeClaim>> = vec![Vec::new(); polynomials.len()];
    let mut beta_h = Fe::ONE;
    for (_, _, mine) in &claims {
        for claim in mine {
            let mut claim = claim.clone();
            claim.weight = claim.weight * beta_h;
            by_polynomial[claim.polynomial].push(claim);
        }
        beta_h = beta_h * merge_beta;
    }
    let instances: Vec<Product> = polynomials
        .iter()
        .enumerate()
        .map(|(i, polynomial)| {
            let n = polynomial.variables;
            let order: Vec<u32> = (0..n)
                .filter(|v| !polynomial.group_variables.contains(v))
                .chain(polynomial.group_variables.iter().copied())
                .collect();
            let free = (n as usize) - polynomial.group_variables.len();
            let p: Vec<Fe> = (0..1u64 << n)
                .map(|x| {
                    let index = order
                        .iter()
                        .enumerate()
                        .fold(0u64, |acc, (j, &v)| acc | (x >> j & 1) << v);
                    witness.at(i, index)
                })
                .collect();
            let w: Vec<Fe> = (0..1u64 << n)
                .map(|x| {
                    by_polynomial[i].iter().fold(Fe::ZERO, |sum, claim| {
                        if x >> free != u64::from(claim.bits) {
                            return sum;
                        }
                        sum + claim.weight * eq_boolean(&claim.z, x & ((1 << free) - 1))
                    })
                })
                .collect();
            Product {
                tables: vec![p],
                public: vec![w],
                lambda: None,
                form: Form::Product,
            }
        })
        .collect();
    let merge = prove_after_gamma(t, "merge", 2, Fe::ONE, &instances, None);
    Proved {
        logup: LogUpProof {
            roots: vec![],
            values: vec![],
        },
        trees: TreeProof {
            levels: table_levels,
            groups,
            table_roots: fp2s(&table_roots),
            merge: proof_of(&merge.rounds, &merge.evals),
            splits: vec![],
        },
        points,
        merge_point: fp2s(&merge.challenges),
    }
}
