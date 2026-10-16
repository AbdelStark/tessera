// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A reference prover and an oracle for advice openings, for tests only.
//!
//! Both are written from RFC-0011 ('Encoding', 'Column digests and root', 'Opening protocol'),
//! RFC-0004 ('Column digests', 'Merkle trees') and the crate documentation's order of checks, and
//! compute with no code of `tv-core`, `tv-field` or `tv-hash`, whose types they use only to hand
//! the verifier its inputs: the field is the test kit's `Fe`, `u128` arithmetic modulo `P`;
//! hashes are the test kit's reference BLAKE3; a codeword symbol is the value of Lagrange's
//! interpolant through the row's values on `H`, plus the mask term, at `7·ω_n^j`, with no NTT;
//! `eq` is its product formula at each row; a multiproof is built and read level by level.
//!
//! - [`Matrix`] commits seeded rows and masks of any shape and opens them honestly
//!   ([`Matrix::open`]). A test can change its committed codewords first, as a cheating prover
//!   would: the opening then carries honest combined rows against changed columns.
//! - [`predict`] gives the verdict of the crate documentation's checks for any opening, checking
//!   the columns query by query and row by row, where `tv-core` checks them row by row.

use std::collections::BTreeSet;

use tv_core::opening::{Claim, CombinedRow, Group, OpeningError, Params};
use tv_field::{Fp, Fp2};
use tv_testkit::SplitMix64;
use tv_testkit::blake3::derive_key;
use tv_testkit::sumcheck::{Fe, P};

use super::Case;

/// A 32-byte digest.
pub type Digest = [u8; 32];

/// `H(domain, bytes) = BLAKE3.derive_key("tessera/v1/" + domain, bytes)`.
pub fn hash(domain: &str, bytes: &[u8]) -> Digest {
    derive_key(&format!("tessera/v1/{domain}"), bytes)
}

// ----- Conversions --------------------------------------------------------------------------------

/// An `Fe` as the verifier's `Fp2`.
pub fn to_fp2(x: Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(x.0).expect("canonical"),
        Fp::from_canonical(x.1).expect("canonical"),
    )
}

/// The verifier's `Fp2` as an `Fe`.
pub fn to_fe(x: Fp2) -> Fe {
    Fe(x.c0.to_canonical(), x.c1.to_canonical())
}

/// An `Fe` of the base field as the verifier's `Fp`.
pub fn to_fp(x: Fe) -> Fp {
    assert_eq!(x.1, 0, "a base-field element");
    Fp::from_canonical(x.0).expect("canonical")
}

/// A seeded element of `Fp2`.
pub fn random_fe(rng: &mut SplitMix64) -> Fe {
    Fe::new(rng.next_u64(), rng.next_u64())
}

/// A seeded element of `Fp`, as an `Fe`.
pub fn random_base(rng: &mut SplitMix64) -> Fe {
    Fe::base(rng.next_u64())
}

/// `rng`'s next value below `bound`.
pub fn below(rng: &mut SplitMix64, bound: u64) -> u64 {
    rng.next_u64() % bound
}

// ----- The field's structure (RFC-0002 'Constants', RFC-0011 'Encoding') -------------------------

/// `ω_m = 7^((P − 1)/m)`.
pub fn root_of_unity(m: u64) -> Fe {
    Fe::base(7).pow(u128::from((P - 1) / m))
}

/// `x_j = 7·ω_n^j`, the point of column `j`.
pub fn column_point(n: u64, j: u64) -> Fe {
    Fe::base(7) * root_of_unity(n).pow(u128::from(j))
}

/// `eq(r, x) = Π_t (r_t·x_t + (1 − r_t)·(1 − x_t))`, `x_t` bit `t` of `x`.
pub fn eq_at(r: &[Fe], x: u64) -> Fe {
    r.iter().enumerate().fold(Fe::ONE, |product, (t, &rt)| {
        let bit = t < 64 && (x >> t) & 1 == 1;
        product * if bit { rt } else { Fe::ONE - rt }
    })
}

/// `Σ_i coefficients[i]·x^i`, by Horner's rule.
pub fn horner(coefficients: &[Fe], x: Fe) -> Fe {
    coefficients
        .iter()
        .rev()
        .fold(Fe::ZERO, |acc, &c| acc * x + c)
}

/// Lagrange's interpolant through `k` values on `H = {ω_k^i}`, evaluated anywhere:
/// `Σ_i v_i·Π_{m ≠ i} (x − h_m)/(h_i − h_m)`, the denominators inverted once.
pub struct Interpolator {
    nodes: Vec<Fe>,
    weights: Vec<Fe>,
}

impl Interpolator {
    pub fn new(k: u64) -> Self {
        let w = root_of_unity(k);
        let nodes: Vec<Fe> = (0..k).map(|i| w.pow(u128::from(i))).collect();
        let weights = nodes
            .iter()
            .enumerate()
            .map(|(i, &hi)| {
                let denominator = nodes
                    .iter()
                    .enumerate()
                    .filter(|&(m, _)| m != i)
                    .fold(Fe::ONE, |product, (_, &hm)| product * (hi - hm));
                denominator.inverse().expect("distinct nodes")
            })
            .collect();
        Self { nodes, weights }
    }

    /// The interpolant of `values` at `x`.
    pub fn at(&self, values: &[Fe], x: Fe) -> Fe {
        assert_eq!(values.len(), self.nodes.len());
        let factors: Vec<Fe> = self.nodes.iter().map(|&h| x - h).collect();
        // Π_{m ≠ i} (x − h_m) from prefix and suffix products.
        let mut prefix = vec![Fe::ONE; factors.len() + 1];
        for (i, &f) in factors.iter().enumerate() {
            prefix[i + 1] = prefix[i] * f;
        }
        let mut suffix = Fe::ONE;
        let mut sum = Fe::ZERO;
        for i in (0..factors.len()).rev() {
            sum = sum + values[i] * self.weights[i] * prefix[i] * suffix;
            suffix = suffix * factors[i];
        }
        sum
    }

    /// A combined row's value at `x`: `enc(u)(x) + (x^k − 1)·M(x)`.
    pub fn encode_at(&self, u: &[Fe], mask: &[Fe], x: Fe) -> Fe {
        let k = self.nodes.len() as u128;
        self.at(u, x) + (x.pow(k) - Fe::ONE) * horner(mask, x)
    }
}

// ----- Hashing (RFC-0004 'Column digests', 'Merkle trees'; RFC-0011 'Column digests and root') ---

/// Segment `symbols`' chained digest at column `j`.
pub fn column_digest(commitment_id: &Digest, j: u64, block: usize, symbols: &[Fe]) -> Digest {
    let mut chain = hash(
        "column-init",
        &[&commitment_id[..], &j.to_le_bytes()].concat(),
    );
    for step in symbols.chunks(block) {
        let mut input = chain.to_vec();
        for symbol in step {
            assert_eq!(symbol.1, 0, "a base-field symbol");
            input.extend_from_slice(&symbol.0.to_le_bytes());
        }
        chain = hash("column-step", &input);
    }
    chain
}

/// `leaf_j`: the segment digests of column `j` merged, the column's symbols split by segment.
pub fn leaf(
    commitment_id: &Digest,
    block: usize,
    segments: &[Vec<u64>],
    j: u64,
    column: &[Fe],
) -> Digest {
    let mut merged = Vec::new();
    let mut start = 0;
    for segment in segments {
        let own = &column[start..start + segment.len()];
        merged.extend_from_slice(&column_digest(commitment_id, j, block, own));
        start += segment.len();
    }
    assert_eq!(start, column.len());
    hash("column-merge", &merged)
}

/// `H("merkle-node", left ‖ right)`.
pub fn node(left: &Digest, right: &Digest) -> Digest {
    hash("merkle-node", &[&left[..], &right[..]].concat())
}

/// The levels of a Merkle tree, the leaves first and the root last.
pub fn tree(leaves: Vec<Digest>) -> Vec<Vec<Digest>> {
    assert!(leaves.len().is_power_of_two());
    let mut levels = vec![leaves];
    while levels.last().unwrap().len() > 1 {
        let next = levels
            .last()
            .unwrap()
            .chunks(2)
            .map(|pair| node(&pair[0], &pair[1]))
            .collect();
        levels.push(next);
    }
    levels
}

/// The canonical multiproof of the sorted, distinct `indices`: level by level from the leaves,
/// left to right, every sibling not itself known.
pub fn multiproof(levels: &[Vec<Digest>], indices: &[u64]) -> Vec<Digest> {
    let mut known: Vec<u64> = indices.to_vec();
    let mut siblings = Vec::new();
    for level in &levels[..levels.len() - 1] {
        let mut parents = Vec::new();
        let mut i = 0;
        while i < known.len() {
            let index = known[i];
            if index % 2 == 0 && known.get(i + 1) == Some(&(index + 1)) {
                i += 2;
            } else {
                siblings.push(level[(index ^ 1) as usize]);
                i += 1;
            }
            parents.push(index / 2);
        }
        known = parents;
    }
    siblings
}

/// The root that `siblings`, read as a canonical multiproof, gives the sorted, distinct leaves of
/// a tree of `2^levels` leaves, or `None` when it holds too few or too many digests.
pub fn multiproof_root(
    leaves: &[(u64, Digest)],
    levels: u32,
    siblings: &[Digest],
) -> Option<Digest> {
    let mut known = leaves.to_vec();
    let mut proof = siblings.iter();
    for _ in 0..levels {
        let mut parents = Vec::new();
        let mut i = 0;
        while i < known.len() {
            let (index, digest) = known[i];
            let (left, right) =
                if index % 2 == 0 && known.get(i + 1).map(|n| n.0) == Some(index + 1) {
                    i += 2;
                    (digest, known[i - 1].1)
                } else {
                    i += 1;
                    let sibling = *proof.next()?;
                    if index % 2 == 0 {
                        (digest, sibling)
                    } else {
                        (sibling, digest)
                    }
                };
            parents.push((index / 2, node(&left, &right)));
        }
        known = parents;
    }
    if proof.next().is_some() {
        return None;
    }
    Some(known[0].1)
}

// ----- The prover ---------------------------------------------------------------------------------

/// A claim to open: its row point and batching challenge. Its value is the matrix's.
#[derive(Clone, Debug)]
pub struct ClaimSpec {
    pub row_point: Vec<Fe>,
    pub gamma: Fe,
}

/// A group to open: its column point and claims.
#[derive(Clone, Debug)]
pub struct GroupSpec {
    pub column_point: Vec<Fe>,
    pub claims: Vec<ClaimSpec>,
}

/// A committed advice matrix: its non-zero rows' values and masks, and the codewords committed,
/// which a cheating prover may have changed.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub params: Params,
    pub row_vars: u32,
    pub segments: Vec<Vec<u64>>,
    /// Each non-zero row's `k` values in `Fp`, in layout order.
    pub values: Vec<Vec<Fe>>,
    /// Each non-zero row's `P_A` mask coefficients in `Fp`.
    pub masks: Vec<Vec<Fe>>,
    /// Each non-zero row's committed codeword, `n` symbols.
    pub codewords: Vec<Vec<Fe>>,
}

impl Matrix {
    /// Seeded rows and masks for the layout, committed honestly.
    pub fn random(
        params: Params,
        row_vars: u32,
        segments: Vec<Vec<u64>>,
        rng: &mut SplitMix64,
    ) -> Self {
        let k = 1usize << params.log2_k;
        let count: usize = segments.iter().map(Vec::len).sum();
        let values: Vec<Vec<Fe>> = (0..count)
            .map(|_| (0..k).map(|_| random_base(rng)).collect())
            .collect();
        let masks: Vec<Vec<Fe>> = (0..count)
            .map(|_| (0..params.mask_coeffs).map(|_| random_base(rng)).collect())
            .collect();
        let mut matrix = Self {
            params,
            row_vars,
            segments,
            values,
            masks,
            codewords: Vec::new(),
        };
        matrix.codewords = (0..count).map(|r| matrix.encode_row(r)).collect();
        matrix
    }

    pub fn k(&self) -> u64 {
        1 << self.params.log2_k
    }

    pub fn n(&self) -> u64 {
        1 << (self.params.log2_k + self.params.log2_inv_rate)
    }

    /// The non-zero rows in layout order.
    pub fn rows(&self) -> Vec<u64> {
        self.segments.iter().flatten().copied().collect()
    }

    /// Row `r`'s honest codeword: `g(x_j) = f(x_j) + (x_j^k − 1)·m(x_j)` for each column.
    pub fn encode_row(&self, r: usize) -> Vec<Fe> {
        let interpolator = Interpolator::new(self.k());
        (0..self.n())
            .map(|j| {
                interpolator.encode_at(&self.values[r], &self.masks[r], column_point(self.n(), j))
            })
            .collect()
    }

    /// Column `j` of the committed codewords.
    pub fn column(&self, j: u64) -> Vec<Fe> {
        self.codewords
            .iter()
            .map(|codeword| codeword[j as usize])
            .collect()
    }

    /// The Merkle tree over the committed columns' leaves.
    pub fn tree(&self) -> Vec<Vec<Digest>> {
        let block = self.params.row_block as usize;
        tree(
            (0..self.n())
                .map(|j| {
                    leaf(
                        &self.params.commitment_id,
                        block,
                        &self.segments,
                        j,
                        &self.column(j),
                    )
                })
                .collect(),
        )
    }

    /// `Ã(column_point ‖ row_point)`: every non-zero row's every value, weighted by `eq`.
    pub fn value(&self, column_point: &[Fe], row_point: &[Fe]) -> Fe {
        self.rows()
            .iter()
            .zip(&self.values)
            .fold(Fe::ZERO, |sum, (&row, values)| {
                let in_row = values
                    .iter()
                    .enumerate()
                    .fold(Fe::ZERO, |s, (i, &v)| s + v * eq_at(column_point, i as u64));
                sum + eq_at(row_point, row) * in_row
            })
    }

    /// The combined row of `coefficients`, one per non-zero row: `Σ c·A_row` and `Σ c·m_row`.
    pub fn combine(&self, coefficients: &[Fe]) -> (Vec<Fe>, Vec<Fe>) {
        let mut u = vec![Fe::ZERO; self.k() as usize];
        let mut mask = vec![Fe::ZERO; self.params.mask_coeffs as usize];
        for ((c, values), m) in coefficients.iter().zip(&self.values).zip(&self.masks) {
            for (slot, &v) in u.iter_mut().zip(values) {
                *slot = *slot + *c * v;
            }
            for (slot, &v) in mask.iter_mut().zip(m) {
                *slot = *slot + *c * v;
            }
        }
        (u, mask)
    }

    /// The honest opening of `groups` and `ξ*` at the draws `queries`, against the committed
    /// codewords.
    pub fn open(&self, groups: &[GroupSpec], xi: &[Fe], queries: &[u64]) -> Case {
        let rows = self.rows();
        let mut combined = Vec::new();
        let mut opened = Vec::new();
        for group in groups {
            let coefficients: Vec<Fe> = rows
                .iter()
                .map(|&row| {
                    group
                        .claims
                        .iter()
                        .fold(Fe::ZERO, |s, c| s + c.gamma * eq_at(&c.row_point, row))
                })
                .collect();
            combined.push(self.combine(&coefficients));
            opened.push(Group {
                column_point: group.column_point.iter().copied().map(to_fp2).collect(),
                claims: group
                    .claims
                    .iter()
                    .map(|claim| Claim {
                        row_point: claim.row_point.iter().copied().map(to_fp2).collect(),
                        gamma: to_fp2(claim.gamma),
                        value: to_fp2(self.value(&group.column_point, &claim.row_point)),
                    })
                    .collect(),
            });
        }
        let xi_coefficients: Vec<Fe> = rows.iter().map(|&row| eq_at(xi, row)).collect();
        combined.push(self.combine(&xi_coefficients));

        let distinct: Vec<u64> = queries
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let levels = self.tree();
        Case {
            name: String::from("reference"),
            params: self.params,
            row_vars: self.row_vars,
            segments: self.segments.clone(),
            root: levels.last().unwrap()[0],
            groups: opened,
            xi_point: xi.iter().copied().map(to_fp2).collect(),
            combined: combined
                .into_iter()
                .map(|(u, mask)| CombinedRow {
                    u: u.into_iter().map(to_fp2).collect(),
                    mask: mask.into_iter().map(to_fp2).collect(),
                })
                .collect(),
            queries: queries.to_vec(),
            columns: distinct
                .iter()
                .map(|&j| self.column(j).into_iter().map(to_fp).collect())
                .collect(),
            siblings: multiproof(&levels, &distinct),
            intermediate: None,
        }
    }
}

/// Seeded groups for a layout: `claims[g]` claims in group `g`, at random points.
pub fn random_groups(
    rng: &mut SplitMix64,
    log2_k: u32,
    row_vars: u32,
    claims: &[usize],
) -> Vec<GroupSpec> {
    claims
        .iter()
        .map(|&count| GroupSpec {
            column_point: (0..log2_k).map(|_| random_fe(rng)).collect(),
            claims: (0..count)
                .map(|_| ClaimSpec {
                    row_point: (0..row_vars).map(|_| random_fe(rng)).collect(),
                    gamma: random_fe(rng),
                })
                .collect(),
        })
        .collect()
}

/// `count` sorted, distinct rows below `2^row_vars`, split into `segments` segments at random
/// places, some possibly empty.
pub fn random_layout(
    rng: &mut SplitMix64,
    row_vars: u32,
    count: usize,
    segments: usize,
) -> Vec<Vec<u64>> {
    let mut rows = BTreeSet::new();
    let space = if row_vars >= 64 {
        u64::MAX
    } else {
        (1u64 << row_vars) - 1
    };
    assert!(count as u128 <= u128::from(space) + 1);
    // The extremes when they fit, then random rows.
    for row in [0, space] {
        if rows.len() < count && rng.next_u64() % 2 == 0 {
            rows.insert(row);
        }
    }
    while rows.len() < count {
        let row = if row_vars >= 64 {
            rng.next_u64()
        } else {
            below(rng, space + 1)
        };
        rows.insert(row);
    }
    let rows: Vec<u64> = rows.into_iter().collect();
    let mut cuts: Vec<usize> = (0..segments.saturating_sub(1))
        .map(|_| below(rng, count as u64 + 1) as usize)
        .collect();
    cuts.sort_unstable();
    let mut out = Vec::new();
    let mut start = 0;
    for cut in cuts.into_iter().chain([count]) {
        out.push(rows[start..cut].to_vec());
        start = cut;
    }
    out
}

// ----- The oracle ---------------------------------------------------------------------------------

/// The verdict of the crate documentation's checks on `case`, in their order, computed without
/// `tv-core`: the verifier's inputs; the combined rows' shape and the group sums; the queries;
/// the answer's shape, leaves and multiproof; then, query by query in increasing order and row by
/// row, groups then `ξ`, the consistency of every combined row with every column.
pub fn predict(case: &Case) -> Result<(), OpeningError> {
    use OpeningError::*;
    let Params {
        log2_k,
        log2_inv_rate,
        mask_coeffs,
        row_block,
        commitment_id,
    } = case.params;

    // The verifier's inputs.
    if log2_inv_rate == 0 || u64::from(log2_k) + u64::from(log2_inv_rate) > 32 {
        return Err(CodeLength {
            log2_k,
            log2_inv_rate,
        });
    }
    let k = 1u64 << log2_k;
    let log2_n = log2_k + log2_inv_rate;
    let n = 1u64 << log2_n;
    if mask_coeffs == 0 || u64::from(mask_coeffs) > k {
        return Err(MaskCoefficients { mask_coeffs });
    }
    if row_block == 0 {
        return Err(RowBlock);
    }
    if case.row_vars > 64 {
        return Err(RowVars {
            row_vars: case.row_vars,
        });
    }
    if case.segments.is_empty() {
        return Err(NoSegments);
    }
    let mut previous = None;
    for (segment, rows) in case.segments.iter().enumerate() {
        for (position, &row) in rows.iter().enumerate() {
            if previous.is_some_and(|p| row <= p) {
                return Err(RowOrder { segment, position });
            }
            if case.row_vars < 64 && row >= 1u64 << case.row_vars {
                return Err(RowRange { segment, position });
            }
            previous = Some(row);
        }
    }
    for (g, group) in case.groups.iter().enumerate() {
        if group.column_point.len() != log2_k as usize {
            return Err(ColumnPoint {
                group: g,
                length: group.column_point.len(),
            });
        }
        for (c, claim) in group.claims.iter().enumerate() {
            if claim.row_point.len() != case.row_vars as usize {
                return Err(RowPoint {
                    group: g,
                    claim: c,
                    length: claim.row_point.len(),
                });
            }
        }
    }
    if case.xi_point.len() != case.row_vars as usize {
        return Err(XiPoint {
            length: case.xi_point.len(),
        });
    }

    // The evaluation check.
    let shaped = case.combined.len() == case.groups.len() + 1
        && case
            .combined
            .iter()
            .all(|row| row.u.len() as u64 == k && row.mask.len() == mask_coeffs as usize);
    if !shaped {
        return Err(Evaluation { group: None });
    }
    for (g, (group, row)) in case.groups.iter().zip(&case.combined).enumerate() {
        let claimed = group
            .claims
            .iter()
            .fold(Fe::ZERO, |s, c| s + to_fe(c.gamma) * to_fe(c.value));
        let point: Vec<Fe> = group.column_point.iter().copied().map(to_fe).collect();
        let opened = row.u.iter().enumerate().fold(Fe::ZERO, |s, (i, &u)| {
            s + to_fe(u) * eq_at(&point, i as u64)
        });
        if claimed != opened {
            return Err(Evaluation { group: Some(g) });
        }
    }

    // The column check: the queries.
    if case.queries.is_empty() {
        return Err(NoQueries);
    }
    if let Some(position) = case.queries.iter().position(|&j| j >= n) {
        return Err(QueryRange { position });
    }
    let distinct: Vec<u64> = case
        .queries
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let merkle_path = MerklePath {
        column: distinct[0] as u32,
    };

    // The answer's shape, leaves and multiproof.
    let rows = case.rows();
    if case.columns.len() != distinct.len() || case.columns.iter().any(|c| c.len() != rows.len()) {
        return Err(merkle_path);
    }
    let columns: Vec<Vec<Fe>> = case
        .columns
        .iter()
        .map(|column| column.iter().map(|s| Fe(s.to_canonical(), 0)).collect())
        .collect();
    let leaves: Vec<(u64, Digest)> = distinct
        .iter()
        .zip(&columns)
        .map(|(&j, column)| {
            (
                j,
                leaf(
                    &commitment_id,
                    row_block as usize,
                    &case.segments,
                    j,
                    column,
                ),
            )
        })
        .collect();
    if multiproof_root(&leaves, log2_n, &case.siblings) != Some(case.root) {
        return Err(merkle_path);
    }

    // Consistency, query by query, row by row.
    let interpolator = Interpolator::new(k);
    let coefficients: Vec<Vec<Fe>> = case
        .groups
        .iter()
        .map(|group| {
            rows.iter()
                .map(|&row| {
                    group.claims.iter().fold(Fe::ZERO, |s, c| {
                        let point: Vec<Fe> = c.row_point.iter().copied().map(to_fe).collect();
                        s + to_fe(c.gamma) * eq_at(&point, row)
                    })
                })
                .collect()
        })
        .chain([{
            let xi: Vec<Fe> = case.xi_point.iter().copied().map(to_fe).collect();
            rows.iter().map(|&row| eq_at(&xi, row)).collect()
        }])
        .collect();
    let combined: Vec<(Vec<Fe>, Vec<Fe>)> = case
        .combined
        .iter()
        .map(|row| {
            (
                row.u.iter().copied().map(to_fe).collect(),
                row.mask.iter().copied().map(to_fe).collect(),
            )
        })
        .collect();
    for (&j, column) in distinct.iter().zip(&columns) {
        let x = column_point(n, j);
        for ((u, mask), c) in combined.iter().zip(&coefficients) {
            let sum = c
                .iter()
                .zip(column)
                .fold(Fe::ZERO, |s, (&c, &symbol)| s + c * symbol);
            if sum != interpolator.encode_at(u, mask, x) {
                return Err(Proximity { column: j as u32 });
            }
        }
    }
    Ok(())
}
