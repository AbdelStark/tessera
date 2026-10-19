// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The canonical advice layout (RFC-0011 'Canonical advice layout', RFC-0034 'Causal layout' and
//! 'Zero rows', RFC-0041 'Production order', RFC-0046, RFC-0047), from tiles given as plain data.
//!
//! - **The order.** A dense tensor's variables take RFC-0047's layout order: its padded axes, those
//!   whose length is not a power of two, in axis order, then the others in axis order.
//! - **Claims map to rows.** On random tiles, each tensor's elements are written into a resident
//!   `Ã` where the layout places them, and a claim `X̃(r)` evaluated on the tensor directly equals
//!   `Ã(r ‖ base_X)` at the layout's point. Every element lies in its tensor's rows and no two
//!   share an entry. Through the reference commitment, which holds the non-zero rows only, the
//!   independent verifier `tv_core::opening` accepts every such claim.
//! - **Zero rows.** Only the outermost length padding is skipped; inner padded axes' rows
//!   remain committed (RFC-0047). Alignment gaps and a score tile's masked rows are skipped. The descriptor's `rows` counts
//!   the others. At 405B shapes, the padding of the 53,248-wide MLP tensors costs no row, and
//!   the score tiles commit RFC-0034's closed form.
//! - **Segments.** They partition the rows, each boundary at a tile's first row, and hold the
//!   layers of RFC-0046: 16, 16, 16, 16, 16, 16, 15 and 15 at `L = 126, S = 8`, with empty
//!   segments when `S > L`.
//! - **Short sequences.** A sequence of `n < Q` takes `Q' = N`, and a tile of `g·N·N ≤ k`
//!   entries takes one row.
//! - **Refusals.** `g·min(Q, N_max) > k`, a `query_block` that is no power of two and a layout of
//!   more than `2^32` rows are `Layout`, and each malformed input is refused with its rule.
//! - **INV-PCS-1.** The layout reads public shapes only, and each public input that places a
//!   tensor moves it when it changes.
//!
//! The expected positions, rows and segments are computed here from the RFCs' formulas, never
//! from the crate's.

mod common;

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::ops::Range;

use common::openings::{TvFp2, Verifier, tv_columns, tv_combined, tv_fp2};
use common::{Fp, SplitMix64};
use tessera_codec::Secret;
use tessera_field::{Field, GoldilocksExt2, QuadraticExtension};
use tessera_pcs::{
    AdviceLayout, AdviceLayoutCounter, Axis, ClaimGroup, CommitmentParams, DenseShape,
    LayoutParams, Padding, PcsError, ReferenceCommitment, RowClaim, TensorShape, TileShape,
    layout_order, segment_layers,
};
use tv_core::opening as tv;

type Fp2 = GoldilocksExt2;

// ----- What the tests compute from the RFCs ----------------------------------------------------

/// The attention geometry of RFC-0034 and RFC-0046, from their formulas.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    n: u32,
    /// `log2 N`.
    n_bits: u32,
    /// `Q' = min(Q, N)`.
    queries: u32,
    blocks: u32,
    group: u32,
    /// `w = min(log2 N, log2 k − log2 Q' − log2 g)`.
    w: u32,
}

impl Geometry {
    fn new(params: &LayoutParams, n: u32) -> Self {
        let padded = n.next_power_of_two();
        let queries = params.query_block.min(padded);
        let n_bits = padded.trailing_zeros();
        let w =
            n_bits.min(params.log2_k - queries.trailing_zeros() - params.group.trailing_zeros());
        Self {
            n,
            n_bits,
            queries,
            blocks: padded / queries,
            group: params.group,
            w,
        }
    }

    /// The real queries of block `b`: those below `n`.
    fn held(&self, block: u32) -> u64 {
        u64::from(
            self.n
                .saturating_sub(block * self.queries)
                .min(self.queries),
        )
    }

    /// A per-row tensor of block `b`: `[g, Q']`, its queries past `n` padding. Both lengths are
    /// powers of two, so its layout order is row-major.
    fn row_shape(&self, block: u32) -> DenseShape {
        let shape = DenseShape::new(
            vec![
                Axis::new(u64::from(self.group)),
                Axis {
                    len: u64::from(self.queries),
                    real: self.held(block),
                },
            ],
            Padding::Zero,
        );
        assert_eq!(shape.order(), [0, 1]);
        shape
    }
}

/// One element of a tensor's padded domain: its coordinates, outermost axis first, its index in
/// the tensor's row-major table, and what it holds.
struct Element {
    coords: Vec<u64>,
    index: usize,
    kind: Entry,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Entry {
    /// A real entry.
    Real,
    /// Padding that holds 0, or a masked score, 0 by RFC-0030.
    Zero,
    /// Inner-axis length padding stays committed even when its value is zero (RFC-0047).
    RetainedZero,
    /// Padding that holds a public constant (RFC-0035).
    Constant,
}

/// `⌈log2 len⌉`: the bits of an axis of `len ≥ 1` positions padded to a power of two.
fn bits(len: u64) -> u32 {
    len.next_power_of_two().trailing_zeros()
}

/// The axes of a tensor's natural row-major domain: a dense tensor's own, or a score tile's
/// `[g, Q', N]`.
fn natural_bits(shape: &TensorShape, geometry: &Geometry) -> Vec<u32> {
    match shape {
        TensorShape::Dense(dense) => dense.axes().iter().map(|a| bits(a.len)).collect(),
        TensorShape::Scores { .. } => vec![
            geometry.group.trailing_zeros(),
            geometry.queries.trailing_zeros(),
            geometry.n_bits,
        ],
    }
}

/// Every element of the tensor's padded domain, in row-major order.
fn elements(shape: &TensorShape, geometry: &Geometry) -> Vec<Element> {
    let bits = natural_bits(shape, geometry);
    let total: u32 = bits.iter().sum();
    (0..1usize << total)
        .map(|index| {
            // Row-major: the last axis is the lowest.
            let mut coords = vec![0u64; bits.len()];
            let mut rest = index as u64;
            for (axis, &b) in bits.iter().enumerate().rev() {
                coords[axis] = rest & ((1u64 << b) - 1);
                rest >>= b;
            }
            let kind =
                match shape {
                    TensorShape::Dense(dense) => {
                        let real = dense.axes().iter().zip(&coords).all(|(a, &x)| x < a.real);
                        match (real, dense.padding()) {
                            (true, _) => Entry::Real,
                            (false, Padding::Zero) => {
                                let outermost = dense.axes().iter().position(|a| a.is_padded());
                                let retained = dense.axes().iter().enumerate().zip(&coords).all(
                                    |((i, a), &x)| {
                                        a.real > 0
                                            && (x < a.real
                                                || (a.is_padded() && Some(i) != outermost))
                                    },
                                );
                                if retained {
                                    Entry::RetainedZero
                                } else {
                                    Entry::Zero
                                }
                            }
                            (false, Padding::Constants) => Entry::Constant,
                        }
                    }
                    TensorShape::Scores { block } => {
                        let (q, i) = (
                            u64::from(*block) * u64::from(geometry.queries) + coords[1],
                            coords[2],
                        );
                        let n = u64::from(geometry.n);
                        // Valid: a real query attends to a real key at or before it (RFC-0030).
                        match q < n && i < n && i <= q {
                            true => Entry::Real,
                            false => Entry::Zero,
                        }
                    }
                };
            Element {
                coords,
                index,
                kind,
            }
        })
        .collect()
}

/// `f̃(r)` for a table of `2^m` values and `m` coordinates, the lowest variable first: `x_1` is the
/// least significant bit of the index (RFC-0010 'Multilinear extensions').
fn mle(table: &[Fp], r: &[Fp2]) -> Fp2 {
    assert_eq!(table.len(), 1 << r.len(), "a table of 2^m values");
    let mut values: Vec<Fp2> = table.iter().map(|&x| Fp2::from(x)).collect();
    for &coordinate in r {
        values = values
            .chunks_exact(2)
            .map(|pair| pair[0] + coordinate * (pair[1] - pair[0]))
            .collect();
    }
    values[0]
}

/// Runs of rows as pairs, to compare.
fn spans(runs: &[Range<u64>]) -> Vec<(u64, u64)> {
    runs.iter().map(|r| (r.start, r.end)).collect()
}

fn fp2(rng: &mut SplitMix64) -> Fp2 {
    let c0 = rng.fp();
    Fp2::new(c0, rng.fp())
}

// ----- Random tiles ----------------------------------------------------------------------------

/// A sequence's tiles, as a plan's expansion and the gadgets would give them, with random shapes.
struct Model {
    params: LayoutParams,
    positions: u32,
    tiles: Vec<TileShape>,
}

/// A random dense tensor of one to three axes, each the sequence's, a model dimension of any
/// length, one of a power of two, or a share's, a power of two of which fewer positions, or none,
/// are real, as a short query block's. Its order is RFC-0047's.
fn dense(rng: &mut SplitMix64, n: u32) -> DenseShape {
    let rank = 1 + rng.below(3);
    let axes: Vec<Axis> = (0..rank)
        .map(|_| match rng.below(5) {
            0 => Axis::new(u64::from(n)),
            1 => {
                let len = 1u64 << (1 + rng.below(3));
                Axis {
                    len,
                    real: rng.below(len as usize + 1) as u64,
                }
            }
            2 => Axis::new(1 << rng.below(4)),
            _ => Axis::new(1 + rng.below(9) as u64),
        })
        .collect();
    let padding = match rng.below(6) {
        0 => Padding::Constants,
        _ => Padding::Zero,
    };
    DenseShape::new(axes, padding)
}

fn node(rng: &mut SplitMix64, layer: Option<u32>, n: u32) -> TileShape {
    TileShape {
        layer,
        values: (0..1 + rng.below(2))
            .map(|_| TensorShape::Dense(dense(rng, n)))
            .collect(),
    }
}

/// Entries the tensor's allocation takes, at most: its domain or a row, and as much again for its
/// alignment gap.
fn footprint(shape: &TensorShape, geometry: &Geometry, log2_k: u32) -> u64 {
    let vars: u32 = natural_bits(shape, geometry).iter().sum();
    2u64 << vars.max(log2_k)
}

/// Random tiles in production order: the tokens, a prologue, each layer's node and attention tiles,
/// an epilogue and the tables, small enough that `Ã` is resident.
fn model(rng: &mut SplitMix64, max_entries: u64) -> Model {
    loop {
        let log2_k = 2 + rng.below(5) as u32;
        let params = LayoutParams {
            log2_k,
            segments: 1 + rng.below(6) as u32,
            layers: 1 + rng.below(3) as u32,
            group: 1 << rng.below(2),
            query_block: 1 << rng.below(4),
            max_context: 1 + rng.below(16) as u32,
        };
        // RFC-0046 'Rows too short for a key': refused, tested apart.
        let n_max = params.max_context.next_power_of_two();
        if u64::from(params.group) * u64::from(params.query_block.min(n_max)) > 1 << log2_k {
            continue;
        }
        let n = 1 + rng.below(params.max_context as usize) as u32;
        let geometry = Geometry::new(&params, n);
        let tokens = DenseShape::new(vec![Axis::new(u64::from(n))], Padding::Zero);
        let mut tiles = vec![TileShape {
            layer: None,
            values: vec![TensorShape::Dense(tokens)],
        }];
        for _ in 0..rng.below(3) {
            tiles.push(node(rng, None, n));
        }
        for layer in 0..params.layers {
            for _ in 0..1 + rng.below(2) {
                tiles.push(node(rng, Some(layer), n));
            }
            for _kv_head in 0..1 + rng.below(2) {
                for block in 0..geometry.blocks {
                    let mut values = vec![TensorShape::Scores { block }];
                    for _ in 0..1 + rng.below(3) {
                        values.push(TensorShape::Dense(geometry.row_shape(block)));
                    }
                    tiles.push(TileShape {
                        layer: Some(layer),
                        values,
                    });
                }
            }
            for _ in 0..rng.below(2) {
                tiles.push(node(rng, Some(layer), n));
            }
        }
        for _ in 0..rng.below(3) {
            tiles.push(node(rng, None, n));
        }
        for _ in 0..1 + rng.below(2) {
            let rows = DenseShape::new(vec![Axis::new(1 + rng.below(40) as u64)], Padding::Zero);
            tiles.push(TileShape {
                layer: None,
                values: vec![TensorShape::Dense(rows)],
            });
        }
        let entries: u64 = tiles
            .iter()
            .flat_map(|t| &t.values)
            .map(|v| footprint(v, &geometry, log2_k))
            .sum();
        if entries <= max_entries {
            return Model {
                params,
                positions: n,
                tiles,
            };
        }
    }
}

/// `Ã` written from random tensors where the layout places them: each tensor's row-major table, the
/// matrix of `2^ν` entries, and whether each row holds real entries, constants or retained padding.
struct Written {
    tables: Vec<Vec<Fp>>,
    matrix: Vec<Fp>,
    holds: Vec<bool>,
}

fn write(rng: &mut SplitMix64, model: &Model, layout: &AdviceLayout) -> Written {
    let geometry = Geometry::new(&model.params, model.positions);
    let log2_k = model.params.log2_k;
    let k = 1u64 << log2_k;
    let size = 1usize << layout.variables();
    let mut matrix = vec![Fp::ZERO; size];
    let mut taken = vec![false; size];
    let mut holds = vec![false; size >> log2_k];
    let mut tables = Vec::new();
    let shapes = model.tiles.iter().flat_map(|t| &t.values);
    assert_eq!(
        shapes.clone().count(),
        layout.tensors().len(),
        "one placement per value"
    );
    for (shape, placement) in shapes.zip(layout.tensors()) {
        let all = elements(shape, &geometry);
        let mut table = vec![Fp::ZERO; all.len()];
        for element in &all {
            let value = match element.kind {
                Entry::Real => rng.nonzero(),
                Entry::Zero | Entry::RetainedZero => Fp::ZERO,
                Entry::Constant => rng.fp(),
            };
            table[element.index] = value;
            let (row, column) = placement.locate(&element.coords).expect("an element");
            assert!(column < k, "a column of the row");
            let rows = placement.first_row()..placement.first_row() + placement.row_count();
            assert!(rows.contains(&row), "an element lies in its tensor's rows");
            let at = (row * k + column) as usize;
            assert!(!taken[at], "two elements share an entry of Ã");
            taken[at] = true;
            matrix[at] = value;
            if element.kind != Entry::Zero {
                holds[row as usize] = true;
            }
        }
        tables.push(table);
    }
    Written {
        tables,
        matrix,
        holds,
    }
}

/// The cases random tiles must reach for the tests on them to mean anything.
#[derive(Debug, Default)]
struct Seen {
    /// A tensor smaller than a row, and one of several rows.
    sub_row: bool,
    multi_row: bool,
    /// A score tile whose real block skips rows of masked positions.
    masked: bool,
    /// A block of padding alone.
    padding_block: bool,
    /// Padding of constants, an axis of no real position, a permuted tensor of several rows.
    constants: bool,
    empty_axis: bool,
    permuted: bool,
    /// Tensors of zero, one and two or more padded axes (RFC-0047 'Testing Strategy').
    padded_axes: [bool; 3],
    /// A tensor whose non-zero rows are not one run.
    runs: bool,
    /// An alignment gap, and a segment without a tile.
    gap: bool,
    empty_segment: bool,
}

impl Seen {
    fn tally(&mut self, model: &Model, layout: &AdviceLayout) {
        let geometry = Geometry::new(&model.params, model.positions);
        let shapes = model.tiles.iter().flat_map(|t| &t.values);
        let mut end = 0;
        for (shape, placement) in shapes.zip(layout.tensors()) {
            self.sub_row |= placement.vars() < model.params.log2_k;
            self.multi_row |= placement.row_count() > 1;
            self.runs |= placement.nonzero().len() > 1;
            self.gap |= placement.first_row() > end;
            end = placement.first_row() + placement.row_count();
            match shape {
                TensorShape::Dense(dense) => {
                    let axes = dense.axes();
                    let padding = axes.iter().any(|a| a.real < 1 << bits(a.len));
                    self.constants |= dense.padding() == Padding::Constants && padding;
                    self.empty_axis |= axes.iter().any(|a| a.real == 0);
                    let identity = dense.order().iter().enumerate().all(|(i, &a)| i == a);
                    self.permuted |= !identity && placement.row_count() > 1;
                    let padded = axes.iter().filter(|a| !a.len.is_power_of_two()).count();
                    self.padded_axes[padded.min(2)] = true;
                }
                TensorShape::Scores { block } => {
                    let real = block * geometry.queries < geometry.n;
                    let committed: u64 = placement.nonzero().iter().map(|r| r.end - r.start).sum();
                    let keys = u64::from(geometry.n).div_ceil(1 << geometry.w);
                    self.masked |= real && committed < keys;
                    self.padding_block |= !real;
                }
            }
        }
        self.empty_segment |= layout.segments().iter().any(|s| s.rows.is_empty());
    }

    fn check(&self) {
        let all = [
            self.sub_row,
            self.multi_row,
            self.masked,
            self.padding_block,
            self.constants,
            self.empty_axis,
            self.permuted,
            self.runs,
            self.gap,
            self.empty_segment,
            self.padded_axes[0],
            self.padded_axes[1],
            self.padded_axes[2],
        ];
        assert!(
            all.iter().all(|&seen| seen),
            "random tiles reach every case: {self:?}"
        );
    }
}

// ----- Claims map to rows ----------------------------------------------------------------------

/// RFC-0011 rule 3: `X̃(r) = Ã(r ‖ base_X)`, at random points, on every tensor of random tiles.
#[test]
fn a_claim_on_a_tensor_is_the_claim_on_a_tilde_at_its_point() {
    let mut rng = SplitMix64::new(0x0106_0001);
    let mut seen = Seen::default();
    for _ in 0..60 {
        let model = model(&mut rng, 1 << 13);
        let layout =
            AdviceLayout::new(model.params, model.positions, &model.tiles).expect("a valid layout");
        seen.tally(&model, &layout);
        let written = write(&mut rng, &model, &layout);
        for (table, placement) in written.tables.iter().zip(layout.tensors()) {
            assert_eq!(1usize << placement.vars(), table.len(), "n_X variables");
            for _ in 0..2 {
                let r: Vec<Fp2> = (0..placement.vars()).map(|_| fp2(&mut rng)).collect();
                let point = placement.point(&r).expect("n_X coordinates");
                assert_eq!(point.len() as u32, layout.variables(), "ν coordinates");
                assert_eq!(mle(table, &r), mle(&written.matrix, &point));
            }
        }
    }
    seen.check();
}

/// RFC-0011/0034/0047: rows hold real entries, constants or retained inner-axis padding.
/// The descriptor's `rows` counts all of them (RFC-0031).
#[test]
fn zero_rows_skip_only_the_padding_rfc_0047_permits() {
    let mut rng = SplitMix64::new(0x0106_0002);
    let mut seen = Seen::default();
    for _ in 0..60 {
        let model = model(&mut rng, 1 << 13);
        let layout =
            AdviceLayout::new(model.params, model.positions, &model.tiles).expect("a valid layout");
        seen.tally(&model, &layout);
        let written = write(&mut rng, &model, &layout);
        let committed: Vec<u64> = (0..written.holds.len() as u64)
            .filter(|&row| written.holds[row as usize])
            .collect();
        for (row, &holds) in written.holds.iter().enumerate() {
            assert_eq!(layout.is_zero_row(row as u64), !holds, "row {row}");
        }
        assert_eq!(layout.rows(), committed.len() as u64);
        let listed: Vec<u64> = layout
            .row_layout()
            .expect("a row layout")
            .row_indices()
            .collect();
        assert_eq!(listed, committed);
        // Every row past the allocation is zero, and the matrix is the smallest that holds it.
        let allocated = layout.allocated_rows();
        assert!(allocated <= 1 << layout.row_vars());
        assert!(layout.row_vars() == 0 || allocated > 1 << (layout.row_vars() - 1));
        assert_eq!(layout.variables(), model.params.log2_k + layout.row_vars());
    }
    seen.check();
}

/// The claims of random tiles, opened through the reference commitment of their non-zero rows
/// alone, are accepted by the independent verifier: the rows the layout skips hold nothing a
/// claim reads, and the layout's point is the commitment's row and column convention.
#[test]
fn claims_open_through_the_commitment_of_the_non_zero_rows() {
    let mut rng = SplitMix64::new(0x0106_0003);
    for case in 0..12 {
        let model = model(&mut rng, 1 << 11);
        let layout =
            AdviceLayout::new(model.params, model.positions, &model.tiles).expect("a valid layout");
        let written = write(&mut rng, &model, &layout);
        let log2_k = model.params.log2_k;
        let k = 1usize << log2_k;
        let rows = layout.row_layout().expect("a row layout");
        let values: Vec<Fp> = rows
            .row_indices()
            .flat_map(|row| written.matrix[row as usize * k..(row as usize + 1) * k].to_vec())
            .collect();
        let params = CommitmentParams {
            log2_k,
            log2_inv_rate: 1,
            mask_coeffs: 1 + rng.below(k),
            row_block: NonZeroUsize::new(1 + rng.below(3)).expect("non-zero"),
            commitment_id: [case as u8; 32],
        };
        let mut masks = || rng.fp();
        let commitment =
            ReferenceCommitment::commit(params, rows.clone(), Secret::new(values), &mut masks)
                .expect("a commitment");
        let mut rng = SplitMix64::new(0x0106_0103 + case as u64);
        // One claim per tensor, each its own group.
        let mut groups = Vec::new();
        let mut claimed = Vec::new();
        for (table, placement) in written.tables.iter().zip(layout.tensors()) {
            let r: Vec<Fp2> = (0..placement.vars()).map(|_| fp2(&mut rng)).collect();
            let point = placement.point(&r).expect("n_X coordinates");
            let (column_point, row_point) = point.split_at(log2_k as usize);
            groups.push(ClaimGroup {
                column_point: column_point.to_vec(),
                claims: vec![RowClaim {
                    row_point: row_point.to_vec(),
                    gamma: fp2(&mut rng),
                }],
            });
            claimed.push(vec![tv_fp2(mle(table, &r))]);
        }
        let xi_point: Vec<Fp2> = (0..layout.row_vars()).map(|_| fp2(&mut rng)).collect();
        let combined = commitment.open(&groups, &xi_point).expect("valid points");
        let queries: Vec<u64> = (0..4).map(|_| rng.below(2 * k) as u64).collect();
        let answer = commitment.answer(&queries).expect("valid queries");
        let tv_groups: Vec<tv::Group> = groups
            .iter()
            .zip(&claimed)
            .map(|(group, values)| tv::Group {
                column_point: group.column_point.iter().copied().map(tv_fp2).collect(),
                claims: group
                    .claims
                    .iter()
                    .zip(values)
                    .map(|(claim, &value)| tv::Claim {
                        row_point: claim.row_point.iter().copied().map(tv_fp2).collect(),
                        gamma: tv_fp2(claim.gamma),
                        value,
                    })
                    .collect(),
            })
            .collect();
        let xi: Vec<TvFp2> = xi_point.iter().copied().map(tv_fp2).collect();
        let verifier = Verifier {
            params: &params,
            layout: &rows,
            root: commitment.root(),
            groups: &tv_groups,
            xi_point: &xi,
        };
        verifier
            .verify(
                &tv_combined(&combined),
                &queries,
                &tv_columns(&answer),
                &answer.proof.siblings,
            )
            .expect("the verifier accepts every claim");
    }
}

/// RFC-0011 rules 2 and 3 and RFC-0034 'Causal layout', on positions computed by hand: aligned
/// offsets, a tensor smaller than a row at the start of its own row, a permuted dense tensor, and
/// a score tile's `(i_hi, h, q, i_lo)` order with its masked rows skipped.
#[test]
fn positions_follow_the_rfcs_formulas() {
    // k = 16; one layer; g = 2; Q = 4; n = 10, so N = 16, Q' = 4, four blocks and
    // w = min(4, 4 − 2 − 1) = 1: a score row holds 2 keys against 4 queries and 2 heads.
    let params = LayoutParams {
        log2_k: 4,
        segments: 1,
        layers: 1,
        group: 2,
        query_block: 4,
        max_context: 16,
    };
    let tokens = DenseShape::new(vec![Axis::new(10)], Padding::Zero);
    let small = DenseShape::new(vec![Axis::new(3)], Padding::Zero);
    // [8, 3]: its padded axis, of 3 positions, is the most significant (RFC-0047), so the entry
    // of (x0, x1) is x0 + 8·x1.
    let swapped = DenseShape::new(vec![Axis::new(8), Axis::new(3)], Padding::Zero);
    assert_eq!(swapped.order(), [1, 0]);
    let geometry = Geometry::new(&params, 10);
    let tiles = vec![
        TileShape {
            layer: None,
            values: vec![TensorShape::Dense(tokens)],
        },
        TileShape {
            layer: None,
            values: vec![TensorShape::Dense(small)],
        },
        TileShape {
            layer: Some(0),
            values: vec![TensorShape::Dense(swapped)],
        },
        TileShape {
            layer: Some(0),
            values: vec![
                TensorShape::Scores { block: 0 },
                TensorShape::Dense(geometry.row_shape(0)),
            ],
        },
        TileShape {
            layer: Some(0),
            values: vec![
                TensorShape::Scores { block: 3 },
                TensorShape::Dense(geometry.row_shape(3)),
            ],
        },
    ];
    let layout = AdviceLayout::new(params, 10, &tiles).expect("a valid layout");
    let at = |tile: usize, value: usize| layout.tensor(tile, value).expect("a tensor");

    // The tokens fill row 0: n_X = log2 k.
    assert_eq!((at(0, 0).first_row(), at(0, 0).row_count()), (0, 1));
    assert_eq!(at(0, 0).locate(&[7]), Ok((0, 7)));
    // Four entries take row 1 whole, at its start; base_X = 1·16/4 over ν − 2 bits.
    assert_eq!((at(1, 0).first_row(), at(1, 0).row_count()), (1, 1));
    assert_eq!(at(1, 0).locate(&[2]), Ok((1, 2)));
    assert_eq!(at(1, 0).base(), 4);
    assert_eq!(at(1, 0).base_bits(), layout.variables() - 2);
    // 32 entries: two rows aligned to 2, from row 2. (4, 2) is entry 4 + 8·2 = 20.
    assert_eq!((at(2, 0).first_row(), at(2, 0).row_count()), (2, 2));
    assert_eq!(at(2, 0).locate(&[4, 2]), Ok((3, 4)));
    assert_eq!(at(2, 0).base(), 1);
    // Block 0's scores: n_X = 4 + 2 + 1 = 7, eight rows aligned to 8, from row 8. (h, q, i) =
    // (1, 3, 2) has i_lo = 0 and i_hi = 1: column i_lo + 2·q + 8·h = 14 of row 8 + 1.
    let scores = at(3, 0);
    assert_eq!((scores.first_row(), scores.row_count()), (8, 8));
    assert_eq!(scores.locate(&[1, 3, 2]), Ok((9, 14)));
    assert_eq!(scores.locate(&[0, 0, 15]), Ok((15, 1)));
    // Its rows hold keys 0–1, 2–3, …: those after its queries 0–3 are masked, so rows 8 and 9
    // are committed, rows 10 to 15 skipped.
    assert_eq!(spans(scores.nonzero()), [(8, 10)]);
    // The tile's reciprocals [2, 4]: eight entries in row 16, (1, 2) at 4·1 + 2.
    assert_eq!((at(3, 1).first_row(), at(3, 1).row_count()), (16, 1));
    assert_eq!(at(3, 1).locate(&[1, 2]), Ok((16, 6)));
    // Block 3 holds queries 12–15, all past n = 10: padding alone, every row skipped.
    assert_eq!(at(4, 0).first_row(), 24);
    assert!(at(4, 0).nonzero().is_empty());
    assert!(at(4, 1).nonzero().is_empty());
    // 33 rows: ν = 4 + 6.
    assert_eq!(layout.allocated_rows(), 33);
    assert_eq!((layout.row_vars(), layout.variables()), (6, 10));
    // Rows 0, 1, 2, 3, 8, 9 and 16: the tokens, the small tensor, the swapped tensor (its real
    // entries x1 < 3 reach both rows), block 0's committed blocks of keys and its reciprocals.
    let committed: Vec<u64> = layout.row_layout().expect("rows").row_indices().collect();
    assert_eq!(committed, [0, 1, 2, 3, 8, 9, 16]);
    assert_eq!(layout.rows(), 7);
}

/// RFC-0034's zero rows on a causal score tile with several rows: block `b` commits
/// `min(⌈n / 2^w⌉, ⌊((b + 1)·Q' − 1) / 2^w⌋ + 1)` rows, and a block of padding none.
#[test]
fn a_causal_score_tile_skips_its_all_masked_rows() {
    // k = 64, g = 2, Q = 4: w = min(5, 6 − 2 − 1) = 3, a row of 8 keys; N = 32, 4 rows per tile.
    for n in [17, 30, 32] {
        let params = LayoutParams {
            log2_k: 6,
            segments: 1,
            layers: 1,
            group: 2,
            query_block: 4,
            max_context: 32,
        };
        let geometry = Geometry::new(&params, n);
        assert_eq!((geometry.w, geometry.blocks), (3, 8));
        let tiles: Vec<TileShape> = (0..geometry.blocks)
            .map(|block| TileShape {
                layer: Some(0),
                values: vec![TensorShape::Scores { block }],
            })
            .collect();
        let layout = AdviceLayout::new(params, n, &tiles).expect("a valid layout");
        for block in 0..geometry.blocks {
            let scores = layout.tensor(block as usize, 0).expect("the scores");
            assert_eq!(scores.row_count(), 4);
            let expected = match block * 4 < n {
                true => n.div_ceil(8).min((4 * (block + 1) - 1) / 8 + 1),
                false => 0,
            };
            let first = scores.first_row();
            let committed: Vec<Vec<u64>> = scores
                .nonzero()
                .iter()
                .map(|r| r.clone().collect())
                .collect();
            let expected: Vec<Vec<u64>> = match expected {
                0 => vec![],
                m => vec![(first..first + u64::from(m)).collect()],
            };
            assert_eq!(committed, expected, "n = {n}, block {block}");
        }
    }
}

/// RFC-0034 'Cost': at 405B shapes, a layer's score tiles commit `S²/2 + b·S/2` exponentials per
/// head, `b = max(Q, 2^w) = 512`, at 1K and 16K positions, with `w = 9`.
#[test]
fn score_tiles_commit_rfc_0034s_closed_form_at_405b() {
    for n in [1024u32, 16_384] {
        let params = LayoutParams {
            log2_k: 22,
            segments: 1,
            layers: 1,
            group: 16,
            query_block: 512,
            max_context: 131_072,
        };
        let geometry = Geometry::new(&params, n);
        assert_eq!(geometry.w, 9);
        // One KV head of 16 query heads, every block.
        let tiles: Vec<TileShape> = (0..geometry.blocks)
            .map(|block| TileShape {
                layer: Some(0),
                values: vec![TensorShape::Scores { block }],
            })
            .collect();
        let layout = AdviceLayout::new(params, n, &tiles).expect("a valid layout");
        let s = u64::from(n);
        let per_head = s * s / 2 + 512 * s / 2;
        assert_eq!(layout.rows() << 22, 16 * per_head, "n = {n}");
    }
}

// ----- Zero padding is free --------------------------------------------------------------------

/// RFC-0047 'The order of a tensor's variables', its example: at 405B shapes, 16,384 positions and
/// `k = 2^22`, a 53,248-wide MLP tensor `[16384, 53248]` lays out as `[53248 → 65536, 16384]`,
/// its one padded axis the most significant. It commits 208 rows, exactly its elements'
/// `16384·53248 / 2^22`, and skips its 48 rows of padding. Row-major, which RFC-0011 Alternative 6
/// rejects, its padding would lie in each of its 256 rows.
#[test]
fn padding_of_405bs_mlp_tensors_costs_no_row() {
    let params = LayoutParams {
        log2_k: 22,
        segments: 8,
        layers: 126,
        group: 16,
        query_block: 512,
        max_context: 131_072,
    };
    let mlp = DenseShape::new(vec![Axis::new(16_384), Axis::new(53_248)], Padding::Zero);
    assert_eq!(mlp.order(), [1, 0]);
    // The silu, up and product tensors of each layer, after the tokens.
    let mut tiles = vec![TileShape {
        layer: None,
        values: vec![TensorShape::Dense(DenseShape::new(
            vec![Axis::new(16_384)],
            Padding::Zero,
        ))],
    }];
    for layer in 0..126 {
        for _ in 0..3 {
            tiles.push(TileShape {
                layer: Some(layer),
                values: vec![TensorShape::Dense(mlp.clone())],
            });
        }
    }
    let layout = AdviceLayout::new(params, 16_384, &tiles).expect("a valid layout");
    let mlp_tensors = &layout.tensors()[1..];
    assert_eq!(mlp_tensors.len(), 378);
    for tensor in mlp_tensors {
        assert_eq!(tensor.row_count(), 256);
        let rows: u64 = tensor.nonzero().iter().map(|r| r.end - r.start).sum();
        assert_eq!(rows, (16_384 * 53_248) >> 22);
        let first = tensor.first_row();
        assert_eq!(spans(tensor.nonzero()), [(first, first + 208)]);
    }
    // The tokens' row and 208 per MLP tensor: padding costs nothing.
    assert_eq!(layout.rows(), 1 + 378 * 208);
}

#[test]
fn inner_padded_axes_remain_committed_under_rfc_0047() {
    let layout = AdviceLayout::new(
        LayoutParams {
            log2_k: 1,
            segments: 1,
            layers: 1,
            group: 1,
            query_block: 1,
            max_context: 3,
        },
        3,
        &[TileShape {
            layer: Some(0),
            values: vec![TensorShape::Dense(DenseShape::new(
                vec![Axis::new(3), Axis::new(5)],
                Padding::Zero,
            ))],
        }],
    )
    .unwrap();
    assert_eq!(layout.allocated_rows(), 16);
    assert_eq!(layout.rows(), 12);
    assert_eq!(spans(layout.tensors()[0].nonzero()), [(0, 12)]);
    for row in 0..16 {
        assert_eq!(layout.is_zero_row(row), row >= 12);
    }
}

#[test]
fn repeated_tile_counter_matches_full_allocation_with_heterogeneous_alignments() {
    let mut rng = SplitMix64::new(0x0106_0047);
    for _ in 0..100 {
        let model = model(&mut rng, 1 << 12);
        let mut counter = AdviceLayoutCounter::new(model.params, model.positions).unwrap();
        let mut expanded = Vec::new();
        for tile in &model.tiles {
            let repeats = 1 + rng.below(7);
            counter.append(&tile.values, repeats as u64).unwrap();
            expanded.extend(std::iter::repeat_n(tile.clone(), repeats));
        }
        let allocated = AdviceLayout::new(model.params, model.positions, &expanded).unwrap();
        assert_eq!(counter.allocated_rows(), allocated.allocated_rows());
    }
    let params = LayoutParams {
        log2_k: 10,
        segments: 1,
        layers: 1,
        group: 1,
        query_block: 1,
        max_context: 1,
    };
    let shape = |vars| {
        TensorShape::Dense(DenseShape::new(
            vec![Axis::new(1u64 << vars)],
            Padding::Zero,
        ))
    };
    let mut counter = AdviceLayoutCounter::new(params, 1).unwrap();
    counter
        .append(&[shape(10), shape(13), shape(11)], 3)
        .unwrap();
    assert_eq!(counter.allocated_rows(), 50);
}

#[test]
fn repeated_tile_counter_checks_the_raw_row_limit_before_expansion_and_preserves_state() {
    let params = LayoutParams {
        log2_k: 10,
        segments: 1,
        layers: 1,
        group: 1,
        query_block: 1,
        max_context: 1,
    };
    let one = TensorShape::Dense(DenseShape::new(vec![Axis::new(1)], Padding::Zero));
    let rejection = PcsError::Layout {
        rule: "a layout allocates at most 2^32 rows (RFC-0047 'The maximum layout')",
    };
    let mut counter = AdviceLayoutCounter::new(params, 1).unwrap();
    counter.append(std::slice::from_ref(&one), 1 << 32).unwrap();
    assert_eq!(counter.allocated_rows(), 1 << 32);
    assert_eq!(
        counter.append(std::slice::from_ref(&one), 1),
        Err(rejection)
    );
    assert_eq!(counter.allocated_rows(), 1 << 32);
    for repeats in [(1 << 32) + 1, u64::MAX] {
        let mut counter = AdviceLayoutCounter::new(params, 1).unwrap();
        assert_eq!(
            counter.append(std::slice::from_ref(&one), repeats),
            Err(rejection)
        );
        assert_eq!(counter.allocated_rows(), 0);
    }
    assert_eq!(
        counter.append(&[], 1),
        Err(PcsError::LayoutData {
            rule: "a tile holds at least one committed value (RFC-0041 'Tiles')"
        })
    );
    assert_eq!(
        counter.append(&[one], 0),
        Err(PcsError::LayoutData {
            rule: "a repeated tile occurs at least once"
        })
    );
}

// ----- The order and the maximum (RFC-0047) ----------------------------------------------------

/// RFC-0047 'The order of a tensor's variables': the padded axes, those whose length is not a power
/// of two, in axis order, then the others in axis order. Expected orders written out by hand.
#[test]
fn the_layout_order_puts_padded_axes_first_in_axis_order() {
    let order = |lengths: &[u64]| {
        let axes: Vec<Axis> = lengths.iter().map(|&len| Axis::new(len)).collect();
        layout_order(&axes)
    };
    // 405B's MLP tensors: the channels, 53,248, are padded.
    assert_eq!(order(&[16_384, 53_248]), [1, 0]);
    // Both padded, at a length that is no power of two: axis order.
    assert_eq!(order(&[16_000, 53_248]), [0, 1]);
    // None padded: row-major.
    assert_eq!(order(&[16_384, 16_384]), [0, 1]);
    assert_eq!(order(&[8, 3]), [1, 0]);
    assert_eq!(order(&[3, 8, 5]), [0, 2, 1]);
    // A length of 1 is 2^0, unpadded.
    assert_eq!(order(&[4, 6, 1, 3, 2]), [1, 3, 0, 2, 4]);
    assert_eq!(order(&[7]), [0]);
    assert_eq!(order(&[]), [] as [usize; 0]);
    // An axis is judged by its length, not by its real positions: a block's queries, of which 88
    // of 512 are real, are no padded axis, and a tile's `[g, Q']` share stays row-major.
    let share = [Axis::new(4), Axis { len: 512, real: 88 }];
    assert_eq!(layout_order(&share), [0, 1]);
    // A dense shape takes the rule's order: there is no other way to build one.
    let shape = DenseShape::new(share.to_vec(), Padding::Zero);
    assert_eq!(shape.order(), [0, 1]);
    let mlp = DenseShape::new(vec![Axis::new(16_384), Axis::new(53_248)], Padding::Zero);
    assert_eq!(mlp.order(), [1, 0]);
}

/// RFC-0047 'The maximum layout': a layout allocates at most `2^32` rows. A tensor of `2^36`
/// entries in rows of 16 takes exactly `2^32` and is accepted, with a row index of 32 bits; one
/// row more is refused as `Layout`, and so is a tensor of `2^33` rows alone.
#[test]
fn a_layout_of_more_than_two_to_the_32_rows_is_refused() {
    let params = LayoutParams {
        log2_k: 4,
        segments: 1,
        layers: 1,
        group: 1,
        query_block: 4,
        max_context: 8,
    };
    let dense = |axes: Vec<Axis>| TensorShape::Dense(DenseShape::new(axes, Padding::Zero));
    let largest = TileShape {
        layer: Some(0),
        values: vec![dense(vec![Axis::new(1 << 32), Axis::new(16)])],
    };
    let layout = AdviceLayout::new(params, 5, std::slice::from_ref(&largest))
        .expect("2^32 rows are accepted");
    assert_eq!(layout.allocated_rows(), 1 << 32);
    assert_eq!((layout.row_vars(), layout.variables()), (32, 36));
    assert_eq!(layout.rows(), 1 << 32);
    let one_more = TileShape {
        layer: None,
        values: vec![dense(vec![Axis::new(1)])],
    };
    let refused = Err(PcsError::Layout { rule: MAX_ROWS });
    assert_eq!(
        AdviceLayout::new(params, 5, &[largest, one_more]).map(|_| ()),
        refused
    );
    let larger = TileShape {
        layer: Some(0),
        values: vec![dense(vec![Axis::new(1 << 32), Axis::new(32)])],
    };
    assert_eq!(AdviceLayout::new(params, 5, &[larger]).map(|_| ()), refused);
}

// ----- Segments --------------------------------------------------------------------------------

/// RFC-0046 'Segments', example: 126 layers in 8 segments.
#[test]
fn segments_of_405b_hold_16_16_16_16_16_16_15_15_layers() {
    let sizes: Vec<u32> = segment_layers(126, 8)
        .iter()
        .map(|r| r.end - r.start)
        .collect();
    assert_eq!(sizes, [16, 16, 16, 16, 16, 16, 15, 15]);
}

/// RFC-0046 'Testing Strategy': for `L` from 1 to 130 and `S` from 1 to 16, every layer lies in
/// exactly one segment, in order, the first `L mod S` segments one layer larger.
#[test]
fn segments_hold_every_layer_once_in_order() {
    for layers in 1..=130u32 {
        for segments in 1..=16u32 {
            let ranges = segment_layers(layers, segments);
            assert_eq!(ranges.len(), segments as usize);
            let mut next = 0;
            for (j, range) in ranges.iter().enumerate() {
                assert_eq!(range.start, next, "L = {layers}, S = {segments}");
                let expected = layers / segments + u32::from((j as u32) < layers % segments);
                assert_eq!(range.end - range.start, expected);
                next = range.end;
            }
            assert_eq!(next, layers);
        }
    }
}

/// RFC-0046 'Rows' and 'An empty segment': on random tiles, the segments partition the rows in
/// order, each non-empty one from its first tile's first row, every tile in the segment of its
/// layer, the prologue in the first and the epilogue in the last.
#[test]
fn segments_cover_the_rows_once_each_boundary_at_a_tiles_first_row() {
    let mut rng = SplitMix64::new(0x0106_0004);
    for _ in 0..200 {
        let model = model(&mut rng, 1 << 16);
        let layout =
            AdviceLayout::new(model.params, model.positions, &model.tiles).expect("a valid layout");
        let s = model.params.segments as usize;
        let layers = segment_layers(model.params.layers, model.params.segments);
        assert_eq!(layout.segments().len(), s);
        let mut next = 0;
        for (j, segment) in layout.segments().iter().enumerate() {
            assert_eq!(segment.layers, layers[j]);
            assert_eq!(
                segment.rows.start,
                next,
                "segment {j} starts where {} ends",
                j.max(1) - 1
            );
            next = segment.rows.end;
            let first = layout.tiles().iter().find(|t| t.segment as usize == j);
            match first {
                Some(tile) => assert_eq!(segment.rows.start, tile.first_row),
                None => assert!(segment.rows.is_empty() && segment.nonzero.is_empty()),
            }
            for run in &segment.nonzero {
                assert!(segment.rows.start <= run.start && run.end <= segment.rows.end);
            }
        }
        assert_eq!(next, layout.allocated_rows());
        let first_layered = model.tiles.iter().position(|t| t.layer.is_some());
        for (t, (tile, placed)) in model.tiles.iter().zip(layout.tiles()).enumerate() {
            let expected = match tile.layer {
                Some(layer) => layers
                    .iter()
                    .position(|r| r.contains(&layer))
                    .expect("a layer"),
                None if Some(t) < first_layered => 0,
                None => s - 1,
            };
            assert_eq!(placed.segment as usize, expected, "tile {t}");
            let rows = &layout.segments()[expected].rows;
            for tensor in &layout.tensors()[placed.tensors.clone()] {
                assert!(rows.start <= tensor.first_row());
                assert!(tensor.first_row() + tensor.row_count() <= rows.end);
            }
        }
    }
}

/// RFC-0046 'An empty segment': 2 layers in 8 segments leave segments 2 to 6 without a tile or a
/// row, and segment 7 holds the epilogue and the tables.
#[test]
fn segments_past_the_layers_are_empty_when_s_exceeds_l() {
    let params = LayoutParams {
        log2_k: 3,
        segments: 8,
        layers: 2,
        group: 1,
        query_block: 4,
        max_context: 8,
    };
    let dense = |len| TensorShape::Dense(DenseShape::new(vec![Axis::new(len)], Padding::Zero));
    let tile = |layer, len| TileShape {
        layer,
        values: vec![dense(len)],
    };
    let tiles = vec![
        tile(None, 5),
        tile(Some(0), 8),
        tile(Some(1), 16),
        tile(None, 3),
        tile(None, 9),
    ];
    let layout = AdviceLayout::new(params, 5, &tiles).expect("a valid layout");
    let segments = layout.segments();
    let layers: Vec<_> = segments.iter().map(|s| s.layers.clone()).collect();
    assert_eq!(layers, [0..1, 1..2, 2..2, 2..2, 2..2, 2..2, 2..2, 2..2]);
    // Rows: the tokens 0, layer 0's tensor 1, layer 1's 2–3, the epilogue 4, the table 6–7 after
    // an alignment gap at row 5, which segment 7 holds.
    let rows: Vec<_> = segments.iter().map(|s| s.rows.clone()).collect();
    assert_eq!(rows, [0..2, 2..4, 4..4, 4..4, 4..4, 4..4, 4..4, 4..8]);
    let placed: Vec<u32> = layout.tiles().iter().map(|t| t.segment).collect();
    assert_eq!(placed, [0, 0, 1, 7, 7]);
    for segment in &segments[2..7] {
        assert!(segment.nonzero.is_empty());
    }
    let row_layout = layout.row_layout().expect("rows");
    let per_segment: Vec<Vec<u64>> = row_layout.segments().to_vec();
    assert_eq!(
        per_segment,
        [
            vec![0, 1],
            vec![2, 3],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![4, 6, 7]
        ]
    );
}

// ----- Short sequences -------------------------------------------------------------------------

/// RFC-0046 'Short query blocks': below `Q`, a tile takes `Q' = N` queries, and a tile of
/// `g·N·N ≤ k` entries takes one row, as RFC-0011 rule 2 places any tensor smaller than a row.
#[test]
fn a_sequence_shorter_than_q_takes_q_prime_n_and_one_row() {
    let params = LayoutParams {
        log2_k: 10,
        segments: 1,
        layers: 1,
        group: 2,
        query_block: 512,
        max_context: 2048,
    };
    for n in [1u32, 2, 5, 9, 16, 17, 32, 300] {
        let attention = params.attention(n).expect("a geometry");
        let padded = n.next_power_of_two();
        assert_eq!(
            (attention.padded, attention.queries, attention.blocks),
            (padded, padded, 1)
        );
        assert_eq!(
            attention.row_shape(0, Padding::Zero),
            Geometry::new(&params, n).row_shape(0)
        );
        let tiles = vec![TileShape {
            layer: Some(0),
            values: vec![
                TensorShape::Scores { block: 0 },
                TensorShape::Dense(attention.row_shape(0, Padding::Zero)),
            ],
        }];
        let layout = AdviceLayout::new(params, n, &tiles).expect("a valid layout");
        let scores = layout.tensor(0, 0).expect("the scores");
        let entries = 2 * u64::from(padded) * u64::from(padded);
        assert_eq!(1u64 << scores.vars(), entries);
        if entries <= 1024 {
            // g·N·N ≤ k: w = n, one row of every key, committed.
            assert_eq!(attention.key_bits, padded.trailing_zeros());
            assert_eq!(scores.row_count(), 1, "n = {n}");
            assert_eq!(spans(scores.nonzero()), [(0, 1)]);
        } else {
            // Past it, rows of 2^w keys, w = log2 k − log2 Q' − log2 g with Q' = N: 16 keys at
            // N = 32, one at N = 512. Of N = 512's rows, the 300 of real keys are committed.
            let w = 10 - padded.trailing_zeros() - 1;
            assert_eq!(attention.key_bits, w, "n = {n}");
            assert_eq!(scores.row_count(), entries >> 10, "n = {n}");
            let committed = u64::from(n).div_ceil(1 << w);
            assert_eq!(spans(scores.nonzero()), [(0, committed)], "n = {n}");
        }
    }
}

// ----- Refusals --------------------------------------------------------------------------------

const ROWS_TOO_SHORT: &str =
    "a row holds one key against a tile's queries and heads: g·min(Q, N_max) ≤ k (RFC-0046)";
const QUERY_BLOCK: &str =
    "query_block is a power of two: RFC-0034's layout takes its logarithm (RFC-0041)";
const SEGMENTS: &str = "the advice has at least one segment (RFC-0046 'Segments')";
const ROW_WIDTH: &str =
    "a row of k = 2^log2_k values has a codeword of 2k ≤ 2^32 symbols (RFC-0011 'Encoding')";

fn base_params() -> LayoutParams {
    LayoutParams {
        log2_k: 10,
        segments: 1,
        layers: 1,
        group: 4,
        query_block: 512,
        max_context: 200,
    }
}

/// RFC-0046 'Rows too short for a key' and RFC-0007 step 6: `g·min(Q, N_max) > k` is refused as
/// `Layout`, at `N_max`, the padded context, whatever the sequence; equality is accepted.
#[test]
fn rows_too_short_for_a_key_are_refused_as_layout() {
    // 4·min(512, 256) = 1024 = k: accepted.
    assert_eq!(base_params().check(), Ok(()));
    // A context of 257 pads to 512: 4·512 > 1024, even for a sequence of one position.
    let wider = LayoutParams {
        max_context: 257,
        ..base_params()
    };
    let refused = Err(PcsError::Layout {
        rule: ROWS_TOO_SHORT,
    });
    assert_eq!(wider.check(), refused);
    assert_eq!(wider.attention(1).map(|_| ()), refused);
    let tiles = [TileShape {
        layer: Some(0),
        values: vec![TensorShape::Scores { block: 0 }],
    }];
    assert_eq!(AdviceLayout::new(wider, 1, &tiles).map(|_| ()), refused);
    // A smaller block fits again: 4·256 = 1024.
    let smaller = LayoutParams {
        query_block: 256,
        ..wider
    };
    assert_eq!(smaller.check(), Ok(()));
    // So does a wider row.
    let wider_row = LayoutParams {
        log2_k: 11,
        ..wider
    };
    assert_eq!(wider_row.check(), Ok(()));
}

/// RFC-0041 'Production order' and RFC-0007 step 6: the layout's other refusals, each its own rule.
#[test]
fn layout_parameters_without_a_layout_are_refused() {
    let refuse = |params: LayoutParams, rule| {
        assert_eq!(params.check(), Err(PcsError::Layout { rule }), "{params:?}");
    };
    refuse(
        LayoutParams {
            query_block: 384,
            ..base_params()
        },
        QUERY_BLOCK,
    );
    refuse(
        LayoutParams {
            query_block: 0,
            ..base_params()
        },
        QUERY_BLOCK,
    );
    refuse(
        LayoutParams {
            segments: 0,
            ..base_params()
        },
        SEGMENTS,
    );
    refuse(
        LayoutParams {
            log2_k: 32,
            ..base_params()
        },
        ROW_WIDTH,
    );
}

const NO_LAYER: &str = "a model has at least one layer";
const NO_CONTEXT: &str = "a model's context holds at least one position";
const NO_HEAD: &str = "an attention tile holds at least one query head";
const POSITIONS: &str = "a sequence has 1 to max_context positions";
const LAYER_WITHOUT_TILE: &str = "every layer has a tile";
const EMPTY_TILE: &str = "a tile holds at least one committed value (RFC-0041 'Tiles')";
const LAYER_PAST: &str = "a tile's layer is below n_layers";
const LAYER_ORDER: &str =
    "the layers' tiles come in layer order, after the prologue's and before the epilogue's";
const GROUP: &str = "g = n_heads / n_kv_heads is a power of two (RFC-0007 step 2, RFC-0047)";
const AXIS_EMPTY: &str = "an axis has at least one position";
const AXIS_BITS: &str = "an axis has at most 2^32 positions";
const AXIS_LEN: &str = "an axis's real positions lie within its length";
const TOO_LARGE: &str = "a tensor has at most 2^64 entries";
const BLOCK: &str = "a score tile's query block is below N / Q'";
const MAX_ROWS: &str = "a layout allocates at most 2^32 rows (RFC-0047 'The maximum layout')";

/// Each rule of the plain data, broken alone, is refused with its rule; the layout derives from
/// the same data once the rule holds.
#[test]
fn malformed_data_is_refused_with_its_rule() {
    let params = LayoutParams {
        log2_k: 4,
        segments: 2,
        layers: 2,
        group: 1,
        query_block: 4,
        max_context: 8,
    };
    let dense = |axes: Vec<Axis>| TensorShape::Dense(DenseShape::new(axes, Padding::Zero));
    let tile = |layer| TileShape {
        layer,
        values: vec![dense(vec![Axis::new(3)])],
    };
    let good = vec![tile(None), tile(Some(0)), tile(Some(1)), tile(None)];
    assert!(AdviceLayout::new(params, 5, &good).is_ok());

    let data = |rule: &'static str| -> Result<(), PcsError> { Err(PcsError::LayoutData { rule }) };
    let refused =
        |p: LayoutParams, n: u32, tiles: &[TileShape]| AdviceLayout::new(p, n, tiles).map(|_| ());
    assert_eq!(
        refused(
            LayoutParams {
                layers: 0,
                ..params
            },
            5,
            &good
        ),
        data(NO_LAYER)
    );
    assert_eq!(
        refused(
            LayoutParams {
                max_context: 0,
                ..params
            },
            5,
            &good
        ),
        data(NO_CONTEXT)
    );
    assert_eq!(
        refused(LayoutParams { group: 0, ..params }, 5, &good),
        data(NO_HEAD)
    );
    // RFC-0007 step 2 refuses n_heads = 24, n_kv_heads = 8 before the layout; the layout checks it
    // again, as data it never receives.
    assert_eq!(
        refused(LayoutParams { group: 3, ..params }, 5, &good),
        data(GROUP)
    );
    assert_eq!(refused(params, 0, &good), data(POSITIONS));
    assert_eq!(refused(params, 9, &good), data(POSITIONS));
    assert_eq!(
        refused(params, 5, &[tile(None), tile(Some(0)), tile(None)]),
        data(LAYER_WITHOUT_TILE)
    );

    let at = |t: usize, rule: &'static str| -> Result<(), PcsError> {
        Err(PcsError::Tile { tile: t, rule })
    };
    let with = |t: usize, replace: TileShape| {
        let mut tiles = good.clone();
        tiles[t] = replace;
        refused(params, 5, &tiles)
    };
    assert_eq!(
        with(
            1,
            TileShape {
                layer: Some(0),
                values: vec![]
            }
        ),
        at(1, EMPTY_TILE)
    );
    assert_eq!(with(2, tile(Some(2))), at(2, LAYER_PAST));
    // The first layer's tiles come first, and each layer's follow the one before.
    assert_eq!(
        refused(
            params,
            5,
            &[tile(None), tile(Some(1)), tile(Some(0)), tile(None)]
        ),
        at(1, LAYER_ORDER)
    );
    assert_eq!(
        refused(params, 5, &[tile(Some(0)), tile(Some(1)), tile(Some(0))]),
        at(2, LAYER_ORDER)
    );
    // No tile of the epilogue precedes a layer's.
    assert_eq!(
        refused(params, 5, &[tile(Some(0)), tile(None), tile(Some(1))]),
        at(2, LAYER_ORDER)
    );
    assert_eq!(
        with(
            0,
            TileShape {
                layer: None,
                values: vec![dense(vec![Axis::new((1 << 32) + 1)])]
            }
        ),
        at(0, AXIS_BITS)
    );
    assert_eq!(
        with(
            0,
            TileShape {
                layer: None,
                values: vec![dense(vec![Axis { len: 4, real: 5 }])]
            }
        ),
        at(0, AXIS_LEN)
    );
    assert_eq!(
        with(
            0,
            TileShape {
                layer: None,
                values: vec![dense(vec![Axis::new(2), Axis::new(0)])]
            }
        ),
        at(0, AXIS_EMPTY)
    );
    let huge = vec![Axis::new(1 << 32), Axis::new(1 << 32), Axis::new(2)];
    assert_eq!(
        with(
            3,
            TileShape {
                layer: None,
                values: vec![dense(huge)]
            }
        ),
        at(3, TOO_LARGE)
    );
    // n = 5: N = 8, Q' = 4, two blocks.
    assert_eq!(
        with(
            1,
            TileShape {
                layer: Some(0),
                values: vec![TensorShape::Scores { block: 2 }]
            }
        ),
        at(1, BLOCK)
    );
    assert!(
        with(
            1,
            TileShape {
                layer: Some(0),
                values: vec![TensorShape::Scores { block: 1 }]
            }
        )
        .is_ok()
    );
}

/// A claim's point and an element's coordinates must fit their tensor.
#[test]
fn points_and_elements_outside_their_tensor_are_refused() {
    let params = base_params();
    let shape = DenseShape::new(vec![Axis::new(5), Axis::new(3)], Padding::Zero);
    let tiles = [TileShape {
        layer: Some(0),
        values: vec![TensorShape::Dense(shape)],
    }];
    let layout = AdviceLayout::new(params, 5, &tiles).expect("a valid layout");
    let tensor = layout.tensor(0, 0).expect("a tensor");
    assert_eq!(tensor.vars(), 5);
    let r = vec![Fp2::ZERO; 4];
    assert_eq!(
        tensor.point(&r),
        Err(PcsError::ClaimPoint {
            expected: 5,
            found: 4
        })
    );
    assert_eq!(
        tensor.locate(&[1]),
        Err(PcsError::ElementAxes {
            expected: 2,
            found: 1
        })
    );
    assert_eq!(
        tensor.locate(&[1, 4]),
        Err(PcsError::ElementRange { axis: 1 })
    );
    assert_eq!(
        tensor.locate(&[8, 0]),
        Err(PcsError::ElementRange { axis: 0 })
    );
    // Padding is in the domain: (7, 3) lies past the real lengths, not past the padded ones.
    assert!(tensor.locate(&[7, 3]).is_ok());
}

// ----- INV-PCS-1 -------------------------------------------------------------------------------

/// The first rows of every tensor.
fn offsets(layout: &AdviceLayout) -> Vec<u64> {
    layout.tensors().iter().map(|t| t.first_row()).collect()
}

/// INV-PCS-1: the layout is a function of public shapes alone, and every public input that places
/// a tensor moves it when it changes: the sequence's length, `Q`, `k`, a tensor's shape before it
/// and the production order. `S` places segments, not tensors.
#[test]
fn every_public_input_that_places_a_tensor_moves_it() {
    let params = LayoutParams {
        log2_k: 4,
        segments: 1,
        layers: 1,
        group: 1,
        query_block: 4,
        max_context: 16,
    };
    let dense = |len| TensorShape::Dense(DenseShape::new(vec![Axis::new(len)], Padding::Zero));
    let tiles = |n: u32, params: &LayoutParams, first: u64| {
        let geometry = Geometry::new(params, n);
        let mut tiles = vec![TileShape {
            layer: None,
            values: vec![dense(first)],
        }];
        for block in 0..geometry.blocks {
            tiles.push(TileShape {
                layer: Some(0),
                values: vec![
                    TensorShape::Scores { block },
                    TensorShape::Dense(geometry.row_shape(block)),
                ],
            });
        }
        tiles.push(TileShape {
            layer: None,
            values: vec![dense(40)],
        });
        tiles
    };
    let last = |layout: &AdviceLayout| *offsets(layout).last().expect("a tensor");
    let layout = |p: LayoutParams, n: u32, first: u64| {
        AdviceLayout::new(p, n, &tiles(n, &p, first)).expect("a valid layout")
    };
    // n = 9: N = 16, four blocks of Q' = 4, score tiles of four rows. The table lies at row 36.
    let base = layout(params, 9, 8);
    assert_eq!(last(&base), 36);
    // Recomputed from the same public inputs, the layout is the same.
    assert_eq!(base, layout(params, 9, 8));
    // The sequence's length: at n = 5, N = 8 and two blocks of tiles of two rows.
    assert_ne!(last(&layout(params, 5, 8)), last(&base));
    // Within one N, n places nothing: it decides which rows are skipped.
    let longer = layout(params, 16, 8);
    assert_eq!(offsets(&longer), offsets(&base));
    assert!(longer.rows() > base.rows());
    // Q: blocks of another size. Block 0's reciprocals follow its four rows of scores at Q = 4,
    // its two at Q = 2.
    let halved = layout(
        LayoutParams {
            query_block: 2,
            ..params
        },
        9,
        8,
    );
    assert_eq!((offsets(&base)[2], offsets(&halved)[2]), (8, 4));
    // k: rows of another width.
    assert_ne!(
        last(&layout(
            LayoutParams {
                log2_k: 5,
                ..params
            },
            9,
            8
        )),
        last(&base)
    );
    // A tensor before it: 100 entries take eight rows, where 8 took one.
    assert_ne!(last(&layout(params, 9, 100)), last(&base));
    // The production order: with the table first, the first tensor follows its four rows.
    let mut swapped = tiles(9, &params, 8);
    let table = swapped.pop().expect("the table");
    swapped.insert(0, table);
    let reordered = AdviceLayout::new(params, 9, &swapped).expect("a valid layout");
    assert_eq!((offsets(&base)[0], offsets(&reordered)[1]), (0, 4));
    // S: the same tensors in more segments.
    let segmented = layout(
        LayoutParams {
            segments: 3,
            ..params
        },
        9,
        8,
    );
    assert_eq!(offsets(&segmented), offsets(&base));
    assert_ne!(segmented.segments(), base.segments());
}

/// What the tests above take for granted about the tiles of a block: the geometry the crate gives
/// is RFC-0034's and RFC-0046's.
#[test]
fn the_attention_geometry_is_rfc_0034s() {
    let mut seen = BTreeSet::new();
    for log2_k in 2..=8 {
        for group in [1, 2, 4] {
            for query_block in [1, 2, 4, 8, 512] {
                for max_context in [1, 3, 8, 13, 16] {
                    let params = LayoutParams {
                        log2_k,
                        segments: 1,
                        layers: 1,
                        group,
                        query_block,
                        max_context,
                    };
                    if params.check().is_err() {
                        continue;
                    }
                    for n in 1..=max_context {
                        let expected = Geometry::new(&params, n);
                        let attention = params.attention(n).expect("a geometry");
                        assert_eq!(attention.positions, n);
                        assert_eq!(attention.padded, n.next_power_of_two());
                        assert_eq!(attention.queries, expected.queries);
                        assert_eq!(attention.blocks, expected.blocks);
                        assert_eq!(attention.heads, group);
                        assert_eq!(attention.key_bits, expected.w);
                        for block in 0..expected.blocks {
                            assert_eq!(
                                attention.row_shape(block, Padding::Zero),
                                expected.row_shape(block)
                            );
                        }
                        seen.insert(expected.w < expected.n_bits);
                    }
                }
            }
        }
    }
    // Both kinds of tile occur: of one row, and of several.
    assert_eq!(seen.len(), 2);
}
