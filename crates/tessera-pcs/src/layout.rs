// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The canonical advice layout (RFC-0011 'Canonical advice layout', as RFC-0034, RFC-0035,
//! RFC-0041, RFC-0046 and RFC-0047 amend it): where each committed tensor of a proof unit lies in
//! the advice matrix, which rows hold nothing, and which segment holds each row.
//!
//! The verifier derives the layout itself, so it is a function of public data alone (INV-PCS-1):
//! the claim list's [`LayoutParams`], the sequence's length, and the sequence's tiles in production
//! order, each with the shapes of the values it commits ([`TileShape`]). A proof unit holds one
//! request until #357, of `n = prompt_len + output_len` positions (RFC-0047 'A request's sequence
//! length'). `tessera-pcs` precedes the compiler and the gadgets in the crate order (01 §3), so the
//! tiles come as plain data: the caller derives them from the execution plan's expansion (RFC-0041
//! 'Tiles') and the gadgets' committed values (RFC-0009).
//!
//! # The rules
//!
//! 1. **Domains and order.** A value is a tensor whose multilinear domain pads each axis to a
//!    power of two ([`Axis`]). A dense tensor's variables follow RFC-0047's layout order, most
//!    significant first: its padded axes, those whose length is not a power of two, in axis order,
//!    then its other axes in axis order ([`layout_order`]). [`DenseShape::new`] derives it, and a
//!    dense shape has no other order. An attention tile's exponentials ([`TensorShape::Scores`])
//!    take RFC-0034's order `(i_hi, h, q, i_lo)` instead, with `i = i_hi·2^w + i_lo` and
//!    `w = min(log2 N, log2 k − log2 Q' − log2 g)` ([`Attention`]).
//! 2. **Allocation.** Tile by tile and, within a tile, value by value, each tensor of `n_X`
//!    variables takes the next offset aligned to `2^{max(n_X, log2 k)}`. A tensor therefore spans
//!    `2^{max(n_X − log2 k, 0)}` whole rows, and a tensor smaller than a row takes one row, from
//!    its first column.
//! 3. **The map.** `Ã` has `ν = log2 k + ⌈log2 R⌉` variables, with `R` the rows the layout
//!    allocates, the last tile's end included (RFC-0047 '`ν`'). A claim `X̃(r) = v` is
//!    `Ã(r' ‖ base_X) = v`, where `r'` is `r` with its coordinates permuted into the layout order
//!    and `base_X` is the index of the tensor's first entry over `2^{n_X}`
//!    ([`Placement::point`]). The low `log2 k` coordinates are the column sub-point, the others the
//!    row point.
//! 4. **Zero rows.** Skipped rows are those of outermost length padding, alignment gaps,
//!    rows past the allocation, and a score tile's rows whose keys all follow its
//!    queries (RFC-0034 'Zero rows'). A tensor whose padding holds public constants that may be
//!    non-zero ([`Padding::Constants`], RFC-0035) commits its padding's rows. [`AdviceLayout::rows`]
//!    counts the others, the advice descriptor's `rows` (RFC-0031).
//! 5. **Segments.** Segment `j` holds layers `j·⌊L/S⌋ + min(j, L mod S)` onward, `⌈L/S⌉` of them
//!    for the first `L mod S` segments and `⌊L/S⌋` for the rest ([`segment_layers`]); the first
//!    segment also holds the tiles before the first layer's, and the last those after the last
//!    layer's. A segment's rows run from its first tile's first row to the next segment's first
//!    row, so the alignment gap before a tile belongs to the segment before it, and a segment with
//!    no tile holds no row (RFC-0046 'Segments').
//!
//! # Refusals
//!
//! [`LayoutParams::check`] refuses, as `Layout` (RFC-0007 step 6, RFC-0039), a claim list from
//! which no layout derives: a `query_block` that is no power of two, whose logarithm RFC-0034
//! takes; a row too short for one key against a tile's queries and heads, `g·min(Q, N_max) > k`
//! (RFC-0046 'Rows too short for a key'); no segment; or a row too wide for a codeword.
//! [`AdviceLayout::new`] refuses, as `Layout` too, a layout of more than `2^32` rows (RFC-0047 'The
//! maximum layout'). RFC-0007 step 6 applies that bound at `n = max_context`, where the layout is
//! largest; [`AdviceLayoutCounter`] checks repeated tile patterns there before the caller
//! expands them into individual placements. A `g = n_heads / n_kv_heads` that is
//! no power of two is an architecture rule of RFC-0007 step 2 (RFC-0047 'Heads'), which validation
//! applies before any layout: the layout refuses it again as [`PcsError::LayoutData`], data it
//! never receives, like any other of another form ([`PcsError::Tile`]).
//!
//! # Cost
//!
//! Nothing is materialised: the layout of a 405B proof unit at 16K positions is a few hundred
//! thousand placements, each with its runs of non-zero rows. A dense tensor's runs are at most
//! its non-zero rows, and one or a few for the tensors of a lowering.

use core::ops::Range;

use tessera_field::Field;

use crate::{PcsError, RowLayout};

/// The widest row: a codeword of `2k` symbols needs a subgroup of order `2k ≤ 2^32` (RFC-0002).
const MAX_LOG2_K: u32 = 31;

/// The longest axis: its padded length has at most 32 bits, as a table's rows and a dimension do.
const MAX_AXIS_LEN: u64 = 1 << 32;

/// The largest layout: at most `2^32` rows, a row index of at most 32 bits (RFC-0047 'The maximum
/// layout').
const MAX_ROWS: u64 = 1 << 32;

/// The largest tensor: its entries have 64-bit indices.
const MAX_TENSOR_VARS: u64 = 64;

const NO_LAYER: &str = "a model has at least one layer";
const NO_CONTEXT: &str = "a model's context holds at least one position";
const NO_HEAD: &str = "an attention tile holds at least one query head";
const ROW_WIDTH: &str =
    "a row of k = 2^log2_k values has a codeword of 2k ≤ 2^32 symbols (RFC-0011 'Encoding')";
const SEGMENTS: &str = "the advice has at least one segment (RFC-0046 'Segments')";
const QUERY_BLOCK: &str =
    "query_block is a power of two: RFC-0034's layout takes its logarithm (RFC-0041)";
const GROUP: &str = "g = n_heads / n_kv_heads is a power of two (RFC-0007 step 2, RFC-0047)";
const ROWS_TOO_SHORT: &str =
    "a row holds one key against a tile's queries and heads: g·min(Q, N_max) ≤ k (RFC-0046)";
const POSITIONS: &str = "a sequence has 1 to max_context positions";
const PADDED: &str = "a sequence pads to at most 2^31 positions";
const SLOT_LEN: &str = "a batch's slot length is a power of two at least max_r n_r";
const SLOT_AXIS: &str = "a slotted axis matches the batch's R · S length";
const LAYER_WITHOUT_TILE: &str = "every layer has a tile";
const EMPTY_TILE: &str = "a tile holds at least one committed value (RFC-0041 'Tiles')";
const LAYER_PAST: &str = "a tile's layer is below n_layers";
const LAYER_ORDER: &str =
    "the layers' tiles come in layer order, after the prologue's and before the epilogue's";
const AXIS_EMPTY: &str = "an axis has at least one position";
const AXIS_BITS: &str = "an axis has at most 2^32 positions";
const AXIS_LEN: &str = "an axis's real positions lie within its length";
const TOO_LARGE: &str = "a tensor has at most 2^64 entries";
const BLOCK: &str = "a score tile's query block is below N / Q'";
const ROWS: &str = "a layout allocates at most 2^32 rows (RFC-0047 'The maximum layout')";

fn refuse<T>(rule: &'static str) -> Result<T, PcsError> {
    Err(PcsError::Layout { rule })
}

fn data<T>(rule: &'static str) -> Result<T, PcsError> {
    Err(PcsError::LayoutData { rule })
}

fn place_rows(cursor: u64, vars: u32, log2_k: u32) -> Result<(u64, u64), PcsError> {
    let span = 1u64
        .checked_shl(vars.saturating_sub(log2_k))
        .filter(|&span| span <= MAX_ROWS)
        .ok_or(PcsError::Layout { rule: ROWS })?;
    // Both operands are at most 2^32.
    let first = cursor.next_multiple_of(span);
    let end = first + span;
    if end > MAX_ROWS {
        return refuse(ROWS);
    }
    Ok((first, end))
}

/// Count a model's full allocation before expanding repeated attention tiles.
/// No row map or placement is materialized; skipped rows still count toward the cap.
#[derive(Clone, Debug)]
pub struct AdviceLayoutCounter {
    attention: Attention,
    log2_k: u32,
    allocated: u64,
}

impl AdviceLayoutCounter {
    /// A counter for the same public parameters and length as [`AdviceLayout::new`].
    ///
    /// # Errors
    /// Those of [`LayoutParams::attention`].
    pub fn new(params: LayoutParams, positions: u32) -> Result<Self, PcsError> {
        Ok(Self {
            attention: params.attention(positions)?,
            log2_k: params.log2_k,
            allocated: 0,
        })
    }

    /// A counter for a batch of requests in slots of `slot_len` (RFC-0071 'The plan and the
    /// layout').
    ///
    /// # Errors
    /// Those of [`LayoutParams::batch_attention`].
    pub fn new_batch(
        params: LayoutParams,
        slot_len: u32,
        lengths: &[u32],
    ) -> Result<Self, PcsError> {
        Ok(Self {
            attention: params.batch_attention(slot_len, lengths)?,
            log2_k: params.log2_k,
            allocated: 0,
        })
    }

    /// Append `repetitions` occurrences of one tile's value shapes.
    /// After one complete pattern the cursor's residue modulo its largest
    /// alignment is fixed. A second occurrence therefore determines the stride
    /// of every remaining occurrence, even for heterogeneous column spans.
    ///
    /// # Errors
    /// `LayoutData` for no value, no repetition or invalid tensor metadata;
    /// `Layout` if the complete allocation exceeds `2^32` rows. Failure leaves
    /// the counter unchanged.
    pub fn append(&mut self, values: &[TensorShape], repetitions: u64) -> Result<(), PcsError> {
        if values.is_empty() {
            return data(EMPTY_TILE);
        }
        if repetitions == 0 {
            return data("a repeated tile occurs at least once");
        }
        for value in values {
            if let Err(rule) = check_value(value, &self.attention) {
                return data(rule);
            }
        }
        let advance = |mut cursor| {
            for value in values {
                cursor = place_rows(cursor, geometry(value, &self.attention).1, self.log2_k)?.1;
            }
            Ok::<_, PcsError>(cursor)
        };
        let first = advance(self.allocated)?;
        let end = if repetitions == 1 {
            first
        } else {
            let step = advance(first)? - first;
            repetitions
                .checked_sub(1)
                .and_then(|n| n.checked_mul(step))
                .and_then(|n| first.checked_add(n))
                .filter(|&n| n <= MAX_ROWS)
                .ok_or(PcsError::Layout { rule: ROWS })?
        };
        self.allocated = end;
        Ok(())
    }

    /// Full allocated rows, including padding, causal zero rows and alignment gaps.
    pub fn allocated_rows(&self) -> u64 {
        self.allocated
    }
}

/// The claim list's data that the advice layout reads besides the tiles (INV-PCS-1). Every field
/// is public: the claim list states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutParams {
    /// `log2 k`, `AdviceLayoutParams.log2_k` (RFC-0026): a row holds `k = 2^log2_k` values. At
    /// most 31.
    pub log2_k: u32,
    /// `S`, `AdviceLayoutParams.segments`: the segments the rows are partitioned into, at least 1
    /// (RFC-0046). The parameter set fixes it, 8 or 1 (RFC-0026), and the layout holds one
    /// [`Segment`] for each.
    pub segments: u32,
    /// `L`, `ArchSpec.n_layers`, at least 1.
    pub layers: u32,
    /// `g = n_heads / n_kv_heads`, the query heads of an attention tile (RFC-0041 'Tiles'): a power
    /// of two, an architecture rule of RFC-0007 step 2 (RFC-0047 'Heads').
    pub group: u32,
    /// `Q`, `GadgetPolicy.query_block`: a power of two (RFC-0041 'Production order').
    pub query_block: u32,
    /// `ArchSpec.max_context`, at least 1: `N_max` is its next power of two.
    pub max_context: u32,
}

impl LayoutParams {
    /// Checks that a layout derives from the parameters for every sequence the claim list admits,
    /// as RFC-0007 step 6 does.
    ///
    /// # Errors
    ///
    /// In this order:
    ///
    /// - [`PcsError::LayoutData`] for no layer, a context of no position, a tile of no query head
    ///   or a `g` that is no power of two: architecture rules that RFC-0007 step 2 refuses before
    ///   step 6 (RFC-0047 'Heads'), checked again here as data the layout never receives;
    /// - [`PcsError::Layout`] for `log2 k > 31`, no segment, a `query_block` that is no power of
    ///   two, and `g·min(Q, N_max) > k`, with `N_max` the padded `max_context`: a row that cannot
    ///   hold one key against a tile's queries and heads (RFC-0046 'Rows too short for a key'). It
    ///   suffices at `N_max`, since `min(Q, N)` grows with `N`.
    pub fn check(&self) -> Result<(), PcsError> {
        if self.layers == 0 {
            return data(NO_LAYER);
        }
        if self.max_context == 0 {
            return data(NO_CONTEXT);
        }
        if self.group == 0 {
            return data(NO_HEAD);
        }
        if !self.group.is_power_of_two() {
            return data(GROUP);
        }
        if self.log2_k > MAX_LOG2_K {
            return refuse(ROW_WIDTH);
        }
        if self.segments == 0 {
            return refuse(SEGMENTS);
        }
        if !self.query_block.is_power_of_two() {
            return refuse(QUERY_BLOCK);
        }
        // N_max ≤ 2^32 and Q ≤ 2^31, g ≤ 2^31: the product stays below 2^62.
        let n_max = u64::from(self.max_context).next_power_of_two();
        let queries = u64::from(self.query_block).min(n_max);
        if u64::from(self.group) * queries > 1 << self.log2_k {
            return refuse(ROWS_TOO_SHORT);
        }
        Ok(())
    }

    /// The attention tiles' geometry for a sequence of `positions` (RFC-0034 'Causal layout',
    /// RFC-0046 'Short query blocks'): for a request, `n = prompt_len + output_len` (RFC-0047 'A
    /// request's sequence length'), at most `max_context`.
    ///
    /// # Errors
    ///
    /// Those of [`check`](Self::check), then [`PcsError::LayoutData`] for a sequence of no
    /// position, of more than `max_context`, or longer than `2^31`.
    pub fn attention(&self, positions: u32) -> Result<Attention, PcsError> {
        self.check()?;
        if positions == 0 || positions > self.max_context {
            return data(POSITIONS);
        }
        let padded = match positions.checked_next_power_of_two() {
            Some(padded) => padded,
            None => return data(PADDED),
        };
        let queries = self.query_block.min(padded);
        // `check` gave g·min(Q, N_max) ≤ k, and N ≤ N_max: log2 g + log2 Q' ≤ log2 k.
        let free = self.log2_k - queries.trailing_zeros() - self.group.trailing_zeros();
        Ok(Attention {
            positions,
            padded,
            queries,
            blocks: padded / queries,
            heads: self.group,
            key_bits: padded.trailing_zeros().min(free),
            slot_lengths: None,
        })
    }

    /// The attention tiles' geometry for a batch of `lengths.len() ≥ 1` requests in slots of
    /// `slot_len = S = 2^b` positions (RFC-0071 'The plan and the layout').
    ///
    /// With one request and `slot_len = 2^{⌈log2 lengths[0]⌉}` this reduces to
    /// [`attention`](Self::attention). For `R ≥ 2`, each slot spans `S` positions, `Q' = min(Q, S)`
    /// divides `S`, and the batch's padded token axis `2^{b + ⌈log2 R⌉}` expands into
    /// `2^{b + ⌈log2 R⌉} / Q'` query blocks.
    ///
    /// # Errors
    ///
    /// Those of [`check`](Self::check), [`PcsError::LayoutData`] for no request, a request outside
    /// `1..=max_context`, a `slot_len` that is not a power of two at least `max_r n_r` or above
    /// `2^31`, and [`PcsError::Layout`] when the padded token axis or query-block count exceeds the
    /// `2^32`-row layout bound (RFC-0047, RFC-0071).
    pub fn batch_attention(&self, slot_len: u32, lengths: &[u32]) -> Result<Attention, PcsError> {
        self.check()?;
        if lengths.is_empty() || lengths.iter().any(|&n| n == 0 || n > self.max_context) {
            return data(POSITIONS);
        }
        let n_max = lengths.iter().copied().max().unwrap_or(0);
        let Some(min_slot) = n_max.checked_next_power_of_two() else {
            return data(PADDED);
        };
        if !slot_len.is_power_of_two() || slot_len < min_slot {
            return data(SLOT_LEN);
        }
        if lengths.len() == 1 {
            let queries = self.query_block.min(slot_len);
            let free = self.log2_k - queries.trailing_zeros() - self.group.trailing_zeros();
            return Ok(Attention {
                positions: lengths[0],
                padded: slot_len,
                queries,
                blocks: slot_len / queries,
                heads: self.group,
                key_bits: slot_len.trailing_zeros().min(free),
                slot_lengths: None,
            });
        }
        let r = u64::try_from(lengths.len()).map_err(|_| PcsError::Layout { rule: ROWS })?;
        let r_padded = r
            .checked_next_power_of_two()
            .ok_or(PcsError::Layout { rule: ROWS })?;
        let padded_tokens = u64::from(slot_len)
            .checked_mul(r_padded)
            .filter(|&p| p <= MAX_AXIS_LEN)
            .ok_or(PcsError::Layout { rule: ROWS })?;
        let positions = u64::from(slot_len)
            .checked_mul(r)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(PcsError::Layout { rule: ROWS })?;
        let queries = self.query_block.min(slot_len);
        let blocks = u32::try_from(padded_tokens / u64::from(queries))
            .map_err(|_| PcsError::Layout { rule: ROWS })?;
        let free = self.log2_k - queries.trailing_zeros() - self.group.trailing_zeros();
        Ok(Attention {
            positions,
            padded: slot_len,
            queries,
            blocks,
            heads: self.group,
            key_bits: slot_len.trailing_zeros().min(free),
            slot_lengths: Some(lengths.to_vec()),
        })
    }
}

/// The attention tiles' geometry for one proof unit (RFC-0034 'Causal layout', RFC-0041 'Tiles',
/// RFC-0046 'Short query blocks', RFC-0071 'The plan and the layout'): the tile of a KV head and
/// block `b` holds `e[h, q, i]` for its `g` query heads, the `Q'` queries `b·Q'` to
/// `(b + 1)·Q' − 1` and its slot's `S` keys, each of its rows `2^w` keys against all its queries
/// and heads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Attention {
    /// `n` for one request, or `N = R · S` for a multi-request batch: the unpadded token axis.
    pub positions: u32,
    /// `S = 2^b`: the padded sequence for one request (`2^{⌈log2 n⌉}`), or the slot length for a
    /// batch (RFC-0071 'The plan and the layout').
    pub padded: u32,
    /// `Q' = min(Q, S)`, the queries of a block.
    pub queries: u32,
    /// `2^{b + ⌈log2 R⌉} / Q'`, the blocks of an attention node's KV head across the padded token
    /// axis.
    pub blocks: u32,
    /// `g`, the query heads of a tile.
    pub heads: u32,
    /// `w = min(b, log2 k − log2 Q' − log2 g)`: a row of a score tile holds `2^w` keys. A tile of
    /// `g·Q'·S ≤ k` entries has `w = b` and takes one row.
    pub key_bits: u32,
    slot_lengths: Option<Vec<u32>>,
}

impl Attention {
    /// Per-request real lengths when this geometry spans `R ≥ 2` request slots, or `None` for a
    /// single sequence.
    pub fn slot_lengths(&self) -> Option<&[u32]> {
        self.slot_lengths.as_deref()
    }

    /// The number of real queries held by query block `block` (`0..=queries`).
    pub fn held_queries(&self, block: u32) -> u32 {
        match &self.slot_lengths {
            None => {
                let first = block.saturating_mul(self.queries);
                self.positions.saturating_sub(first).min(self.queries)
            }
            Some(lengths) => {
                let first = u64::from(block) * u64::from(self.queries);
                let slot = u64::from(self.padded);
                let r = first / slot;
                let q0 = (first % slot) as u32;
                usize::try_from(r)
                    .ok()
                    .and_then(|r| lengths.get(r))
                    .map_or(0, |&n_r| n_r.saturating_sub(q0).min(self.queries))
            }
        }
    }

    /// The shape of a tensor an attention tile holds one value of per query row, at `block`: its
    /// reciprocals `R[h, q]`, its maxima `c[h, q]` and its bracket limbs (RFC-0041 'Tiles').
    ///
    /// It is the tile's share, allocated as a tensor of its own (RFC-0011 rule 2): `[g, Q']`, the
    /// queries at or past the slot's `n_r` its padding, and a block of padding alone without a real
    /// query. Both lengths are powers of two, so its layout order is row-major, in every block
    /// alike.
    pub fn row_shape(&self, block: u32, padding: Padding) -> DenseShape {
        let held = self.held_queries(block);
        DenseShape::new(
            vec![
                Axis::new(u64::from(self.heads)),
                Axis {
                    len: u64::from(self.queries),
                    real: u64::from(held),
                },
            ],
            padding,
        )
    }
}

/// One axis of a dense tensor: its length, padded to the next power of two in the tensor's domain
/// (RFC-0011 rule 1), and its real positions, `0` to `real − 1`; the rest is padding.
///
/// A tensor's axis of `d` positions is [`Axis::new`]`(d)`, every position real. A tile's share of
/// a tensor has the share's length, and may hold fewer real positions: a query block past the
/// sequence's end has `Q'` positions and fewer real queries, or none. RFC-0047 judges an axis
/// padded by its length alone ([`is_padded`](Self::is_padded)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis {
    /// The length, from 1 to `2^32`.
    pub len: u64,
    /// The real positions, at most `len`.
    pub real: u64,
}

impl Axis {
    /// An axis of `len` positions, every one real.
    pub fn new(len: u64) -> Self {
        Self { len, real: len }
    }

    /// `⌈log2 len⌉`, the axis's variables in the tensor's domain.
    pub fn bits(&self) -> u32 {
        // A length of 0 or 1 takes no variable; above 2^63 no length is valid.
        self.len
            .checked_next_power_of_two()
            .map_or(u64::BITS, u64::trailing_zeros)
    }

    /// Whether the axis is padded: its length is not a power of two (RFC-0047 'The order of a
    /// tensor's variables').
    pub fn is_padded(&self) -> bool {
        !self.len.is_power_of_two()
    }
}

/// RFC-0047's layout order of a dense tensor of axes `axes`, outermost first: the positions of its
/// padded axes, those whose length is not a power of two, in axis order, then those of its other
/// axes in axis order, from the most significant to the least.
///
/// At 405B, `[16384, 53248]` takes `[1, 0]`: its padding, the channels 53,248 to 65,535, fills
/// whole rows, and a second padded axis would pad within them (RFC-0047 'Padding').
/// [`DenseShape::new`] takes this order, so a tensor laid out by this module has no other.
pub fn layout_order(axes: &[Axis]) -> Vec<usize> {
    let padded = axes.iter().enumerate().filter(|(_, a)| a.is_padded());
    let others = axes.iter().enumerate().filter(|(_, a)| !a.is_padded());
    padded.chain(others).map(|(axis, _)| axis).collect()
}

/// What a dense tensor's padding holds (RFC-0011 rule 1, RFC-0035 'What a padded row holds').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Padding {
    /// Zero, as RFC-0035 has it for every tensor but those [`Constants`](Self::Constants) names: a
    /// row of padding alone is a zero row, skipped.
    Zero,
    /// The values its gadget computes on zero inputs, public constants some of which are not zero:
    /// the scoring bound's `Exp2Frac` and `Pow2` lookups, the AES S-box's and the anchors'
    /// constants (RFC-0035). Rule 4 skips a row only when all of it is zero, so every row of the
    /// tensor is committed.
    Constants,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AxisSlots {
    axis: usize,
    slot_len: u32,
    lengths: Vec<u32>,
}

/// The shape of a dense tensor: its axes in its own row-major order, RFC-0047's layout order of
/// their variables, and what its padding holds.
///
/// The order is derived ([`layout_order`]), never given: a tile's derivation cannot lay a tensor
/// out in another.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DenseShape {
    axes: Vec<Axis>,
    order: Vec<usize>,
    padding: Padding,
    slots: Option<AxisSlots>,
}

impl DenseShape {
    /// The tensor of axes `axes`, outermost first, in the order its relations name its indices
    /// (RFC-0010 'Multilinear extensions'), laid out in RFC-0047's order; its padding holds
    /// `padding`.
    pub fn new(axes: Vec<Axis>, padding: Padding) -> Self {
        let order = layout_order(&axes);
        Self {
            axes,
            order,
            padding,
            slots: None,
        }
    }

    /// A dense tensor whose axis `slot_axis` is a multi-request batch's slotted token axis of
    /// `lengths.len() · slot_len` positions, with request `r`'s real positions `0..lengths[r]`
    /// inside slot `r` (RFC-0071 'The plan and the layout').
    ///
    /// When `lengths` holds at most one request, this is [`new`](Self::new).
    pub fn with_slots(
        axes: Vec<Axis>,
        padding: Padding,
        slot_axis: usize,
        slot_len: u32,
        lengths: Vec<u32>,
    ) -> Self {
        let order = layout_order(&axes);
        let slots = (lengths.len() >= 2).then_some(AxisSlots {
            axis: slot_axis,
            slot_len,
            lengths,
        });
        Self {
            axes,
            order,
            padding,
            slots,
        }
    }

    /// The axes, outermost first: row-major, the last axis the lowest variables of a claim.
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// The axes, by position in [`axes`](Self::axes), from the most significant to the least in
    /// the layout: [`layout_order`]'s.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// What its padding holds.
    pub fn padding(&self) -> Padding {
        self.padding
    }
}

/// A committed value of a tile, allocated as one tensor (RFC-0011 rule 2).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorShape {
    /// A tensor laid out by whole axes.
    Dense(DenseShape),
    /// The exponentials of an attention tile at query block `block`: `e[h, q, i]` for the tile's
    /// `g` heads, `Q'` queries and `N` keys, in RFC-0034's order `(i_hi, h, q, i_lo)`. Its rows are
    /// blocks of keys, and those whose keys all follow the block's queries, or that hold padding
    /// alone, are zero rows (RFC-0034 'Zero rows'). Its natural axes are `[g, Q', N]`.
    Scores {
        /// The query block `b`, from 0 to `N / Q' − 1`.
        block: u32,
    },
}

/// One tile of a sequence (RFC-0041 'Tiles'), in production order: the values it commits, each
/// allocated as a tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileShape {
    /// The layer whose nodes the tile belongs to (RFC-0037 'Layers and kinds'), or `None` for the
    /// tokens, the prologue's and the epilogue's nodes and the tables. The layers' tiles come in
    /// layer order, the first from layer 0; the tiles before them are the prologue's, segment 0's,
    /// and those after them the epilogue's, segment `S − 1`'s (RFC-0046 'Segments').
    pub layer: Option<u32>,
    /// The values, in the tile's order (RFC-0041 'Production order'): the node's outputs, then the
    /// gadget's committed values in the order RFC-0009 lists them. An attention tile holds its
    /// exponentials and reciprocals, then its maxima, then `NonNeg(a_q)`'s limbs and
    /// `NonNeg(b_q)`'s, in relation 3's order (RFC-0009 'SoftmaxRow').
    pub values: Vec<TensorShape>,
}

/// The variable order of a placed tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Geometry {
    /// By whole axes: the axes, outermost first, and RFC-0047's layout order.
    Dense { axes: Vec<Axis>, order: Vec<usize> },
    /// RFC-0034's `(i_hi, h, q, i_lo)`, over natural axes `[g, Q', N]`.
    Scores {
        /// `w`.
        key_bits: u32,
        /// `log2 N`.
        key_vars: u32,
        /// `log2 Q'`.
        query_vars: u32,
        /// `log2 g`.
        head_vars: u32,
    },
}

/// One natural source axis reconstructed from public storage bits. Score keys have two
/// fragments; all other axes have only the low fragment. Never holds witness data.
struct SourceAxis {
    len: usize,
    stride: usize,
    shift: u32,
    mask: u128,
    high_shift: u32,
    high_mask: u128,
    low_bits: u32,
}

/// Where one tensor lies in `Ã` (RFC-0011 rules 2 to 4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    first_row: u64,
    /// `max(n_X − log2 k, 0)`: the tensor spans `2^row_bits` rows.
    row_bits: u32,
    /// `n_X`.
    vars: u32,
    log2_k: u32,
    /// `base_X`: the tensor's first entry is `base_X · 2^{n_X}`.
    base: u128,
    /// `ν − n_X`.
    base_bits: u32,
    geometry: Geometry,
    /// The rows that hold a real entry, in increasing order.
    nonzero: Vec<Range<u64>>,
}

impl Placement {
    /// The tensor's first row: aligned to [`row_count`](Self::row_count).
    pub fn first_row(&self) -> u64 {
        self.first_row
    }

    /// The rows the tensor spans, `2^{max(n_X − log2 k, 0)}`, from its first row. A tensor smaller
    /// than a row takes one, from its first column.
    pub fn row_count(&self) -> u64 {
        1 << self.row_bits
    }

    /// `n_X`, the variables of the tensor's padded domain.
    pub fn vars(&self) -> u32 {
        self.vars
    }

    /// The tensor's rows that hold a real entry, as runs of consecutive rows in increasing order;
    /// its other rows are zero rows (RFC-0011 rule 4).
    pub fn nonzero(&self) -> &[Range<u64>] {
        &self.nonzero
    }

    /// `base_X`, the tensor's fixed row-offset bits (RFC-0011 rule 3): its entries are `Ã`'s
    /// `base_X · 2^{n_X}` to `(base_X + 1) · 2^{n_X} − 1`. A tensor smaller than a row has
    /// `log2 k − n_X` zero bits of column below its row.
    pub fn base(&self) -> u128 {
        self.base
    }

    /// The bits of `base_X`, `ν − n_X`.
    pub fn base_bits(&self) -> u32 {
        self.base_bits
    }

    /// The point of `Ã` at which a claim `X̃(r)` on this tensor is the claim `Ã(r' ‖ base_X)`
    /// (RFC-0011 rule 3): `ν` coordinates, the lowest variable first, the first `log2 k` the
    /// column sub-point and the rest the row point.
    ///
    /// `r` names the tensor's variables in its own row-major order, the lowest first: the last
    /// axis's coordinates, its lowest bit first, then the axis before it, up to the first axis's
    /// (RFC-0010 'Multilinear extensions'). For a score tile that is `r_i ‖ r_q ‖ r_h` over
    /// `[g, Q', N]`, and its layout puts them in RFC-0034's order:
    /// `r_{i,lo} ‖ r_q ‖ r_h ‖ r_{i,hi}`.
    ///
    /// # Errors
    ///
    /// [`PcsError::ClaimPoint`] unless `r` has `n_X` coordinates.
    pub fn point<F: Field>(&self, r: &[F]) -> Result<Vec<F>, PcsError> {
        self.point_geometry(r, F::ZERO, F::ONE)
    }

    /// The coordinate permutation of [`Self::point`], for values or symbolic origins.
    /// Only copies, reordering and Boolean placement coordinates are permitted: no arithmetic
    /// on `C` is available. A public memory planner can therefore count distinct column-point
    /// origins before Fiat-Shamir challenges are drawn (RFC-0011 rules 1 and 3).
    ///
    /// `zero` and `one` represent the two distinct Boolean constants in the caller's domain.
    ///
    /// # Errors
    /// [`PcsError::ClaimPoint`] unless `r` has this tensor's number of variables.
    pub fn point_geometry<C: Copy>(&self, r: &[C], zero: C, one: C) -> Result<Vec<C>, PcsError> {
        let vars = self.vars as usize;
        if r.len() != vars {
            return Err(PcsError::ClaimPoint {
                expected: vars,
                found: r.len(),
            });
        }
        let mut point = Vec::with_capacity(vars + self.base_bits as usize);
        match &self.geometry {
            Geometry::Dense { axes, order } => {
                let starts = natural_starts(axes);
                // The least significant axis first.
                for &axis in order.iter().rev() {
                    let start = starts[axis];
                    point.extend_from_slice(&r[start..start + axes[axis].bits() as usize]);
                }
            }
            &Geometry::Scores {
                key_bits,
                key_vars,
                query_vars,
                head_vars,
            } => {
                let (w, n, q, h) = (
                    key_bits as usize,
                    key_vars as usize,
                    query_vars as usize,
                    head_vars as usize,
                );
                point.extend_from_slice(&r[..w]);
                point.extend_from_slice(&r[n..n + q + h]);
                point.extend_from_slice(&r[w..n]);
            }
        }
        point.extend((0..self.base_bits).map(|bit| match (self.base >> bit) & 1 {
            1 => one,
            _ => zero,
        }));
        Ok(point)
    }

    /// Requested heap bytes of the axis descriptors retained by [`Self::row_indices`].
    /// This reads public rank metadata and allocates nothing. Caller-owned source
    /// shapes, field rows, iterator stack storage and allocator overhead are separate.
    /// Returns `None` if the single vector would be unaddressable or overflow.
    pub fn row_mapping_memory_bytes(&self) -> Option<usize> {
        let rank = match &self.geometry {
            Geometry::Dense { axes, .. } => axes.len(),
            Geometry::Scores { .. } => 3,
        };
        let bytes = rank.checked_mul(size_of::<SourceAxis>())?;
        (bytes <= isize::MAX as usize).then_some(bytes)
    }

    /// Natural row-major source indices for the tensor-covered columns of `row`.
    /// `shape` is the source's public, unpadded shape in natural axis order. Each item is
    /// `Some(index)` for an element of that shape or `None` for per-axis length padding.
    ///
    /// The iterator covers `min(k, 2^n_X)` columns, starting at column zero. A caller supplies
    /// zero for the remainder of a row when the tensor is smaller than `k`. Scalar tensors
    /// therefore yield one item. A zero-length source axis yields only padding. The caller
    /// still follows [`Self::nonzero`] for structural/causal/slot zero rows; this mapping does
    /// not replace that public schedule or the witness's computed padding values.
    ///
    /// Only axis metadata is allocated, proportional to rank and independent of tensor rows.
    /// No witness values influence the mapping (INV-PCS-1, RFC-0047, RFC-0034).
    ///
    /// # Errors
    /// [`PcsError::ElementAxes`] for a rank mismatch, [`PcsError::ElementRange`] for a source
    /// axis larger than its padded domain, or [`PcsError::LayoutData`] for a row outside this
    /// placement, a source product that overflows `usize`, or [`PcsError::LayoutMemory`] for a failed metadata reservation.
    pub fn row_indices(
        &self,
        row: u64,
        shape: &[u32],
    ) -> Result<impl ExactSizeIterator<Item = Option<usize>>, PcsError> {
        let bad = |rule| PcsError::LayoutData { rule };
        let local_row = row
            .checked_sub(self.first_row)
            .filter(|&r| r < self.row_count())
            .ok_or_else(|| bad("source row lies inside its tensor placement"))?;
        let rank = match &self.geometry {
            Geometry::Dense { axes, .. } => axes.len(),
            Geometry::Scores { .. } => 3,
        };
        if shape.len() != rank {
            return Err(PcsError::ElementAxes {
                expected: rank,
                found: shape.len(),
            });
        }
        let mut axes = Vec::new();
        axes.try_reserve_exact(rank)
            .map_err(|_| PcsError::LayoutMemory)?;
        for &len in shape {
            axes.push(SourceAxis {
                len: len as usize,
                stride: 0,
                shift: 0,
                mask: 0,
                high_shift: 0,
                high_mask: 0,
                low_bits: 0,
            });
        }
        match &self.geometry {
            Geometry::Dense {
                axes: domain,
                order,
            } => {
                let mut shift = 0;
                for &axis in order.iter().rev() {
                    let bits = domain[axis].bits();
                    axes[axis].shift = shift;
                    axes[axis].mask = (1u128 << bits) - 1;
                    shift += bits;
                }
            }
            &Geometry::Scores {
                key_bits,
                key_vars,
                query_vars,
                head_vars,
            } => {
                // RFC-0034: storage low-to-high is i_lo, q, h, i_hi.
                axes[0].shift = key_bits + query_vars;
                axes[0].mask = (1u128 << head_vars) - 1;
                axes[1].shift = key_bits;
                axes[1].mask = (1u128 << query_vars) - 1;
                axes[2].mask = (1u128 << key_bits) - 1;
                axes[2].high_shift = key_bits + query_vars + head_vars;
                axes[2].high_mask = (1u128 << (key_vars - key_bits)) - 1;
                axes[2].low_bits = key_bits;
            }
        }
        // An empty source has no addressable elements. Its otherwise unused trailing
        // strides need not fit usize (notably on a 32-bit host); every result is padding.
        let mut stride = usize::from(!shape.contains(&0));
        for (axis, a) in axes.iter_mut().enumerate().rev() {
            let capacity = (a.mask + 1) * (a.high_mask + 1);
            if a.len as u128 > capacity {
                return Err(PcsError::ElementRange { axis });
            }
            a.stride = stride;
            stride = stride
                .checked_mul(a.len)
                .ok_or_else(|| bad("source shape product fits usize"))?;
        }
        let start = u128::from(local_row) << self.log2_k;
        let count = 1usize
            .checked_shl(self.vars.min(self.log2_k))
            .ok_or_else(|| bad("source row width fits usize"))?;
        Ok((0..count).map(move |column| {
            let index = start + column as u128;
            axes.iter().try_fold(0usize, |flat, a| {
                let coordinate = (((index >> a.shift) & a.mask)
                    | (((index >> a.high_shift) & a.high_mask) << a.low_bits))
                    as usize;
                // Each coordinate is below its source length; the checked source product
                // above bounds every offset and sum. Branches depend on public indices only.
                (coordinate < a.len).then(|| flat + coordinate * a.stride)
            })
        }))
    }

    /// The row and column of `Ã` that hold the element at `coords`, one coordinate per natural
    /// axis, outermost first: a dense tensor's axes, or `[h, q, i]` for a score tile, `q` within
    /// its block. Padding positions are in the domain.
    ///
    /// # Errors
    ///
    /// [`PcsError::ElementAxes`] unless `coords` has one coordinate per axis, then
    /// [`PcsError::ElementRange`] for the first that lies past its axis's padded length.
    pub fn locate(&self, coords: &[u64]) -> Result<(u64, u64), PcsError> {
        let inside = |bits: &[u32]| -> Result<(), PcsError> {
            if coords.len() != bits.len() {
                return Err(PcsError::ElementAxes {
                    expected: bits.len(),
                    found: coords.len(),
                });
            }
            match coords.iter().zip(bits).position(|(&x, &b)| x >> b != 0) {
                Some(axis) => Err(PcsError::ElementRange { axis }),
                None => Ok(()),
            }
        };
        let local: u128 = match &self.geometry {
            Geometry::Dense { axes, order } => {
                let bits: Vec<u32> = axes.iter().map(Axis::bits).collect();
                inside(&bits)?;
                let mut local = 0u128;
                let mut shift = 0;
                for &axis in order.iter().rev() {
                    local |= u128::from(coords[axis]) << shift;
                    shift += axes[axis].bits();
                }
                local
            }
            &Geometry::Scores {
                key_bits,
                key_vars,
                query_vars,
                head_vars,
            } => {
                inside(&[head_vars, query_vars, key_vars])?;
                let (h, q, i) = (coords[0], coords[1], coords[2]);
                let low = i & ((1 << key_bits) - 1);
                u128::from(low)
                    | u128::from(q) << key_bits
                    | u128::from(h) << (key_bits + query_vars)
                    | u128::from(i >> key_bits) << (key_bits + query_vars + head_vars)
            }
        };
        let index = self.base << self.vars | local;
        // The row is below the allocation's 2^32 rows, and the column below k.
        Ok((
            (index >> self.log2_k) as u64,
            (index & ((1 << self.log2_k) - 1)) as u64,
        ))
    }
}

/// Where each natural axis's coordinates start in a claim's point: the last axis's at 0.
fn natural_starts(axes: &[Axis]) -> Vec<usize> {
    let mut starts = vec![0; axes.len()];
    let mut start = 0;
    for (axis, a) in axes.iter().enumerate().rev() {
        starts[axis] = start;
        start += a.bits() as usize;
    }
    starts
}

/// Where a tile lies: its layer, its segment, its first row and its tensors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TilePlacement {
    /// The tile's layer, as given.
    pub layer: Option<u32>,
    /// The segment that holds it (RFC-0046 'Segments').
    pub segment: u32,
    /// Its first tensor's first row, where its segment starts when it is the segment's first.
    pub first_row: u64,
    /// Its tensors, as positions in [`AdviceLayout::tensors`].
    pub tensors: Range<usize>,
}

/// A segment of the rows (RFC-0011 rule 5, RFC-0046 'Segments'): what one GPU commits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The layers it holds.
    pub layers: Range<u32>,
    /// Its rows: from its first tile's first row to the next segment's, the alignment gaps
    /// included. A segment without a tile has none.
    pub rows: Range<u64>,
    /// Its non-zero rows, as runs in increasing order: the rows its column chains absorb.
    pub nonzero: Vec<Range<u64>>,
}

/// The canonical advice layout of one proof unit (RFC-0011 'Canonical advice layout'): every
/// tensor's placement, the zero-row map and the segments.
///
/// [`AdviceLayout::new`] derives it from public data alone (INV-PCS-1). The verifier derives its
/// own from the claim list and the batch, and hashes [`rows`](Self::rows) into the advice
/// commitment's `commitment_id` (RFC-0031).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceLayout {
    params: LayoutParams,
    attention: Attention,
    row_vars: u32,
    allocated: u64,
    rows: u64,
    tensors: Vec<Placement>,
    tiles: Vec<TilePlacement>,
    segments: Vec<Segment>,
}

impl AdviceLayout {
    /// Requested heap bytes retained by this public layout, including all nested
    /// vectors. A cloned layout uses no more: its vectors reserve their lengths,
    /// which are bounded by these capacities. Does not include the inline struct,
    /// source tile shapes, construction scratch or allocator bookkeeping.
    /// Returns `None` on byte-count overflow.
    pub fn heap_bytes(&self) -> Option<usize> {
        fn vector<T>(v: &Vec<T>) -> Option<usize> {
            v.capacity().checked_mul(size_of::<T>())
        }
        let mut bytes = vector(&self.tensors)?
            .checked_add(vector(&self.tiles)?)?
            .checked_add(vector(&self.segments)?)?;
        if let Some(lengths) = &self.attention.slot_lengths {
            bytes = bytes.checked_add(vector(lengths)?)?;
        }
        for tensor in &self.tensors {
            bytes = bytes.checked_add(vector(&tensor.nonzero)?)?;
            if let Geometry::Dense { axes, order } = &tensor.geometry {
                bytes = bytes
                    .checked_add(vector(axes)?)?
                    .checked_add(vector(order)?)?;
            }
        }
        for segment in &self.segments {
            bytes = bytes.checked_add(vector(&segment.nonzero)?)?;
        }
        Some(bytes)
    }
    /// The layout of a sequence of `positions` whose tiles, in production order (RFC-0041
    /// 'Production order'), are `tiles`, under `params`.
    ///
    /// A proof unit holds one request (#357): `positions` is its `n = prompt_len + output_len`,
    /// every token it commits a position (RFC-0047 'A request's sequence length', 03 §4), at most
    /// `max_context`. The verifier refuses a longer request (RFC-0019 step 4, `Batch`).
    ///
    /// # Errors
    ///
    /// In this order:
    ///
    /// - those of [`LayoutParams::attention`];
    /// - tile by tile, [`PcsError::Tile`] for a tile that holds no value, a layer of `L` or more,
    ///   a layer out of order (the first layer's tiles not first, a layer's tiles not after the
    ///   one before's, or a layer's after a tile of the epilogue), then, value by value, an axis
    ///   of no position, of more than `2^32`, or with more real positions than its length, a
    ///   tensor of more than `2^64` entries, or a score tile's block of `N / Q'` or more;
    /// - [`PcsError::LayoutData`] when the last layer has no tile;
    /// - [`PcsError::Layout`] for a layout of more than `2^32` rows (RFC-0047 'The maximum
    ///   layout').
    pub fn new(
        params: LayoutParams,
        positions: u32,
        tiles: &[TileShape],
    ) -> Result<Self, PcsError> {
        let attention = params.attention(positions)?;
        Self::from_attention(params, attention, tiles)
    }

    /// The layout of a batch of `lengths.len() ≥ 1` requests in slots of `slot_len` positions
    /// whose tiles, in production order, are `tiles`, under `params` (RFC-0071 'The plan and the
    /// layout').
    ///
    /// # Errors
    ///
    /// Those of [`LayoutParams::batch_attention`], followed by the tile, layer and `2^32`-row
    /// checks of [`new`](Self::new).
    pub fn new_batch(
        params: LayoutParams,
        slot_len: u32,
        lengths: &[u32],
        tiles: &[TileShape],
    ) -> Result<Self, PcsError> {
        let attention = params.batch_attention(slot_len, lengths)?;
        Self::from_attention(params, attention, tiles)
    }

    fn from_attention(
        params: LayoutParams,
        attention: Attention,
        tiles: &[TileShape],
    ) -> Result<Self, PcsError> {
        let segment_of = segment_of(params.layers, params.segments);
        let last_segment = params.segments - 1;
        // The tiles' forms and segments.
        let mut segments_of = Vec::with_capacity(tiles.len());
        let mut last_layer: Option<u32> = None;
        let mut epilogue = false;
        for (index, tile) in tiles.iter().enumerate() {
            let fail = |rule| Err(PcsError::Tile { tile: index, rule });
            if tile.values.is_empty() {
                return fail(EMPTY_TILE);
            }
            let segment = match tile.layer {
                Some(layer) => {
                    if layer >= params.layers {
                        return fail(LAYER_PAST);
                    }
                    let follows = match last_layer {
                        None => layer == 0,
                        Some(last) => layer == last || layer.checked_sub(1) == Some(last),
                    };
                    if epilogue || !follows {
                        return fail(LAYER_ORDER);
                    }
                    last_layer = Some(layer);
                    segment_of(layer)
                }
                None if last_layer.is_some() => {
                    epilogue = true;
                    last_segment
                }
                None => 0,
            };
            for value in &tile.values {
                if let Err(rule) = check_value(value, &attention) {
                    return fail(rule);
                }
            }
            segments_of.push(segment);
        }
        if last_layer != Some(params.layers - 1) {
            return data(LAYER_WITHOUT_TILE);
        }
        // Rule 2: each tensor at the next offset aligned to its span, the layout within RFC-0047's
        // maximum of 2^32 rows.
        let log2_k = params.log2_k;
        let mut cursor: u64 = 0;
        let mut tensors = Vec::new();
        let mut placed = Vec::with_capacity(tiles.len());
        for (tile, &segment) in tiles.iter().zip(&segments_of) {
            let first = tensors.len();
            for value in &tile.values {
                let (geometry, vars) = geometry(value, &attention);
                let row_bits = vars.saturating_sub(log2_k);
                let (first_row, end) = place_rows(cursor, vars, log2_k)?;
                cursor = end;
                let nonzero = nonzero_rows(value, &attention, row_bits)
                    .into_iter()
                    .map(|run| run.start + first_row..run.end + first_row)
                    .collect();
                tensors.push(Placement {
                    first_row,
                    row_bits,
                    vars,
                    log2_k,
                    base: 0,
                    base_bits: 0,
                    geometry,
                    nonzero,
                });
            }
            placed.push(TilePlacement {
                layer: tile.layer,
                segment,
                first_row: tensors[first].first_row,
                tensors: first..tensors.len(),
            });
        }
        // Rule 3: ν = log2 k + ⌈log2 R⌉ (RFC-0047 '`ν`'), then each tensor's base over ν − n_X
        // bits. R ≤ 2^32, so a row index has at most 32 bits.
        let allocated = cursor;
        let row_vars = match allocated {
            0 | 1 => 0,
            rows => u64::BITS - (rows - 1).leading_zeros(),
        };
        for tensor in &mut tensors {
            if tensor.vars >= log2_k {
                tensor.base = u128::from(tensor.first_row >> tensor.row_bits);
                tensor.base_bits = row_vars - tensor.row_bits;
            } else {
                tensor.base = u128::from(tensor.first_row) << (log2_k - tensor.vars);
                tensor.base_bits = row_vars + (log2_k - tensor.vars);
            }
        }
        let segments = segments(&params, &placed, &tensors, allocated);
        let rows = segments
            .iter()
            .flat_map(|s| &s.nonzero)
            .map(|run| run.end - run.start)
            .sum();
        Ok(Self {
            params,
            attention,
            row_vars,
            allocated,
            rows,
            tensors,
            tiles: placed,
            segments,
        })
    }

    /// The parameters it was derived under.
    pub fn params(&self) -> &LayoutParams {
        &self.params
    }

    /// The sequence's attention geometry.
    pub fn attention(&self) -> &Attention {
        &self.attention
    }

    /// `ν = log2 k + ⌈log2 R⌉`, the variables of `Ã` (RFC-0047 '`ν`'): `log2 k` of column and
    /// [`row_vars`](Self::row_vars) of row.
    pub fn variables(&self) -> u32 {
        self.params.log2_k + self.row_vars
    }

    /// `ν − log2 k = ⌈log2 R⌉`, the bits of a row index, with `R` the
    /// [allocated rows](Self::allocated_rows) (RFC-0047 '`ν`'): at most 32, the proximity point's
    /// coordinates (RFC-0011 step 3) and RFC-0014's `ϑ` for advice.
    pub fn row_vars(&self) -> u32 {
        self.row_vars
    }

    /// `R`, the rows the tensors and their alignment gaps take, the last tile's end included, at
    /// most `2^32`: rows `R` to `2^{ν − log2 k} − 1` are zero rows.
    pub fn allocated_rows(&self) -> u64 {
        self.allocated
    }

    /// The non-zero rows, every segment's: the advice descriptor's `rows`, which the
    /// `commitment_id` hashes (RFC-0031 'The descriptor').
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The tiles, in production order.
    pub fn tiles(&self) -> &[TilePlacement] {
        &self.tiles
    }

    /// The tensors, in production order: tile by tile, each tile's values in order.
    pub fn tensors(&self) -> &[Placement] {
        &self.tensors
    }

    /// Value `value` of tile `tile`, both counted from 0, or `None` when there is none.
    pub fn tensor(&self, tile: usize, value: usize) -> Option<&Placement> {
        let range = self.tiles.get(tile)?.tensors.clone();
        self.tensors.get(range)?.get(value)
    }

    /// The `S` segments, in order.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Whether `row` is a zero row (RFC-0011 rule 4): it holds no real entry, so it is neither
    /// encoded, nor hashed, nor opened.
    pub fn is_zero_row(&self, row: u64) -> bool {
        let segment = self
            .segments
            .iter()
            .find(|s| s.rows.contains(&row) && !s.nonzero.is_empty());
        let Some(segment) = segment else {
            return true;
        };
        let after = segment.nonzero.partition_point(|run| run.end <= row);
        segment
            .nonzero
            .get(after)
            .is_none_or(|run| !run.contains(&row))
    }

    /// The non-zero rows of each segment, as the commitment takes them ([`RowLayout`]): one index
    /// per non-zero row.
    ///
    /// # Errors
    ///
    /// Those of [`RowLayout::new`], which a derived layout never gives.
    pub fn row_layout(&self) -> Result<RowLayout, PcsError> {
        let segments = self
            .segments
            .iter()
            .map(|s| s.nonzero.iter().flat_map(Clone::clone).collect())
            .collect();
        RowLayout::new(self.row_vars, segments)
    }
}

/// Checks a value's form, returning the rule it breaks. A dense shape's order is
/// [`layout_order`]'s by construction.
fn check_value(value: &TensorShape, attention: &Attention) -> Result<(), &'static str> {
    match value {
        TensorShape::Dense(dense) => {
            let mut vars: u64 = 0;
            for axis in dense.axes() {
                if axis.len == 0 {
                    return Err(AXIS_EMPTY);
                }
                if axis.len > MAX_AXIS_LEN {
                    return Err(AXIS_BITS);
                }
                if axis.real > axis.len {
                    return Err(AXIS_LEN);
                }
                vars += u64::from(axis.bits());
            }
            if vars > MAX_TENSOR_VARS {
                return Err(TOO_LARGE);
            }
            if let Some(slots) = &dense.slots {
                let Some(&axis) = dense.axes().get(slots.axis) else {
                    return Err(SLOT_AXIS);
                };
                if !slots.slot_len.is_power_of_two()
                    || slots.lengths.iter().any(|&n| n == 0 || n > slots.slot_len)
                {
                    return Err(SLOT_LEN);
                }
                let expected = u64::try_from(slots.lengths.len())
                    .ok()
                    .and_then(|r| r.checked_mul(u64::from(slots.slot_len)));
                if expected != Some(axis.len) || axis.real != axis.len {
                    return Err(SLOT_AXIS);
                }
            }
            Ok(())
        }
        TensorShape::Scores { block } => match *block < attention.blocks {
            true => Ok(()),
            false => Err(BLOCK),
        },
    }
}

/// A checked value's variable order and `n_X`.
fn geometry(value: &TensorShape, attention: &Attention) -> (Geometry, u32) {
    match value {
        TensorShape::Dense(dense) => {
            // At most 64, checked.
            let vars = dense.axes().iter().map(Axis::bits).sum();
            let geometry = Geometry::Dense {
                axes: dense.axes().to_vec(),
                order: dense.order().to_vec(),
            };
            (geometry, vars)
        }
        TensorShape::Scores { .. } => {
            let key_vars = attention.padded.trailing_zeros();
            let query_vars = attention.queries.trailing_zeros();
            let head_vars = attention.heads.trailing_zeros();
            let geometry = Geometry::Scores {
                key_bits: attention.key_bits,
                key_vars,
                query_vars,
                head_vars,
            };
            (geometry, key_vars + query_vars + head_vars)
        }
    }
}

/// A checked value's rows that hold a real entry, relative to its first row, in runs.
fn nonzero_rows(value: &TensorShape, attention: &Attention, row_bits: u32) -> Vec<Range<u64>> {
    match value {
        TensorShape::Dense(dense) => dense_rows(dense, row_bits),
        &TensorShape::Scores { block } => score_rows(attention, block),
    }
}

/// A dense tensor's committed rows: all when its padding holds constants. RFC-0047 retains
/// inner padded axes' rows, even when every entry in such a row is zero.
///
/// A row fixes the tensor's `row_bits` most significant variables: whole axes, the most
/// significant first, and the high bits of the axis they end in. The free axes below have a real
/// position each, their position 0, unless one has none. So a row holds a real entry exactly when
/// each whole axis it fixes is below its length, and the axis it splits has a real position in
/// the part it fixes: its high bits `h` with `h·2^{low} < len`. Inner length padding counts
/// as held for this purpose (RFC-0047); a share's explicit real-position cutoff still applies.
fn dense_rows(dense: &DenseShape, row_bits: u32) -> Vec<Range<u64>> {
    if dense.padding() == Padding::Constants {
        return one_run(0..1 << row_bits);
    }
    if dense.axes().iter().any(|a| a.real == 0) {
        return Vec::new();
    }
    // RFC-0071 'The plan and the layout': when a multi-request batch's slotted token axis sits
    // outermost in RFC-0047's order (`order[0]`), rule 4 skips rows whose covered token positions
    // all lie in inter-slot or end-of-axis padding. When another padded axis (such as `d_ff`) sits
    // outermost instead, the token axis is inner and RFC-0047 commits its padding rows.
    let slotted_outer = dense
        .slots
        .as_ref()
        .filter(|slots| row_bits > 0 && dense.order().first() == Some(&slots.axis));
    // The fixed parts, the most significant first: how many of their values are real, and their
    // bits.
    let mut parts = Vec::new();
    let mut remaining = row_bits;
    let mut outer_runs: Option<(Vec<Range<u64>>, u32)> = None;
    let outermost_padded = dense
        .order()
        .iter()
        .copied()
        .find(|&axis| dense.axes()[axis].is_padded());
    for (pos, &index) in dense.order().iter().enumerate() {
        if remaining == 0 {
            break;
        }
        let axis = dense.axes()[index];
        let bits = axis.bits();
        let fixed = bits.min(remaining);
        if pos == 0
            && let Some(slots) = slotted_outer
        {
            let chunk = 1u64 << (bits - fixed);
            let slot = u64::from(slots.slot_len);
            let runs = if chunk >= slot {
                let slots_per_row = chunk / slot;
                one_run(0..(slots.lengths.len() as u64).div_ceil(slots_per_row))
            } else {
                let per_slot = slot / chunk;
                let mut runs = Vec::new();
                for (r, &n_r) in slots.lengths.iter().enumerate() {
                    let base = (r as u64) * per_slot;
                    let held = u64::from(n_r).div_ceil(chunk);
                    if held > 0 {
                        push_run(&mut runs, base..base + held);
                    }
                }
                runs
            };
            outer_runs = Some((runs, fixed));
            remaining -= fixed;
            continue;
        }
        let real = if axis.is_padded() && Some(index) != outermost_padded {
            // RFC-0047: only the outermost length padding gives skipped rows.
            1 << bits
        } else {
            axis.real
        };
        parts.push((real.div_ceil(1 << (bits - fixed)), fixed));
        remaining -= fixed;
    }
    // From the least significant part up: the runs within `width` rows, repeated for each real
    // value of the part above.
    let mut runs = one_run(0..1);
    let mut width = 1u64;
    for &(real, bits) in parts.iter().rev() {
        runs = match runs.as_slice() {
            [run] if *run == (0..width) => one_run(0..real * width),
            _ => {
                let mut repeated: Vec<Range<u64>> = Vec::with_capacity(runs.len());
                for value in 0..real {
                    for run in &runs {
                        push_run(
                            &mut repeated,
                            run.start + value * width..run.end + value * width,
                        );
                    }
                }
                repeated
            }
        };
        width <<= bits;
    }
    if let Some((outer, _fixed)) = outer_runs {
        runs = match runs.as_slice() {
            [run] if *run == (0..width) => {
                let mut scaled = Vec::with_capacity(outer.len());
                for r in outer {
                    push_run(&mut scaled, r.start * width..r.end * width);
                }
                scaled
            }
            _ => {
                let mut repeated = Vec::new();
                for range in outer {
                    for value in range {
                        for run in &runs {
                            push_run(
                                &mut repeated,
                                run.start + value * width..run.end + value * width,
                            );
                        }
                    }
                }
                repeated
            }
        };
    }
    runs
}

/// A score tile's rows that hold a valid position (RFC-0034 'Zero rows', reading `Q'` for `Q` as
/// RFC-0046, RFC-0047 and RFC-0071 have it): row `i_hi` holds keys
/// `i_hi·2^w` to `(i_hi + 1)·2^w − 1` against the block's slot-local queries `q_0` to
/// `q_0 + Q' − 1`. It holds only masked positions when every key follows every query,
/// `i_hi·2^w > q_0 + Q' − 1`, or when its keys or the block's queries are all padding,
/// `i_hi·2^w ≥ n_r` or `q_0 ≥ n_r`.
/// The others are rows `0` to `min(⌈n_r / 2^w⌉, ⌊(q_0 + Q' − 1) / 2^w⌋ + 1) − 1`.
fn score_rows(attention: &Attention, block: u32) -> Vec<Range<u64>> {
    let queries = u64::from(attention.queries);
    let (n_r, q0) = match &attention.slot_lengths {
        None => (u64::from(attention.positions), u64::from(block) * queries),
        Some(lengths) => {
            let first = u64::from(block) * queries;
            let slot = u64::from(attention.padded);
            let r = first / slot;
            let Some(&len) = usize::try_from(r).ok().and_then(|r| lengths.get(r)) else {
                return Vec::new();
            };
            (u64::from(len), first % slot)
        }
    };
    if q0 >= n_r {
        return Vec::new();
    }
    let keys = 1u64 << attention.key_bits;
    let last_query = q0 + queries - 1;
    one_run(0..n_r.div_ceil(keys).min(last_query / keys + 1))
}

/// The runs of one run of rows.
fn one_run(rows: Range<u64>) -> Vec<Range<u64>> {
    core::iter::once(rows).collect()
}

/// Appends `run` to `runs`, merged with the last one when they touch.
fn push_run(runs: &mut Vec<Range<u64>>, run: Range<u64>) {
    match runs.last_mut() {
        Some(last) if last.end == run.start => last.end = run.end,
        _ => runs.push(run),
    }
}

/// The segments of the rows (RFC-0046 'Segments'): each from its first tile's first row to the
/// next segment's, or to the allocation's end; a segment without a tile holds no row.
fn segments(
    params: &LayoutParams,
    tiles: &[TilePlacement],
    tensors: &[Placement],
    allocated: u64,
) -> Vec<Segment> {
    let mut segments: Vec<Segment> = segment_layers(params.layers, params.segments)
        .into_iter()
        .map(|layers| Segment {
            layers,
            rows: 0..0,
            nonzero: Vec::new(),
        })
        .collect();
    let mut starts: Vec<Option<u64>> = vec![None; segments.len()];
    for tile in tiles {
        let start = &mut starts[tile.segment as usize];
        start.get_or_insert(tile.first_row);
        for tensor in &tensors[tile.tensors.clone()] {
            for run in &tensor.nonzero {
                push_run(&mut segments[tile.segment as usize].nonzero, run.clone());
            }
        }
    }
    let mut end = allocated;
    for (segment, start) in segments.iter_mut().zip(&starts).rev() {
        let start = start.unwrap_or(end);
        segment.rows = start..end;
        end = start;
    }
    segments
}

/// The layer ranges of `segments` contiguous segments over `layers` layers (RFC-0046 'Segments'):
/// segment `j` holds layers `j·⌊L/S⌋ + min(j, L mod S)` to `(j+1)·⌊L/S⌋ + min(j+1, L mod S) − 1`.
/// The first `L mod S` segments hold `⌈L/S⌉` layers and the others `⌊L/S⌋`, so a segment past
/// the layers, when `S > L`, holds none. No segment gives no range.
pub fn segment_layers(layers: u32, segments: u32) -> Vec<Range<u32>> {
    if segments == 0 {
        return Vec::new();
    }
    let (size, rest) = (layers / segments, layers % segments);
    // Each bound is at most L.
    let bound = |j: u32| j * size + j.min(rest);
    (0..segments).map(|j| bound(j)..bound(j + 1)).collect()
}

/// The segment that holds layer `layer < L`, for `S ≥ 1` segments: the inverse of
/// [`segment_layers`].
fn segment_of(layers: u32, segments: u32) -> impl Fn(u32) -> u32 {
    let (size, rest) = (u64::from(layers / segments), u64::from(layers % segments));
    move |layer| {
        let layer = u64::from(layer);
        // The first `rest` segments hold `size + 1` layers each.
        let wide = rest * (size + 1);
        let segment = match layer < wide {
            true => layer / (size + 1),
            // Past them, `size ≥ 1`: otherwise `wide = rest = L` and no layer is past them.
            false => rest + (layer - wide) / size,
        };
        // Below S.
        segment as u32
    }
}
