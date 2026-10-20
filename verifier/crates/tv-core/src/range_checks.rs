// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0007 steps 5.2/5.3, derived from the public rules of RFC-0006/0009.

use std::collections::HashSet;

use tv_field::Fp;
use tv_tables::{TableError, TableId, TableKind, TableSpec};

use crate::claims::GadgetPolicy;
use crate::lowering::{Graph, LoweredTable, Node, NodeId, Op};
use crate::ranges::{Interval, RangeError, analyze};
use crate::selection::{Choice, Pattern, Selection};
use crate::validation::{ClaimsError, GadgetId, StatementField};

fn gadget(id: GadgetId) -> ClaimsError {
    ClaimsError::StatementMismatch {
        field: StatementField::Gadget { gadget: id },
    }
}

fn table(id: TableId) -> ClaimsError {
    ClaimsError::StatementMismatch {
        field: StatementField::Table { table: id },
    }
}

fn table_error(id: GadgetId, error: TableError) -> ClaimsError {
    match error {
        TableError::Memory => ClaimsError::Memory,
        error if error.is_rejection() => gadget(id),
        _ => ClaimsError::Internal,
    }
}

fn interval(ranges: &[Interval], node: &Node, input: usize) -> Result<Interval, ClaimsError> {
    let id = node.inputs().get(input).ok_or(ClaimsError::Internal)?;
    ranges
        .get(usize::try_from(id.0).map_err(|_| ClaimsError::Internal)?)
        .copied()
        .ok_or(ClaimsError::Internal)
}

fn within(range: Interval, lo: i128, hi: i128) -> bool {
    range
        .bounds()
        .is_some_and(|(a, b)| lo <= a && a <= b && b <= hi)
}

fn centered(range: Interval) -> bool {
    // The open centered range of RFC-0006, not the signed 64-bit type's range.
    let limit = i128::from(Fp::MODULUS.checked_div(2).unwrap_or(0));
    range
        .bounds()
        .is_some_and(|(lo, hi)| lo > limit.saturating_neg() && hi < limit)
}

/// Small fixed keys avoid heap clones while interning definitions; shifts are at most four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Range(u8),
    Requant {
        bits: u8,
        shifts: [u8; 4],
        classes: u8,
        lo: i32,
        hi: i32,
    },
}

impl Key {
    fn requant(
        bits: u32,
        shifts: &[u8],
        drop_bits: u32,
        lo: i64,
        hi: i64,
        id: GadgetId,
    ) -> Result<Self, ClaimsError> {
        if shifts.is_empty() || shifts.len() > 4 {
            return Err(gadget(id));
        }
        let mut held = [0; 4];
        for (slot, &shift) in held.iter_mut().zip(shifts) {
            *slot = u8::try_from(
                u32::from(shift)
                    .checked_sub(drop_bits)
                    .ok_or_else(|| gadget(id))?,
            )
            .map_err(|_| gadget(id))?;
        }
        Ok(Self::Requant {
            bits: u8::try_from(bits).map_err(|_| gadget(id))?,
            shifts: held,
            classes: u8::try_from(shifts.len()).map_err(|_| gadget(id))?,
            lo: i32::try_from(lo).map_err(|_| gadget(id))?,
            hi: i32::try_from(hi).map_err(|_| gadget(id))?,
        })
    }

    fn matches(self, kind: &TableKind) -> bool {
        match (self, kind) {
            (Self::Range(bits), TableKind::Range { bits: b }) => bits == *b,
            (
                Self::Requant {
                    bits,
                    shifts,
                    classes,
                    lo,
                    hi,
                },
                TableKind::Requant {
                    b,
                    r_by_class,
                    lo: low,
                    hi: high,
                },
            ) => {
                bits == *b
                    && lo == *low
                    && hi == *high
                    && shifts.get(..usize::from(classes)) == Some(r_by_class.as_slice())
            }
            _ => false,
        }
    }

    fn kind(self) -> Result<TableKind, ClaimsError> {
        let kind = match self {
            Self::Range(bits) => TableKind::Range { bits },
            Self::Requant {
                bits,
                shifts,
                classes,
                lo,
                hi,
            } => {
                let slice = shifts
                    .get(..usize::from(classes))
                    .ok_or(ClaimsError::Internal)?;
                let mut r_by_class = Vec::new();
                r_by_class
                    .try_reserve_exact(slice.len())
                    .map_err(|_| ClaimsError::Memory)?;
                r_by_class.extend_from_slice(slice);
                TableKind::Requant {
                    b: bits,
                    r_by_class,
                    lo,
                    hi,
                }
            }
        };
        Ok(kind)
    }

    fn log2_rows(self, id: GadgetId) -> Result<u8, ClaimsError> {
        // Check the full RFC-0008 bounds without regenerating any table values.
        self.kind()?
            .check_parameters()
            .map_err(|error| table_error(id, error))
    }
}

struct Tables<'a> {
    base: &'a [LoweredTable],
    added: Vec<(Key, u8)>,
    seen: HashSet<Key>,
    capture: bool,
    trace: Vec<Vec<(TableId, Key, u8)>>,
}

impl Tables<'_> {
    fn intern(&mut self, key: Key, max_bits: u8, id: GadgetId) -> Result<(), ClaimsError> {
        let rows = key.log2_rows(id)?;
        if rows > max_bits {
            return Err(gadget(id));
        }
        if self.seen.contains(&key) || self.base.iter().any(|t| key.matches(&t.kind)) {
            if self.capture {
                let index = if let Some(base) = self.base.iter().find(|t| key.matches(&t.kind)) {
                    base.id
                } else {
                    let at = self
                        .added
                        .iter()
                        .position(|(k, _)| *k == key)
                        .ok_or(ClaimsError::Internal)?;
                    TableId(
                        u32::try_from(self.base.len().checked_add(at).ok_or(ClaimsError::Memory)?)
                            .map_err(|_| ClaimsError::Memory)?,
                    )
                };
                self.record(id, index, key, rows)?;
            }
            return Ok(());
        }
        let index = self
            .base
            .len()
            .checked_add(self.added.len())
            .ok_or(ClaimsError::Memory)?;
        let _ = u32::try_from(index).map_err(|_| ClaimsError::Memory)?;
        self.seen.try_reserve(1).map_err(|_| ClaimsError::Memory)?;
        self.added.try_reserve(1).map_err(|_| ClaimsError::Memory)?;
        self.seen.insert(key);
        self.added.push((key, rows));
        if self.capture {
            self.record(
                id,
                TableId(u32::try_from(index).map_err(|_| ClaimsError::Memory)?),
                key,
                rows,
            )?;
        }
        Ok(())
    }

    fn record(
        &mut self,
        id: GadgetId,
        table: TableId,
        key: Key,
        rows: u8,
    ) -> Result<(), ClaimsError> {
        let entries = self
            .trace
            .get_mut(usize::try_from(id.0).map_err(|_| ClaimsError::Memory)?)
            .ok_or(ClaimsError::Internal)?;
        entries.try_reserve(1).map_err(|_| ClaimsError::Memory)?;
        entries.push((table, key, rows));
        Ok(())
    }

    fn compare(&self, claimed: &[TableSpec]) -> Result<(), ClaimsError> {
        let count = self
            .base
            .len()
            .checked_add(self.added.len())
            .ok_or(ClaimsError::Memory)?;
        let mut first: Option<TableId> = None;
        let mut note = |id: TableId| first = Some(first.map_or(id, |old| old.min(id)));
        // The schema requires strictly increasing ids, not merely a matching set.
        for pair in claimed.windows(2) {
            if let [left, right] = pair
                && left.id >= right.id
            {
                note(right.id);
            }
        }
        for index in 0..count.max(claimed.len()) {
            let id = TableId(u32::try_from(index).map_err(|_| ClaimsError::Memory)?);
            let Some(spec) = claimed.get(index) else {
                note(id);
                continue;
            };
            if index >= count {
                note(spec.id);
                continue;
            }
            if spec.id != id {
                note(spec.id.min(id));
                continue;
            }
            let same = if let Some(base) = self.base.get(index) {
                base.id == id && base.kind == spec.kind && base.log2_rows == spec.log2_rows
            } else {
                let added = index
                    .checked_sub(self.base.len())
                    .ok_or(ClaimsError::Internal)?;
                self.added
                    .get(added)
                    .is_some_and(|&(key, rows)| key.matches(&spec.kind) && rows == spec.log2_rows)
            };
            if !same {
                note(id);
            }
        }
        match first {
            Some(id) => Err(table(id)),
            None => Ok(()),
        }
    }
}

/// The narrowest signed box, searched rather than inferred from absolute values at negative
/// powers of two: [-2^(b-1), 2^(b-1)-1]. Only widths at most 62 are integer gadget inputs.
pub(crate) fn signed_width(range: Interval) -> Option<u32> {
    let (lo, hi) = range.bounds()?;
    (1u32..=62).find(|&b| {
        let half = 1i128.checked_shl(b.saturating_sub(1)).unwrap_or(0);
        lo >= half.saturating_neg() && hi < half
    })
}

fn ceil_log2(value: u64) -> u32 {
    u64::BITS.saturating_sub(value.saturating_sub(1).leading_zeros())
}

fn select_requant(
    range: Interval,
    shifts: &[u8],
    lo: i64,
    hi: i64,
    max_bits: u8,
    id: GadgetId,
    tables: &mut Tables<'_>,
) -> Result<(), ClaimsError> {
    let bits = signed_width(range).ok_or_else(|| gadget(id))?;
    let classes = u64::try_from(shifts.len()).map_err(|_| gadget(id))?;
    if !(1..=4).contains(&classes) || shifts.iter().any(|r| !(1..=63).contains(r)) || lo > hi {
        return Err(gadget(id));
    }
    let class_bits = ceil_log2(classes);
    let combined = bits.checked_add(class_bits).ok_or(ClaimsError::Internal)?;
    let max = u32::from(max_bits);
    if combined <= max {
        return tables.intern(Key::requant(bits, shifts, 0, lo, hi, id)?, max_bits, id);
    }
    let least = u32::from(*shifts.iter().min().ok_or(ClaimsError::Internal)?);
    let greatest = u32::from(*shifts.iter().max().ok_or(ClaimsError::Internal)?);
    let low = combined.saturating_sub(max);
    let high = least.saturating_sub(1).min(max);
    let packed = 1u128
        .checked_shl(bits)
        .and_then(|power| power.checked_mul(u128::from(classes)));
    if low <= high && packed.is_some_and(|bound| bound < u128::from(Fp::MODULUS)) {
        let split = combined.div_ceil(2).clamp(low, high);
        tables.intern(
            Key::Range(u8::try_from(split).map_err(|_| gadget(id))?),
            max_bits,
            id,
        )?;
        return tables.intern(
            Key::requant(
                bits.checked_sub(split).ok_or(ClaimsError::Internal)?,
                shifts,
                split,
                lo,
                hi,
                id,
            )?,
            max_bits,
            id,
        );
    }
    // RFC-0024: retain low digits, then saturate the top part before the final lookup.
    let split = least.saturating_sub(1).min(max);
    let magnitude = lo.unsigned_abs().max(hi.unsigned_abs()).max(1);
    let middle = greatest
        .checked_sub(split)
        .and_then(|r| r.checked_add(ceil_log2(magnitude)))
        .ok_or_else(|| gadget(id))?;
    let top = bits
        .checked_sub(split)
        .and_then(|b| b.checked_sub(middle))
        .filter(|&k| k > 0)
        .ok_or_else(|| gadget(id))?;
    let reduced = middle.checked_add(2).ok_or_else(|| gadget(id))?;
    if reduced.checked_add(class_bits).is_none_or(|b| b > max)
        || top.checked_add(1).is_none_or(|b| b > max)
    {
        return Err(gadget(id));
    }
    if split > 0 {
        tables.intern(
            Key::Range(u8::try_from(split).map_err(|_| gadget(id))?),
            max_bits,
            id,
        )?;
    }
    tables.intern(
        Key::Range(u8::try_from(middle).map_err(|_| gadget(id))?),
        max_bits,
        id,
    )?;
    tables.intern(
        Key::Range(u8::try_from(top).map_err(|_| gadget(id))?),
        max_bits,
        id,
    )?;
    tables.intern(
        Key::requant(
            top.checked_add(1).ok_or(ClaimsError::Internal)?,
            &[1],
            0,
            -2,
            1,
            id,
        )?,
        max_bits,
        id,
    )?;
    tables.intern(
        Key::requant(reduced, shifts, split, lo, hi, id)?,
        max_bits,
        id,
    )
}

/// RFC-0009 relation 3, after RFC-0006's public table policy has been checked.
fn softmax_ranges(
    sigma: u8,
    policy: GadgetPolicy,
    id: GadgetId,
    tables: &mut Tables<'_>,
) -> Result<(), ClaimsError> {
    if policy.limb_bits == 0 || policy.limb_bits > policy.max_log2_rows {
        return Err(gadget(id));
    }
    let reciprocal = sigma.checked_add(1).ok_or_else(|| gadget(id))?;
    tables.intern(Key::Range(reciprocal), policy.max_log2_rows, id)?;
    // Walk the public bit budget from low limbs to the top, without a division or an index.
    for _ in 0..2 {
        let mut remaining = sigma.checked_add(18).ok_or_else(|| gadget(id))?;
        while remaining > policy.limb_bits {
            tables.intern(Key::Range(policy.limb_bits), policy.max_log2_rows, id)?;
            remaining = remaining
                .checked_sub(policy.limb_bits)
                .ok_or(ClaimsError::Internal)?;
        }
        tables.intern(Key::Range(remaining), policy.max_log2_rows, id)?;
    }
    Ok(())
}

// RFC-0050: derive raw coefficient bounds directly from digit triples; constants
// are already bytes and have one-bit normalization carry lookups.
fn root_carry_bits(
    width: u8,
    increment: bool,
    limbs: usize,
    id: GadgetId,
) -> Result<[u8; 15], ClaimsError> {
    let coefficients = root_coefficient_bounds(width, increment, limbs, id)?;
    let raw = coefficients.get(..limbs).ok_or(ClaimsError::Internal)?;
    let mut widths = [0u8; 15];
    let mut carry = 0u64;
    let mut previous_box = 0u64;
    for (&a, width) in raw.iter().zip(&mut widths) {
        carry = a.checked_add(carry).ok_or_else(|| gadget(id))? >> 8;
        let bits = u64::BITS.saturating_sub(carry.leading_zeros()).max(1);
        let full_box = 1u64
            .checked_shl(bits)
            .and_then(|x| x.checked_sub(1))
            .ok_or_else(|| gadget(id))?;
        let left = a.checked_add(previous_box).ok_or_else(|| gadget(id))?;
        let right = full_box
            .checked_mul(256)
            .and_then(|x| x.checked_add(255))
            .ok_or_else(|| gadget(id))?;
        if left >= 1 << 40 || right >= 1 << 40 {
            return Err(gadget(id));
        }
        *width = u8::try_from(bits).map_err(|_| gadget(id))?;
        previous_box = full_box;
    }
    if carry != 0 {
        return Err(gadget(id));
    }
    Ok(widths)
}

pub(crate) fn root_coefficient_bounds(
    width: u8,
    increment: bool,
    limbs: usize,
    id: GadgetId,
) -> Result<[u64; 15], ClaimsError> {
    if !(1..=32).contains(&width)
        || limbs
            != usize::from(width)
                .saturating_mul(2)
                .saturating_add(55)
                .div_ceil(8)
    {
        return Err(ClaimsError::Internal);
    }
    let count = usize::from(width).div_ceil(8);
    let mut root = [0u64; 4];
    for (i, digit) in root.iter_mut().take(count).enumerate() {
        let remaining = u32::from(width)
            .checked_sub(
                u32::try_from(i.checked_mul(8).ok_or(ClaimsError::Internal)?)
                    .map_err(|_| ClaimsError::Internal)?,
            )
            .ok_or(ClaimsError::Internal)?;
        *digit = (1u64 << remaining.min(8)).saturating_sub(1);
        if i == 0 && increment {
            *digit = digit.checked_add(1).ok_or_else(|| gadget(id))?;
        }
    }
    let mut coefficients = [0u64; 15];
    let raw = coefficients.get_mut(..limbs).ok_or(ClaimsError::Internal)?;
    for (i, &a) in root.iter().take(count).enumerate() {
        for (j, &b) in root.iter().take(count).enumerate() {
            for (k, u) in [65535u64, 65535, 65535, 127].into_iter().enumerate() {
                let at = i
                    .checked_add(j)
                    .and_then(|x| k.checked_mul(2).and_then(|y| x.checked_add(y)))
                    .ok_or(ClaimsError::Internal)?;
                let slot = raw.get_mut(at).ok_or(ClaimsError::Internal)?;
                *slot = a
                    .checked_mul(b)
                    .and_then(|x| x.checked_mul(u))
                    .and_then(|x| slot.checked_add(x))
                    .ok_or_else(|| gadget(id))?;
            }
        }
    }
    Ok(coefficients)
}

fn rsqrt_ranges(
    width: u8,
    policy: GadgetPolicy,
    id: GadgetId,
    tables: &mut Tables<'_>,
) -> Result<(), ClaimsError> {
    if !(1..=32).contains(&width) {
        return Err(gadget(id));
    }
    let low = width.saturating_sub(1) / 8;
    let limbs = usize::from(width)
        .checked_mul(2)
        .and_then(|x| x.checked_add(55))
        .ok_or(ClaimsError::Internal)?
        .div_ceil(8);
    let lower = root_carry_bits(width, false, limbs, id)?;
    let upper = root_carry_bits(width, true, limbs, id)?;
    let constants = [1u8; 15];
    tables.intern(Key::Range(width), policy.max_log2_rows, id)?;
    for _ in 0..3 {
        tables.intern(Key::Range(16), policy.max_log2_rows, id)?;
    }
    tables.intern(Key::Range(7), policy.max_log2_rows, id)?;
    for _ in 0..low {
        tables.intern(Key::Range(8), policy.max_log2_rows, id)?;
    }
    let top = low
        .checked_mul(8)
        .and_then(|x| width.checked_sub(x))
        .ok_or(ClaimsError::Internal)?;
    tables.intern(Key::Range(top), policy.max_log2_rows, id)?;
    for (left, right) in [(&lower, &constants), (&constants, &upper)] {
        for carry_widths in [left, right] {
            for _ in 0..limbs {
                tables.intern(Key::Range(8), policy.max_log2_rows, id)?;
            }
            for &bits in carry_widths.iter().take(limbs) {
                tables.intern(Key::Range(bits), policy.max_log2_rows, id)?;
            }
        }
        for _ in 0..limbs {
            tables.intern(Key::Range(8), policy.max_log2_rows, id)?;
        }
        for _ in 0..limbs {
            tables.intern(Key::Range(1), policy.max_log2_rows, id)?;
        }
    }
    Ok(())
}

fn classes_fit(ranges: &[Interval], node: &Node, count: usize) -> Result<bool, ClaimsError> {
    match node.inputs().get(1) {
        Some(_) => Ok(within(
            interval(ranges, node, 1)?,
            0,
            i128::try_from(count)
                .map_err(|_| ClaimsError::Internal)?
                .saturating_sub(1),
        )),
        None => Ok(count == 1),
    }
}

/// Both phases finish before layout validation. Digests were checked at step 4, so equality of
/// id, definition and row count now pins them to the independently derived canonical tables.
pub(crate) fn check(
    graph: &Graph,
    policy: GadgetPolicy,
    claimed: &[TableSpec],
) -> Result<Selection, ClaimsError> {
    let (ranges, tables) = walk(graph, policy, true)?;
    tables.compare(claimed)?;
    finish_catalog(graph, ranges, tables)
}

fn walk(
    graph: &Graph,
    policy: GadgetPolicy,
    capture: bool,
) -> Result<(Vec<Interval>, Tables<'_>), ClaimsError> {
    let ranges = analyze(graph).map_err(|e| match e {
        RangeError::Memory => ClaimsError::Memory,
        RangeError::Internal => ClaimsError::Internal,
    })?;
    let mut tables = Tables {
        base: graph.tables(),
        added: Vec::new(),
        seen: HashSet::new(),
        capture,
        trace: Vec::new(),
    };
    let mut next = 0u32;
    for node in graph.nodes() {
        if !is_gadget(node.op()) {
            continue;
        }
        let id = GadgetId(next);
        next = next.checked_add(1).ok_or(ClaimsError::Internal)?;
        if capture {
            tables
                .trace
                .try_reserve(1)
                .map_err(|_| ClaimsError::Memory)?;
            tables.trace.push(Vec::new());
        }
        check_gadget(graph, node, &ranges, policy, id, &mut tables)?;
    }
    Ok((ranges, tables))
}

fn check_gadget(
    graph: &Graph,
    node: &Node,
    ranges: &[Interval],
    policy: GadgetPolicy,
    id: GadgetId,
    tables: &mut Tables<'_>,
) -> Result<(), ClaimsError> {
    let input = interval(ranges, node, 0)?;
    if !centered(input) {
        return Err(gadget(id));
    }
    match *node.op() {
        Op::OneHot { vocab } => {
            if !within(input, 0, i128::from(vocab).saturating_sub(1)) {
                return Err(gadget(id));
            }
        }
        Op::Requant { shift, lo, hi } => {
            let shifts = graph
                .shift_classes()
                .get(usize::try_from(shift).map_err(|_| ClaimsError::Internal)?)
                .ok_or(ClaimsError::Internal)?
                .as_slice();
            if !classes_fit(ranges, node, shifts.len())? {
                return Err(gadget(id));
            }
            select_requant(input, shifts, lo, hi, policy.max_log2_rows, id, tables)?;
        }
        Op::TableMap { table: tid } => {
            let base = graph.table(tid).ok_or(ClaimsError::Internal)?;
            let TableKind::Silu {
                b, ref r_by_class, ..
            } = base.kind
            else {
                return Err(gadget(id));
            };
            let half = 1i128
                .checked_shl(u32::from(b).saturating_sub(1))
                .ok_or_else(|| gadget(id))?;
            if b == 0
                || !within(input, half.saturating_neg(), half.saturating_sub(1))
                || base.log2_rows > policy.max_log2_rows
                || !classes_fit(ranges, node, r_by_class.len())?
            {
                return Err(gadget(id));
            }
        }
        Op::SoftmaxRow { exp_table, sigma } => {
            let base = graph.table(exp_table).ok_or(ClaimsError::Internal)?;
            let TableKind::Exp {
                key_bits,
                sigma: table_sigma,
                scale,
            } = base.kind
            else {
                return Err(gadget(id));
            };
            let (lo, hi) = input.bounds().ok_or_else(|| gadget(id))?;
            let domain = 1i128
                .checked_shl(u32::from(key_bits))
                .ok_or_else(|| gadget(id))?;
            let last = u64::try_from(domain.checked_sub(1).ok_or(ClaimsError::Internal)?)
                .map_err(|_| ClaimsError::Internal)?;
            let masked_is_zero = tv_tables::exp_maps_to_zero(last, sigma, scale)
                .map_err(|error| table_error(id, error))?;
            if !within(input, -2_097_152, 2_097_152)
                || hi.checked_sub(lo).is_none_or(|spread| spread >= domain)
                || table_sigma != sigma
                || u32::from(sigma).saturating_add(1) > u32::from(policy.max_log2_rows)
                || policy.limb_bits == 0
                || policy.limb_bits > policy.max_log2_rows
                || base.log2_rows > policy.max_log2_rows
                || !masked_is_zero
            {
                return Err(gadget(id));
            }
            softmax_ranges(sigma, policy, id, tables)?;
        }
        Op::RsqrtRow { eps_int, .. } => {
            let (lo, hi) = input.bounds().ok_or_else(|| gadget(id))?;
            let output = node
                .outputs()
                .first()
                .and_then(|id| usize::try_from(id.0).ok())
                .and_then(|i| ranges.get(i))
                .and_then(Interval::bounds)
                .ok_or(ClaimsError::Internal)?;
            let width = u128::BITS.saturating_sub(
                u128::try_from(output.1)
                    .map_err(|_| ClaimsError::Internal)?
                    .leading_zeros(),
            );
            if lo < 0
                || eps_int == 0
                || hi
                    .checked_add(i128::from(eps_int))
                    .is_none_or(|u| u >= 36_028_797_018_963_968)
                || width > 32
                || width > u32::from(policy.max_log2_rows)
                || policy.limb_bits == 0
                || policy.limb_bits > policy.max_log2_rows
            {
                return Err(gadget(id));
            }
            rsqrt_ranges(
                u8::try_from(width).map_err(|_| gadget(id))?,
                policy,
                id,
                tables,
            )?;
        }
        _ => return Err(ClaimsError::Internal),
    }
    Ok(())
}

fn is_gadget(op: &Op) -> bool {
    matches!(
        op,
        Op::OneHot { .. }
            | Op::Requant { .. }
            | Op::TableMap { .. }
            | Op::SoftmaxRow { .. }
            | Op::RsqrtRow { .. }
    )
}

fn copy_shifts(shifts: &[u8]) -> Result<Vec<u8>, ClaimsError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(shifts.len())
        .map_err(|_| ClaimsError::Memory)?;
    values.extend_from_slice(shifts);
    Ok(values)
}

fn copy_kind(kind: &TableKind) -> Result<TableKind, ClaimsError> {
    Ok(match kind {
        TableKind::Range { bits } => TableKind::Range { bits: *bits },
        TableKind::Requant {
            b,
            r_by_class,
            lo,
            hi,
        } => TableKind::Requant {
            b: *b,
            r_by_class: copy_shifts(r_by_class)?,
            lo: *lo,
            hi: *hi,
        },
        TableKind::Pow2 { max_exp } => TableKind::Pow2 { max_exp: *max_exp },
        TableKind::Bitwise8 { op } => TableKind::Bitwise8 { op: *op },
        TableKind::AesSbox => TableKind::AesSbox,
        TableKind::Gf256Mul { c } => TableKind::Gf256Mul { c: *c },
        TableKind::Exp {
            key_bits,
            sigma,
            scale,
        } => TableKind::Exp {
            key_bits: *key_bits,
            sigma: *sigma,
            scale: *scale,
        },
        TableKind::Exp2Frac {
            frac_bits,
            out_bits,
        } => TableKind::Exp2Frac {
            frac_bits: *frac_bits,
            out_bits: *out_bits,
        },
        TableKind::Silu {
            b,
            r_by_class,
            in_exp,
            out_exp,
        } => TableKind::Silu {
            b: *b,
            r_by_class: copy_shifts(r_by_class)?,
            in_exp: *in_exp,
            out_exp: *out_exp,
        },
        TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        } => TableKind::Rope {
            theta: *theta,
            factor: *factor,
            low_freq_factor: *low_freq_factor,
            high_freq_factor: *high_freq_factor,
            original_max: *original_max,
            bits: *bits,
            max_context: *max_context,
            d_head: *d_head,
        },
    })
}

fn pattern(op: &Op, requests: &[(TableId, Key, u8)]) -> Result<Pattern, ClaimsError> {
    Ok(match op {
        Op::OneHot { .. } => Pattern::OneHot,
        Op::RsqrtRow { .. } => Pattern::RsqrtRow,
        Op::TableMap { .. } => Pattern::Lookup,
        Op::SoftmaxRow { .. } => Pattern::SoftmaxRow,
        Op::Requant { .. } => match requests {
            [(_, Key::Requant { .. }, _)] => Pattern::Requant1,
            [(_, Key::Range(split), _), (_, Key::Requant { .. }, _)] => {
                Pattern::RequantSplit { split: *split }
            }
            [
                (_, Key::Range(middle), _),
                (_, Key::Range(_), _),
                (_, Key::Requant { .. }, _),
                (_, Key::Requant { .. }, _),
            ] => Pattern::RequantWide {
                split: 0,
                middle: *middle,
            },
            [
                (_, Key::Range(split), _),
                (_, Key::Range(middle), _),
                (_, Key::Range(_), _),
                (_, Key::Requant { .. }, _),
                (_, Key::Requant { .. }, _),
            ] => Pattern::RequantWide {
                split: *split,
                middle: *middle,
            },
            _ => return Err(ClaimsError::Internal),
        },
        _ => return Err(ClaimsError::Internal),
    })
}

pub(crate) fn catalog(graph: &Graph, policy: GadgetPolicy) -> Result<Selection, ClaimsError> {
    let (ranges, state) = walk(graph, policy, true)?;
    finish_catalog(graph, ranges, state)
}

fn finish_catalog(
    graph: &Graph,
    ranges: Vec<Interval>,
    state: Tables<'_>,
) -> Result<Selection, ClaimsError> {
    let count = state
        .base
        .len()
        .checked_add(state.added.len())
        .ok_or(ClaimsError::Memory)?;
    let mut tables = Vec::new();
    tables
        .try_reserve_exact(count)
        .map_err(|_| ClaimsError::Memory)?;
    for t in state.base {
        tables.push(LoweredTable {
            id: t.id,
            kind: copy_kind(&t.kind)?,
            log2_rows: t.log2_rows,
        });
    }
    for &(key, rows) in &state.added {
        tables.push(LoweredTable {
            id: TableId(u32::try_from(tables.len()).map_err(|_| ClaimsError::Memory)?),
            kind: key.kind()?,
            log2_rows: rows,
        });
    }
    let mut choices = Vec::new();
    choices
        .try_reserve_exact(state.trace.len())
        .map_err(|_| ClaimsError::Memory)?;
    let mut request_sets = state.trace.iter();
    for (index, node) in graph.nodes().iter().enumerate() {
        if !is_gadget(node.op()) {
            continue;
        }
        let requests = request_sets.next().ok_or(ClaimsError::Internal)?;
        let prefix = match node.op() {
            Op::TableMap { table } => Some(*table),
            Op::SoftmaxRow { exp_table, .. } => Some(*exp_table),
            _ => None,
        };
        let capacity = requests
            .len()
            .checked_add(usize::from(prefix.is_some()))
            .ok_or(ClaimsError::Memory)?;
        let mut refs = Vec::new();
        refs.try_reserve_exact(capacity)
            .map_err(|_| ClaimsError::Memory)?;
        if let Some(id) = prefix {
            refs.push(id);
        }
        refs.extend(requests.iter().map(|(id, _, _)| *id));
        choices.push(Choice {
            node: NodeId(u32::try_from(index).map_err(|_| ClaimsError::Memory)?),
            pattern: pattern(node.op(), requests)?,
            tables: refs,
        });
    }
    if request_sets.next().is_some() {
        return Err(ClaimsError::Internal);
    }
    Ok(Selection {
        ranges,
        tables,
        choices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_families_match_every_width_of_the_independent_oracle() {
        let vectors = tv_testkit::read_vectors("advice/layout.json");
        let scalar =
            std::fs::read_to_string(tv_testkit::vectors_dir().join("rsqrt/canonical.txt")).unwrap();
        let scalar: Vec<Vec<u64>> = scalar
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                line.split_whitespace()
                    .map(|part| part.parse().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(scalar.len(), 32);
        let roots = vectors
            .get("roots")
            .and_then(tv_testkit::Json::as_array)
            .unwrap();
        assert_eq!(roots.len(), 32);
        let policy = GadgetPolicy {
            max_log2_rows: 32,
            query_block: 512,
            limb_bits: 16,
        };
        for root in roots {
            let width = root
                .get("width")
                .and_then(tv_testkit::Json::as_u64)
                .unwrap() as u8;
            let mut tables = Tables {
                base: &[],
                added: Vec::new(),
                seen: HashSet::new(),
                capture: true,
                trace: vec![Vec::new()],
            };
            rsqrt_ranges(width, policy, GadgetId(0), &mut tables).unwrap();
            let expected: Vec<_> = root
                .get("requests")
                .and_then(tv_testkit::Json::as_array)
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u8)
                .collect();
            let found: Vec<_> = tables.trace[0]
                .iter()
                .map(|(_, key, _)| match key {
                    Key::Range(bits) => *bits,
                    _ => panic!("root requests only ranges"),
                })
                .collect();
            assert_eq!(found, expected, "width {width}: all repeated requests");
            let schema = scalar
                .iter()
                .find(|case| case[0] == u64::from(width))
                .unwrap();
            assert_eq!(
                schema[4],
                4 + u64::from((width - 1) / 8) + 12 * (2 * u64::from(width) + 55).div_ceil(8),
                "RFC-0050 real columns"
            );
            assert_eq!(schema[5], found.len() as u64);
            assert_eq!(
                &schema[6..],
                found
                    .iter()
                    .map(|&bits| u64::from(bits))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                tables.added.len(),
                expected.iter().collect::<HashSet<_>>().len()
            );
        }
        for width in [0, 33] {
            let mut tables = Tables {
                base: &[],
                added: Vec::new(),
                seen: HashSet::new(),
                capture: false,
                trace: Vec::new(),
            };
            assert!(matches!(
                rsqrt_ranges(width, policy, GadgetId(2), &mut tables),
                Err(ClaimsError::StatementMismatch {
                    field: StatementField::Gadget {
                        gadget: GadgetId(2)
                    }
                })
            ));
        }
    }

    fn selected(bits: u32, shifts: &[u8]) -> Vec<(Key, u8)> {
        let half = 1i128 << (bits - 1);
        let mut tables = Tables {
            base: &[],
            added: Vec::new(),
            seen: HashSet::new(),
            capture: false,
            trace: Vec::new(),
        };
        select_requant(
            Interval::test_bounds(-half, half - 1),
            shifts,
            -128,
            127,
            32,
            GadgetId(7),
            &mut tables,
        )
        .unwrap();
        tables.added
    }

    #[test]
    fn softmax_ranges_follow_each_lookup_and_intern_equal_definitions_once() {
        for (sigma, limb, expected) in [
            (4, 4, vec![5, 4, 2]),
            (16, 16, vec![17, 16, 2]),
            (16, 17, vec![17]),
            (0, 32, vec![1, 18]),
            (31, 32, vec![32, 17]),
        ] {
            let mut tables = Tables {
                base: &[],
                added: Vec::new(),
                seen: HashSet::new(),
                capture: false,
                trace: Vec::new(),
            };
            let mut policy = GadgetPolicy {
                max_log2_rows: 32,
                query_block: 512,
                limb_bits: 16,
            };
            policy.limb_bits = limb;
            softmax_ranges(sigma, policy, GadgetId(7), &mut tables).unwrap();
            assert_eq!(
                tables.added,
                expected
                    .into_iter()
                    .map(|bits| (Key::Range(bits), bits))
                    .collect::<Vec<_>>()
            );
            let original = tables.added.clone();
            softmax_ranges(sigma, policy, GadgetId(8), &mut tables).unwrap();
            assert_eq!(tables.added, original);
        }
        let mut tables = Tables {
            base: &[],
            added: Vec::new(),
            seen: HashSet::new(),
            capture: false,
            trace: Vec::new(),
        };
        let mut policy = GadgetPolicy {
            max_log2_rows: 32,
            query_block: 512,
            limb_bits: 16,
        };
        for limb in [0, 33] {
            policy.limb_bits = limb;
            assert_eq!(
                softmax_ranges(16, policy, GadgetId(7), &mut tables),
                Err(gadget(GadgetId(7)))
            );
        }
        policy.limb_bits = 16;
        policy.max_log2_rows = 16;
        assert_eq!(
            softmax_ranges(16, policy, GadgetId(7), &mut tables),
            Err(gadget(GadgetId(7)))
        );
    }

    #[test]
    fn goldilocks_class_packing_chooses_wide_for_four_62_bit_classes() {
        let three = selected(62, &[33, 34, 36]);
        assert_eq!(three.len(), 2);
        assert_eq!(three[0], (Key::Range(32), 32));
        assert_eq!(
            three[1],
            (
                Key::requant(30, &[1, 2, 4], 0, -128, 127, GadgetId(7)).unwrap(),
                32
            )
        );
        let four = selected(62, &[33, 34, 35, 36]);
        assert_eq!(
            four,
            vec![
                (Key::Range(32), 32),
                (Key::Range(11), 11),
                (Key::Range(19), 19),
                (Key::requant(20, &[1], 0, -2, 1, GadgetId(7)).unwrap(), 20),
                (
                    Key::requant(13, &[1, 2, 3, 4], 0, -128, 127, GadgetId(7)).unwrap(),
                    15
                ),
            ]
        );
        // The forbidden split compares boxes of 4*2^62 > P, while three classes fit.
        assert!(3u128 * (1u128 << 62) < u128::from(Fp::MODULUS));
        assert!(4u128 * (1u128 << 62) > u128::from(Fp::MODULUS));
    }

    #[test]
    fn a_split_never_crosses_its_least_rounding_shift() {
        assert_eq!(
            selected(32, &[1]),
            vec![(
                Key::requant(32, &[1], 0, -128, 127, GadgetId(7)).unwrap(),
                32
            )]
        );
        let wide = selected(33, &[1]);
        assert_eq!(
            wide,
            vec![
                (Key::Range(8), 8),
                (Key::Range(25), 25),
                (Key::requant(26, &[1], 0, -2, 1, GadgetId(7)).unwrap(), 26),
                (
                    Key::requant(10, &[1], 0, -128, 127, GadgetId(7)).unwrap(),
                    10
                ),
            ]
        );
        let half = 1i128 << 61;
        let mut tables = Tables {
            base: &[],
            added: Vec::new(),
            seen: HashSet::new(),
            capture: false,
            trace: Vec::new(),
        };
        assert_eq!(
            select_requant(
                Interval::test_bounds(-half, half - 1),
                &[1],
                -128,
                127,
                32,
                GadgetId(7),
                &mut tables
            ),
            Err(gadget(GadgetId(7)))
        );
        assert!(tables.added.is_empty());
        // Class bits enter the balanced split; on a padded-row-cost tie the low part is wider.
        let split = selected(31, &[20, 21, 22]);
        assert_eq!(split[0], (Key::Range(17), 17));
        assert_eq!(
            split[1],
            (
                Key::requant(14, &[3, 4, 5], 0, -128, 127, GadgetId(7)).unwrap(),
                16
            )
        );
    }

    #[test]
    fn integer_widths_and_centered_boundaries_are_exact() {
        let limit = i128::from(Fp::MODULUS / 2);
        assert!(centered(Interval::test_bounds(-limit + 1, limit - 1)));
        assert!(!centered(Interval::test_bounds(-limit, 0)));
        assert!(!centered(Interval::test_bounds(0, limit)));
        for bits in 1..=62 {
            let half = 1i128 << (bits - 1);
            assert_eq!(
                signed_width(Interval::test_bounds(-half, half - 1)),
                Some(bits)
            );
            if bits < 62 {
                assert_eq!(
                    signed_width(Interval::test_bounds(-half - 1, 0)),
                    Some(bits + 1)
                );
                assert_eq!(signed_width(Interval::test_bounds(0, half)), Some(bits + 1));
            }
        }
        assert_eq!(
            signed_width(Interval::test_bounds(-(1i128 << 61) - 1, 0)),
            None
        );
    }

    #[test]
    fn table_errors_keep_their_fault_class() {
        let id = GadgetId(1);
        assert_eq!(table_error(id, TableError::Memory), ClaimsError::Memory);
        assert_eq!(table_error(id, TableError::Internal), ClaimsError::Internal);
        assert_eq!(table_error(id, TableError::KeyBits), gadget(id));
    }

    fn fixture() -> (Graph, GadgetPolicy) {
        let file = tv_testkit::read_vectors("claims/claim-lists.json");
        let case = file
            .get("cases")
            .unwrap()
            .as_array()
            .unwrap()
            .first()
            .unwrap();
        let bytes = tv_testkit::hex(case.get("hex").unwrap().as_str().unwrap());
        let claims =
            crate::claims::ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
        let public = crate::validation::check_public(&claims).unwrap();
        (
            crate::lowering::lower(&public, 13).unwrap(),
            claims.gadget_policy,
        )
    }

    fn backed(op: &Op) -> bool {
        matches!(
            op,
            Op::OneHot { .. }
                | Op::Requant { .. }
                | Op::RsqrtRow { .. }
                | Op::TableMap { .. }
                | Op::SoftmaxRow { .. }
        )
    }

    fn one(
        graph: &Graph,
        node: &Node,
        ranges: &[Interval],
        policy: GadgetPolicy,
        id: GadgetId,
    ) -> Result<(), ClaimsError> {
        let mut tables = Tables {
            base: graph.tables(),
            added: Vec::new(),
            seen: HashSet::new(),
            capture: false,
            trace: Vec::new(),
        };
        check_gadget(graph, node, ranges, policy, id, &mut tables)
    }

    #[test]
    fn each_integer_input_domain_is_checked_at_its_gadget() {
        let (graph, policy) = fixture();
        let original = analyze(&graph).unwrap();
        let mut tested = [false; 5];
        for (index, node) in graph.nodes().iter().filter(|n| backed(n.op())).enumerate() {
            let id = GadgetId(index as u32);
            assert_eq!(one(&graph, node, &original, policy, id), Ok(()));
            let mut bad = original.clone();
            let slot = node.inputs()[0].0 as usize;
            let which = match *node.op() {
                Op::OneHot { vocab } => {
                    bad[slot] = Interval::test_bounds(0, i128::from(vocab));
                    0
                }
                Op::Requant { .. } => {
                    bad[slot] = Interval::test_bounds(0, 1i128 << 61);
                    1
                }
                Op::TableMap { table } => {
                    let TableKind::Silu { b, .. } = graph.table(table).unwrap().kind else {
                        panic!("SiLU")
                    };
                    bad[slot] = Interval::test_bounds(0, 1i128 << (b - 1));
                    2
                }
                Op::SoftmaxRow { .. } => {
                    bad[slot] = Interval::test_bounds(2_097_153, 2_097_153);
                    3
                }
                Op::RsqrtRow { .. } => {
                    bad[slot] = Interval::test_bounds(-1, 0);
                    4
                }
                _ => unreachable!(),
            };
            assert_eq!(
                one(&graph, node, &bad, policy, id),
                Err(gadget(id)),
                "{:?}",
                node.op()
            );
            tested[which] = true;
        }
        assert!(tested.into_iter().all(|seen| seen));
    }

    #[test]
    fn hidden_classes_must_be_in_the_public_class_domain() {
        let (graph, policy) = fixture();
        let original = analyze(&graph).unwrap();
        let mut tested = 0;
        for (index, node) in graph.nodes().iter().filter(|n| backed(n.op())).enumerate() {
            let count = match node.op() {
                Op::Requant { shift, .. } => {
                    graph.shift_classes()[*shift as usize].as_slice().len()
                }
                Op::TableMap { table } => match &graph.table(*table).unwrap().kind {
                    TableKind::Silu { r_by_class, .. } => r_by_class.len(),
                    _ => unreachable!(),
                },
                _ => continue,
            };
            if node.inputs().len() != 2 {
                continue;
            }
            let id = GadgetId(index as u32);
            for bounds in [(-1, 0), (0, count as i128)] {
                let mut bad = original.clone();
                bad[node.inputs()[1].0 as usize] = Interval::test_bounds(bounds.0, bounds.1);
                assert_eq!(one(&graph, node, &bad, policy, id), Err(gadget(id)));
            }
            tested += 1;
        }
        assert!(tested > 0);
    }

    #[test]
    fn softmax_spread_and_norm_boxes_have_strict_upper_bounds() {
        let (graph, policy) = fixture();
        let original = analyze(&graph).unwrap();
        let mut softmax = false;
        let mut norm = false;
        for (index, node) in graph.nodes().iter().filter(|n| backed(n.op())).enumerate() {
            let id = GadgetId(index as u32);
            match *node.op() {
                Op::SoftmaxRow { exp_table, .. } => {
                    let TableKind::Exp { key_bits, .. } = graph.table(exp_table).unwrap().kind
                    else {
                        panic!("Exp")
                    };
                    let mut bad = original.clone();
                    bad[node.inputs()[0].0 as usize] = Interval::test_bounds(0, 1i128 << key_bits);
                    assert_eq!(one(&graph, node, &bad, policy, id), Err(gadget(id)));
                    softmax = true;
                }
                Op::RsqrtRow { eps_int, .. } => {
                    let mut bad = original.clone();
                    bad[node.inputs()[0].0 as usize] =
                        Interval::test_bounds(0, (1i128 << 55) - i128::from(eps_int));
                    assert_eq!(one(&graph, node, &bad, policy, id), Err(gadget(id)));
                    let mut bad = original.clone();
                    bad[node.outputs()[0].0 as usize] = Interval::test_bounds(0, 1i128 << 32);
                    assert_eq!(one(&graph, node, &bad, policy, id), Err(gadget(id)));
                    norm = true;
                }
                _ => {}
            }
        }
        assert!(softmax && norm);
    }

    #[test]
    fn every_lookup_width_respects_the_policy() {
        let (graph, policy) = fixture();
        let ranges = analyze(&graph).unwrap();
        let mut tested = [false; 3];
        for (index, node) in graph.nodes().iter().filter(|n| backed(n.op())).enumerate() {
            let id = GadgetId(index as u32);
            let (width, which) = match node.op() {
                Op::TableMap { table } => (u32::from(graph.table(*table).unwrap().log2_rows), 0),
                Op::SoftmaxRow { exp_table, .. } => {
                    (u32::from(graph.table(*exp_table).unwrap().log2_rows), 1)
                }
                Op::RsqrtRow { .. } => {
                    let last = ranges[node.outputs()[0].0 as usize].bounds().unwrap().1 as u128;
                    (128 - last.leading_zeros(), 2)
                }
                _ => continue,
            };
            let mut narrow = policy;
            narrow.max_log2_rows = (width - 1) as u8;
            assert!(narrow.max_log2_rows >= narrow.limb_bits);
            assert_eq!(one(&graph, node, &ranges, narrow, id), Err(gadget(id)));
            tested[which] = true;
        }
        assert!(tested.into_iter().all(|seen| seen));
        let mut tables = Tables {
            base: &[],
            added: Vec::new(),
            seen: HashSet::new(),
            capture: false,
            trace: Vec::new(),
        };
        assert_eq!(
            tables.intern(Key::Range(32), 31, GadgetId(7)),
            Err(gadget(GadgetId(7)))
        );
        assert!(tables.added.is_empty());
    }

    #[test]
    fn a_multiclass_lookup_requires_its_class_column() {
        let (graph, _) = fixture();
        let ranges = analyze(&graph).unwrap();
        let node = graph
            .nodes()
            .iter()
            .find(|n| matches!(n.op(), Op::Requant { .. }) && n.inputs().len() == 1)
            .unwrap();
        assert_eq!(classes_fit(&ranges, node, 1), Ok(true));
        assert_eq!(classes_fit(&ranges, node, 2), Ok(false));
    }
    #[test]
    fn captured_choices_retain_widths_and_reused_table_ids() {
        for (bits, shifts, expected) in [
            (16, vec![4], Pattern::Requant1),
            (40, vec![24], Pattern::RequantSplit { split: 20 }),
            (
                40,
                vec![8],
                Pattern::RequantWide {
                    split: 7,
                    middle: 8,
                },
            ),
        ] {
            let mut tables = Tables {
                base: &[],
                added: Vec::new(),
                seen: HashSet::new(),
                capture: true,
                trace: vec![Vec::new()],
            };
            let half = 1i128 << (bits - 1);
            select_requant(
                Interval::test_bounds(-half, half - 1),
                &shifts,
                -128,
                127,
                32,
                GadgetId(0),
                &mut tables,
            )
            .unwrap();
            let op = Op::Requant {
                shift: 0,
                lo: -128,
                hi: 127,
            };
            assert_eq!(pattern(&op, &tables.trace[0]).unwrap(), expected);
            let count = tables.added.len();
            tables.trace.push(Vec::new());
            select_requant(
                Interval::test_bounds(-half, half - 1),
                &shifts,
                -128,
                127,
                32,
                GadgetId(1),
                &mut tables,
            )
            .unwrap();
            assert_eq!(tables.added.len(), count);
            assert_eq!(tables.trace[0], tables.trace[1]);
        }
    }

    #[test]
    fn captured_softmax_retains_all_five_references_to_one_table() {
        let mut tables = Tables {
            base: &[],
            added: Vec::new(),
            seen: HashSet::new(),
            capture: true,
            trace: vec![Vec::new()],
        };
        let policy = GadgetPolicy {
            max_log2_rows: 32,
            query_block: 512,
            limb_bits: 17,
        };
        softmax_ranges(16, policy, GadgetId(0), &mut tables).unwrap();
        assert_eq!(tables.added.len(), 1);
        // Reciprocal plus two limbs of each RFC-0050 slack, all the same family.
        assert_eq!(tables.trace[0], vec![(TableId(0), Key::Range(17), 17); 5]);
    }
}
