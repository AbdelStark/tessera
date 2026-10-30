// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0076: public common-family selection, independently derived from the RFC.

use super::{ClaimsError, GadgetId, Graph, Interval, Key, NodeId, Op, ceil_log2, gadget};
use crate::sumcheck::Protocol;
use tv_field::Fp;

struct Input<'a> {
    bits: u32,
    shifts: &'a [u8],
    lo: i64,
    hi: i64,
    id: GadgetId,
}

fn empty<T>(capacity: usize) -> Result<Vec<T>, ClaimsError> {
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| ClaimsError::Memory)?;
    Ok(out)
}

fn feasible(keys: &[Key], maximum: u8, id: GadgetId) -> Result<bool, ClaimsError> {
    for &key in keys {
        match key.log2_rows(id) {
            Ok(width) if width <= maximum => {}
            Ok(_) | Err(ClaimsError::StatementMismatch { .. }) => return Ok(false),
            Err(error) => return Err(error),
        }
    }
    Ok(true)
}

fn split(input: &Input<'_>, maximum: u8) -> Result<Option<Vec<Key>>, ClaimsError> {
    let b = input.bits.max(2);
    let c = ceil_log2(u64::try_from(input.shifts.len()).map_err(|_| ClaimsError::Memory)?);
    let least = u32::from(*input.shifts.iter().min().ok_or(ClaimsError::Internal)?);
    let a = b
        .checked_add(c)
        .ok_or(ClaimsError::Internal)?
        .saturating_sub(u32::from(maximum))
        .max(1);
    let d = least
        .saturating_sub(1)
        .min(u32::from(maximum))
        .min(b.saturating_sub(1));
    let classes = u128::try_from(input.shifts.len()).map_err(|_| ClaimsError::Memory)?;
    let packed = 1u128
        .checked_shl(b)
        .and_then(|power| power.checked_mul(classes));
    if b > 62 || a > d || packed.is_none_or(|bound| bound >= u128::from(Fp::MODULUS)) {
        return Ok(None);
    }
    let s = b
        .checked_add(c)
        .ok_or(ClaimsError::Internal)?
        .div_ceil(2)
        .clamp(a, d);
    let mut keys = empty(2)?;
    keys.push(Key::Range(
        u8::try_from(s).map_err(|_| ClaimsError::Internal)?,
    ));
    keys.push(Key::requant(
        b.checked_sub(s).ok_or(ClaimsError::Internal)?,
        input.shifts,
        s,
        input.lo,
        input.hi,
        input.id,
    )?);
    Ok(feasible(&keys, maximum, input.id)?.then_some(keys))
}

fn wide(input: &Input<'_>, maximum: u8, zero_low: bool) -> Result<Option<Vec<Key>>, ClaimsError> {
    let least = u32::from(*input.shifts.iter().min().ok_or(ClaimsError::Internal)?);
    let greatest = u32::from(*input.shifts.iter().max().ok_or(ClaimsError::Internal)?);
    let s = if zero_low {
        0
    } else {
        least.saturating_sub(1).min(u32::from(maximum))
    };
    let magnitude = input.lo.unsigned_abs().max(input.hi.unsigned_abs()).max(1);
    let t = greatest
        .checked_sub(s)
        .and_then(|v| v.checked_add(ceil_log2(magnitude)))
        .ok_or(ClaimsError::Internal)?;
    let b = input.bits.max(
        s.checked_add(t)
            .and_then(|v| v.checked_add(1))
            .ok_or(ClaimsError::Internal)?,
    );
    let k = b
        .checked_sub(s)
        .and_then(|v| v.checked_sub(t))
        .ok_or(ClaimsError::Internal)?;
    let c = ceil_log2(u64::try_from(input.shifts.len()).map_err(|_| ClaimsError::Memory)?);
    if b > 62
        || s >= least
        || t == 0
        || k == 0
        || t.checked_add(2)
            .and_then(|v| v.checked_add(c))
            .is_none_or(|v| v > u32::from(maximum))
        || k.checked_add(1).is_none_or(|v| v > u32::from(maximum))
    {
        return Ok(None);
    }
    let mut keys = empty(5)?;
    if s != 0 {
        keys.push(Key::Range(
            u8::try_from(s).map_err(|_| ClaimsError::Internal)?,
        ));
    }
    keys.push(Key::Range(
        u8::try_from(t).map_err(|_| ClaimsError::Internal)?,
    ));
    keys.push(Key::Range(
        u8::try_from(k).map_err(|_| ClaimsError::Internal)?,
    ));
    keys.push(Key::requant(
        k.checked_add(1).ok_or(ClaimsError::Internal)?,
        &[1],
        0,
        -2,
        1,
        input.id,
    )?);
    keys.push(Key::requant(
        t.checked_add(2).ok_or(ClaimsError::Internal)?,
        input.shifts,
        s,
        input.lo,
        input.hi,
        input.id,
    )?);
    Ok(feasible(&keys, maximum, input.id)?.then_some(keys))
}

fn signature(keys: &[Key]) -> Result<u8, ClaimsError> {
    match keys {
        [Key::Requant { .. }] => Ok(0),
        [Key::Range(_), Key::Requant { .. }] => Ok(1),
        [
            Key::Range(_),
            Key::Range(_),
            Key::Requant { .. },
            Key::Requant { .. },
        ] => Ok(2),
        [
            Key::Range(_),
            Key::Range(_),
            Key::Range(_),
            Key::Requant { .. },
            Key::Requant { .. },
        ] => Ok(3),
        _ => Err(ClaimsError::Internal),
    }
}

fn resolve(
    inputs: &[Input<'_>],
    requests: &mut [Vec<Key>],
    maximum: u8,
) -> Result<(), ClaimsError> {
    let first = inputs.first().ok_or(ClaimsError::Internal)?;
    let classed = first.shifts.len() > 1;
    for input in inputs {
        if (input.shifts.len() > 1) != classed {
            return Err(gadget(input.id));
        }
        if !(1..=62).contains(&input.bits)
            || input.shifts.is_empty()
            || input.shifts.len() > 4
            || input.shifts.iter().any(|r| !(1..=63).contains(r))
            || !(-(1i64 << 31)..0).contains(&input.lo)
            || !(0..(1i64 << 31)).contains(&input.hi)
        {
            return Err(gadget(input.id));
        }
    }
    let expected = signature(requests.first().ok_or(ClaimsError::Internal)?)?;
    let mut compatible = true;
    for keys in requests.iter() {
        compatible &= signature(keys)? == expected;
    }
    if compatible {
        return Ok(());
    }
    let mut replacements = empty(inputs.len())?;
    for input in inputs {
        replacements.push(split(input, maximum)?);
    }
    if replacements.iter().any(Option::is_none) {
        replacements.clear();
        let zero_low = inputs.iter().any(|i| i.shifts.contains(&1));
        for input in inputs {
            replacements.push(Some(
                wide(input, maximum, zero_low)?.ok_or_else(|| gadget(input.id))?,
            ));
        }
    }
    for (target, replacement) in requests.iter_mut().zip(replacements) {
        *target = replacement.ok_or(ClaimsError::Internal)?;
    }
    Ok(())
}

pub(super) fn apply(
    graph: &Graph,
    ranges: &[Interval],
    maximum: u8,
    pending: &mut [Vec<Key>],
) -> Result<(), ClaimsError> {
    if graph.slots().is_none_or(|s| s.protocol() != Protocol::One) {
        return Ok(());
    }
    let layout = crate::schedule::Layout::of(graph).map_err(|e| match e {
        crate::schedule::ScheduleError::Memory => ClaimsError::Memory,
        _ => gadget(GadgetId(0)),
    })?;
    let layers = graph
        .checkpoints()
        .len()
        .checked_sub(1)
        .ok_or(ClaimsError::Internal)?;
    let mut numbering = empty(graph.nodes().len())?;
    let mut next = 0usize;
    for node in graph.nodes() {
        let id = if super::is_gadget(node.op())? {
            let id = next;
            next = next.checked_add(1).ok_or(ClaimsError::Memory)?;
            Some(id)
        } else {
            None
        };
        numbering.push(id);
    }
    for (at, node) in graph.nodes().iter().enumerate() {
        if !matches!(node.op(), Op::Requant { .. }) {
            continue;
        }
        let node_id = NodeId(u32::try_from(at).map_err(|_| ClaimsError::Memory)?);
        let Some((0, position)) = layout.position(node_id) else {
            continue;
        };
        let mut inputs = empty(layers)?;
        let mut old = empty(layers)?;
        for layer in 0..layers {
            let id = layout
                .node_id(
                    u32::try_from(layer).map_err(|_| ClaimsError::Memory)?,
                    position,
                )
                .map_err(|_| ClaimsError::Internal)?;
            let n = graph.node(id).ok_or(ClaimsError::Internal)?;
            let Op::Requant { shift, lo, hi } = *n.op() else {
                return Err(ClaimsError::Internal);
            };
            let index = numbering
                .get(usize::try_from(id.0).map_err(|_| ClaimsError::Memory)?)
                .copied()
                .flatten()
                .ok_or(ClaimsError::Internal)?;
            let gid = GadgetId(u32::try_from(index).map_err(|_| ClaimsError::Memory)?);
            inputs.push(Input {
                bits: super::signed_width(super::interval(ranges, n, 0)?)
                    .ok_or_else(|| gadget(gid))?,
                shifts: graph
                    .shift_classes()
                    .get(usize::try_from(shift).map_err(|_| ClaimsError::Memory)?)
                    .ok_or(ClaimsError::Internal)?
                    .as_slice(),
                lo,
                hi,
                id: gid,
            });
            old.push(core::mem::take(
                pending.get_mut(index).ok_or(ClaimsError::Internal)?,
            ));
        }
        resolve(&inputs, &mut old, maximum)?;
        for (input, keys) in inputs.iter().zip(old) {
            *pending
                .get_mut(usize::try_from(input.id.0).map_err(|_| ClaimsError::Memory)?)
                .ok_or(ClaimsError::Internal)? = keys;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tv_testkit::Json;

    fn field<'a>(j: &'a Json, name: &str) -> &'a Json {
        j.get(name).unwrap()
    }
    fn number(j: &Json) -> i64 {
        match j {
            Json::Number(n) => n.parse().unwrap(),
            _ => panic!("oracle integer"),
        }
    }
    fn array(j: &Json) -> &[Json] {
        j.as_array().unwrap()
    }

    fn key(row: &Json) -> Key {
        let r = array(row);
        match r[0].as_str().unwrap() {
            "range" => Key::Range(number(&r[1]) as u8),
            "requant" => Key::requant(
                number(&r[1]) as u32,
                &array(&r[2])
                    .iter()
                    .map(|v| number(v) as u8)
                    .collect::<Vec<_>>(),
                0,
                number(&r[3]),
                number(&r[4]),
                GadgetId(0),
            )
            .unwrap(),
            _ => panic!("unknown oracle request"),
        }
    }

    #[test]
    fn independent_integer_selection_vectors() {
        let vectors = tv_testkit::read_vectors("uniform-requant/cases.json");
        let mut cases = 0;
        let mut promotions = 0;
        for case in array(field(&vectors, "cases")) {
            let name = field(case, "name").as_str().unwrap();
            let maximum = number(field(case, "maximum")) as u8;
            let nodes = array(field(case, "nodes"));
            let shifts: Vec<Vec<u8>> = nodes
                .iter()
                .map(|n| {
                    array(field(n, "shifts"))
                        .iter()
                        .map(|v| number(v) as u8)
                        .collect()
                })
                .collect();
            let inputs: Vec<Input<'_>> = nodes
                .iter()
                .zip(&shifts)
                .enumerate()
                .map(|(i, (n, rs))| Input {
                    bits: number(field(n, "bits")) as u32,
                    shifts: rs,
                    lo: number(field(n, "lo")),
                    hi: number(field(n, "hi")),
                    id: GadgetId(i as u32),
                })
                .collect();
            let mut state = super::super::Tables {
                base: &[],
                added: Vec::new(),
                seen: Default::default(),
                capture: false,
                deferred: true,
                pending: vec![Vec::new(); nodes.len()],
                trace: Vec::new(),
            };
            let old = inputs.iter().try_for_each(|n| {
                let half = 1i128 << (n.bits - 1);
                super::super::select_requant(
                    Interval::test_bounds(-half, half - 1),
                    n.shifts,
                    n.lo,
                    n.hi,
                    maximum,
                    n.id,
                    &mut state,
                )
            });
            let expected = field(case, "expected");
            let outcome = old.and_then(|()| resolve(&inputs, &mut state.pending, maximum));
            if let Some(reason) = expected.get("error") {
                let layer = number(field(expected, "layer")) as u32;
                assert_eq!(outcome, Err(gadget(GadgetId(layer))), "{name}: {reason:?}");
            } else {
                assert_eq!(outcome, Ok(()), "{name}");
                for (actual, expected) in state.pending.iter().zip(array(field(expected, "plans")))
                {
                    let keys: Vec<_> = array(field(expected, "requests")).iter().map(key).collect();
                    assert_eq!(*actual, keys, "{name}");
                    let catalog = crate::selection::Selection {
                        ranges: Vec::new(),
                        choices: Vec::new(),
                        tables: keys
                            .iter()
                            .enumerate()
                            .map(|(i, k)| crate::lowering::LoweredTable {
                                id: tv_tables::TableId(i as u32),
                                kind: k.kind().unwrap(),
                                log2_rows: k.log2_rows(GadgetId(0)).unwrap(),
                            })
                            .collect(),
                    };
                    let pattern = match field(expected, "family").as_str().unwrap() {
                        "one" => crate::selection::Pattern::Requant1,
                        "split" => crate::selection::Pattern::RequantSplit {
                            split: number(field(expected, "split")) as u8,
                        },
                        "wide" => crate::selection::Pattern::RequantWide {
                            split: number(field(expected, "split")) as u8,
                            middle: number(field(expected, "middle")) as u8,
                        },
                        _ => panic!("oracle family"),
                    };
                    let choice = crate::selection::Choice {
                        node: NodeId(0),
                        pattern,
                        tables: catalog.tables.iter().map(|t| t.id).collect(),
                    };
                    assert_eq!(
                        catalog.key_bits(&choice).unwrap(),
                        number(field(expected, "bits")) as u32,
                        "{name}"
                    );
                }
                if matches!(field(expected, "promoted"), Json::Bool(true)) {
                    promotions += 1;
                }
            }
            // No temporary auxiliary ids were assigned during either pass.
            assert!(state.added.is_empty(), "{name}");
            cases += 1;
        }
        assert_eq!((cases, promotions), (2191, 187));
    }
}
