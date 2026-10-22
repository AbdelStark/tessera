// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Clearing CPU exact-greedy certificate sources (draft RFC-0053).

use core::hint::select_unpredictable;
use tessera_codec::SecretBuf;
use tessera_compiler::{GreedyCertificate, RuntimePlan, analyze};

use crate::{TileWitness, WitnessError, structure::Structure, tile::Gadgets};

const METADATA: &str = "runtime metadata belongs to the source's canonical lowering";
const INPUT: &str = "runtime logits and tokens fit their public shapes and integer ranges";

pub(crate) fn configure(
    structure: &Structure<'_>,
    runtime: &RuntimePlan,
    gadgets: &mut Gadgets,
) -> Result<(), WitnessError> {
    if let Some(c) = runtime.certificate() {
        let lowering = structure.lowering;
        let ranges = analyze(&lowering.graph, &lowering.tables)
            .map_err(|_| WitnessError::Usage { rule: METADATA })?;
        if lowering.graph.tensors[lowering.logits.0 as usize].shape != [c.positions, c.vocab]
            || ranges.of(lowering.logits).and_then(|r| r.bounded()) != Some(c.logits)
        {
            return Err(WitnessError::Usage { rule: METADATA });
        }
        if !gadgets.read.contains(&c.table) {
            gadgets.read.push(c.table);
            gadgets.read.sort_unstable();
        }
        gadgets.certificate = Some(c);
    }
    Ok(())
}

pub(crate) fn maximum(
    c: GreedyCertificate,
    tokens: &[i64],
    logits: &[i64],
) -> Result<TileWitness, WitnessError> {
    let (n, v) = (c.positions as usize, c.vocab as usize);
    if tokens.len() != n || n.checked_mul(v) != Some(logits.len()) {
        return Err(WitnessError::Usage { rule: INPUT });
    }
    let memory = WitnessError::MemoryTable { table: c.table };
    let mut maxima = SecretBuf::try_with_capacity(n).map_err(|_| memory)?;
    maxima.resize(n, 0i64);
    let size = usize::try_from(c.entries()).map_err(|_| memory)?;
    let mut keys = SecretBuf::try_with_capacity(size).map_err(|_| memory)?;
    keys.resize(size, 0u32);
    let padded_v = v.next_power_of_two();
    let mut valid = true;
    let mut exact = true;
    for t in c.prompt as usize - 1..n - 1 {
        let token = tokens[t + 1];
        valid &= (0 <= token) & (token < v as i64);
        for (index, &logit) in logits[t * v..(t + 1) * v].iter().enumerate() {
            valid &= (c.logits.lo <= i128::from(logit)) & (i128::from(logit) <= c.logits.hi);
            maxima[t] = select_unpredictable(index as i64 == token, logit, maxima[t]);
        }
        for (index, &logit) in logits[t * v..(t + 1) * v].iter().enumerate() {
            let key =
                i128::from(maxima[t]) - i128::from(logit) - i128::from((index as i64) < token);
            exact &= (0 <= key) & (key < 1i128 << c.range_bits);
            keys[(t + 1 - c.prompt as usize) * padded_v + index] = key as u32;
        }
    }
    if !valid {
        return Err(WitnessError::Internal { rule: INPUT });
    }
    if !exact {
        return Err(WitnessError::ExactMismatch);
    }
    Ok(TileWitness::greedy(c, maxima, keys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_ir::{GadgetId, IntRange, TableId};

    fn certificate(prompt: u32) -> GreedyCertificate {
        GreedyCertificate {
            gadget: GadgetId(29),
            table: TableId(3),
            prompt,
            positions: 5,
            vocab: 3,
            logits: IntRange { lo: -9, hi: 7 },
            range_bits: 5,
            variables: 4,
        }
    }

    #[test]
    fn real_time_maxima_and_independently_padded_keys() {
        let c = certificate(2);
        let tokens = [2, 2, 0, 2, 1];
        let logits = [7, 7, 7, -3, -3, -8, -9, -8, -2, -7, -1, -1, 7, 7, 7];
        let tile = maximum(c, &tokens, &logits).unwrap();
        assert!(tile.shares().is_empty());
        assert!(tile.gadget().is_empty());
        assert_eq!(
            tile.greedy_maximum(),
            Some((c.gadget, &[0, -3, -2, -1, 0][..]))
        );
        let lookup = tile.lookups().next().unwrap();
        assert_eq!(lookup.table(), c.table);
        assert!(lookup.values().is_none());
        assert_eq!(
            lookup.keys(),
            &[0, 0, 5, 0, 6, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
        );
        for (index, wrong) in [(2, 1), (3, 1), (4, 2)] {
            let mut bad = tokens;
            bad[index] = wrong;
            assert_eq!(
                maximum(c, &bad, &logits).unwrap_err(),
                WitnessError::ExactMismatch
            );
        }
        let mut outside = logits;
        outside[5] = -10;
        assert_eq!(
            maximum(c, &tokens, &outside).unwrap_err(),
            WitnessError::Internal { rule: INPUT }
        );
        let mut outside = tokens;
        outside[2] = 3;
        assert_eq!(
            maximum(c, &outside, &logits).unwrap_err(),
            WitnessError::Internal { rule: INPUT }
        );
        assert_eq!(
            maximum(c, &tokens[..4], &logits).unwrap_err(),
            WitnessError::Usage { rule: INPUT }
        );
    }
}
