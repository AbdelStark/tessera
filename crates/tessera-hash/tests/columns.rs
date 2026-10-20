// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Chained column digests (RFC-0004 'Column digests') against the definition written out: per
//! column, over the whole matrix at once, in steps of `B` rows, with a short final step when the
//! number of rows is not a multiple of `B`.

mod common;

use std::num::NonZeroUsize;

use common::SplitMix64;
use tessera_field::{Goldilocks, PrimeField};
use tessera_hash::{ColumnChains, ColumnError, Digest, Domain, hash};

type Absorb = fn(&mut ColumnChains, &[Goldilocks]) -> Result<(), ColumnError>;
const ABSORBERS: [Absorb; 2] = [
    ColumnChains::absorb_block,
    ColumnChains::absorb_block_no_alloc,
];

/// The definition, column by column, from the whole matrix.
fn reference(id: &Digest, rows: &[Vec<Goldilocks>], columns: usize, block: usize) -> Vec<Digest> {
    (0..columns)
        .map(|j| {
            let mut init = id.to_vec();
            init.extend_from_slice(&(j as u64).to_le_bytes());
            let mut chain = hash(Domain::ColumnInit, &init);
            for step in rows.chunks(block) {
                let mut input = chain.to_vec();
                for row in step {
                    input.extend_from_slice(&row[j].to_canonical_u64().to_le_bytes());
                }
                chain = hash(Domain::ColumnStep, &input);
            }
            chain
        })
        .collect()
}

fn nonzero(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("non-zero")
}

#[test]
fn streamed_chains_match_the_definition() {
    let mut rng = SplitMix64::new(0x636f_6c75_6d6e_0001);
    for (m, columns, block) in [
        (0, 3, 4),
        (1, 1, 1),
        (7, 1, 1),
        (5, 3, 2),
        (31, 4, 32),
        (32, 4, 32),
        (33, 4, 32),
        (100, 3, 32),
        (130, 2, 64),
        (17, 9, 5),
    ] {
        let id = rng.digest();
        let rows: Vec<Vec<Goldilocks>> = (0..m)
            .map(|_| {
                (0..columns)
                    .map(|_| Goldilocks::from_u64_reduced(rng.next_u64()))
                    .collect()
            })
            .collect();
        for absorb in ABSORBERS {
            let mut chains = ColumnChains::new(&id, nonzero(columns), nonzero(block));
            let flat: Vec<Goldilocks> = rows.iter().flatten().copied().collect();
            for chunk in flat.chunks(block * columns) {
                absorb(&mut chains, chunk).expect("a valid block");
            }
            assert_eq!(
                chains.finish(),
                reference(&id, &rows, columns, block),
                "{m} rows, {columns} columns, block {block}"
            );
        }
    }
}

#[test]
fn malformed_blocks_are_rejected_and_change_nothing() {
    for absorb in ABSORBERS {
        let id = [9; 32];
        let x = Goldilocks::from_u64_reduced(5);
        let fresh = || ColumnChains::new(&id, nonzero(3), nonzero(4));
        let untouched = fresh().finish();

        let mut chains = fresh();
        assert_eq!(
            absorb(&mut chains, &[x; 7]),
            Err(ColumnError::PartialRow {
                symbols: 7,
                columns: 3
            })
        );
        assert_eq!(
            absorb(&mut chains, &[]),
            Err(ColumnError::BlockSize { rows: 0, block: 4 })
        );
        assert_eq!(
            absorb(&mut chains, &[x; 15]),
            Err(ColumnError::BlockSize { rows: 5, block: 4 })
        );
        assert_eq!(chains.finish(), untouched);

        // A short block ends the matrix: nothing may follow it.
        let mut chains = fresh();
        absorb(&mut chains, &[x; 6]).expect("two rows");
        let after_short = chains.clone().finish();
        assert_eq!(
            absorb(&mut chains, &[x; 12]),
            Err(ColumnError::AfterFinalBlock)
        );
        assert_eq!(chains.finish(), after_short);
    }
}
