// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU thread count never changes a proof (RFC-0018 'CPU backend', #729): a bounded table
//! batch, whose level regeneration and resident rounds split over threads, leaves the transcript
//! in the same state at 1 and at 8 threads. Its own test binary, since the setting is global.

use std::num::NonZeroUsize;

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{
    LogUpInstance, TableBatchBudget, TableSource, TreeId, Tuples, prove_logup,
    prove_table_batch_bounded,
};
use tessera_sumcheck::{SumcheckTranscript, WindowError, set_parallelism};

struct Table {
    rows: usize,
    salt: u64,
}

impl TableSource<B> for Table {
    fn rows(&self) -> usize {
        self.rows
    }
    fn has_values(&self) -> bool {
        true
    }
    fn read(
        &self,
        first: usize,
        keys: &mut [B],
        values: &mut [B],
        counts: &mut [B],
    ) -> Result<(), WindowError> {
        for i in 0..keys.len() {
            let j = (first + i) as u64;
            keys[i] = B::from_u64_reduced(j);
            values[i] = B::from_u64_reduced(j.wrapping_mul(self.salt) >> 7);
            counts[i] = B::from_u64_reduced((j ^ self.salt) % 5);
        }
        Ok(())
    }
}

fn digest(threads: usize) -> F {
    set_parallelism(NonZeroUsize::new(threads).unwrap());
    // Ragged heights so trees join the batch at different levels.
    let tables: Vec<Table> = [(18, 3), (17, 11), (19, 5), (2, 2)]
        .into_iter()
        .map(|(h, salt)| Table {
            rows: (1 << h) - 3,
            salt,
        })
        .collect();
    let refs: Vec<&dyn TableSource<B>> = tables.iter().map(|t| t as _).collect();
    let mut transcript = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [59; 32],
        session_id: [71; 16],
    });
    let _proven = prove_table_batch_bounded(
        &mut transcript,
        F::new(B::from_u64_reduced(3), B::ONE),
        F::new(B::from_u64_reduced(5), B::from_u64_reduced(7)),
        &refs,
        TableBatchBudget {
            allowance: u64::MAX / 4,
            source_bytes: 0,
            chunk_rows: NonZeroUsize::new(1 << 12).unwrap(),
        },
    )
    .unwrap();
    SumcheckTranscript::<F>::challenge(&mut transcript, "test/digest")
}

#[test]
fn thread_count_never_changes_the_table_batch_transcript() {
    let one = digest(1);
    for threads in [2, 8, 33] {
        assert_eq!(digest(threads), one, "{threads} threads");
    }
    // In the same test: the setting is global, so a second test would race this one.
    let one = lookup_digest(1);
    for threads in [2, 8, 33] {
        assert_eq!(
            lookup_digest(threads),
            one,
            "{threads} threads, resident lookups"
        );
    }
}

/// A resident lookup tree of 2^19 leaves and its table tree, built and proven at `threads`: the
/// trees' levels split over threads (#744) and the resident level tables load on them.
fn lookup_digest(threads: usize) -> F {
    set_parallelism(NonZeroUsize::new(threads).unwrap());
    let rows: Vec<B> = (0..1u64 << 12).map(B::from_u64_reduced).collect();
    let values: Vec<B> = (0..1u64 << 12)
        .map(|j| B::from_u64_reduced(j * j + 3))
        .collect();
    let picks: Vec<usize> = (0..(1usize << 19) - 5)
        .map(|i| (i * 2_654_435_761) % rows.len())
        .collect();
    let keys: Vec<B> = picks.iter().map(|&j| rows[j]).collect();
    let looked: Vec<B> = picks.iter().map(|&j| values[j]).collect();
    let mut counts = vec![0u64; rows.len()];
    for &j in &picks {
        counts[j] += 1;
    }
    let counts: Vec<B> = counts.into_iter().map(B::from_u64_reduced).collect();
    let instance = LogUpInstance::new(
        Tuples::new(&rows, Some(&values)),
        Tuples::new(&keys, Some(&looked)),
        &counts,
    );
    let mut transcript = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [61; 32],
        session_id: [73; 16],
    });
    let _proven = prove_logup(&mut transcript, &[instance], &[], &TreeId::paired(1)).unwrap();
    SumcheckTranscript::<F>::challenge(&mut transcript, "test/digest")
}
