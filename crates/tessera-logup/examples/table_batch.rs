// SPDX-License-Identifier: Apache-2.0 OR MIT

//! P2's table batch alone (RFC-0054 `prove_table_batch_bounded`), for profiling: `TABLES` dense
//! `Range` tables of `2^HEIGHT` rows with sparse counts, under an unbounded allowance. Prints the
//! wall time and a digest of the transcript state after the batch, which must not depend on
//! `THREADS` (#729).
//!
//! ```text
//! cargo run --release -p tessera-logup --example table_batch -- HEIGHT TABLES THREADS
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::print_stdout)]

use std::num::NonZeroUsize;
use std::time::Instant;

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{TableBatchBudget, TableSource, prove_table_batch_bounded};
use tessera_sumcheck::{SumcheckTranscript, WindowError};

/// A `Range` table whose row `j` is counted `j mod 7` times when `j` is a multiple of 1024.
struct Dense {
    rows: usize,
}

impl TableSource<B> for Dense {
    fn rows(&self) -> usize {
        self.rows
    }
    fn has_values(&self) -> bool {
        false
    }
    fn fraction_cache_depth(&self) -> Option<usize> {
        // As the prover's table plan picks with an ample allowance: h − 4 (#729).
        (std::env::var_os("NO_CACHE").is_none()).then(|| self.rows.ilog2() as usize - 4)
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
            values[i] = B::ZERO;
            counts[i] = B::from_u64_reduced(if j.is_multiple_of(1024) { j % 7 } else { 0 });
        }
        Ok(())
    }
}

fn main() {
    let arg = |i: usize, default: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(default)
    };
    let (height, tables, threads) = (arg(1, 20), arg(2, 2), arg(3, 1));
    tessera_sumcheck::set_parallelism(NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN));
    let sources: Vec<Dense> = (0..tables).map(|_| Dense { rows: 1 << height }).collect();
    let refs: Vec<&dyn TableSource<B>> = sources.iter().map(|s| s as _).collect();
    let mut transcript = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap_or_else(|_| unreachable!()),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap_or_else(|_| unreachable!())),
        statement_digest: [59; 32],
        session_id: [71; 16],
    });
    let started = Instant::now();
    let proven = prove_table_batch_bounded(
        &mut transcript,
        F::new(B::from_u64_reduced(3), B::ONE),
        F::new(B::from_u64_reduced(5), B::from_u64_reduced(7)),
        &refs,
        TableBatchBudget {
            allowance: u64::MAX / 4,
            source_bytes: 0,
            chunk_rows: NonZeroUsize::new(1 << 20).unwrap_or(NonZeroUsize::MIN),
        },
    );
    let elapsed = started.elapsed();
    assert!(proven.is_ok(), "the batch proves");
    let digest: F = SumcheckTranscript::<F>::challenge(&mut transcript, "bench/digest");
    println!(
        "height {height} tables {tables} threads {threads}: {:.3} s, {:.1} ns/row, digest {digest:?}",
        elapsed.as_secs_f64(),
        elapsed.as_nanos() as f64 / (tables << height) as f64
    );
}
