// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The GPU level offload never changes a proof (#734): a table batch proved with every level
//! instance's rounds on the device leaves the transcript as on the host. Its own test binary,
//! since the registration is global.

#![cfg(feature = "gpu")]

use std::num::NonZeroUsize;
use std::sync::Arc;

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_gpu::{Device, GpuLevels};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{TableBatchBudget, TableSource, prove_table_batch_bounded};
use tessera_sumcheck::{SumcheckTranscript, WindowError, set_level_offload};

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

fn digest() -> F {
    let tables: Vec<Table> = [(16, 3), (15, 11), (17, 5), (2, 2)]
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
            chunk_rows: NonZeroUsize::new(1 << 10).unwrap(),
        },
    )
    .unwrap();
    SumcheckTranscript::<F>::challenge(&mut transcript, "test/digest")
}

#[test]
fn gpu_levels_never_change_the_table_batch_transcript() {
    let host = digest();
    let device = Arc::new(Device::new(0).expect("a CUDA device"));
    set_level_offload(Some(Arc::new(GpuLevels::new(device, 1 << 30, 2))));
    let gpu = digest();
    set_level_offload(None);
    assert_eq!(gpu, host);
}
