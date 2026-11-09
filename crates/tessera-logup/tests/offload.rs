// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A registered level offload never changes a proof (RFC-0018, #734): a host implementation of
//! `LevelOffload`, the GPU's reference, leaves a table batch's transcript as without it. Its own
//! test binary, since the registration is global.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{TableBatchBudget, TableSource, prove_table_batch_bounded};
use tessera_sumcheck::{
    DeviceLevel, LevelOffload, LevelShape, SumcheckTranscript, WindowError, set_level_offload,
};

struct HostLevel {
    shape: LevelShape,
    tables: Mutex<Vec<Vec<F>>>,
}

impl DeviceLevel for HostLevel {
    fn sums(&self, low: &[F], high: &[F], low_bits: usize) -> Result<[F; 3], WindowError> {
        let t = self.tables.lock().unwrap();
        let mut out = [F::ZERO; 3];
        for p in 0..t[0].len() / 2 {
            let w = low[p & (low.len() - 1)] * high[p >> low_bits];
            for (z, x) in [0u64, 2, 3].into_iter().enumerate() {
                let x = F::from(B::from_u64_reduced(x));
                let l: Vec<F> = t
                    .iter()
                    .map(|c| c[2 * p] + x * (c[2 * p + 1] - c[2 * p]))
                    .collect();
                let v = match self.shape {
                    LevelShape::Fraction { lambda } => {
                        l[0] * l[3] + l[1] * l[2] + lambda * l[2] * l[3]
                    }
                    LevelShape::Product => l[0] * l[1],
                };
                out[z] += w * v;
            }
        }
        Ok(out)
    }
    fn bind(&mut self, r: F) -> Result<(), WindowError> {
        for c in self.tables.get_mut().unwrap().iter_mut() {
            *c = (0..c.len() / 2)
                .map(|i| c[2 * i] + r * (c[2 * i + 1] - c[2 * i]))
                .collect();
        }
        Ok(())
    }
    fn firsts(&self) -> Result<Vec<F>, WindowError> {
        Ok(self.tables.lock().unwrap().iter().map(|c| c[0]).collect())
    }
}

struct Host(Mutex<usize>);

impl LevelOffload for Host {
    fn load(
        &self,
        shape: LevelShape,
        tables: &[&[F]],
    ) -> Result<Option<Box<dyn DeviceLevel>>, WindowError> {
        // Take every other instance, so host and offloaded instances share a batch.
        let mut n = self.0.lock().unwrap();
        *n += 1;
        Ok((*n % 2 == 0).then(|| {
            Box::new(HostLevel {
                shape,
                tables: Mutex::new(tables.iter().map(|t| t.to_vec()).collect()),
            }) as Box<dyn DeviceLevel>
        }))
    }
}

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
    let tables: Vec<Table> = [(12, 3), (11, 11), (13, 5), (2, 2)]
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
fn an_offload_never_changes_the_table_batch_transcript() {
    let host = digest();
    let offload = Arc::new(Host(Mutex::new(0)));
    set_level_offload(Some(offload.clone()));
    let offloaded = digest();
    set_level_offload(None);
    assert!(
        *offload.0.lock().unwrap() > 4,
        "the offload saw the batch's instances"
    );
    assert_eq!(offloaded, host);
}
