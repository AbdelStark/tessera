// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An offloaded level instance in a batch of higher degree (#734): the offload returns the
//! instance's sums at 0, 2 and 3 only, and the host must interpolate the batch's further nodes.
//! A batch of degree 4, with product and fraction instances beside a degree-4 instance, is proved
//! with and without a host offload: the proofs are equal. Its own binary: the offload is global.

use std::sync::{Arc, Mutex};

use tessera_codec::Ascii;
use tessera_field::{Field, Goldilocks as B, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_sumcheck::{
    DeviceLevel, LevelOffload, LevelShape, Prefix, PreparedWindow, Protocol, SumcheckTranscript,
    Term, WindowError, WindowInstance, WindowPass, WindowSource, WindowTable, set_level_offload,
};

struct Source(Vec<Vec<F>>);

impl WindowSource<F> for Source {
    fn regenerate(&mut self, _: WindowPass) -> Result<(), WindowError> {
        Ok(())
    }
    fn read(&self, table: usize, start: usize, out: &mut [F]) -> Result<(), WindowError> {
        out.copy_from_slice(&self.0[table][start..start + out.len()]);
        Ok(())
    }
}

struct Host {
    shape: LevelShape,
    tables: Mutex<Vec<Vec<F>>>,
}

impl DeviceLevel for Host {
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
                out[z] += w * match self.shape {
                    LevelShape::Fraction { lambda } => {
                        l[0] * l[3] + l[1] * l[2] + lambda * l[2] * l[3]
                    }
                    LevelShape::Product => l[0] * l[1],
                };
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

struct Offload(Mutex<usize>);

impl LevelOffload for Offload {
    fn load(
        &self,
        shape: LevelShape,
        tables: &[&[F]],
    ) -> Result<Option<Box<dyn DeviceLevel>>, WindowError> {
        *self.0.lock().unwrap() += 1;
        Ok(Some(Box::new(Host {
            shape,
            tables: Mutex::new(tables.iter().map(|t| t.to_vec()).collect()),
        })))
    }
}

fn f(a: u64, b: u64) -> F {
    F::new(B::from_u64_reduced(a), B::from_u64_reduced(b))
}

fn prove() -> (Vec<Vec<F>>, Vec<F>, F) {
    let n = 1usize << 6;
    let table = |s: u64| {
        (0..n as u64)
            .map(|i| f(i * 31 + s, i ^ s))
            .collect::<Vec<_>>()
    };
    let source = Source((0..9).map(|s| table(s * 1000 + 7)).collect());
    let eq = |v: usize| {
        (0..v)
            .map(|j| Some(f(j as u64 + 5, 2 * j as u64 + 1)))
            .collect::<Vec<_>>()
    };
    let private = |id| WindowTable { id, public: false };
    let specs = vec![
        // V0·V1: the product shape, degree 3 with its eq.
        WindowInstance {
            vars: 6,
            tables: vec![private(0), private(1)],
            terms: vec![Term::new(F::ONE, vec![0, 1])],
            eq: eq(6),
            windowed: false,
        },
        // N0·D1 + N1·D0 + λ·D0·D1: the fraction shape.
        WindowInstance {
            vars: 6,
            tables: vec![private(2), private(3), private(4), private(5)],
            terms: vec![
                Term::new(F::ONE, vec![0, 3]),
                Term::new(F::ONE, vec![1, 2]),
                Term::new(f(9, 4), vec![2, 3]),
            ],
            eq: eq(6),
            windowed: false,
        },
        // Three factors: degree 4, which sets the batch's.
        WindowInstance {
            vars: 6,
            tables: vec![private(6), private(7), private(8)],
            terms: vec![Term::new(f(3, 1), vec![0, 1, 2])],
            eq: eq(6),
            windowed: false,
        },
    ];
    let mut source = source;
    let mut transcript = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [3; 32],
        session_id: [4; 16],
    });
    let prepared = PreparedWindow::new(&mut source, Protocol::One, 4, specs, 0).unwrap();
    let proven = prepared
        .prove(&mut transcript, Prefix::Stage(2), &mut source)
        .unwrap();
    let digest = SumcheckTranscript::<F>::challenge(&mut transcript, "after");
    (
        proven.proof().rounds.clone(),
        proven.proof().final_evals.clone(),
        digest,
    )
}

#[test]
fn offloaded_instances_in_a_degree_four_batch_give_the_host_proof() {
    let host = prove();
    let offload = Arc::new(Offload(Mutex::new(0)));
    set_level_offload(Some(offload.clone()));
    let offloaded = prove();
    set_level_offload(None);
    assert_eq!(
        *offload.0.lock().unwrap(),
        2,
        "the product and fraction instances"
    );
    assert_eq!(offloaded, host);
}
