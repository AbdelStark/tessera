// SPDX-License-Identifier: Apache-2.0 OR MIT
//! EXP-0020. Frozen registration is committed at 217238ef0c491d5ffbd677a8353bee79b44a473e.
#![forbid(unsafe_code)]
#![allow(dead_code, unused_imports)]
use tessera_field::{
    Field, Goldilocks as Fp, GoldilocksExt2 as Fp2, PrimeField, QuadraticExtension,
};
use tessera_poly::PolyField;
pub use tessera_sumcheck::{Claim, Prefix, SumcheckError, SumcheckProof, SumcheckTranscript};
mod arith;
mod coeff;
mod contract;
mod counts;
mod driver;
mod forest;
mod instance;
mod kernel;
mod level;
mod logup;
mod measure;
mod prover;
mod script_oracle;
mod trees;
pub use forest::{Forest, check_forests};
pub use instance::{LogUpInstance, Tuples};
pub use prover::ProvenBatch;
pub use tessera_logup::{Column, LogUpError};
pub use trees::{
    FractionClaim, FractionTree, LeafNumerators, LevelClaim, ProvenTreeLevel, Tree, prove_trees,
};
pub trait BenchField: PolyField<Base = Fp> {
    fn note(_slot: usize, _n: u64) {}
    fn words(self) -> [u64; 2];
    fn from_words(x: [Fp; 2]) -> Self;
}
impl BenchField for Fp2 {
    fn words(self) -> [u64; 2] {
        self.coeffs().map(|x| x.to_canonical_u64())
    }
    fn from_words(x: [Fp; 2]) -> Self {
        Self::new(x[0], x[1])
    }
}
thread_local! {static ARM:core::cell::Cell<u8>=const{core::cell::Cell::new(0)}; static WIDTH:core::cell::Cell<usize>=const{core::cell::Cell::new(2)};}
fn arm() -> u8 {
    ARM.with(core::cell::Cell::get)
}
fn width() -> usize {
    WIDTH.with(core::cell::Cell::get)
}
fn main() {
    driver::run();
}
