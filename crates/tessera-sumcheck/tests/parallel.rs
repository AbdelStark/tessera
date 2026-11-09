// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The thread count never changes a batch proof (RFC-0018 'CPU backend'): `prove_batch`, whose
//! round sums split over threads, gives the same proof at 1, 3 and 16 threads, for products and
//! sums of products, base and extension tables, padded instances included. Its own test binary,
//! since the setting is global.

mod common;

use common::{Fp2, Session, SplitMix64};
use std::num::NonZeroUsize;
use tessera_hash::Transcript;
use tessera_sumcheck::{
    Instance, Prefix, Protocol, Table, Term, prove_batch, set_parallelism,
};

#[test]
fn thread_count_never_changes_a_batch_proof() {
    let mut rng = SplitMix64::new(0x7e55_e7a);
    let session = Session::random(&mut rng);
    let a = rng.fp2s(1 << 15);
    let b = rng.fp2s(1 << 15);
    let c = rng.fps(1 << 15);
    let d = rng.fp2s(1 << 13);
    let e = rng.fp2s(1 << 13);
    let w = rng.fp2();
    let prove = |threads: usize| {
        set_parallelism(NonZeroUsize::new(threads).unwrap());
        let instances = vec![
            Instance::new(vec![Table::ext(&a), Table::ext(&b), Table::base(&c)]),
            Instance::sum_of_products(
                vec![Table::ext(&a), Table::base(&c), Table::ext(&b)],
                vec![Term::new(w, vec![0, 1]), Term::new(Fp2::from(c[3]), vec![2, 0, 1])],
            ),
            Instance::new(vec![Table::ext(&d), Table::ext(&e)]),
        ];
        let mut transcript = Transcript::new(&session.context());
        let proven =
            prove_batch(&mut transcript, Protocol::One, Prefix::Stage(1), 3, &instances).unwrap();
        format!("{:?}", proven.proof())
    };
    let one = prove(1);
    for threads in [3, 16] {
        assert_eq!(prove(threads), one, "{threads} threads");
    }
}
