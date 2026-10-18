// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Requant1`'s uniqueness, RFC-0009 Lemma 1, over `F65537` and Goldilocks: given `z` and `c`,
//! exactly one `y` passes, `rq(z, r_c, lo, hi)`, and every other value is rejected at the one
//! lookup. By exhaustion over every key and class of gadgets the compiler selects, with one class
//! and with four, and every `y'` from `lo − 2` to `hi + 2`.

mod common;

use common::{Fp, rejected, rq, select_requant};
use tessera_compiler::Gadget;
use tessera_field::{F65537, PrimeField};
use tessera_gadgets::{Committed, Requant, Term};
use tessera_poly::PolyField;

fn exhaust<F: PrimeField + PolyField<Base = F>>(b: u32, shifts: &[u8], (lo, hi): (i64, i64)) {
    let (selected, specs) = select_requant(b, shifts, lo, hi, 8);
    assert!(matches!(selected, Gadget::Requant1 { .. }));
    let gadget = Requant::new(&selected, &specs).expect("Requant1");
    let constraints = gadget.constraints::<F>();
    let tables = gadget.tables();
    let half = 1i64 << (b - 1);
    for z in -half..half {
        for (class, &r) in shifts.iter().enumerate() {
            let y = rq(z, r, lo, hi);
            for candidate in lo - 2..=hi + 2 {
                let found = rejected(&constraints, &tables, |term| match term {
                    Term::Input => F::from_i64(z),
                    Term::Class => F::from_u64_reduced(class as u64),
                    Term::Committed(Committed::Y) => F::from_i64(candidate),
                    Term::Committed(other) => panic!("Requant1 commits no {other:?}"),
                });
                let expected: &[usize] = if candidate == y { &[] } else { &[0] };
                assert_eq!(found, expected, "z = {z}, class {class}, y' = {candidate}");
            }
        }
    }
}

#[test]
fn only_the_requantized_value_passes_by_exhaustion() {
    for (b, shifts, bounds) in [
        (8u32, &[3u8][..], (-16i64, 15i64)),
        (6, &[1, 2, 3, 4][..], (-20, 20)),
        (7, &[2, 5][..], (0, 3)),
    ] {
        exhaust::<F65537>(b, shifts, bounds);
        exhaust::<Fp>(b, shifts, bounds);
    }
}
