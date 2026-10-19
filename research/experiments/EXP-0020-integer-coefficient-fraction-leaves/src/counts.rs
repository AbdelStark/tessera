// SPDX-License-Identifier: Apache-2.0 OR MIT
//! E0 operation trace using the same generic algorithms, never instantiated by timed Fp2.
use crate::*;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use tessera_codec::Zeroize;
thread_local! {static COUNTS:core::cell::RefCell<[u64;23]>=const{core::cell::RefCell::new([0;23])};}
fn tick(i: usize, n: u64) {
    COUNTS.with(|c| c.borrow_mut()[i] += n);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct C(pub Fp2);
impl Zeroize for C {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}
macro_rules! binary{($tr:ident,$fn:ident,$op:tt,$slot:expr)=>{impl $tr for C{type Output=Self;fn $fn(self,b:Self)->Self{tick($slot,1);Self(self.0 $op b.0)}}};}
binary!(Add,add,+,1);
binary!(Sub,sub,-,2);
binary!(Mul,mul,*,0);
impl Neg for C {
    type Output = Self;
    fn neg(self) -> Self {
        tick(3, 1);
        Self(-self.0)
    }
}
impl AddAssign for C {
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}
impl SubAssign for C {
    fn sub_assign(&mut self, b: Self) {
        *self = *self - b;
    }
}
impl MulAssign for C {
    fn mul_assign(&mut self, b: Self) {
        *self = *self * b;
    }
}
impl Field for C {
    const ZERO: Self = Self(Fp2::ZERO);
    const ONE: Self = Self(Fp2::ONE);
    fn square(&self) -> Self {
        tick(22, 1);
        Self(self.0.square())
    }
    fn double(&self) -> Self {
        *self + *self
    }
    fn inverse(&self) -> Option<Self> {
        tick(6, 1);
        self.0.inverse().map(Self)
    }
    fn pow(&self, e: u64) -> Self {
        tick(19, 1);
        Self(self.0.pow(e))
    }
}
impl From<Fp> for C {
    fn from(x: Fp) -> Self {
        tick(21, 1);
        Self(x.into())
    }
}
impl PolyField for C {
    type Base = Fp;
    fn mul_by_base(&self, x: Fp) -> Self {
        tick(4, 1);
        Self(self.0.mul_by_base(x))
    }
    fn add_base(&self, x: Fp) -> Self {
        tick(5, 1);
        Self(self.0.add_base(x))
    }
}
impl BenchField for C {
    fn note(i: usize, n: u64) {
        tick(i, n)
    }
    fn words(self) -> [u64; 2] {
        tick(7, 2);
        self.0.words()
    }
    fn from_words(x: [Fp; 2]) -> Self {
        tick(8, 1);
        Self(Fp2::from_words(x))
    }
}
struct T(tessera_hash::Transcript);
impl SumcheckTranscript<C> for T {
    fn absorb_message(&mut self, l: &str, m: &[C]) {
        self.0
            .absorb_message(l, &m.iter().map(|x| x.0).collect::<Vec<_>>());
    }
    fn challenge(&mut self, l: &str) -> C {
        C(self.0.challenge(l))
    }
}
pub fn run(case: &driver::Case, arm: u8) -> serde_json::Value {
    COUNTS.with(|x| *x.borrow_mut() = [0; 23]);
    ARM.with(|x| x.set(arm));
    WIDTH.with(|x| x.set(2));
    coeff::DOMAIN.with(|x| x.set(Some(case.domain)));
    let mut t = T(case.transcript());
    if arm == 0 {
        let p = tessera_logup::prove_logup(
            &mut t,
            &[case.production()],
            &[],
            &tessera_logup::TreeId::paired(1),
        )
        .unwrap();
        drop(std::hint::black_box(p));
    } else {
        let p =
            logup::prove_logup(&mut t, &[case.local()], &[], &logup::TreeId::paired(1)).unwrap();
        drop(std::hint::black_box(p));
    }
    let counts = COUNTS.with(|c| *c.borrow());
    let names = [
        "Fp2_mul",
        "Fp2_add",
        "Fp2_sub",
        "Fp2_neg",
        "Fp2_by_base_mul",
        "Fp2_add_base",
        "Fp2_inverse_opaque",
        "canonical_component_reads",
        "from_reduced_components",
        "signed_weight_wide_products",
        "signed192_additions",
        "signed192_reductions",
        "deferred_wide_products",
        "unsigned192_additions",
        "unsigned192_reductions",
        "integer_coefficient_products",
        "integer_coefficient_additions",
        "integer_grid_extrapolations",
        "once_bind_column_reads",
        "Fp2_pow_opaque",
        "raw_integer_conversions",
        "Fp2_from_base",
        "Fp2_square",
    ];
    let operations: serde_json::Map<_, _> = names
        .iter()
        .zip(counts)
        .map(|(&k, n)| (k.into(), serde_json::json!(n)))
        .collect();
    serde_json::json!({"case":case.name,"arm":arm,"statement_state":t.0.state(),"operations":operations,"scope":"semantic operation counts from same generic loops; base inverse opaque, no lane/time fit"})
}
