// SPDX-License-Identifier: Apache-2.0 OR MIT
use tessera_field::Field;
pub(crate) fn eq_weights<F: Field>(p: &[F]) -> Vec<F> {
    tessera_poly::eq_table(p).expect("public shape")
}
