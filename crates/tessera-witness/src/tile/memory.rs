// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained public source metadata; constructor and tile work remain separate (RFC-0018).

use super::*;
use tessera_gadgets::{Lookup, Virtual};

pub(super) fn array<T>(count: usize) -> Option<usize> {
    count.checked_mul(size_of::<T>())
}

pub(super) fn affine<T>(form: &Affine<Goldilocks, T>) -> Option<usize> {
    array::<(Goldilocks, T)>(form.terms.capacity())
}

pub(super) fn lookups<T>(values: &Vec<Lookup<Goldilocks, T>>) -> Option<usize> {
    values.iter().try_fold(
        array::<Lookup<Goldilocks, T>>(values.capacity())?,
        |sum, value| {
            sum.checked_add(affine(&value.key)?)?
                .checked_add(value.value.as_ref().map_or(Some(0), affine)?)
        },
    )
}

fn constraints(value: &Constraints<Goldilocks>) -> Option<usize> {
    let bytes = array::<Committed>(value.committed.capacity())?
        .checked_add(array::<(Virtual, Affine<Goldilocks>)>(
            value.virtuals.capacity(),
        )?)?
        .checked_add(lookups(&value.lookups)?)?;
    value
        .virtuals
        .iter()
        .try_fold(bytes, |sum, (_, form)| sum.checked_add(affine(form)?))
}

impl Gadgets {
    pub(crate) fn heap_bytes(&self) -> Option<usize> {
        let mut bytes = array::<Option<RequantGadget>>(self.requant.capacity())?;
        for size in [
            array::<Vec<TableId>>(self.requests.capacity())?,
            array::<Option<Prepared>>(self.auxiliary.capacity())?,
            array::<TableId>(self.read.capacity())?,
            array::<TableId>(self.incomplete.capacity())?,
            array::<tessera_compiler::GreedyKeyColumn>(self.key_columns.capacity())?,
            array::<(TableId, u64)>(self.columns.capacity())?,
        ] {
            bytes = bytes.checked_add(size)?;
        }
        for request in &self.requests {
            bytes = bytes.checked_add(array::<TableId>(request.capacity())?)?;
        }
        for value in self.requant.iter().flatten() {
            bytes = bytes
                .checked_add(value.requant.heap_bytes()?)?
                .checked_add(constraints(&value.constraints)?)?
                .checked_add(array::<(TableId, u64)>(value.rows.capacity())?)?;
        }
        self.auxiliary
            .iter()
            .flatten()
            .try_fold(bytes, |sum, value| sum.checked_add(value.heap_bytes()?))
    }
}
