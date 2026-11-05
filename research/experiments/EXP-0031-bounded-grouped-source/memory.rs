// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Allocation-free public source bill; no production admission authority.

use serde::Serialize;
use tessera_codec::SecretBuf;
use tessera_field::Goldilocks;
use tessera_prover::leaf::GroupLayout;
use tessera_tables::TableId;

use crate::adapter::{ForestRecord, IndexRecord};
use crate::fixture::ForestGeometry;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Memory,
    Shape,
}

pub fn add(a: u64, b: u64) -> Result<u64, Refusal> {
    a.checked_add(b).ok_or(Refusal::Memory)
}
pub fn mul(a: u64, b: u64) -> Result<u64, Refusal> {
    a.checked_mul(b).ok_or(Refusal::Memory)
}

pub fn array<T>(n: u64) -> Result<u64, Refusal> {
    let bytes = mul(n, size_of::<T>() as u64)?;
    if usize::try_from(n).is_err() || bytes > isize::MAX as u64 {
        Err(Refusal::Memory)
    } else {
        Ok(bytes)
    }
}

pub fn growing<T>(n: u64) -> Result<u64, Refusal> {
    if n == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    array::<T>(mul(4, n.max(minimum))?)
}

#[derive(Serialize)]
pub struct SourceBill {
    pub columns: u64,
    pub forest: u64,
    pub records: u64,
    pub earlier: u64,
    pub peak: u64,
    pub source_payload: u64,
    pub row_records: u64,
    pub forest_records: u64,
}

impl SourceBill {
    pub fn new(
        layout: &GroupLayout,
        forest: &ForestGeometry,
        gt: u32,
        gl: u32,
        earlier: u64,
    ) -> Result<Self, Refusal> {
        let fixed = gt.checked_add(gl).ok_or(Refusal::Memory)?;
        let vars = forest.vars.checked_sub(fixed).ok_or(Refusal::Shape)?;
        let forest_records = 1u64.checked_shl(vars).ok_or(Refusal::Memory)?;
        type Column = (
            TableId,
            SecretBuf<Goldilocks>,
            Option<SecretBuf<Goldilocks>>,
        );
        let mut columns = growing::<Column>(layout.instances().len() as u64)?;
        let mut source_payload = 0;
        let mut row_records = 0;
        for instance in layout.instances() {
            let n = 1u64.checked_shl(instance.height).ok_or(Refusal::Memory)?;
            let copies = if instance.table == TableId(2) { 2 } else { 1 };
            columns = add(columns, mul(copies, growing::<Goldilocks>(n)?)?)?;
            source_payload = add(source_payload, mul(copies, array::<Goldilocks>(n)?)?)?;
            for block in &instance.blocks {
                row_records = add(
                    row_records,
                    1u64.checked_shl(block.vars).ok_or(Refusal::Memory)?,
                )?;
            }
        }
        let forest = growing::<Goldilocks>(forest_records)?;
        source_payload = add(source_payload, array::<Goldilocks>(forest_records)?)?;
        let records = add(
            growing::<IndexRecord>(row_records)?,
            growing::<ForestRecord>(forest_records)?,
        )?;
        let peak = add(add(add(columns, forest)?, records)?, earlier)?;
        Ok(Self {
            columns,
            forest,
            records,
            earlier,
            peak,
            source_payload,
            row_records,
            forest_records,
        })
    }

    pub fn check(&self, allowance: u64) -> Result<(), Refusal> {
        if self.peak <= allowance {
            Ok(())
        } else {
            Err(Refusal::Memory)
        }
    }
}
