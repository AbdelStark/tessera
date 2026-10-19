// SPDX-License-Identifier: Apache-2.0 OR MIT

//! First-pass certificate oracle; no witness tile assembly or multiplicity code is called.

use tessera_compiler::{AuxColumn, CompareColumn as Array, Gadget, Selection};
use tessera_field::{Goldilocks, PrimeField};
use tessera_gadgets::{Bracket, CompareColumn, RsqrtColumn, RsqrtRow, RsqrtTerm, Side};
use tessera_ir::{Node, Op, TableId};
use tessera_tables::TableKind;

pub type Columns = Vec<(AuxColumn, Vec<i64>)>;
pub type Lookups = Vec<(TableId, Vec<u32>, Option<Vec<i64>>)>;

pub fn root(selection: &Selection, node: &Node, sums: &[i64], real: &[bool]) -> (Columns, Lookups) {
    let Op::RsqrtRow {
        dim,
        kappa,
        eps_int,
    } = node.op
    else {
        panic!("root")
    };
    let gadget = RsqrtRow::new(dim, kappa, eps_int, &selection.tables).expect("scalar root");
    let input: Vec<_> = sums
        .iter()
        .zip(real)
        .map(|(&s, &real)| if real { s + eps_int as i64 } else { 0 })
        .collect();
    let witness = gadget.witness(&input, real).expect("scalar witness");
    let constraints = gadget.constraints::<Goldilocks>().expect("constraints");
    let mut columns = Vec::new();
    for c in constraints.committed {
        let role = match c {
            RsqrtColumn::Root => continue,
            RsqrtColumn::InputLimb(j) => AuxColumn::InputLimb(u32::from(j)),
            RsqrtColumn::RootLimb(j) => AuxColumn::RootByte(u32::from(j)),
            RsqrtColumn::Compare(b, c) => {
                let (array, j) = match c {
                    CompareColumn::Digit(Side::Left, j) => (Array::LeftByte, j),
                    CompareColumn::Carry(Side::Left, j) => (Array::LeftCarry, j),
                    CompareColumn::Digit(Side::Right, j) => (Array::RightByte, j),
                    CompareColumn::Carry(Side::Right, j) => (Array::RightCarry, j),
                    CompareColumn::Difference(j) => (Array::Difference, j),
                    CompareColumn::Borrow(j) => (Array::Borrow, j),
                    _ => panic!("virtual input"),
                };
                AuxColumn::Comparison {
                    upper: b == Bracket::Upper,
                    column: array,
                    limb: j as u32,
                }
            }
        };
        columns.push((role, witness.tensor(c).expect("column").to_vec()));
    }
    let lookups = constraints
        .lookups
        .into_iter()
        .map(|l| {
            assert!(l.value.is_none());
            let keys = (0..input.len())
                .map(|i| {
                    let key = l.key.evaluate::<Goldilocks>(|t| match t {
                        RsqrtTerm::Input => Goldilocks::from_i64(input[i]),
                        RsqrtTerm::RowSelector => Goldilocks::from_u64_reduced(u64::from(real[i])),
                        RsqrtTerm::Column(c) => {
                            Goldilocks::from_i64(witness.tensor(c).expect("column")[i])
                        }
                    });
                    u32::try_from(key.to_canonical_u64()).expect("key")
                })
                .collect();
            (l.table, keys, None)
        })
        .collect();
    (columns, lookups)
}

/// Integer max and reciprocal-slack arithmetic, independent of SoftmaxRow::evaluate/witness.
pub fn softmax(
    selection: &Selection,
    node: &Node,
    scores: &[i64],
    valid: usize,
    exponentials: &[i64],
) -> (Columns, Lookups) {
    let Op::SoftmaxRow { sigma, exp_table } = node.op else {
        panic!("Softmax")
    };
    let Some((_, Gadget::SoftmaxRow { limb, .. })) = selection.of(node.id) else {
        panic!("Softmax selection")
    };
    let bits = |table| match selection
        .tables
        .iter()
        .find(|t| t.id == table)
        .expect("table")
        .kind
    {
        TableKind::Range { bits } => u32::from(bits),
        _ => panic!("range"),
    };
    let k0 = match selection
        .tables
        .iter()
        .find(|t| t.id == exp_table)
        .expect("Exp")
        .kind
    {
        TableKind::Exp { key_bits, .. } => (1u32 << key_bits) - 1,
        _ => panic!("Exp"),
    };
    let maximum = scores[..valid].iter().copied().max().unwrap_or(0);
    let sum: i64 = exponentials.iter().sum();
    let (r, a, b) = if valid == 0 {
        (0, 0, 0)
    } else {
        let mass = 1i64 << (2 * sigma);
        let r = mass / sum;
        let a = mass - r * sum;
        (r, a, sum - 1 - a)
    };
    assert!(a >= 0 && b >= 0);
    let requests = selection.reads(node.id).expect("requests");
    let count = (requests.len() - 2) / 2;
    let width = limb.map_or(u32::from(sigma) + 18, bits);
    let mut columns = vec![(AuxColumn::Maximum, vec![maximum])];
    let mut lookups = vec![
        (
            exp_table,
            scores
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if i < valid {
                        u32::try_from(maximum - x).expect("spread")
                    } else {
                        k0
                    }
                })
                .collect(),
            Some(exponentials.to_vec()),
        ),
        (
            requests[1],
            vec![u32::try_from(r).expect("reciprocal")],
            None,
        ),
    ];
    for (second, value) in [(false, a), (true, b)] {
        for j in 0..count {
            let table = requests[2 + usize::from(second) * count + j];
            let digit = (value as u64 >> (j as u32 * width)) & ((1u64 << bits(table)) - 1);
            lookups.push((table, vec![digit as u32], None));
            if j + 1 < count {
                columns.push((
                    AuxColumn::Slack {
                        second,
                        limb: j as u32,
                    },
                    vec![digit as i64],
                ));
            }
        }
    }
    (columns, lookups)
}
