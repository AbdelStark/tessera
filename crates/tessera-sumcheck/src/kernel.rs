// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's inner loops: the values of a product of tables over pairs of entries, and
//! binding the lowest variable.
//!
//! A pair is `(T[2t], T[2t + 1])`: the lowest variable is the least significant bit of the index
//! (RFC-0010 'Multilinear extensions'). On the line through a pair the table is `a + z·Δ`, with
//! `a = T[2t]` and `Δ = T[2t + 1] − a`, so its values at consecutive nodes differ by `Δ` and come
//! by addition alone; the product across `k` tables costs `k − 1` multiplications per node.
//!
//! Three loops form the sums, chosen by the tables' number and field and the number of nodes,
//! after measuring each loop alone on one Apple M4 Max core (`benches/throughput.rs` measures the
//! whole prover):
//!
//! - one table: `Σ_t (a + z·Δ) = A + z·(B − A)`, so two sums with four accumulators each give
//!   every node, about four times faster than a sum per node, whose additions wait on each other;
//! - two to four tables in the prime subfield, or two or three over `F` at up to three nodes:
//!   one pass that zips the tables' pairs and keeps every product in registers, 20 to 45% faster
//!   than the rows below in the prime subfield;
//! - otherwise, and for instances that mix the two fields: a block of pairs at a time, each table
//!   multiplying its values into rows of [`Scratch`], which keeps register pressure flat as the
//!   degree grows.
//!
//! Every loop walks its slices with `chunks_exact` and `zip`, without bounds checks. The sums are
//! taken over a range of pairs, so a parallel backend can split the range into chunks and add
//! their sums (RFC-0018). Control flow depends on lengths and on which tables are over the prime
//! subfield, never on a value (INV-ARCH-2).

use core::ops::Range;

use tessera_field::Field;
use tessera_poly::PolyField;

use crate::MAX_DEGREE;

/// The most nodes a product is evaluated at: `0, 1, …, MAX_DEGREE`.
pub(crate) const NODES: usize = MAX_DEGREE + 1;

/// Pairs per block of the row loop. A block's rows, 128 bytes per pair over `Fp2` and 64 over
/// `Fp`, stay in the first-level cache while each table multiplies into them.
const BLOCK: usize = 128;

/// The most nodes at which tables over the extension field are zipped: beyond, the values they
/// keep in registers spill, and the row loop is faster.
const ZIP_EXT_NODES: usize = 3;

/// Rows of products for one block of pairs, in the field and in its prime subfield: entry `m` of
/// row `t` is the product, so far, of the tables' values at node `m` on pair `t`.
pub(crate) struct Scratch<F: PolyField> {
    ext: Vec<[F; NODES]>,
    base: Vec<[F::Base; NODES]>,
}

impl<F: PolyField> Scratch<F> {
    /// Rows for one block, allocated once per proof.
    pub(crate) fn new() -> Self {
        Self {
            ext: vec![[F::ZERO; NODES]; BLOCK],
            base: vec![[F::Base::ZERO; NODES]; BLOCK],
        }
    }
}

/// The values of `a + z·(b − a)` at the `N` nodes, in entries `0..N`: `0, 2, 3, …, N` when
/// `SKIP_ONE` (a round message, which never carries `p(1)`), `0, 1, …, N − 1` otherwise. Every
/// value after the first two comes by adding `Δ = b − a`.
#[inline(always)]
fn line<E: Field, const N: usize, const SKIP_ONE: bool>(a: E, b: E) -> [E; N] {
    let delta = b - a;
    let mut values = [a; N];
    let mut value = if SKIP_ONE { b + delta } else { b };
    for slot in values.iter_mut().skip(1) {
        *slot = value;
        value += delta;
    }
    values
}

/// `Σ_t` of a single table's values at the nodes: with `A = Σ_t a` and `B = Σ_t b`, the sum at
/// node `z` is `A + z·(B − A)`. The two sums take four accumulators each, so that consecutive
/// additions do not wait on each other.
fn single<E: Field, const N: usize, const SKIP_ONE: bool>(table: &[E]) -> [E; NODES] {
    let mut a = [E::ZERO; 4];
    let mut b = [E::ZERO; 4];
    let mut quads = table.chunks_exact(8);
    for q in &mut quads {
        a[0] += q[0];
        b[0] += q[1];
        a[1] += q[2];
        b[1] += q[3];
        a[2] += q[4];
        b[2] += q[5];
        a[3] += q[6];
        b[3] += q[7];
    }
    for pair in quads.remainder().chunks_exact(2) {
        a[0] += pair[0];
        b[0] += pair[1];
    }
    let values =
        line::<E, N, SKIP_ONE>((a[0] + a[1]) + (a[2] + a[3]), (b[0] + b[1]) + (b[2] + b[3]));
    let mut sums = [E::ZERO; NODES];
    sums[..N].copy_from_slice(&values);
    sums
}

/// The left-nested pattern of a chain of zips, `((p0, p1), p2)`, binding each table's pair.
macro_rules! nested {
    ($p0:ident) => { $p0 };
    ($p0:ident, $p1:ident $(, $rest:ident)*) => { nested!(@acc ($p0, $p1) $(, $rest)*) };
    (@acc $acc:tt) => { $acc };
    (@acc $acc:tt, $next:ident $(, $rest:ident)*) => { nested!(@acc ($acc, $next) $(, $rest)*) };
}

/// A loop over the pairs of a fixed number of tables, zipped, with the product at every node in
/// registers.
macro_rules! zipped {
    ($(#[$doc:meta])* $name:ident: $t0:ident $(, $t:ident)+) => {
        $(#[$doc])*
        fn $name<E: Field, const N: usize, const SKIP_ONE: bool>(
            $t0: &[E],
            $($t: &[E],)+
        ) -> [E; NODES] {
            let mut sums = [E::ZERO; NODES];
            for nested!($t0 $(, $t)+) in $t0.chunks_exact(2)$(.zip($t.chunks_exact(2)))+ {
                let mut product = line::<E, N, SKIP_ONE>($t0[0], $t0[1]);
                $(
                    let values = line::<E, N, SKIP_ONE>($t[0], $t[1]);
                    for (p, v) in product.iter_mut().zip(values) {
                        *p *= v;
                    }
                )+
                for (sum, p) in sums.iter_mut().zip(product) {
                    *sum += p;
                }
            }
            sums
        }
    };
}

zipped!(
    /// `Σ_t` of the product of two tables at the nodes.
    zip2: t0, t1
);
zipped!(
    /// `Σ_t` of the product of three tables at the nodes.
    zip3: t0, t1, t2
);
zipped!(
    /// `Σ_t` of the product of four tables at the nodes.
    zip4: t0, t1, t2, t3
);

/// Calls `at(m, value)` with the value of `a + z·(b − a)` at each of the `N` nodes of [`line`],
/// so that a row loop writes each value where it goes.
#[inline(always)]
fn each_node<E: Field, const N: usize, const SKIP_ONE: bool>(
    a: E,
    b: E,
    mut at: impl FnMut(usize, E),
) {
    let delta = b - a;
    at(0, a);
    let mut value = if SKIP_ONE { b + delta } else { b };
    for m in 1..N {
        at(m, value);
        value += delta;
    }
}

/// Writes the first table's values on each pair of a block into its row.
fn first_factor<E: Field, const N: usize, const SKIP_ONE: bool>(
    pairs: &[E],
    rows: &mut [[E; NODES]],
) {
    for (pair, row) in pairs.chunks_exact(2).zip(rows) {
        each_node::<E, N, SKIP_ONE>(pair[0], pair[1], |m, value| row[m] = value);
    }
}

/// Multiplies another table's values on each pair of a block into its row.
fn next_factor<E: Field, const N: usize, const SKIP_ONE: bool>(
    pairs: &[E],
    rows: &mut [[E; NODES]],
) {
    for (pair, row) in pairs.chunks_exact(2).zip(rows) {
        each_node::<E, N, SKIP_ONE>(pair[0], pair[1], |m, value| row[m] *= value);
    }
}

/// The product of `tables` on the pairs whose entries are `entries`, into `rows`.
fn block_product<E: Field, T: AsRef<[E]>, const N: usize, const SKIP_ONE: bool>(
    tables: &[T],
    entries: Range<usize>,
    rows: &mut [[E; NODES]],
) {
    if let Some((first, rest)) = tables.split_first() {
        first_factor::<E, N, SKIP_ONE>(&first.as_ref()[entries.clone()], rows);
        for table in rest {
            next_factor::<E, N, SKIP_ONE>(&table.as_ref()[entries.clone()], rows);
        }
    }
}

/// The entries of a table's pairs `pairs`.
fn at<'t, E, T: AsRef<[E]>>(table: &'t T, pairs: &Range<usize>) -> &'t [E] {
    &table.as_ref()[2 * pairs.start..2 * pairs.end]
}

/// The block boundaries of a range of pairs.
fn blocks(pairs: Range<usize>) -> impl Iterator<Item = Range<usize>> {
    let end = pairs.end;
    pairs
        .step_by(BLOCK)
        .map(move |start| start..end.min(start.saturating_add(BLOCK)))
}

/// `Σ_t Π_k` of tables of one field at the nodes, by rows.
fn rows_sums<E: Field, T: AsRef<[E]>, const N: usize, const SKIP_ONE: bool>(
    tables: &[T],
    pairs: Range<usize>,
    rows: &mut [[E; NODES]],
) -> [E; NODES] {
    let mut sums = [E::ZERO; NODES];
    if tables.is_empty() {
        return sums;
    }
    for block in blocks(pairs) {
        let rows = &mut rows[..block.len()];
        block_product::<E, T, N, SKIP_ONE>(tables, 2 * block.start..2 * block.end, rows);
        for row in rows.iter() {
            for (sum, &value) in sums[..N].iter_mut().zip(&row[..N]) {
                *sum += value;
            }
        }
    }
    sums
}

/// `Σ_t Π_k` of tables of one field at the nodes, by the loop the module documentation picks;
/// `zip_nodes` is the most nodes at which this field's tables are zipped.
fn homogeneous_sums<E: Field, T: AsRef<[E]>, const N: usize, const SKIP_ONE: bool>(
    tables: &[T],
    pairs: Range<usize>,
    rows: &mut [[E; NODES]],
    zip_nodes: usize,
) -> [E; NODES] {
    let zip = N <= zip_nodes;
    let p = &pairs;
    match tables {
        [t0] => single::<E, N, SKIP_ONE>(at(t0, p)),
        [t0, t1] if zip => zip2::<E, N, SKIP_ONE>(at(t0, p), at(t1, p)),
        [t0, t1, t2] if zip => zip3::<E, N, SKIP_ONE>(at(t0, p), at(t1, p), at(t2, p)),
        [t0, t1, t2, t3] if zip => {
            zip4::<E, N, SKIP_ONE>(at(t0, p), at(t1, p), at(t2, p), at(t3, p))
        }
        _ => rows_sums::<E, T, N, SKIP_ONE>(tables, pairs, rows),
    }
}

/// `Σ_t Π_k (a_k + z·Δ_k)` at the `N` nodes of [`line`], over the pairs `t` in `pairs`, of the
/// product of the prime-subfield tables `base` and the tables `ext`, entries `N` and after of the
/// result being zero; zero for no table.
///
/// Tables of one field alone are summed in that field, prime-subfield sums lifted once at the
/// end. Beside extension tables, the prime-subfield tables multiply among themselves in the prime
/// subfield, and each pair's product enters the extension product by
/// [`PolyField::mul_by_base`]. Every table has at least `2·pairs.end` entries.
pub(crate) fn product_sums<F, B, X, const N: usize, const SKIP_ONE: bool>(
    base: &[B],
    ext: &[X],
    pairs: Range<usize>,
    scratch: &mut Scratch<F>,
) -> [F; NODES]
where
    F: PolyField,
    B: AsRef<[F::Base]>,
    X: AsRef<[F]>,
{
    if ext.is_empty() {
        return homogeneous_sums::<F::Base, B, N, SKIP_ONE>(base, pairs, &mut scratch.base, NODES)
            .map(F::from);
    }
    if base.is_empty() {
        return homogeneous_sums::<F, X, N, SKIP_ONE>(ext, pairs, &mut scratch.ext, ZIP_EXT_NODES);
    }
    let mut sums = [F::ZERO; NODES];
    for block in blocks(pairs) {
        let entries = 2 * block.start..2 * block.end;
        let base_rows = &mut scratch.base[..block.len()];
        let ext_rows = &mut scratch.ext[..block.len()];
        block_product::<_, _, N, SKIP_ONE>(base, entries.clone(), base_rows);
        block_product::<_, _, N, SKIP_ONE>(ext, entries, ext_rows);
        for (ext_row, base_row) in ext_rows.iter().zip(base_rows.iter()) {
            for ((sum, x), &b) in sums[..N].iter_mut().zip(&ext_row[..N]).zip(&base_row[..N]) {
                *sum += x.mul_by_base(b);
            }
        }
    }
    sums
}

/// `out[t] = a + r·(b − a)` for each pair of a prime-subfield table, into `out`, whose capacity
/// is reserved: the first binding of a committed table, two base multiplications per pair
/// ([`PolyField::mul_by_base`]).
pub(crate) fn bind_base<F: PolyField>(table: &[F::Base], r: F, out: &mut Vec<F>) {
    out.clear();
    out.extend(
        table
            .chunks_exact(2)
            .map(|pair| r.mul_by_base(pair[1] - pair[0]).add_base(pair[0])),
    );
}

/// `out[t] = a + r·(b − a)` for each pair of a borrowed table over `F`, into `out`, whose
/// capacity is reserved.
pub(crate) fn bind_ext<F: Field>(table: &[F], r: F, out: &mut Vec<F>) {
    out.clear();
    out.extend(
        table
            .chunks_exact(2)
            .map(|pair| pair[0] + r * (pair[1] - pair[0])),
    );
}

/// Binds the lowest variable of `table` to `r` in place, keeping its allocation: entry `t`
/// becomes `T[2t] + r·(T[2t + 1] − T[2t])` and the table halves.
///
/// Ascending `t` is safe in place. After `t = 0`, the outputs are written in ranges `[s, 2s)` for
/// `s = 1, 2, 4, …`: the inputs of a range, `[2s, 4s)`, lie past every output written so far, and
/// its outputs were the inputs of the range before, already read. The split borrow makes each
/// range a plain zip, as in `tessera_poly::bind_lowest_in_place`.
pub(crate) fn bind_in_place<F: Field>(table: &mut Vec<F>, r: F) {
    let half = table.len() / 2;
    if let [a, b, ..] = table[..] {
        table[0] = a + r * (b - a);
    }
    let mut start = 1;
    while start < half {
        let end = (2 * start).min(half);
        let (outputs, inputs) = table.split_at_mut(2 * start);
        for (out, pair) in outputs[start..end].iter_mut().zip(inputs.chunks_exact(2)) {
            *out = pair[0] + r * (pair[1] - pair[0]);
        }
        start = end;
    }
    table.truncate(half);
}
