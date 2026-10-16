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
//! Sums of products ([`term_sums`]) take the row loop too, a block of 64 pairs at a time: each
//! table some term reads writes its line values into a row once, and the terms read the rows.
//! Register pressure then stays flat however many tables the terms read, where a zipped loop
//! would hold every table's line values at once: five tables at a fraction tree's level.
//!
//! Every loop walks its slices with `chunks_exact` and `zip`, without bounds checks. The sums are
//! taken over a range of pairs, so a parallel backend can split the range into chunks and add
//! their sums (RFC-0018). Control flow depends on lengths and on which tables are over the prime
//! subfield, never on a value (INV-ARCH-2).

use core::ops::Range;

use tessera_field::Field;
use tessera_poly::PolyField;

use crate::MAX_DEGREE;
use crate::terms::Plan;

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
    /// The term loop's rows, empty until a sum-of-products instance fits them.
    pub(crate) terms: TermScratch<F>,
}

impl<F: PolyField> Scratch<F> {
    /// Rows for one block, allocated once per proof.
    pub(crate) fn new() -> Self {
        Self {
            ext: vec![[F::ZERO; NODES]; BLOCK],
            base: vec![[F::Base::ZERO; NODES]; BLOCK],
            terms: TermScratch::new(),
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

/// Pairs per block of the term loop. Its rows hold `N` values per pair, at most 4 in a tree level,
/// so the line rows of five tables and three work rows take about 32 KB over `Fp2`.
const TERM_BLOCK: usize = 64;

/// A table's entries as the term loop reads them, unbound: in the prime subfield or in `F`.
pub(crate) enum Entries<'a, F: PolyField> {
    Base(&'a [F::Base]),
    Ext(&'a [F]),
}

impl<F: PolyField> Clone for Entries<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: PolyField> Copy for Entries<'_, F> {}

/// Tables addressed by position, whose values on the line through each pair the term loop writes
/// a block at a time: an instance's unbound tables, or its tables bound over `F`.
pub(crate) trait Lines<F: PolyField> {
    /// Writes table `k`'s values at the `N` nodes of [`line`] on each pair of `pairs`, `N` values
    /// per pair, into `out`, lifted to `F`.
    fn write<const N: usize, const SKIP_ONE: bool>(
        &self,
        k: usize,
        pairs: Range<usize>,
        out: &mut [F],
    );
}

/// Writes the values of each pair of `entries` at the `N` nodes, `N` per pair, into `out`.
fn write_lines<E: Field, F: From<E>, const N: usize, const SKIP_ONE: bool>(
    entries: &[E],
    out: &mut [F],
) {
    for (pair, row) in entries.chunks_exact(2).zip(out.chunks_exact_mut(N)) {
        each_node::<E, N, SKIP_ONE>(pair[0], pair[1], |m, value| row[m] = F::from(value));
    }
}

impl<F: PolyField> Lines<F> for [Entries<'_, F>] {
    fn write<const N: usize, const SKIP_ONE: bool>(
        &self,
        k: usize,
        pairs: Range<usize>,
        out: &mut [F],
    ) {
        match self[k] {
            Entries::Base(table) => {
                write_lines::<F::Base, F, N, SKIP_ONE>(at(&table, &pairs), out);
            }
            Entries::Ext(table) => write_lines::<F, F, N, SKIP_ONE>(at(&table, &pairs), out),
        }
    }
}

impl<F: PolyField> Lines<F> for [Vec<F>] {
    fn write<const N: usize, const SKIP_ONE: bool>(
        &self,
        k: usize,
        pairs: Range<usize>,
        out: &mut [F],
    ) {
        write_lines::<F, F, N, SKIP_ONE>(at(&self[k], &pairs), out);
    }
}

/// Rows for one block of the term loop, allocated once per proof: a line row per table that a
/// term reads, the common factors' product, a group's sum and a product beside it; and one sum
/// per group.
pub(crate) struct TermScratch<F> {
    lines: Vec<F>,
    common: Vec<F>,
    sum: Vec<F>,
    product: Vec<F>,
    groups: Vec<[F; NODES]>,
}

impl<F: Field> TermScratch<F> {
    /// No rows: a batch of products never allocates them.
    pub(crate) fn new() -> Self {
        Self {
            lines: Vec::new(),
            common: Vec::new(),
            sum: Vec::new(),
            product: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// Grows the rows to fit a plan of `lines` line slots and `groups` groups, before any round.
    pub(crate) fn fit(&mut self, lines: usize, groups: usize) {
        let row = TERM_BLOCK * NODES;
        if self.lines.len() < lines * row {
            self.lines.resize(lines * row, F::ZERO);
        }
        for rows in [&mut self.common, &mut self.sum, &mut self.product] {
            if rows.len() < row {
                rows.resize(row, F::ZERO);
            }
        }
        if self.groups.len() < groups {
            self.groups.resize(groups, [F::ZERO; NODES]);
        }
    }
}

/// `out[i] = a[i]·b[i]`.
fn product_into<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x * y;
    }
}

/// `out[i] += a[i]·b[i]`.
fn add_product<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o += x * y;
    }
}

/// `out[i] *= a[i]`.
fn multiply<F: Field>(out: &mut [F], a: &[F]) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o *= x;
    }
}

/// `out[i] += a[i]`.
fn add<F: Field>(out: &mut [F], a: &[F]) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o += x;
    }
}

/// The product of the rows `slots` (at least two) into `out`.
fn product_of<'r, F: Field>(out: &mut [F], line: impl Fn(usize) -> &'r [F], slots: &[usize]) {
    if let [first, second, rest @ ..] = slots {
        product_into(out, line(*first), line(*second));
        for &slot in rest {
            multiply(out, line(slot));
        }
    }
}

/// `acc[m] += Σ_t c[t][m]·r[t][m]` over the rows of `N` values, `c` absent meaning 1; `r` absent
/// meaning 1 too, the sum of `c` alone.
fn accumulate<F: Field, const N: usize>(acc: &mut [F; NODES], c: Option<&[F]>, r: Option<&[F]>) {
    match (c, r) {
        (Some(c), Some(r)) => {
            for (c, r) in c.chunks_exact(N).zip(r.chunks_exact(N)) {
                for m in 0..N {
                    acc[m] += c[m] * r[m];
                }
            }
        }
        (Some(row), None) | (None, Some(row)) => {
            for row in row.chunks_exact(N) {
                for m in 0..N {
                    acc[m] += row[m];
                }
            }
        }
        (None, None) => {}
    }
}

/// `Σ_t Σ_k w_k·Π_{f ∈ k} (a_f + z·Δ_f)` at the `N` nodes of [`line`], over the pairs `t` in
/// `pairs`, of the terms of an instance whose tables are `tables`, by its [`Plan`]; entries `N`
/// and after of the result are zero.
///
/// A block of pairs at a time: each table some term reads writes its line values into a row,
/// once, whatever the number of terms that read it; the common factors multiply into one row; each
/// group sums its terms' remaining products into a row and adds its dot product with the common
/// row into the group's sum. Each group's weight multiplies its sum once, at the end.
pub(crate) fn term_sums<F, L, const N: usize, const SKIP_ONE: bool>(
    tables: &L,
    plan: &Plan<F>,
    pairs: Range<usize>,
    scratch: &mut TermScratch<F>,
) -> [F; NODES]
where
    F: PolyField,
    L: Lines<F> + ?Sized,
{
    let TermScratch {
        lines,
        common,
        sum,
        product,
        groups,
    } = scratch;
    let groups = &mut groups[..plan.groups.len()];
    for acc in groups.iter_mut() {
        *acc = [F::ZERO; NODES];
    }
    let row = TERM_BLOCK * NODES;
    let end = pairs.end;
    for start in pairs.step_by(TERM_BLOCK) {
        let block = start..end.min(start.saturating_add(TERM_BLOCK));
        let len = block.len() * N;
        for (slot, &table) in plan.lines.iter().enumerate() {
            let out = &mut lines[slot * row..slot * row + len];
            tables.write::<N, SKIP_ONE>(table, block.clone(), out);
        }
        let lines = &*lines;
        let line = |slot: usize| &lines[slot * row..slot * row + len];
        let c: Option<&[F]> = match plan.common.as_slice() {
            [] => None,
            [only] => Some(line(*only)),
            slots => {
                let out = &mut common[..len];
                product_of(out, line, slots);
                Some(&*out)
            }
        };
        for (group, acc) in plan.groups.iter().zip(groups.iter_mut()) {
            let r: Option<&[F]> = match group.rests.as_slice() {
                [only] if only.is_empty() => None,
                [only] if only.len() == 1 => Some(line(only[0])),
                rests => {
                    let sum = &mut sum[..len];
                    for (k, rest) in rests.iter().enumerate() {
                        match (k, rest.as_slice()) {
                            (0, []) => sum.fill(F::ONE),
                            (0, [a]) => sum.copy_from_slice(line(*a)),
                            (0, slots) => product_of(sum, line, slots),
                            (_, []) => sum.iter_mut().for_each(|x| *x += F::ONE),
                            (_, [a]) => add(sum, line(*a)),
                            (_, [a, b]) => add_product(sum, line(*a), line(*b)),
                            (_, slots) => {
                                let product = &mut product[..len];
                                product_of(product, line, slots);
                                add(sum, product);
                            }
                        }
                    }
                    Some(&*sum)
                }
            };
            accumulate::<F, N>(acc, c, r);
        }
    }
    let mut sums = [F::ZERO; NODES];
    for (group, acc) in plan.groups.iter().zip(groups.iter()) {
        for (s, &a) in sums[..N].iter_mut().zip(&acc[..N]) {
            *s += group.weight * a;
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
