// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tables a relation builds from its operands before its sumcheck: `eq` tables, and contractions
//! of an operand over some of its variables (RFC-0010 'Relations', RFC-0012 'Opening by the
//! folding pass', step 2).
//!
//! An operand over the prime subfield, flattened row-major, is contracted with the weights
//! `eq(r, ·)` of a point over `F`: each entry costs one product of an element of `F` by a base
//! element ([`PolyField::mul_by_base`]: two base multiplications in a quadratic extension) and
//! one addition in `F`. A contraction reads the operand once, in order, and writes a table `2^m`
//! times smaller, `m` the variables it sums out.
//!
//! - **Middle variables** ([`contract_middle`]): `out[o·I + i] = Σ_m e[m]·T[(o·M + m)·I + i]`,
//!   the operand a stack of `M × I` matrices whose rows are summed with the weights `e`, such as a
//!   weight matrix's output rows or an activation matrix's tokens.
//! - **Low variables** ([`contract_low`]): `out[k] = Σ_c e[c]·T[k·C + c]`, each row of `C` entries
//!   summed with the weights, such as the columns of `V` in `PV`.
//!
//! The loops were chosen by measuring several orders alone on one Apple M4 Max core, in two runs,
//! on operands of `2^14` to `2^24` entries shaped from rows of one entry to rows of `2^20`.
//! Orders that read the operand in sequence and differ only in how often they touch the output,
//! once per row, per 2 or 4 rows or per block of columns, all take about the same time, and those
//! that keep 4 independent sums in registers take less: the additions' dependency chains, not the
//! reads, set the pace. The fastest on wide rows also read the operand in at most 8 streams:
//!
//! - middle variables: panels of 8 rows, 4 columns at a time, the 4 sums run down the panel and
//!   are added to the output once per panel: 1.1 to 1.4 ns per entry, the fastest order on rows
//!   of `2^14` entries or more and within 20% of the fastest on every shape. A pass over the
//!   output per row, the plain order, takes 1.5 to 1.8 ns, and 3.7 to 3.9 ns on rows of 2
//!   entries; two or four rows per pass, or blocks of 2048 columns, 1.4 to 2.0 ns. Sums down every
//!   row of a layer, column by column, take 1.1 to 1.5 ns while the layer stays in cache and 2.0
//!   to 3.7 ns once it does not. Panels of 16 or 32 rows take 1.0 to 1.3 ns on rows of up to
//!   `2^10` entries, but 2.2 to 2.7 ns on rows of `2^14` or more. Rows of fewer than 4 entries
//!   take the sums down every row of a layer, 1.1 ns on rows of 2.
//! - low variables: 4 rows at a time, zipped, one sum per row: 0.9 to 1.2 ns per entry for rows
//!   of 1 to `2^12` entries, within 6% of the fastest order on every shape. One row at a time
//!   takes 1.1 to 2.6 ns with four sums and 1.2 to 3.6 ns with one, and the transposed order, a
//!   pass over the output per column, 1.0 to 6.5 ns.
//!
//! Control flow depends on lengths only, never on a value (INV-ARCH-2). Operands and their
//! contractions are secret-class values (05 §3) that nothing here shows.

use core::alloc::Layout;

use tessera_field::Field;
use tessera_poly::{PolyField, eq_table};

/// Rows of a panel of the middle contraction. Panels of 16 or 32 rows measure up to 15% faster on
/// rows of up to `2^10` entries, but about twice as slow on rows of `2^14` or more, as weight
/// matrices' rows are.
const PANEL: usize = 8;

/// Columns summed at a time down a panel of the middle contraction, and rows at a time of the
/// low contraction: 4 independent sums in registers.
const GROUP: usize = 4;

/// `eq(point, ·)` on the point's `2^n` vertices ([`eq_table`]). `eq_table` fails only when it
/// cannot allocate its table, and every other allocation of the prover aborts the process when it
/// fails, as `Vec`'s do: so does this one.
pub(crate) fn eq_weights<F: Field>(point: &[F]) -> Vec<F> {
    eq_table(point).unwrap_or_else(|_| {
        let layout = u32::try_from(point.len())
            .ok()
            .and_then(|vars| 1usize.checked_shl(vars))
            .and_then(|len| Layout::array::<F>(len).ok())
            .unwrap_or_else(Layout::new::<F>);
        std::alloc::handle_alloc_error(layout)
    })
}

/// `eq(point, ·)` on the high variables of a table whose `low` lowest variables it does not
/// depend on: entry `y·2^low + x` is `eq(point, y)`.
pub(crate) fn eq_high<F: Field>(point: &[F], low: usize) -> Vec<F> {
    let weights = eq_weights(point);
    let width = 1usize << low;
    let mut table = Vec::with_capacity(weights.len() * width);
    for &e in &weights {
        table.extend(core::iter::repeat_n(e, width));
    }
    table
}

/// Contracts the middle variables of `table` with `weights`: with `M = weights.len()` and
/// `I = inner`, `out[o·I + i] = Σ_m weights[m]·table[(o·M + m)·I + i]` for every `o` below
/// `table.len() / (M·I)`.
///
/// `M` and `I` are powers of two and `M·I` divides `table.len()`, as the shapes of the caller
/// ensure.
pub(crate) fn contract_middle<F: PolyField>(
    table: &[F::Base],
    weights: &[F],
    inner: usize,
) -> Vec<F> {
    let layer = weights.len() * inner;
    let mut out = Vec::with_capacity(table.len() / weights.len());
    if inner < GROUP {
        for layer in table.chunks_exact(layer) {
            narrow_columns(layer, weights, inner, &mut out);
        }
        return out;
    }
    out.resize(table.len() / weights.len(), F::ZERO);
    let rows = PANEL.min(weights.len());
    for (layer, sums) in table.chunks_exact(layer).zip(out.chunks_exact_mut(inner)) {
        for (panel, weights) in layer
            .chunks_exact(rows * inner)
            .zip(weights.chunks_exact(rows))
        {
            add_panel(panel, weights, inner, sums);
        }
    }
    out
}

/// Adds `Σ_m weights[m]·panel[m·I + i]` into `sums[i]` for each column `i` of a panel of
/// `weights.len() ≤ PANEL` rows of `I = sums.len()` entries, `I` a multiple of `GROUP`: the
/// `GROUP` sums of a group of columns run down the panel in registers.
fn add_panel<F: PolyField>(panel: &[F::Base], weights: &[F], inner: usize, sums: &mut [F]) {
    let mut rows = panel.chunks_exact(inner);
    let mut groups: [core::slice::ChunksExact<'_, F::Base>; PANEL] =
        core::array::from_fn(|_| rows.next().unwrap_or(&[]).chunks_exact(GROUP));
    for out in sums.chunks_exact_mut(GROUP) {
        let mut column = [F::ZERO; GROUP];
        for (group, &e) in groups.iter_mut().zip(weights) {
            if let Some(entries) = group.next() {
                for (sum, &t) in column.iter_mut().zip(entries) {
                    *sum += e.mul_by_base(t);
                }
            }
        }
        for (out, sum) in out.iter_mut().zip(column) {
            *out += sum;
        }
    }
}

/// Appends the `inner < GROUP` column sums of one layer of `weights.len()` rows: `GROUP` sums
/// run over the layer `GROUP` entries at a time, sum `k` taking column `k mod inner`.
fn narrow_columns<F: PolyField>(layer: &[F::Base], weights: &[F], inner: usize, out: &mut Vec<F>) {
    let mut sums = [F::ZERO; GROUP];
    let mut chunks = layer.chunks_exact(GROUP);
    for (chunk, rows) in (&mut chunks).zip(weights.chunks_exact(GROUP / inner)) {
        for (k, (sum, &t)) in sums.iter_mut().zip(chunk).enumerate() {
            if let Some(&e) = rows.get(k / inner) {
                *sum += e.mul_by_base(t);
            }
        }
    }
    // A layer of fewer than `GROUP` entries: every row here, one at a time.
    for (row, &e) in chunks.remainder().chunks_exact(inner).zip(weights) {
        for (sum, &t) in sums.iter_mut().zip(row) {
            *sum += e.mul_by_base(t);
        }
    }
    for column in 0..inner {
        out.push(
            sums.iter()
                .skip(column)
                .step_by(inner)
                .fold(F::ZERO, |total, &sum| total + sum),
        );
    }
}

/// Contracts the low variables of `table` with `weights`: with `C = weights.len()`,
/// `out[k] = Σ_c weights[c]·table[k·C + c]` for every `k` below `table.len() / C`.
///
/// `C` is a power of two that divides `table.len()`, as the shapes of the caller ensure.
pub(crate) fn contract_low<F: PolyField>(table: &[F::Base], weights: &[F]) -> Vec<F> {
    let width = weights.len();
    let mut out = Vec::with_capacity(table.len() / width);
    let mut groups = table.chunks_exact(GROUP * width);
    for group in &mut groups {
        let (r01, r23) = group.split_at(2 * width);
        let (r0, r1) = r01.split_at(width);
        let (r2, r3) = r23.split_at(width);
        let mut sums = [F::ZERO; GROUP];
        for ((((&e, &t0), &t1), &t2), &t3) in weights.iter().zip(r0).zip(r1).zip(r2).zip(r3) {
            sums[0] += e.mul_by_base(t0);
            sums[1] += e.mul_by_base(t1);
            sums[2] += e.mul_by_base(t2);
            sums[3] += e.mul_by_base(t3);
        }
        out.extend(sums);
    }
    // A table of fewer than `GROUP` rows: each row alone.
    for row in groups.remainder().chunks_exact(width) {
        out.push(
            row.iter()
                .zip(weights)
                .fold(F::ZERO, |sum, (&t, &e)| sum + e.mul_by_base(t)),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    //! The contractions against their definitions on every small shape, which reaches each
    //! remainder of the loops: layers of fewer than `GROUP` entries, rows narrower than `GROUP`,
    //! panels of fewer than `PANEL` rows, and tables of fewer than `GROUP` rows.

    use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};

    use super::{contract_low, contract_middle};

    type Fp = Goldilocks;
    type Fp2 = GoldilocksExt2;

    fn random(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *seed >> 1
    }

    fn fp2(seed: &mut u64) -> Fp2 {
        let c0 = Fp::from_u64_reduced(random(seed));
        Fp2::new(c0, Fp::from_u64_reduced(random(seed)))
    }

    #[test]
    fn contractions_match_their_definitions() {
        let mut seed = 0x57_0001;
        for outer in 0..=3 {
            for mid in 0..=5 {
                for inner in 0..=4 {
                    let (o, m, i) = (1 << outer, 1 << mid, 1 << inner);
                    let table: Vec<Fp> = (0..o * m * i)
                        .map(|_| Fp::from_u64_reduced(random(&mut seed)))
                        .collect();
                    let weights: Vec<Fp2> = (0..m).map(|_| fp2(&mut seed)).collect();
                    let expected: Vec<Fp2> = (0..o * i)
                        .map(|k| {
                            let (layer, column) = (k / i, k % i);
                            (0..m).fold(Fp2::ZERO, |sum, row| {
                                sum + weights[row]
                                    * Fp2::from(table[(layer * m + row) * i + column])
                            })
                        })
                        .collect();
                    let shape = format!("2^{outer} x 2^{mid} x 2^{inner}");
                    assert!(
                        contract_middle(&table, &weights, i) == expected,
                        "middle, {shape}"
                    );
                    // The same table as 2^(outer + mid) rows of 2^inner entries.
                    let weights: Vec<Fp2> = (0..i).map(|_| fp2(&mut seed)).collect();
                    let expected: Vec<Fp2> = table
                        .chunks_exact(i)
                        .map(|row| {
                            row.iter()
                                .zip(&weights)
                                .fold(Fp2::ZERO, |sum, (&t, &e)| sum + e * Fp2::from(t))
                        })
                        .collect();
                    assert!(contract_low(&table, &weights) == expected, "low, {shape}");
                }
            }
        }
    }
}
