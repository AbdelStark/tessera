// SPDX-License-Identifier: Apache-2.0 OR MIT
use tessera_codec::SecretBuf;
use tessera_field::Field;
use tessera_poly::PolyField;
pub(crate) fn bind_base<F: PolyField>(table: &[F::Base], r: F, out: &mut SecretBuf<F>) {
    out.clear();
    out.extend(
        table
            .chunks_exact(2)
            .map(|pair| r.mul_by_base(pair[1] - pair[0]).add_base(pair[0])),
    );
}

/// `out[t] = a + r·(b − a)` for each pair of a borrowed table over `F`, into `out`, whose
/// capacity is reserved.
pub(crate) fn bind_ext<F: Field>(table: &[F], r: F, out: &mut SecretBuf<F>) {
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
pub(crate) fn bind_in_place<F: Field>(table: &mut SecretBuf<F>, r: F) {
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
