// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The RoPE constants (RFC-0008 'Fixed-point generators'): for each position `p` and frequency
//! `i`, `(C, S) = (round(2^bits · cos(p·ω_i)), round(2^bits · sin(p·ω_i)))`, with `ω_i = θ^{−2i/d}`
//! scaled as Llama-3 scales it. They are the public constants of `Rotate`, not a lookup table.

use tessera_codec::Encode;
use tessera_field::{Goldilocks, PrimeField};
use tessera_hash::{Domain, Hasher};

use crate::fixed::{TAU, cos_sin, exp_neg, ln};
use crate::{TableError, TableId, TableKind, TableSpec};

/// The most pairs a RoPE table may hold: its core is all of them (RFC-0008 'Structured
/// evaluation').
const MAX_PAIRS: u64 = 1 << 24;

/// The constants of one RoPE table, every pair of which its digest binds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RopeTable {
    spec: TableSpec,
    rope: RopeConstants,
}

/// The constants of a RoPE kind computed on demand, without the digest: what the witness
/// engine's `Rotate` reads. Built from the scaled frequencies in Q64 once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RopeConstants {
    omegas: Vec<u128>,
    bits: u8,
    max_context: u32,
}

impl RopeTable {
    /// The constants of `kind`, a `Rope`, with index `id`: its parameters checked, its
    /// `log2_rows` and digest computed. The digest enumerates every pair, `max_context · d_head
    /// / 2` of them.
    ///
    /// # Errors
    ///
    /// [`TableError::Parameter`] for another kind or parameters RFC-0008 does not allow.
    pub fn new(id: TableId, kind: TableKind) -> Result<Self, TableError> {
        let rope = RopeConstants::new(&kind)?;
        let pairs = u64::from(rope.max_context) * rope.omegas.len() as u64;
        let mut table = Self {
            spec: TableSpec {
                id,
                kind,
                log2_rows: u8::try_from(u64::BITS - (pairs - 1).leading_zeros()).unwrap_or(0),
                digest: [0; 32],
            },
            rope,
        };
        let mut hasher = Hasher::new(Domain::Table);
        table.spec.id.encode_to(&mut hasher);
        table.spec.kind.encode_to(&mut hasher);
        table.spec.log2_rows.encode_to(&mut hasher);
        Vec::<u64>::new().encode_to(&mut hasher);
        (2 * pairs).encode_to(&mut hasher);
        for p in 0..table.rope.max_context {
            for omega in &table.rope.omegas {
                let (c, s) = table.rope.pair(p, *omega);
                Goldilocks::from_i64(c).encode_to(&mut hasher);
                Goldilocks::from_i64(s).encode_to(&mut hasher);
            }
        }
        table.spec.digest = hasher.finalize();
        Ok(table)
    }

    /// The table a claim list names, regenerated from its kind and id, as claim-list validation
    /// does (RFC-0007): the specification must be the one its kind generates.
    ///
    /// # Errors
    ///
    /// [`TableError::Parameter`], then [`TableError::Log2Rows`] and [`TableError::Digest`], as
    /// for a lookup table.
    pub fn from_spec(spec: &TableSpec) -> Result<Self, TableError> {
        let table = Self::new(spec.id, spec.kind.clone())?;
        if spec.log2_rows != table.spec.log2_rows {
            return Err(TableError::Log2Rows {
                stated: spec.log2_rows,
                expected: table.spec.log2_rows,
            });
        }
        if spec.digest != table.spec.digest {
            return Err(TableError::Digest);
        }
        Ok(table)
    }

    /// The specification, digest included.
    pub fn spec(&self) -> &TableSpec {
        &self.spec
    }

    /// The constants it holds.
    pub fn constants(&self) -> &RopeConstants {
        &self.rope
    }
}

impl RopeConstants {
    /// The constants of `kind`, a `Rope`, after checking its parameters: the scaled frequencies
    /// are computed once, the pairs on demand.
    ///
    /// # Errors
    ///
    /// [`TableError::Parameter`] for another kind or parameters RFC-0008 does not allow.
    pub fn new(kind: &TableKind) -> Result<Self, TableError> {
        let refuse = |rule| Err(TableError::Parameter { kind: "Rope", rule });
        let TableKind::Rope {
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
            bits,
            max_context,
            d_head,
        } = *kind
        else {
            return Err(TableError::Parameter {
                kind: "lookup",
                rule: "built as a GeneratedTable: only Rope is a RopeTable",
            });
        };
        if theta < 2 {
            return refuse("θ ≥ 2");
        }
        if factor < 1 || low_freq_factor < 1 || low_freq_factor >= high_freq_factor {
            return refuse("factor ≥ 1 and 1 ≤ low_freq_factor < high_freq_factor");
        }
        if original_max < 1 {
            return refuse("original_max ≥ 1");
        }
        if !(1..=30).contains(&bits) {
            return refuse("1 ≤ bits ≤ 30");
        }
        if d_head < 2 || d_head % 2 != 0 {
            return refuse("an even d_head of at least 2");
        }
        let pairs = u64::from(max_context) * u64::from(d_head / 2);
        if max_context < 1 || pairs > MAX_PAIRS {
            return refuse("1 ≤ max_context and max_context · d_head / 2 ≤ 2^24");
        }
        let ln_theta = ln(u64::from(theta));
        let (factor, low, high) = (
            u128::from(factor),
            u128::from(low_freq_factor),
            u128::from(high_freq_factor),
        );
        let omegas = (0..u128::from(d_head / 2))
            .map(|i| {
                let (y, n) = exp_neg(2 * i * ln_theta / u128::from(d_head));
                let omega = if n < 128 { y >> n } else { 0 };
                let l_omega = u128::from(original_max) * omega;
                if TAU * high < l_omega {
                    // A wavelength below original_max / high: kept.
                    omega
                } else if TAU * low > l_omega {
                    // A wavelength above original_max / low: slowed by the factor.
                    omega / factor
                } else {
                    // Between them, interpolated by `(L·ω/2π − low) / (high − low)` in Q40.
                    let smooth = ((l_omega - low * TAU) << 40) / ((high - low) * TAU);
                    omega / factor + ((omega * smooth) >> 40) * (factor - 1) / factor
                }
            })
            .collect();
        Ok(Self {
            omegas,
            bits,
            max_context,
        })
    }

    /// The scaled frequencies `ω_i` in Q64, one per frequency: the intermediate RFC-0008 pins,
    /// so that implementations can be compared before any rounding to `bits`.
    pub fn frequencies(&self) -> &[u128] {
        &self.omegas
    }

    /// `(C, S)` at `position` and `frequency`, or `None` past the table.
    pub fn at(&self, position: u32, frequency: u32) -> Option<(i64, i64)> {
        let omega = *self.omegas.get(usize::try_from(frequency).ok()?)?;
        (position < self.max_context).then(|| self.pair(position, omega))
    }

    /// `(C, S)` for the position `p` and the frequency `omega`, rounded half up.
    fn pair(&self, p: u32, omega: u128) -> (i64, i64) {
        let (c, s) = cos_sin(u128::from(p) * omega);
        let round = |v: i128| {
            let rounded = ((v << self.bits) + (1 << 61)) >> 62;
            // `|v| ≤ 2^62`, so the constant is at most `2^bits ≤ 2^30` in magnitude.
            i64::try_from(rounded).unwrap_or(0)
        };
        (round(c), round(s))
    }
}
