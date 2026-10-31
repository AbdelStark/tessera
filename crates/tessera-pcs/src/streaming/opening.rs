// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Regenerated advice openings (RFC-0011): original-claim batching, P4 self-check and P5 answers.
//!
//! The caller supplies public points mapped by the compiler, and replays the same per-proof mask
//! stream in P1, P4 and P5 (RFC-0018). P4 returns no message until its sampled column matches P1
//! and the completed combinations agree with that column. The driver absorbs those combined
//! rows before drawing P5's queries. Protocol/2 blinding and the full driver remain separate.

mod memory;
pub use memory::{AdviceColumnMemory, AdviceCommitmentMemory, AdviceOpeningMemory};

use core::fmt;

use tessera_codec::{SecretBuf, Zeroize, error::ErrorCode};
use tessera_field::{Field, Goldilocks as Fp, GoldilocksExt2 as Fp2, PrimeField, batch_inverse};
use tessera_hash::{Digest, Domain, Hasher, MerkleTree};

use super::{AdviceCommitment, Rows, SegmentDigests, check_parts, merged_column};
use crate::opening::{COMBINE_SCRATCH, add_scaled_eq, combine_into};
use crate::{
    AdviceLayout, ClaimGroup, ColumnAnswer, CombinedRow, MaskSource, PcsError, RowClaim, RowEncoder,
};

/// One weighted, full advice point obtained from a public tensor/layer/tile image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdvicePoint {
    /// The full `nu` coordinates, LSB first, including the tensor's fixed offset bits.
    pub point: Vec<Fp2>,
    /// The piece, actual layer and image weight; batching is applied once per original claim.
    pub weight: Fp2,
}

/// One original combination obligation, before its images are batched (RFC-0037).
/// Its value is unnecessary to the prover's opening pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaim {
    /// All weighted images of this original claim, sharing one column sub-point; nonempty.
    pub pieces: Vec<AdvicePoint>,
}

/// Why a regenerated opening failed. No variant carries a value, mask or codeword.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdviceOpeningError {
    /// Invalid public arguments or an unaddressable buffer.
    Invalid(PcsError),
    /// P4's regenerated leaf or completed combinations disagree (`ERR-INTERNAL`).
    SelfCheck,
    /// The operating system's random source failed (`ERR-RES-IO`).
    Randomness,
}

impl AdviceOpeningError {
    /// The stable outcome code (04 §3); no message is returned on an error.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Invalid(PcsError::TooLarge { .. }) => ErrorCode::ResMemory,
            Self::Invalid(_) => ErrorCode::Usage,
            Self::SelfCheck => ErrorCode::Internal,
            Self::Randomness => ErrorCode::ResIo,
        }
    }
}
impl From<PcsError> for AdviceOpeningError {
    fn from(error: PcsError) -> Self {
        Self::Invalid(error)
    }
}
impl fmt::Display for AdviceOpeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => error.fmt(f),
            Self::SelfCheck => f.write_str("ERR-INTERNAL: advice opening self-check failed"),
            Self::Randomness => f.write_str("ERR-RES-IO: the random source failed"),
        }
    }
}
impl std::error::Error for AdviceOpeningError {}

type Result<T> = core::result::Result<T, AdviceOpeningError>;
fn invalid(rule: &'static str) -> AdviceOpeningError {
    PcsError::LayoutData { rule }.into()
}
fn capacity(rows: u64, row_len: usize) -> AdviceOpeningError {
    PcsError::TooLarge { rows, row_len }.into()
}
fn public_capacity<T>(len: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| capacity(len as u64, 1))?;
    Ok(out)
}
fn copied<T: Copy>(values: &[T]) -> Result<Vec<T>> {
    let mut out = public_capacity(values.len())?;
    out.extend_from_slice(values);
    Ok(out)
}
fn public_zeros<T: Copy + Default>(len: usize) -> Result<Vec<T>> {
    let mut out = public_capacity(len)?;
    out.resize(len, T::default());
    Ok(out)
}

fn zeros<T: Copy + Default>(len: usize) -> Result<SecretBuf<T>> {
    let mut out = SecretBuf::try_with_capacity(len).map_err(|_| capacity(len as u64, 1))?;
    out.resize(len, T::default());
    Ok(out)
}

/// Public P1 opening metadata: canonical context and a Merkle tree, never a witness/codeword matrix.
/// The root-only [`super::merge_segments`] keeps its original zero-heap implementation.
pub struct AdviceOpening<'layout> {
    layout: std::borrow::Cow<'layout, AdviceLayout>,
    params: crate::CommitmentParams,
    tree: MerkleTree,
    sample: usize,
    leaf: Digest,
}
impl fmt::Debug for AdviceOpening<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdviceOpening")
            .field("columns", &self.tree.leaves())
            .field("rows", &self.layout.rows())
            .finish_non_exhaustive()
    }
}
impl<'layout> AdviceOpening<'layout> {
    /// Builds the public tree from the complete ordered P1 segment digests and samples the
    /// private self-check column uniformly from the OS random source. It retains no segment chains.
    ///
    /// # Errors
    /// The context errors of [`super::merge_segments`], `Randomness`, or an allocation refusal.
    pub fn new(parts: &[SegmentDigests<'layout>]) -> Result<Self> {
        Self::build(parts, |bytes| {
            getrandom::fill(bytes).map_err(|_| AdviceOpeningError::Randomness)
        })
    }

    fn build(
        parts: &[SegmentDigests<'layout>],
        draw: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        let first = check_parts(parts)?;
        if first.params.log2_inv_rate != 1 {
            return Err(invalid("advice openings use rate 1/2 (RFC-0011)"));
        }
        let n = first.digests.len();
        let mut bytes = [0; 4];
        draw(&mut bytes)?;
        // A checked codeword has a power-of-two length <= 2^32: masking is exactly uniform.
        let sample = (u32::from_le_bytes(bytes) as usize) & (n - 1);
        let size = n
            .checked_mul(2)
            .and_then(|x| x.checked_sub(1))
            .ok_or_else(|| capacity(2, n))?;
        let mut leaves = Vec::new();
        leaves.try_reserve_exact(size).map_err(|_| capacity(2, n))?;
        let mut leaf = [0; 32];
        for column in 0..n {
            let digest = merged_column(parts, column);
            if column == sample {
                leaf = digest;
            }
            leaves.push(digest);
        }
        let tree = MerkleTree::new(leaves)
            .map_err(|_| invalid("the P1 tree has a checked power-of-two length"))?;
        Ok(Self {
            layout: std::borrow::Cow::Borrowed(first.layout),
            params: first.params,
            tree,
            sample,
            leaf,
        })
    }

    /// Own the public layout so a CPU session can retain P1 across network rounds.
    /// Secret witness rows and codewords are never copied into this metadata.
    pub fn into_owned(self) -> AdviceOpening<'static> {
        AdviceOpening {
            layout: std::borrow::Cow::Owned(self.layout.into_owned()),
            params: self.params,
            tree: self.tree,
            sample: self.sample,
            leaf: self.leaf,
        }
    }

    /// Start P5 with the opaque permit released by the successful P4 of this same P1.
    ///
    /// # Errors
    /// A permit for another commitment, invalid queries, an encoder mismatch or allocation failure.
    pub fn begin_columns<'state, 'work>(
        &'state self,
        permit: &ColumnPermit,
        encoder: &'work RowEncoder,
        masks: &'work mut dyn MaskSource,
        queries: &[u64],
    ) -> Result<ColumnPass<'work, 'state, 'layout>> {
        if permit.commitment != self.commitment() {
            return Err(invalid("P5 uses the P1 whose P4 self-check passed"));
        }
        check_encoder(self, encoder)?;
        if queries.is_empty() {
            return Err(PcsError::NoQueries.into());
        }
        if let Some(position) = queries
            .iter()
            .position(|&q| q >= encoder.codeword_len() as u64)
        {
            return Err(PcsError::QueryRange {
                position,
                query: queries[position],
                columns: encoder.codeword_len(),
            }
            .into());
        }
        let mut distinct = copied(queries)?;
        distinct.sort_unstable();
        distinct.dedup();
        let n = self.layout.rows();
        let count = usize::try_from(n).map_err(|_| capacity(n, distinct.len()))?;
        count
            .checked_mul(distinct.len())
            .filter(|&len| len <= isize::MAX as usize / size_of::<Fp>())
            .ok_or_else(|| capacity(n, distinct.len()))?;
        let mut columns = public_capacity(distinct.len())?;
        for _ in &distinct {
            columns.push(
                SecretBuf::try_with_capacity(count).map_err(|_| capacity(n, distinct.len()))?,
            );
        }
        Ok(ColumnPass {
            state: self,
            encoder,
            masks,
            cursor: Cursor::new(&self.layout),
            codeword: zeros(encoder.codeword_len())?,
            distinct,
            columns,
        })
    }

    /// The exact Round1 commitment represented by this tree.
    pub fn commitment(&self) -> AdviceCommitment {
        AdviceCommitment {
            root: self.tree.root(),
            layout_digest: self.params.commitment_id,
        }
    }

    /// Groups original claims in first-appearance order and preserves original order within each
    /// group. One gamma power covers every weighted piece; powers restart at gamma^1 per group.
    /// The caller fixes original claims before drawing gamma (RFC-0014).
    ///
    /// # Errors
    /// Wrong full-point or proximity arity, empty combinations, mixed column points, or capacity.
    pub fn prepare<'s>(
        &'s self,
        claims: &[AdviceClaim],
        gamma: Fp2,
        xi: &[Fp2],
    ) -> Result<OpeningGroups<'s, 'layout>> {
        let k_bits = self.params.log2_k as usize;
        let variables = self.layout.variables() as usize;
        let row_vars = self.layout.row_vars() as usize;
        if xi.len() != row_vars {
            return Err(PcsError::XiPoint {
                expected: row_vars,
                found: xi.len(),
            }
            .into());
        }
        let mut groups: Vec<ClaimGroup> = public_capacity(claims.len())?;
        let mut powers = public_capacity(claims.len())?;
        for original in claims {
            let first = original
                .pieces
                .first()
                .ok_or_else(|| invalid("an original advice combination has at least one image"))?;
            for piece in &original.pieces {
                if piece.point.len() != variables {
                    return Err(PcsError::ClaimPoint {
                        expected: variables,
                        found: piece.point.len(),
                    }
                    .into());
                }
                if piece.point[..k_bits] != first.point[..k_bits] {
                    return Err(invalid(
                        "an original combination shares one column sub-point",
                    ));
                }
            }
            let point = &first.point[..k_bits];
            let index = match groups.iter().position(|g| g.column_point == point) {
                Some(index) => index,
                None => {
                    groups.push(ClaimGroup {
                        column_point: copied(point)?,
                        claims: Vec::new(),
                    });
                    powers.push(Fp2::ONE);
                    groups.len() - 1
                }
            };
            powers[index] *= gamma;
            groups[index]
                .claims
                .try_reserve_exact(original.pieces.len())
                .map_err(|_| capacity(original.pieces.len() as u64, variables))?;
            for piece in &original.pieces {
                groups[index].claims.push(RowClaim {
                    row_point: copied(&piece.point[k_bits..])?,
                    gamma: powers[index] * piece.weight,
                });
            }
        }
        Ok(OpeningGroups {
            state: self,
            groups,
            xi: copied(xi)?,
        })
    }
}

/// Batched public coefficients bound to one P1 state, with no dense row-coefficient matrix.
pub struct OpeningGroups<'state, 'layout> {
    state: &'state AdviceOpening<'layout>,
    groups: Vec<ClaimGroup>,
    xi: Vec<Fp2>,
}
impl<'s, 'l> OpeningGroups<'s, 'l> {
    /// The number of derived column-point groups, excluding proximity.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Starts P4 with a fresh replay of P1's mask stream. Encoder/context/size checks precede draws.
    ///
    /// # Errors
    /// Incompatible encoder dimensions or a buffer that cannot be allocated.
    pub fn begin<'g, 'work>(
        &'g self,
        encoder: &'work RowEncoder,
        masks: &'work mut dyn MaskSource,
    ) -> Result<OpeningPass<'g, 'work, 's, 'l>> {
        OpeningPass::new(self, encoder, masks)
    }
}

struct Cursor<'layout> {
    layout: &'layout AdviceLayout,
    segment: usize,
    rows: Rows<'layout>,
}
impl<'a> Cursor<'a> {
    fn new(layout: &'a AdviceLayout) -> Self {
        Self {
            layout,
            segment: 0,
            rows: layout.segments()[0]
                .nonzero
                .iter()
                .cloned()
                .flatten()
                .peekable(),
        }
    }
    fn expected(&mut self) -> Option<(usize, u64)> {
        if self.segment >= self.layout.segments().len() {
            return None;
        }
        loop {
            if let Some(row) = self.rows.peek() {
                return Some((self.segment, *row));
            }
            self.segment += 1;
            let part = self.layout.segments().get(self.segment)?;
            self.rows = part.nonzero.iter().cloned().flatten().peekable();
        }
    }
    fn check(&mut self, row: u64) -> Result<usize> {
        let expected = self.expected();
        if expected.map(|(_, r)| r) != Some(row) {
            return Err(PcsError::StreamRow {
                segment: self.segment as u32,
                expected: expected.map(|(_, r)| r),
                found: Some(row),
            }
            .into());
        }
        Ok(self.segment)
    }
    fn finish(&mut self) -> Result<()> {
        if let Some((segment, row)) = self.expected() {
            return Err(PcsError::StreamRow {
                segment: segment as u32,
                expected: Some(row),
                found: None,
            }
            .into());
        }
        Ok(())
    }
}

fn check_encoder(state: &AdviceOpening<'_>, encoder: &RowEncoder) -> Result<()> {
    let shape = state.params.shape()?;
    if encoder.message_len() != shape.k
        || encoder.codeword_len() != shape.n
        || encoder.mask_coeffs() != shape.mask_coeffs
    {
        return Err(invalid("the opening encoder uses P1's k, n and P"));
    }
    Ok(())
}
fn init(state: &AdviceOpening<'_>) -> Digest {
    Hasher::new(Domain::ColumnInit)
        .update(&state.params.commitment_id)
        .update(&(state.sample as u64).to_le_bytes())
        .finalize()
}

/// P4's private block and accumulators. Dropping an incomplete or failed pass clears every
/// derived allocation; only [`Self::finish`] can return combined-row messages (OBS-6).
pub struct OpeningPass<'groups, 'work, 'state, 'layout> {
    groups: &'groups OpeningGroups<'state, 'layout>,
    encoder: &'work RowEncoder,
    masks: &'work mut dyn MaskSource,
    cursor: Cursor<'state>,
    values: SecretBuf<Fp>,
    mask_values: SecretBuf<Fp>,
    codeword: SecretBuf<Fp>,
    samples: SecretBuf<Fp>,
    numbers: Vec<u64>,
    u: Vec<SecretBuf<Fp2>>,
    m: Vec<SecretBuf<Fp2>>,
    sums: SecretBuf<Fp2>,
    block: usize,
    segment: usize,
    chain: Digest,
    merge: Hasher,
}
impl<'g, 'w, 's, 'l> OpeningPass<'g, 'w, 's, 'l> {
    fn new(
        groups: &'g OpeningGroups<'s, 'l>,
        encoder: &'w RowEncoder,
        masks: &'w mut dyn MaskSource,
    ) -> Result<Self> {
        let state = groups.state;
        check_encoder(state, encoder)?;
        let k = encoder.message_len();
        let p = encoder.mask_coeffs();
        let count = groups
            .groups
            .len()
            .checked_add(1)
            .ok_or_else(|| capacity(u64::MAX, k))?;
        let rows =
            usize::try_from(state.layout.rows()).map_err(|_| capacity(state.layout.rows(), k))?;
        let b = state.params.row_block.get().min(rows);
        let value_size = b.checked_mul(k).ok_or_else(|| capacity(b as u64, k))?;
        let mask_size = b.checked_mul(p).ok_or_else(|| capacity(b as u64, p))?;
        count
            .checked_mul(k.checked_add(p).ok_or_else(|| capacity(count as u64, k))?)
            .filter(|&len| len <= isize::MAX as usize / size_of::<Fp2>())
            .ok_or_else(|| capacity(count as u64, k))?;
        count
            .checked_mul(b)
            .filter(|&len| len <= isize::MAX as usize / size_of::<Fp2>())
            .ok_or_else(|| capacity(count as u64, b))?;
        count
            .checked_mul(COMBINE_SCRATCH)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| capacity(count as u64, COMBINE_SCRATCH))?;
        let mut u = public_capacity(count)?;
        let mut m = public_capacity(count)?;
        for _ in 0..count {
            u.push(zeros(k)?);
            m.push(zeros(p)?);
        }
        Ok(Self {
            groups,
            encoder,
            masks,
            cursor: Cursor::new(&state.layout),
            values: zeros(value_size)?,
            mask_values: zeros(mask_size)?,
            codeword: zeros(encoder.codeword_len())?,
            samples: zeros(b)?,
            numbers: public_capacity(b)?,
            u,
            m,
            sums: zeros(count)?,
            block: b,
            segment: 0,
            chain: init(state),
            merge: Hasher::new(Domain::ColumnMerge),
        })
    }

    /// Adds the next canonical retained row. Wrong order/length draws no mask and consumes no row.
    /// Padded values remain committed when the public layout retains their row.
    ///
    /// # Errors
    /// `StreamRow` or `RowLength` for public shape/order, otherwise the encoder's error.
    pub fn push_row(&mut self, row: u64, values: &[Fp]) -> Result<()> {
        let segment = self.cursor.check(row)?;
        let k = self.encoder.message_len();
        if values.len() != k {
            return Err(PcsError::RowLength {
                expected: k,
                found: values.len(),
            }
            .into());
        }
        while self.segment < segment {
            self.end_segment();
        }
        let r = self.numbers.len();
        let p = self.encoder.mask_coeffs();
        let own = &mut self.mask_values[r * p..(r + 1) * p];
        let mut next = 0;
        let mut capture = || {
            let x = self.masks.draw();
            own[next] = x;
            next += 1;
            x
        };
        self.encoder
            .encode_row(values, &mut capture, &mut self.codeword)?;
        self.values[r * k..(r + 1) * k].copy_from_slice(values);
        self.samples[r] = self.codeword[self.groups.state.sample];
        self.numbers.push(row);
        self.cursor.rows.next();
        if self.numbers.len() == self.block {
            self.flush();
        }
        Ok(())
    }

    fn flush(&mut self) {
        let len = self.numbers.len();
        if len == 0 {
            return;
        }
        let mut coefficients = Vec::with_capacity(self.groups.groups.len() + 1);
        for group in &self.groups.groups {
            let mut row = vec![Fp2::ZERO; len];
            for claim in &group.claims {
                add_scaled_eq(
                    &mut row,
                    self.numbers.iter().copied(),
                    &claim.row_point,
                    claim.gamma,
                );
            }
            coefficients.push(row);
        }
        let mut xi = vec![Fp2::ZERO; len];
        add_scaled_eq(
            &mut xi,
            self.numbers.iter().copied(),
            &self.groups.xi,
            Fp2::ONE,
        );
        coefficients.push(xi);
        let k = self.encoder.message_len();
        let p = self.encoder.mask_coeffs();
        combine_into(&coefficients, &self.values[..len * k], k, |g, i, x| {
            self.u[g][i] += x
        });
        combine_into(&coefficients, &self.mask_values[..len * p], p, |g, i, x| {
            self.m[g][i] += x
        });
        for (sum, weights) in self.sums.iter_mut().zip(&coefficients) {
            for (&weight, &symbol) in weights.iter().zip(&self.samples[..len]) {
                *sum += weight * Fp2::from(symbol);
            }
        }
        let mut hash = Hasher::new(Domain::ColumnStep);
        hash.update(&self.chain);
        for symbol in &self.samples[..len] {
            hash.update(&symbol.to_canonical_u64().to_le_bytes());
        }
        self.chain = hash.finalize();
        self.values.iter_mut().for_each(Zeroize::zeroize);
        self.mask_values.iter_mut().for_each(Zeroize::zeroize);
        self.samples.iter_mut().for_each(Zeroize::zeroize);
        self.numbers.clear();
    }
    fn end_segment(&mut self) {
        self.flush();
        self.merge.update(&self.chain);
        self.chain = init(self.groups.state);
        self.segment += 1;
    }

    /// Completes P4: checks every row arrived, the sampled merged leaf, and each combined row
    /// against its sampled-column sum, before declassifying any combined row as a message.
    ///
    /// # Errors
    /// `StreamRow` for missing rows, `SelfCheck` (`ERR-INTERNAL`) for a regeneration or
    /// accumulation fault. Every private buffer clears on return, including errors.
    pub fn finish(mut self) -> Result<CheckedOpening<'s, 'l>> {
        self.cursor.finish()?;
        while self.segment < self.groups.state.layout.segments().len() {
            self.end_segment();
        }
        if self.merge.finalize() != self.groups.state.leaf {
            return Err(AdviceOpeningError::SelfCheck);
        }
        let (weights, x, vanishing) = sample_weights(self.encoder, self.groups.state.sample)?;
        for ((values, masks), &expected) in self.u.iter().zip(&self.m).zip(self.sums.iter()) {
            let value = values
                .iter()
                .zip(&weights)
                .fold(Fp2::ZERO, |sum, (&v, &w)| sum + v * Fp2::from(w));
            let mask = masks
                .iter()
                .rev()
                .fold(Fp2::ZERO, |sum, &v| sum * Fp2::from(x) + v);
            if value + mask * Fp2::from(vanishing) != expected {
                return Err(AdviceOpeningError::SelfCheck);
            }
        }
        // Allocate every output while it still contains only zeros. No fallible operation follows
        // copying private accumulators into the protocol message (OBS-6).
        let mut combined = public_capacity(self.u.len())?;
        for _ in &self.u {
            combined.push(CombinedRow {
                u: public_zeros(self.encoder.message_len())?,
                mask: public_zeros(self.encoder.mask_coeffs())?,
            });
        }
        for ((row, u), m) in combined.iter_mut().zip(&self.u).zip(&self.m) {
            row.u.copy_from_slice(u);
            row.mask.copy_from_slice(m);
        }
        Ok(CheckedOpening {
            state: self.groups.state,
            combined,
        })
    }
}

// RFC-0018: w_i = omega_k^i * Z_H(x) / (k*(x-omega_k^i)), all public metadata.
fn sample_weights(encoder: &RowEncoder, column: usize) -> Result<(Vec<Fp>, Fp, Fp)> {
    let k = encoder.message_len();
    let n = encoder.codeword_len();
    let x = Fp::from_u64_reduced(7) * Fp::two_adic_generator(n.ilog2()).pow(column as u64);
    let omega = Fp::two_adic_generator(k.ilog2());
    let vanishing = x.pow(k as u64) - Fp::ONE;
    let mut power = Fp::ONE;
    let mut weights: Vec<_> = (0..k)
        .map(|_| {
            let d = x - power;
            power *= omega;
            d
        })
        .collect();
    batch_inverse(&mut weights).map_err(|_| AdviceOpeningError::SelfCheck)?;
    let factor = vanishing
        * Fp::from_u64_reduced(k as u64)
            .inverse()
            .ok_or(AdviceOpeningError::SelfCheck)?;
    power = Fp::ONE;
    for weight in &mut weights {
        *weight *= factor * power;
        power *= omega;
    }
    Ok((weights, x, vanishing))
}

/// Opaque P4 completion marker, bound to its P1 commitment for a later session phase.
#[must_use = "answer only the late queries of this completed opening"]
pub struct ColumnPermit {
    commitment: AdviceCommitment,
}

/// Combined rows released only after P4's self-check, bound to their P1 tree.
#[must_use = "absorb the combined rows before drawing column queries, then answer those queries"]
pub struct CheckedOpening<'state, 'layout> {
    state: &'state AdviceOpening<'layout>,
    combined: Vec<CombinedRow>,
}
impl<'s, 'l> CheckedOpening<'s, 'l> {
    /// The checked combined-row message, in group order followed by proximity.
    pub fn combined(&self) -> &[CombinedRow] {
        &self.combined
    }

    /// Release the checked rows and an owned P5 permit without cloning secret or public matrices.
    pub fn into_parts(self) -> (Vec<CombinedRow>, ColumnPermit) {
        (
            self.combined,
            ColumnPermit {
                commitment: self.state.commitment(),
            },
        )
    }

    /// Starts P5 after the driver has absorbed the combined rows and drawn queries.
    /// Repeated/unsorted draws are answered as their sorted distinct set.
    ///
    /// # Errors
    /// Encoder mismatch, `NoQueries`, `QueryRange`, or output capacity refusal, before any draw.
    pub fn answer<'work>(
        &self,
        encoder: &'work RowEncoder,
        masks: &'work mut dyn MaskSource,
        queries: &[u64],
    ) -> Result<ColumnPass<'work, 's, 'l>> {
        self.state.begin_columns(
            &ColumnPermit {
                commitment: self.state.commitment(),
            },
            encoder,
            masks,
            queries,
        )
    }
}

/// P5's regenerated columns. It retains one codeword and the eventual output columns, never
/// the witness or codeword matrix; dropping an incomplete pass clears partial output columns.
pub struct ColumnPass<'work, 'state, 'layout> {
    state: &'state AdviceOpening<'layout>,
    encoder: &'work RowEncoder,
    masks: &'work mut dyn MaskSource,
    cursor: Cursor<'state>,
    codeword: SecretBuf<Fp>,
    distinct: Vec<u64>,
    columns: Vec<SecretBuf<Fp>>,
}
impl ColumnPass<'_, '_, '_> {
    /// Encodes the next retained row with the replayed P1 mask stream and keeps just queried symbols.
    ///
    /// # Errors
    /// `StreamRow` before the encoder's `RowLength`; refused rows consume no mask or cursor entry.
    pub fn push_row(&mut self, row: u64, values: &[Fp]) -> Result<()> {
        self.cursor.check(row)?;
        self.encoder
            .encode_row(values, self.masks, &mut self.codeword)?;
        for (column, &index) in self.columns.iter_mut().zip(&self.distinct) {
            column.push(self.codeword[index as usize]);
        }
        self.cursor.rows.next();
        Ok(())
    }

    /// Returns every retained symbol, in segment order, and the P1 tree's canonical multiproof.
    ///
    /// # Errors
    /// `StreamRow` for missing rows. No partial answer is returned on failure.
    pub fn finish(mut self) -> Result<ColumnAnswer> {
        self.cursor.finish()?;
        let proof = self
            .state
            .tree
            .open(&self.distinct)
            .map_err(|_| invalid("checked distinct queries index the P1 tree"))?;
        let mut columns = public_capacity(self.columns.len())?;
        for column in &self.columns {
            columns.push(public_zeros(column.len())?);
        }
        for (out, column) in columns.iter_mut().zip(&self.columns) {
            out.copy_from_slice(column);
        }
        Ok(ColumnAnswer { columns, proof })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Axis, DenseShape, LayoutParams, Padding, StreamingCommitter, TensorShape, TileShape,
    };
    use core::num::NonZeroUsize;
    use std::cell::Cell;
    use tessera_field::QuadraticExtension;

    fn fixture() -> (AdviceLayout, crate::CommitmentParams, RowEncoder) {
        let layout = AdviceLayout::new(
            LayoutParams {
                log2_k: 3,
                segments: 2,
                layers: 1,
                group: 1,
                query_block: 1,
                max_context: 1,
            },
            1,
            &[TileShape {
                layer: Some(0),
                values: vec![TensorShape::Dense(DenseShape::new(
                    vec![Axis::new(13)],
                    Padding::Zero,
                ))],
            }],
        )
        .unwrap();
        let params = crate::CommitmentParams {
            log2_k: 3,
            log2_inv_rate: 1,
            mask_coeffs: 3,
            row_block: NonZeroUsize::new(3).unwrap(),
            commitment_id: [37; 32],
        };
        (layout, params, RowEncoder::new(8, 1, 3).unwrap())
    }
    fn parts<'a>(
        layout: &'a AdviceLayout,
        params: crate::CommitmentParams,
        encoder: &RowEncoder,
    ) -> Vec<SegmentDigests<'a>> {
        let mut mask = || Fp::from_u64_reduced(7);
        layout
            .segments()
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut stream =
                    StreamingCommitter::new(params, layout, i as u32, encoder, &mut mask).unwrap();
                for row in s.nonzero.iter().cloned().flatten() {
                    stream
                        .push_row(row, &[Fp::from_u64_reduced(row + 3); 8])
                        .unwrap();
                }
                stream.finish().unwrap()
            })
            .collect()
    }

    #[test]
    fn commitment_memory_bounds_encoder_segments_block_tree_and_owned_layout() {
        use allocation_counter::measure;
        let (_, base, _) = fixture();
        for (bits, segments, rows) in [
            (0, 1, 0),
            (0, 1, 2),
            (3, 1, 2),
            (6, 2, 1),
            (6, 3, 10),
            (10, 5, 3),
        ] {
            let layout = AdviceLayout::new(
                LayoutParams {
                    log2_k: bits,
                    segments,
                    layers: 2,
                    group: 1,
                    query_block: 1,
                    max_context: 2,
                },
                2,
                &(0..2)
                    .map(|layer| TileShape {
                        layer: Some(layer),
                        values: vec![TensorShape::Dense(DenseShape::new(
                            vec![
                                {
                                    let real = if rows == 0 {
                                        0
                                    } else {
                                        rows + 4 * u64::from(layer)
                                    };
                                    Axis {
                                        len: real.max(1),
                                        real,
                                    }
                                },
                                Axis::new(1 << bits),
                            ],
                            Padding::Zero,
                        ))],
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let params = crate::CommitmentParams {
                log2_k: bits,
                mask_coeffs: base.mask_coeffs.min(1 << bits),
                ..base
            };
            let mut plan = None;
            let planning =
                measure(|| plan = Some(AdviceCommitmentMemory::new(&layout, params).unwrap()));
            assert_eq!(planning.count_total, 0);
            let plan = plan.unwrap();
            let mut encoder = None;
            let construction = measure(|| {
                encoder = Some(RowEncoder::new(1 << bits, 1, params.mask_coeffs).unwrap());
            });
            assert_eq!(
                u64::try_from(construction.bytes_current).unwrap(),
                plan.encoder_bytes()
            );
            assert!(construction.bytes_max <= plan.encoder_peak_bytes());
            let encoder_bytes = u64::try_from(construction.bytes_current).unwrap();
            let encoder = encoder.unwrap();
            let mut row_values = vec![Fp::ZERO; encoder.message_len()];
            let mut masks = || Fp::from_u64_reduced(7);
            let mut parts = Vec::new();
            let committing = measure(|| {
                parts.try_reserve_exact(segments as usize).unwrap();
                for segment in 0..segments {
                    let mut pass =
                        StreamingCommitter::new(params, &layout, segment, &encoder, &mut masks)
                            .unwrap();
                    for row in layout.segments()[segment as usize]
                        .nonzero
                        .iter()
                        .cloned()
                        .flatten()
                    {
                        row_values.fill(Fp::from_u64_reduced(row + 3));
                        pass.push_row(row, &row_values).unwrap();
                    }
                    parts.push(pass.finish().unwrap());
                }
            });
            let segments_bytes = u64::try_from(committing.bytes_current).unwrap();
            assert_eq!(segments_bytes, plan.segments_bytes());
            assert!(
                encoder_bytes + committing.bytes_max <= plan.commit_peak_bytes(),
                "P1 commit: {committing:?} {plan:?}"
            );
            let mut owned = None;
            let merging = measure(|| {
                owned = Some(
                    AdviceOpening::build(&parts, |out| {
                        out.fill(0);
                        Ok(())
                    })
                    .unwrap()
                    .into_owned(),
                );
            });
            assert!(
                encoder_bytes + segments_bytes + merging.bytes_max <= plan.merge_peak_bytes(),
                "P1 merge: {merging:?} {plan:?}"
            );
            assert!(u64::try_from(merging.bytes_current).unwrap() <= plan.retained_bytes());
            assert!(
                construction
                    .bytes_max
                    .max(encoder_bytes + committing.bytes_max)
                    .max(encoder_bytes + segments_bytes + merging.bytes_max)
                    <= plan.peak_bytes()
            );
            assert_eq!(owned.unwrap().tree.leaves(), encoder.codeword_len());
        }
    }

    #[test]
    fn commitment_memory_refuses_invalid_or_overflowing_geometry_without_allocating() {
        let (layout, params, _) = fixture();
        for bad in [
            crate::CommitmentParams {
                log2_k: 4,
                ..params
            },
            crate::CommitmentParams {
                log2_inv_rate: 2,
                ..params
            },
        ] {
            let mut result = None;
            let info = allocation_counter::measure(|| {
                result = Some(AdviceCommitmentMemory::new(&layout, bad))
            });
            assert_eq!(info.count_total, 0);
            assert!(matches!(
                result.unwrap(),
                Err(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
            ));
        }
        #[cfg(target_pointer_width = "64")]
        {
            let layout = AdviceLayout::new(
                LayoutParams {
                    log2_k: 31,
                    segments: 1,
                    layers: 1,
                    group: 1,
                    query_block: 1,
                    max_context: 2,
                },
                2,
                &[TileShape {
                    layer: Some(0),
                    values: vec![TensorShape::Dense(DenseShape::new(
                        vec![Axis::new(1u64 << 32), Axis::new(1u64 << 31)],
                        Padding::Zero,
                    ))],
                }],
            )
            .unwrap();
            let params = crate::CommitmentParams {
                log2_k: 31,
                log2_inv_rate: 1,
                mask_coeffs: 1,
                row_block: NonZeroUsize::new(usize::MAX).unwrap(),
                ..params
            };
            let mut result = None;
            let info = allocation_counter::measure(|| {
                result = Some(AdviceCommitmentMemory::new(&layout, params))
            });
            assert_eq!(info.count_total, 0);
            assert_eq!(result.unwrap().unwrap_err().code(), ErrorCode::ResMemory);
        }
    }

    #[test]
    fn column_memory_bounds_private_public_columns_and_multiproof() {
        use allocation_counter::measure;
        let small = fixture();
        let make = |bits, shape: Vec<Axis>| {
            let layout = AdviceLayout::new(
                LayoutParams {
                    log2_k: bits,
                    ..*small.0.params()
                },
                1,
                &[TileShape {
                    layer: Some(0),
                    values: vec![TensorShape::Dense(DenseShape::new(shape, Padding::Zero))],
                }],
            )
            .unwrap();
            let params = crate::CommitmentParams {
                log2_k: bits,
                ..small.1
            };
            (
                layout,
                params,
                RowEncoder::new(1 << bits, 1, params.mask_coeffs).unwrap(),
            )
        };
        let medium = make(6, vec![Axis::new(9), Axis::new(64)]);
        let large = make(10, vec![Axis::new(13)]);
        for (layout, params, encoder) in [small, medium, large] {
            let mut row_values = vec![Fp::ZERO; encoder.message_len()];
            let mut masks = || Fp::from_u64_reduced(7);
            let parts = layout
                .segments()
                .iter()
                .enumerate()
                .map(|(segment, part)| {
                    let mut committer = StreamingCommitter::new(
                        params,
                        &layout,
                        segment as u32,
                        &encoder,
                        &mut masks,
                    )
                    .unwrap();
                    for row in part.nonzero.iter().cloned().flatten() {
                        row_values.fill(Fp::from_u64_reduced(row + 3));
                        committer.push_row(row, &row_values).unwrap();
                    }
                    committer.finish().unwrap()
                })
                .collect::<Vec<_>>();
            let state = AdviceOpening::build(&parts, |out| {
                out.fill(0);
                Ok(())
            })
            .unwrap();
            let groups = state
                .prepare(&[], Fp2::ONE, &vec![Fp2::ONE; layout.row_vars() as usize])
                .unwrap();
            let mut p4 = groups.begin(&encoder, &mut masks).unwrap();
            for segment in layout.segments() {
                for row in segment.nonzero.iter().cloned().flatten() {
                    row_values.fill(Fp::from_u64_reduced(row + 3));
                    p4.push_row(row, &row_values).unwrap();
                }
            }
            let (_, permit) = p4.finish().unwrap().into_parts();
            let n = encoder.codeword_len() as u64;
            let cases = [
                vec![0],
                vec![n - 1],
                vec![n - 1, 0, n - 1, 1],
                (0..193).map(|i| (i * 337 + 17) % n).collect(),
                vec![0; n as usize + 7],
                (0..n).collect(),
            ];
            for queries in cases {
                let mut plan = None;
                let planning = measure(|| {
                    plan = Some(
                        AdviceColumnMemory::new(&layout, params, queries.len() as u64).unwrap(),
                    )
                });
                assert_eq!(planning.count_total, 0);
                let plan = plan.unwrap();
                let mut pass = None;
                let borrowed_masks = &mut masks;
                let starting = measure(|| {
                    let masks = borrowed_masks;
                    pass = Some(
                        state
                            .begin_columns(&permit, &encoder, masks, &queries)
                            .unwrap(),
                    );
                });
                let private = u64::try_from(starting.bytes_current).unwrap();
                assert!(
                    starting.bytes_max <= plan.private_pass_bytes(),
                    "P5 construct: {starting:?} {plan:?}"
                );
                let mut pass = pass.unwrap();
                let mut answer = None;
                let finishing = measure(|| {
                    for segment in layout.segments() {
                        for row in segment.nonzero.iter().cloned().flatten() {
                            row_values.fill(Fp::from_u64_reduced(row + 3));
                            pass.push_row(row, &row_values).unwrap();
                        }
                    }
                    answer = Some(pass.finish().unwrap());
                });
                assert!(
                    private + finishing.bytes_max <= plan.peak_bytes(),
                    "P5 finish: {private} {finishing:?} {plan:?}"
                );
                let retained =
                    u64::try_from(starting.bytes_current + finishing.bytes_current).unwrap();
                assert!(
                    retained <= plan.output_bytes(),
                    "P5 output: {retained} {plan:?}"
                );
                let answer = answer.unwrap();
                assert!(
                    (answer.proof.siblings.capacity() * size_of::<Digest>()) as u64
                        <= plan.proof_bytes()
                );
                let mut distinct = queries.clone();
                distinct.sort_unstable();
                distinct.dedup();
                assert_eq!(answer.columns.len(), distinct.len());
                assert!(
                    answer
                        .columns
                        .iter()
                        .all(|column| column.len() == layout.rows() as usize)
                );
            }
        }
    }

    #[test]
    fn column_memory_invalid_and_unaddressable_arguments_allocate_nothing() {
        let (layout, params, _) = fixture();
        for (queries, code) in [(0, ErrorCode::Usage), (u64::MAX, ErrorCode::ResMemory)] {
            let mut result = None;
            let info = allocation_counter::measure(|| {
                result = Some(AdviceColumnMemory::new(&layout, params, queries))
            });
            assert_eq!(info.count_total, 0);
            assert_eq!(result.unwrap().unwrap_err().code(), code);
        }
        for other in [
            crate::CommitmentParams {
                log2_inv_rate: 2,
                ..params
            },
            crate::CommitmentParams {
                log2_k: 4,
                ..params
            },
        ] {
            assert!(matches!(
                AdviceColumnMemory::new(&layout, other, 1),
                Err(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
            ));
        }
    }

    #[test]
    fn opening_memory_bounds_groups_private_buffers_and_publication_overlap() {
        use allocation_counter::measure;
        let small = fixture();
        let large_layout = AdviceLayout::new(
            LayoutParams {
                log2_k: 10,
                ..*small.0.params()
            },
            1,
            &[TileShape {
                layer: Some(0),
                values: vec![TensorShape::Dense(DenseShape::new(
                    vec![Axis::new(13)],
                    Padding::Zero,
                ))],
            }],
        )
        .unwrap();
        let large_params = crate::CommitmentParams {
            log2_k: 10,
            ..small.1
        };
        let large = (
            large_layout,
            large_params,
            RowEncoder::new(1024, 1, 3).unwrap(),
        );
        for (layout, params, encoder) in [small, large] {
            let mut row_values = vec![Fp::ZERO; encoder.message_len()];
            let mut masks = || Fp::from_u64_reduced(7);
            let parts = layout
                .segments()
                .iter()
                .enumerate()
                .map(|(segment, part)| {
                    let mut committer = StreamingCommitter::new(
                        params,
                        &layout,
                        segment as u32,
                        &encoder,
                        &mut masks,
                    )
                    .unwrap();
                    for row in part.nonzero.iter().cloned().flatten() {
                        row_values.fill(Fp::from_u64_reduced(row + 3));
                        committer.push_row(row, &row_values).unwrap();
                    }
                    committer.finish().unwrap()
                })
                .collect::<Vec<_>>();
            let state = AdviceOpening::build(&parts, |out| {
                out.fill(0);
                Ok(())
            })
            .unwrap();
            let variables = layout.variables() as usize;
            let k_bits = params.log2_k as usize;
            let gamma = Fp2::new(Fp::from_u64_reduced(3), Fp::from_u64_reduced(5));
            let xi = vec![gamma; layout.row_vars() as usize];
            for (originals, pieces, distinct) in
                [(0, 0, 0), (1, 1, 1), (1024, 1, 1), (67, 3, 8), (513, 2, 64)]
            {
                let claims = (0..originals)
                    .map(|c| AdviceClaim {
                        pieces: (0..pieces)
                            .map(|image| {
                                let column =
                                    Fp2::from(Fp::from_u64_reduced((c % distinct + 1) as u64));
                                let mut point = vec![column; variables];
                                point[k_bits..]
                                    .fill(Fp2::from(Fp::from_u64_reduced((image + 2) as u64)));
                                AdvicePoint {
                                    point,
                                    weight: if image % 2 == 0 { Fp2::ONE } else { Fp2::ZERO },
                                }
                            })
                            .collect(),
                    })
                    .collect::<Vec<_>>();
                let mut plan = None;
                let planning = measure(|| {
                    plan = Some(
                        AdviceOpeningMemory::new(
                            &layout,
                            params,
                            originals as u64,
                            (originals * pieces) as u64,
                            distinct as u64,
                        )
                        .unwrap(),
                    )
                });
                assert_eq!(
                    planning.count_total, 0,
                    "planning must not allocate payload"
                );
                let plan = plan.unwrap();
                let mut groups = None;
                let preparing =
                    measure(|| groups = Some(state.prepare(&claims, gamma, &xi).unwrap()));
                let groups = groups.unwrap();
                assert_eq!(groups.group_count(), distinct);
                let retained = u64::try_from(preparing.bytes_current).unwrap();
                assert!(
                    preparing.bytes_max <= plan.prepare_peak_bytes(),
                    "prepare {originals}/{pieces}/{distinct}: {preparing:?} {plan:?}"
                );
                assert!(
                    retained <= plan.groups_bytes(),
                    "retained groups: {preparing:?} {plan:?}"
                );
                let mut checked = None;
                let opening = measure(|| {
                    let mut masks = || Fp::from_u64_reduced(7);
                    let mut pass = groups.begin(&encoder, &mut masks).unwrap();
                    for segment in layout.segments() {
                        for row in segment.nonzero.iter().cloned().flatten() {
                            row_values.fill(Fp::from_u64_reduced(row + 3));
                            pass.push_row(row, &row_values).unwrap();
                        }
                    }
                    checked = Some(pass.finish().unwrap());
                });
                assert!(
                    retained + opening.bytes_max <= plan.peak_bytes(),
                    "P4 {originals}/{pieces}/{distinct}: {opening:?} {plan:?}"
                );
                assert!(
                    u64::try_from(opening.bytes_current).unwrap() <= plan.output_bytes(),
                    "public rows: {opening:?} {plan:?}"
                );
                assert_eq!(checked.unwrap().combined().len(), distinct + 1);
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn opening_memory_plans_production_width_without_allocating_large_buffers() {
        use allocation_counter::measure;
        // Public geometry from the two-position1B diagnostic. This synthetic layout
        // matches its row width/arity; it is not a real-model proof or a whole phase bound.
        let k = 1u64 << 22;
        let layout = AdviceLayout::new(
            LayoutParams {
                log2_k: 22,
                segments: 1,
                layers: 1,
                group: 1,
                query_block: 32,
                max_context: 2,
            },
            2,
            &[TileShape {
                layer: Some(0),
                values: vec![TensorShape::Dense(DenseShape::new(
                    vec![Axis::new(8758), Axis::new(k)],
                    Padding::Zero,
                ))],
            }],
        )
        .unwrap();
        assert_eq!(layout.row_vars(), 14);
        assert_eq!(layout.rows(), 8758);
        let params = crate::CommitmentParams {
            log2_k: 22,
            log2_inv_rate: 1,
            mask_coeffs: 200,
            row_block: NonZeroUsize::new(32).unwrap(),
            commitment_id: [0; 32],
        };
        let mut plan = None;
        let allocations = measure(|| {
            plan = Some(AdviceOpeningMemory::new(&layout, params, 7470, 16860, 56).unwrap())
        });
        assert_eq!(allocations.count_total, 0);
        // Independently enumerated payload/descriptor totals, Fp8/Fp2 16 on this target.
        let plan = plan.unwrap();
        assert_eq!(plan.groups_bytes(), 4_829_536);
        assert_eq!(plan.private_pass_bytes(), 4_966_293_696);
        assert_eq!(plan.output_bytes(), 3_825_390_384);
        assert_eq!(plan.peak_bytes(), 8_830_068_048);
        let mut columns = None;
        let allocations = measure(|| {
            columns = Some(AdviceColumnMemory::new(&layout, params, 193).unwrap());
        });
        assert_eq!(allocations.count_total, 0);
        let columns = columns.unwrap();
        assert_eq!(columns.private_pass_bytes(), 80_637_392);
        assert_eq!(columns.proof_bytes(), 142_048);
        assert_eq!(columns.output_bytes(), 13_669_032);
        assert_eq!(columns.peak_bytes(), 94_306_424);
        let mut p1 = None;
        let allocations =
            measure(|| p1 = Some(AdviceCommitmentMemory::new(&layout, params).unwrap()));
        assert_eq!(allocations.count_total, 0);
        let p1 = p1.unwrap();
        assert_eq!(p1.encoder_bytes(), 83_886_080);
        assert_eq!(p1.encoder_peak_bytes(), 83_886_256);
        assert_eq!(p1.commit_peak_bytes(), 2_499_805_280);
        assert_eq!(p1.peak_bytes(), 2_499_805_280);
    }

    #[test]
    fn opening_memory_refuses_inconsistent_or_unaddressable_geometry_without_allocating() {
        use allocation_counter::measure;
        let (layout, params, _) = fixture();
        for (claims, images, groups) in [
            (2, 1, 1),
            (1, 0, 1),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 1, 2),
        ] {
            let mut result = None;
            let measured = measure(|| {
                result = Some(AdviceOpeningMemory::new(
                    &layout, params, claims, images, groups,
                ))
            });
            assert_eq!(measured.count_total, 0);
            assert!(matches!(
                result.unwrap(),
                Err(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
            ));
        }
        let mut result = None;
        let measured = measure(|| {
            result = Some(AdviceOpeningMemory::new(
                &layout,
                params,
                u64::MAX,
                u64::MAX,
                1,
            ))
        });
        assert_eq!(measured.count_total, 0);
        assert_eq!(result.unwrap().unwrap_err().code(), ErrorCode::ResMemory);
        for other in [
            crate::CommitmentParams {
                log2_inv_rate: 2,
                ..params
            },
            crate::CommitmentParams {
                log2_k: 4,
                ..params
            },
        ] {
            assert!(matches!(
                AdviceOpeningMemory::new(&layout, other, 1, 1, 1),
                Err(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
            ));
        }
    }

    #[test]
    fn os_failure_and_invalid_context_never_fall_back_to_a_fixed_column() {
        let called = Cell::new(false);
        let error = AdviceOpening::build(&[], |_| {
            called.set(true);
            Ok(())
        })
        .err()
        .unwrap();
        assert!(!called.get());
        assert_eq!(error, AdviceOpeningError::Invalid(PcsError::NoSegments));
        let (layout, params, encoder) = fixture();
        let parts_at_other_rate = parts;
        let parts = parts(&layout, params, &encoder);
        let other_encoder = RowEncoder::new(8, 2, 3).unwrap();
        let other_params = crate::CommitmentParams {
            log2_inv_rate: 2,
            ..params
        };
        let other = parts_at_other_rate(&layout, other_params, &other_encoder);
        let called = Cell::new(false);
        assert!(matches!(
            AdviceOpening::build(&other, |_| {
                called.set(true);
                Ok(())
            })
            .err(),
            Some(AdviceOpeningError::Invalid(PcsError::LayoutData { .. }))
        ));
        assert!(!called.get());
        let error = AdviceOpening::build(&parts, |_| Err(AdviceOpeningError::Randomness))
            .err()
            .unwrap();
        assert_eq!(error, AdviceOpeningError::Randomness);
        assert_eq!(error.code(), ErrorCode::ResIo);
        assert_eq!(
            zeros::<Fp>(usize::MAX).err().unwrap().code(),
            ErrorCode::ResMemory
        );
        assert_eq!(
            public_capacity::<Fp>(usize::MAX).err().unwrap().code(),
            ErrorCode::ResMemory
        );
    }

    #[test]
    fn every_possible_sample_checks_the_leaf_and_each_private_accumulator() {
        let (layout, params, encoder) = fixture();
        let parts = parts(&layout, params, &encoder);
        for sample in 0..16u32 {
            let state = AdviceOpening::build(&parts, |out| {
                out.copy_from_slice(&sample.to_le_bytes());
                Ok(())
            })
            .unwrap();
            assert_eq!(state.sample, sample as usize);
            let gamma = Fp2::new(Fp::from_u64_reduced(3), Fp::from_u64_reduced(5));
            let claim = AdviceClaim {
                pieces: vec![AdvicePoint {
                    point: vec![gamma; layout.variables() as usize],
                    weight: gamma,
                }],
            };
            let groups = state
                .prepare(&[claim], gamma, &vec![gamma; layout.row_vars() as usize])
                .unwrap();
            for fault in 0..4 {
                let mut mask = || Fp::from_u64_reduced(7);
                let mut pass = groups.begin(&encoder, &mut mask).unwrap();
                for segment in layout.segments() {
                    for row in segment.nonzero.iter().cloned().flatten() {
                        pass.push_row(row, &[Fp::from_u64_reduced(row + 3); 8])
                            .unwrap();
                    }
                }
                match fault {
                    1 => pass.u[0][0] += Fp2::ONE,
                    2 => pass.m[1][0] += Fp2::ONE,
                    3 => pass.sums[1] += Fp2::ONE,
                    _ => {}
                }
                let result = pass.finish();
                if fault == 0 {
                    assert!(result.is_ok());
                } else {
                    assert_eq!(result.err(), Some(AdviceOpeningError::SelfCheck));
                }
            }
        }
    }

    #[test]
    fn a_refused_boundary_row_cannot_flush_or_change_private_accumulators() {
        let (_, params, encoder) = fixture();
        let layout = AdviceLayout::new(
            LayoutParams {
                log2_k: 3,
                segments: 2,
                layers: 2,
                group: 1,
                query_block: 1,
                max_context: 1,
            },
            1,
            &[
                TileShape {
                    layer: Some(0),
                    values: vec![TensorShape::Dense(DenseShape::new(
                        vec![Axis::new(13)],
                        Padding::Zero,
                    ))],
                },
                TileShape {
                    layer: Some(1),
                    values: vec![TensorShape::Dense(DenseShape::new(
                        vec![Axis::new(13)],
                        Padding::Zero,
                    ))],
                },
            ],
        )
        .unwrap();
        let parts = parts(&layout, params, &encoder);
        let state = AdviceOpening::new(&parts).unwrap();
        let xi = vec![Fp2::from(Fp::from_u64_reduced(7)); layout.row_vars() as usize];
        let groups = state.prepare(&[], Fp2::ONE, &xi).unwrap();
        let mut mask = || Fp::from_u64_reduced(7);
        let mut pass = groups.begin(&encoder, &mut mask).unwrap();
        for r in 0..2 {
            pass.push_row(r, &[Fp::from_u64_reduced(r + 3); 8]).unwrap();
        }
        assert_eq!(pass.segment, 0);
        assert_eq!(pass.numbers, [0, 1]);
        assert_eq!(
            pass.push_row(2, &[]),
            Err(PcsError::RowLength {
                expected: 8,
                found: 0
            }
            .into())
        );
        assert_eq!(pass.segment, 0);
        assert_eq!(pass.numbers, [0, 1]);
        assert!(pass.u[0].iter().all(|v| *v == Fp2::ZERO));
        pass.push_row(2, &[Fp::from_u64_reduced(5); 8]).unwrap();
        assert!(pass.values[8..].iter().all(|v| *v == Fp::ZERO));
        assert!(pass.mask_values[3..].iter().all(|v| *v == Fp::ZERO));
        assert!(pass.samples[1..].iter().all(|v| *v == Fp::ZERO));
        pass.push_row(3, &[Fp::from_u64_reduced(6); 8]).unwrap();
        assert!(pass.finish().is_ok());
    }

    #[test]
    fn sampled_evaluation_weights_match_independent_lagrange_at_every_small_coset_point() {
        for log in 0..=6 {
            let k = 1usize << log;
            let encoder = RowEncoder::new(k, 1, 1).unwrap();
            let root = Fp::from_u64_reduced(7).pow((Fp::MODULUS - 1) / k as u64);
            let h: Vec<_> = (0..k).map(|i| root.pow(i as u64)).collect();
            for column in 0..2 * k {
                let (weights, x, vanishing) = sample_weights(&encoder, column).unwrap();
                assert_ne!(vanishing, Fp::ZERO);
                for i in 0..k {
                    let numerator = h
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .fold(Fp::ONE, |p, (_, &h)| p * (x - h));
                    let denominator = h
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .fold(Fp::ONE, |p, (_, &value)| p * (h[i] - value));
                    assert_eq!(weights[i], numerator * denominator.inverse().unwrap());
                }
            }
        }
    }
}
