// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Per-proof private BLAKE3 XOF streams; only this reader touches the heap-resident seed.
use super::{ErrorCode, MaskRng, ProverError};
use tessera_codec::Secret;
use tessera_field::{Goldilocks, PrimeField};
use zeroize::{Zeroize, Zeroizing};

struct Heap<T: Zeroize>(Box<T>);
impl<T: Zeroize> Zeroize for Heap<T> {
    fn zeroize(&mut self) {
        self.0.as_mut().zeroize();
    }
}

pub(super) struct Seed(Secret<Vec<u8>>);
pub(super) struct Masks(Secret<Heap<blake3::OutputReader>>);
impl Seed {
    #[allow(clippy::disallowed_methods)]
    pub(super) fn new(mode: MaskRng) -> Result<Self, ProverError> {
        let mut seed = Secret::new(vec![0; 32]);
        match mode {
            MaskRng::OsRng => getrandom::fill(seed.expose_secret_mut())
                .map_err(|_| ProverError::Resource(ErrorCode::ResIo))?,
            #[cfg(feature = "test-rng")]
            MaskRng::Seeded(mut state) => {
                for chunk in seed.expose_secret_mut().chunks_exact_mut(8) {
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut value = state;
                    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
                    chunk.copy_from_slice(&(value ^ (value >> 31)).to_le_bytes());
                }
            }
        }
        Ok(Self(seed))
    }
    #[allow(clippy::disallowed_methods)]
    fn reader(&self, label: &'static str) -> Secret<Heap<blake3::OutputReader>> {
        let mut hash = Secret::new(Heap(Box::new(blake3::Hasher::new_derive_key(label))));
        hash.expose_secret_mut().0.update(self.0.expose_secret());
        Secret::new(Heap(Box::new(hash.expose_secret().0.finalize_xof())))
    }
    pub(super) fn masks(&self) -> Masks {
        Masks(self.reader("tessera/prover/cpu/advice-masks/1"))
    }
    #[allow(clippy::disallowed_methods)]
    pub(super) fn share(&self) -> [u8; 32] {
        let mut out = [0; 32];
        self.reader("tessera/prover/cpu/prover-share/1")
            .expose_secret_mut()
            .0
            .fill(&mut out);
        out
    }
    #[allow(clippy::disallowed_methods)]
    pub(super) fn session(&self) -> [u8; 16] {
        let mut out = [0; 16];
        self.reader("tessera/prover/cpu/ni-session/1")
            .expose_secret_mut()
            .0
            .fill(&mut out);
        out
    }
}
impl tessera_pcs::MaskSource for Masks {
    #[allow(clippy::disallowed_methods)]
    fn draw(&mut self) -> Goldilocks {
        field(|| {
            let mut bytes = Zeroizing::new([0; 8]);
            self.0.expose_secret_mut().0.fill(bytes.as_mut());
            u64::from_le_bytes(*bytes)
        })
    }
}
fn field(mut next: impl FnMut() -> u64) -> Goldilocks {
    loop {
        if let Some(v) = Goldilocks::from_canonical_u64(next()) {
            return v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_codec::SecretBuf;
    use tessera_pcs::MaskSource;
    #[test]
    fn rejects_noncanonical_words_instead_of_reducing_them() {
        let words = [u64::MAX, Goldilocks::MODULUS, 7];
        let mut n = 0;
        let v = field(|| {
            let v = words[n];
            n += 1;
            v
        });
        assert_eq!(n, 3);
        assert_eq!(v.to_canonical_u64(), 7);
    }
    #[test]
    fn private_streams_replay_and_fresh_proofs_change_their_public_share() {
        let a = Seed::new(MaskRng::OsRng).unwrap();
        let b = Seed::new(MaskRng::OsRng).unwrap();
        let x: SecretBuf<_> = (0..17).map(|_| a.masks().draw()).collect();
        let mut first = a.masks();
        let mut replay = a.masks();
        let mut next = b.masks();
        let original: SecretBuf<_> = (0..17).map(|_| first.draw()).collect();
        let repeated: SecretBuf<_> = (0..17).map(|_| replay.draw()).collect();
        let fresh: SecretBuf<_> = (0..17).map(|_| next.draw()).collect();
        assert!(original[..] == repeated[..]);
        assert!(original[..] != x[..]);
        assert!(original[..] != fresh[..]);
        assert_ne!(a.share(), b.share());
        assert_ne!(a.session(), a.share()[..16]);
        let first = Zeroizing::new(a.masks().draw().to_canonical_u64().to_le_bytes());
        assert!(first[..] != a.share()[..8]);
    }
}
