// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One-sequence TokenFile adapter: decoded tokens enter their clearing wrapper directly.
use tessera_codec::{Decode, DecodeError, Reader, Secret, error::ErrorCode};

pub(super) struct Tokens {
    pub(super) values: Secret<Vec<u32>>,
    pub(super) len: u32,
}
impl Decode for Tokens {
    const MIN_SIZE: usize = 8;
    #[allow(clippy::disallowed_methods)]
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        r.enter()?;
        r.enter()?;
        let at = r.position();
        let sequences = r.sequence_len(8)?;
        if sequences != 1 {
            return Err(DecodeError {
                code: ErrorCode::Usage,
                offset: at as u64,
            });
        }
        r.enter()?;
        let at = r.position();
        let n = r.sequence_len(4)?;
        let len = u32::try_from(n).map_err(|_| r.limit(at))?;
        let mut values = Secret::new(Vec::new());
        values
            .expose_secret_mut()
            .try_reserve_exact(n)
            .map_err(|_| DecodeError {
                code: ErrorCode::ResMemory,
                offset: at as u64,
            })?;
        for _ in 0..n {
            values.expose_secret_mut().push(u32::decode_from(r)?);
        }
        r.leave();
        r.leave();
        r.leave();
        Ok(Self { values, len })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_codec::DecodeLimits;
    fn open(count: u64, n: u64, tokens: &[u32]) -> Result<Tokens, DecodeError> {
        let mut payload = count.to_le_bytes().to_vec();
        payload.extend_from_slice(&n.to_le_bytes());
        for t in tokens {
            payload.extend_from_slice(&t.to_le_bytes());
        }
        tessera_codec::decode(&payload, DecodeLimits::MESSAGE)
    }
    #[test]
    #[allow(clippy::disallowed_methods)]
    fn bounded_tokens_enter_the_clearing_wrapper_directly() {
        let t = open(1, 3, &[1, 2, 7]).unwrap();
        assert_eq!(t.len, 3);
        assert_eq!(t.values.expose_secret(), &[1, 2, 7]);
        assert!(matches!(open(2, 0, &[0, 0]), Err(e) if e.code == ErrorCode::Usage));
        assert!(matches!(open(1, u64::MAX, &[]), Err(e) if e.code == ErrorCode::CodecLimit));
        assert!(
            matches!(open(1, 3, &[1, 2]), Err(e) if e.code == ErrorCode::CodecTruncated && e.offset == 8)
        );
    }
}
