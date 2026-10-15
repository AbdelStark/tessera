// SPDX-License-Identifier: Apache-2.0 OR MIT

//! TCE encodings (RFC-0003 'Primitive types'). A prime-field element is its canonical value as a
//! little-endian `u64`, and decoding rejects any value at or above the modulus. An extension
//! element is `c0` then `c1`.

use tessera_codec::{Decode, DecodeError, Encode, Reader, Sink};

use crate::{ExtensionBase, F65537, Goldilocks, PrimeField, QuadraticExt, QuadraticExtension};

macro_rules! prime_fields {
    ($($field:ty),*) => {$(
        impl Encode for $field {
            fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
                self.to_canonical_u64().encode_to(out);
            }
        }

        impl Decode for $field {
            const MIN_SIZE: usize = 8;

            fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
                let at = reader.position();
                let value = u64::decode_from(reader)?;
                Self::from_canonical_u64(value).ok_or_else(|| reader.noncanonical(at))
            }
        }
    )*};
}

prime_fields!(Goldilocks, F65537);

impl<B: ExtensionBase + Encode> Encode for QuadraticExt<B> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        let [c0, c1] = self.coeffs();
        c0.encode_to(out);
        c1.encode_to(out);
    }
}

impl<B: ExtensionBase + Decode> Decode for QuadraticExt<B> {
    const MIN_SIZE: usize = 2 * B::MIN_SIZE;

    fn decode_from(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        // A pair of coefficients: one nesting level, like any value holding others.
        reader.enter()?;
        let c0 = B::decode_from(reader)?;
        let c1 = B::decode_from(reader)?;
        reader.leave();
        Ok(Self::new(c0, c1))
    }
}
