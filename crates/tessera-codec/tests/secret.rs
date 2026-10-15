// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `Secret` and `Public` (05 §3): a secret is zeroized when its wrapper drops (OBS-2), and a
//! public value encodes and prints exactly as the value it wraps, so marking a value public never
//! changes a wire format or a report. That a secret cannot be printed, cloned or encoded is shown
//! by the compile-fail cases in `tests/ui/`.

// Reading the value back is what these tests check; `secret-access.toml` lists this file.
#![allow(clippy::disallowed_methods)]

use std::cell::RefCell;
use std::rc::Rc;

use tessera_codec::{DecodeLimits, Public, Secret, Zeroize, decode, encode};

/// A value whose zeroization outlives it: it zeroes a buffer it shares with the test.
struct Shared(Rc<RefCell<[u8; 16]>>);

impl Zeroize for Shared {
    fn zeroize(&mut self) {
        self.0.borrow_mut().zeroize();
    }
}

#[test]
fn dropping_a_secret_zeroizes_its_value() {
    let memory = Rc::new(RefCell::new([0xA5u8; 16]));
    let secret = Secret::new(Shared(Rc::clone(&memory)));
    assert_eq!(*memory.borrow(), [0xA5; 16], "zeroized while still in use");
    assert_eq!(*secret.expose_secret().0.borrow(), [0xA5; 16]);
    drop(secret);
    assert_eq!(*memory.borrow(), [0; 16]);
}

#[test]
fn a_public_value_encodes_as_what_it_wraps() {
    let limits = DecodeLimits {
        max_total_bytes: 64,
        max_vec_len: 8,
        max_depth: 4,
    };
    let count = Public(7u64);
    assert_eq!(encode(&count), encode(&7u64));
    assert_eq!(decode::<Public<u64>>(&encode(&count), limits), Ok(count));
    assert_eq!(count.to_string(), "7");
    // Wrapping adds no nesting level: the same inputs decode under the same limits.
    let some = encode(&Some(7u64));
    for max_depth in [0, 1] {
        let limits = DecodeLimits {
            max_depth,
            ..limits
        };
        assert_eq!(
            decode::<Public<Option<u64>>>(&some, limits).map(|public| public.0),
            decode::<Option<u64>>(&some, limits)
        );
    }
}
