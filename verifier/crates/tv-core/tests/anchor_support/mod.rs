// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support for the anchor's log checks: TCE encoders for log entries and manifests written
//! from RFC-0015, RFC-0059, RFC-0060, RFC-0062 and RFC-0063, the envelope sealed with the test
//! kit's reference BLAKE3, RFC 6962's recursive tree hash, and chains of periods built from a
//! short description of their entries.
//!
//! The encoders and the tree are this file's own: `tv-core` decodes and streams, these encode
//! and recurse. Manifests are signed for real, so every check runs `tv-core`'s production path,
//! strict Ed25519 included: the test recorder signs with a fixed secret scalar and a fixed nonce
//! ([`sign`]), its points computed offline with RFC 8032 §5.1's arithmetic, and `S` here, modulo
//! `L`, bit by bit.

#![allow(dead_code)]

use tv_codec::DecodeLimits;
use tv_core::anchor::{
    CheckedManifest, Direction, LogManifest, RecorderKeys, SamplerPins, SamplerPolicy, VerifiedLog,
    Window,
};
use tv_core::protocol::{BatchDescriptor, Decoding, PredictorSpec, PublicStatement, RequestRef};
use tv_hash::sha256::sha256;
use tv_hash::sha512::sha512;

/// The test recorder's public key, pinned: `A = encode([a]B)` for the secret scalar
/// [`SECRET`], computed offline with RFC 8032 §5.1's arithmetic.
pub const KEY: [u8; 32] = hex32("c8f4cfd7ec9c81e16c77144a4d1ce0dca0339661df313cceb9203113468fb608");
/// The test recorder's secret scalar `a`, `0x0707…07` little-endian, below `L`.
pub const SECRET: [u8; 32] = [0x07; 32];
/// The fixed nonce `r` of every test signature, below `L`. Two signatures with one nonce reveal
/// the secret; the key signs nothing but tests.
pub const NONCE: [u8; 32] = {
    let mut r = [0x2a; 32];
    r[31] = 0x0a;
    r
};
/// `encode([r]B)`, the `R` of every test signature, computed offline.
pub const NONCE_POINT: [u8; 32] =
    hex32("69e8ec7eb4586b63bf87a39b211ee47d660dc5d39c11b2981de188f1336a08f6");
/// `encode([r]B + T)`, `T` a point of order 8, computed offline: a point of mixed order, not of
/// small order. A signature with it as `R` and `S = r + k·a` passes the cofactored equation
/// `[8][S]B = [8]R + [8][k]A` and fails the cofactorless one.
pub const MIXED_NONCE_POINT: [u8; 32] =
    hex32("8bde9cd4f460165ce288d8fd9115cdcce657fc56648639f0172c61a6d073cc54");
/// The encoding of the identity, `y = 1`, `x = 0`: a point of order 1, of small order.
pub const IDENTITY: [u8; 32] = {
    let mut y = [0; 32];
    y[0] = 1;
    y
};
/// The scenarios' log.
pub const LOG_ID: [u8; 16] = [0x10; 16];
/// A period's length in the scenarios.
pub const PERIOD_NS: u64 = 1_000_000;
/// The first period's start.
pub const GENESIS_NS: u64 = 5_000_000;

// ----- Encoders ---------------------------------------------------------------------------------

/// `TCE(SessionOpen)` or, with `carry`, `TCE(SessionCarry)`: tag 0 or 3, `seq`, `t_ns`, the
/// session and the key commitment.
pub fn encode_key_entry(
    carry: bool,
    seq: u64,
    t_ns: u64,
    session: [u8; 16],
    key: [u8; 32],
) -> Vec<u8> {
    let tag: u16 = if carry { 3 } else { 0 };
    [
        &tag.to_le_bytes()[..],
        &seq.to_le_bytes(),
        &t_ns.to_le_bytes(),
        &session,
        &key,
    ]
    .concat()
}

/// `TCE(Frame)`: tag 1, then the header's fields in its order (RFC-0059 'The log entry').
#[allow(clippy::too_many_arguments)]
pub fn encode_frame(
    seq: u64,
    t_ns: u64,
    session: [u8; 16],
    request: u64,
    direction: Direction,
    index: u32,
    length: u32,
    counter: u32,
    digest: [u8; 32],
) -> Vec<u8> {
    let direction: u16 = match direction {
        Direction::In => 0,
        Direction::Out => 1,
    };
    [
        &1u16.to_le_bytes()[..],
        &seq.to_le_bytes(),
        &t_ns.to_le_bytes(),
        &session,
        &request.to_le_bytes(),
        &direction.to_le_bytes(),
        &index.to_le_bytes(),
        &length.to_le_bytes(),
        &counter.to_le_bytes(),
        &digest,
    ]
    .concat()
}

/// `TCE(SessionClose)`: tag 2.
pub fn encode_close(seq: u64, t_ns: u64, session: [u8; 16]) -> Vec<u8> {
    [
        &2u16.to_le_bytes()[..],
        &seq.to_le_bytes(),
        &t_ns.to_le_bytes(),
        &session,
    ]
    .concat()
}

/// `TCE(RequestEnd)`: tag 4 (RFC-0063).
pub fn encode_end(seq: u64, t_ns: u64, session: [u8; 16], request: u64) -> Vec<u8> {
    [
        &4u16.to_le_bytes()[..],
        &seq.to_le_bytes(),
        &t_ns.to_le_bytes(),
        &session,
        &request.to_le_bytes(),
    ]
    .concat()
}

/// A manifest's fields, before encoding.
#[derive(Clone, Debug)]
pub struct Fields {
    pub key_id: [u8; 32],
    pub log_id: [u8; 16],
    pub start: u64,
    pub end: u64,
    pub first_seq: u64,
    pub count: u64,
    pub root: [u8; 32],
    pub prev: [u8; 32],
    pub scheme: u16,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}

impl Fields {
    /// `TCE(LogManifest)`: the fields in order, then the signature envelope, its `Bytes` a `u64`
    /// length and the bytes.
    pub fn encode(&self) -> Vec<u8> {
        [
            &self.key_id[..],
            &self.log_id,
            &self.start.to_le_bytes(),
            &self.end.to_le_bytes(),
            &self.first_seq.to_le_bytes(),
            &self.count.to_le_bytes(),
            &self.root,
            &self.prev,
            &self.scheme.to_le_bytes(),
            &(self.public_key.len() as u64).to_le_bytes(),
            &self.public_key,
            &(self.signature.len() as u64).to_le_bytes(),
            &self.signature,
        ]
        .concat()
    }

    /// The `.tlm` file: RFC-0003's envelope of schema `0x0402`, version 1, the fields as they
    /// are, the signature included.
    pub fn seal(&self) -> Vec<u8> {
        seal(&self.encode())
    }

    /// The signed message: the label, then every field but the signature, the payload's first
    /// 144 bytes (RFC-0060 'Signatures').
    pub fn signed_message(&self) -> Vec<u8> {
        [&b"tessera-manifest/v1"[..], &self.encode()[..144]].concat()
    }

    /// The fields, signed by the test recorder over their signed message.
    pub fn signed(&self) -> Self {
        let mut fields = self.clone();
        fields.signature = sign(&fields.signed_message()).to_vec();
        fields
    }

    /// [`seal`](Self::seal) of [`signed`](Self::signed).
    pub fn seal_signed(&self) -> Vec<u8> {
        self.signed().seal()
    }
}

/// RFC-0003's envelope of schema `0x0402`, version 1, around `payload`, its digest the test kit's
/// BLAKE3.
pub fn seal(payload: &[u8]) -> Vec<u8> {
    let mut out = [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &0x0402u16.to_le_bytes(),
        &1u16.to_le_bytes(),
        &0u16.to_le_bytes(),
        &(payload.len() as u64).to_le_bytes(),
        payload,
    ]
    .concat();
    let digest = tv_testkit::blake3::derive_key("tessera/v1/envelope", &out);
    out.extend_from_slice(&digest);
    out
}

/// RFC 6962's `MTH` over the leaves `SHA-256(0x00 ‖ entry)`, by its recursive definition.
pub fn mth(entries: &[Vec<u8>]) -> [u8; 32] {
    match entries.len() {
        0 => sha256(&[]),
        1 => sha256(&[&[0x00][..], &entries[0]].concat()),
        n => {
            let mut k = 1;
            while k * 2 < n {
                k *= 2;
            }
            sha256(&[&[0x01][..], &mth(&entries[..k]), &mth(&entries[k..])].concat())
        }
    }
}

// ----- Signatures -------------------------------------------------------------------------------

/// 32 bytes from 64 hex digits, at compile time.
const fn hex32(text: &str) -> [u8; 32] {
    const fn digit(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("a hex digit"),
        }
    }
    let text = text.as_bytes();
    assert!(text.len() == 64);
    let mut out = [0; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = digit(text[2 * i]) << 4 | digit(text[2 * i + 1]);
        i += 1;
    }
    out
}

/// The test recorder's signature on `message`, `R ‖ S` (RFC 8032 §5.1.6 with a fixed nonce):
/// `R = encode([r]B)`, `S = r + k·a mod L`, `k = SHA-512(R ‖ A ‖ message) mod L`.
pub fn sign(message: &[u8]) -> [u8; 64] {
    sign_with(&NONCE_POINT, &scalar::reduce(&NONCE), &KEY, message)
}

/// A signature `R ‖ S` on `message` under `key`: `S = nonce + k·a mod L`, with the test secret
/// `a` and `k = SHA-512(R ‖ key ‖ message) mod L`. The cofactorless equation holds when
/// `R = [nonce]B` and `key = [a]B`, and also for `R` the identity with `nonce = 0`, since then
/// `[S]B − [k]A` is the identity.
pub fn sign_with(r: &[u8; 32], nonce: &scalar::Scalar, key: &[u8; 32], message: &[u8]) -> [u8; 64] {
    let k = scalar::reduce(&sha512(&[&r[..], key, message].concat()));
    let s = scalar::add(nonce, &scalar::mul(&k, &scalar::reduce(&SECRET)));
    let mut signature = [0; 64];
    signature[..32].copy_from_slice(r);
    signature[32..].copy_from_slice(&scalar::to_bytes(&s));
    signature
}

/// Arithmetic modulo `L = 2^252 + 27742317777372353535851937790883648493` on four
/// little-endian 64-bit limbs, bit by bit: slow and plain, for a few test signatures. Every
/// value is below `L`, so a sum is below `2^254` and fits.
pub mod scalar {
    /// An integer below `2^256`, least significant limb first.
    pub type Scalar = [u64; 4];

    /// The group order `L`.
    pub const L: Scalar = [
        0x5812_631a_5cf5_d3ed,
        0x14de_f9de_a2f7_9cd6,
        0,
        0x1000_0000_0000_0000,
    ];

    fn below(a: &Scalar, b: &Scalar) -> bool {
        for i in (0..4).rev() {
            if a[i] != b[i] {
                return a[i] < b[i];
            }
        }
        false
    }

    /// `a + b` without reduction, for `a + b < 2^256`.
    pub fn sum(a: &Scalar, b: &Scalar) -> Scalar {
        let mut out = [0; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            let t = u128::from(a[i]) + u128::from(b[i]) + carry;
            out[i] = t as u64;
            carry = t >> 64;
        }
        assert_eq!(carry, 0);
        out
    }

    fn minus_l(a: &Scalar) -> Scalar {
        let mut out = [0; 4];
        let mut borrow = false;
        for i in 0..4 {
            let (t, b1) = a[i].overflowing_sub(L[i]);
            let (t, b2) = t.overflowing_sub(u64::from(borrow));
            out[i] = t;
            borrow = b1 || b2;
        }
        out
    }

    /// `a + b mod L`, for `a, b < L`.
    pub fn add(a: &Scalar, b: &Scalar) -> Scalar {
        let s = sum(a, b);
        if below(&s, &L) { s } else { minus_l(&s) }
    }

    /// `a · b mod L`, for `a < L`: double and add over `b`'s bits.
    pub fn mul(a: &Scalar, b: &Scalar) -> Scalar {
        let mut x = [0; 4];
        for i in (0..256).rev() {
            x = add(&x, &x);
            if b[i / 64] >> (i % 64) & 1 == 1 {
                x = add(&x, a);
            }
        }
        x
    }

    /// A little-endian integer of any length, modulo `L`: Horner's rule over its bits.
    pub fn reduce(bytes: &[u8]) -> Scalar {
        let mut x = [0; 4];
        for byte in bytes.iter().rev() {
            for bit in (0..8).rev() {
                x = add(&x, &x);
                if byte >> bit & 1 == 1 {
                    x = add(&x, &[1, 0, 0, 0]);
                }
            }
        }
        x
    }

    /// The 32 little-endian bytes of `a`.
    pub fn to_bytes(a: &Scalar) -> [u8; 32] {
        let mut out = [0; 32];
        for (i, limb) in a.iter().enumerate() {
            out[8 * i..8 * i + 8].copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    /// `a` from its 32 little-endian bytes, unreduced.
    pub fn from_bytes(bytes: &[u8; 32]) -> Scalar {
        let mut out = [0; 4];
        for (i, limb) in out.iter_mut().enumerate() {
            *limb = u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap());
        }
        out
    }
}

// ----- Opening and checking ---------------------------------------------------------------------

/// The scenarios' pinned keys.
pub fn keys() -> RecorderKeys {
    RecorderKeys::new(vec![KEY])
}

/// Opens and checks a manifest under the test recorder's pinned key, strict Ed25519 included.
pub fn checked(tlm: &[u8]) -> CheckedManifest {
    LogManifest::open(tlm, &DecodeLimits::MESSAGE)
        .unwrap()
        .check(&keys())
        .unwrap()
}

// ----- Scenarios --------------------------------------------------------------------------------

/// The session id of scenario session `s`.
pub fn sid(s: u8) -> [u8; 16] {
    [s; 16]
}

/// One entry of a scenario period, before `seq` and default times are assigned.
#[derive(Clone, Debug)]
pub enum Item {
    Open { s: u8, key: [u8; 32] },
    Carry { s: u8, key: [u8; 32] },
    Close { s: u8 },
    End { s: u8, r: u64, t: Option<u64> },
    Frame(FrameItem),
}

/// A frame of a scenario: by default its length is that of its tokens, its counter the session's
/// next, and its time the period's clock.
#[derive(Clone, Copy, Debug)]
pub struct FrameItem {
    pub s: u8,
    pub r: u64,
    pub direction: Direction,
    pub index: u32,
    pub length: u32,
    pub counter: Option<u32>,
    pub t: Option<u64>,
}

impl FrameItem {
    /// The same frame at time `t`.
    pub fn at(mut self, t: u64) -> Item {
        self.t = Some(t);
        Item::Frame(self)
    }
    /// The same frame with counter `c`.
    pub fn counter(mut self, c: u32) -> Item {
        self.counter = Some(c);
        Item::Frame(self)
    }
    /// The same frame with ciphertext length `length`.
    pub fn length(mut self, length: u32) -> Item {
        self.length = length;
        Item::Frame(self)
    }
    /// The frame as an item.
    pub fn item(self) -> Item {
        Item::Frame(self)
    }
}

impl From<FrameItem> for Item {
    fn from(frame: FrameItem) -> Self {
        Item::Frame(frame)
    }
}

/// Session `s`'s open, its key commitment `[s; 32]`.
pub fn open(s: u8) -> Item {
    Item::Open { s, key: [s; 32] }
}
/// Session `s`'s carry, its key commitment `[s; 32]`.
pub fn carry(s: u8) -> Item {
    Item::Carry { s, key: [s; 32] }
}
/// Session `s`'s close.
pub fn close(s: u8) -> Item {
    Item::Close { s }
}
/// Request `r` of session `s` ends.
pub fn end(s: u8, r: u64) -> Item {
    Item::End { s, r, t: None }
}
/// An inbound frame of `tokens` tokens at position `index`.
pub fn inbound(s: u8, r: u64, index: u32, tokens: u32) -> FrameItem {
    frame_item(s, r, Direction::In, index, tokens)
}
/// An outbound frame of `tokens` tokens at position `index`.
pub fn outbound(s: u8, r: u64, index: u32, tokens: u32) -> FrameItem {
    frame_item(s, r, Direction::Out, index, tokens)
}
fn frame_item(s: u8, r: u64, direction: Direction, index: u32, tokens: u32) -> FrameItem {
    FrameItem {
        s,
        r,
        direction,
        index,
        length: 22 + 4 * tokens,
        counter: None,
        t: None,
    }
}

/// The items of a period, as the test writes them.
pub type Items = Vec<Item>;

/// A built period: its manifest's fields, `.tlm` and `.tdl`, and its entries' encodings.
#[derive(Clone, Debug)]
pub struct Built {
    pub fields: Fields,
    pub tlm: Vec<u8>,
    pub tdl: Vec<u8>,
    pub entries: Vec<Vec<u8>>,
    pub identity: [u8; 32],
}

/// A chain of one log's periods, each `(start, end, items)`, built in order: `seq` contiguous
/// from 0 across the chain, each period linked to the one before, the first a genesis. Default
/// frame counters count per session across the chain from 0, and default times step by 10 ns
/// from each period's start.
pub fn chain(periods: &[(u64, u64, Items)]) -> Vec<Built> {
    let key_id = sha256(&KEY);
    let mut built: Vec<Built> = Vec::new();
    let mut seq = 0u64;
    let mut counters = std::collections::BTreeMap::<u8, u32>::new();
    for (start, end, items) in periods {
        let mut clock = *start;
        let mut next_time = |t: Option<u64>| {
            clock += 10;
            t.unwrap_or(clock)
        };
        let first_seq = seq;
        let mut entries = Vec::new();
        for item in items {
            let entry = match item {
                Item::Open { s, key } => {
                    encode_key_entry(false, seq, next_time(None), sid(*s), *key)
                }
                Item::Carry { s, key } => {
                    encode_key_entry(true, seq, next_time(None), sid(*s), *key)
                }
                Item::Close { s } => encode_close(seq, next_time(None), sid(*s)),
                Item::End { s, r, t } => encode_end(seq, next_time(*t), sid(*s), *r),
                Item::Frame(f) => {
                    let next = counters.entry(f.s).or_insert(0);
                    let counter = f.counter.unwrap_or(*next);
                    *next = (*next).max(counter.saturating_add(1));
                    encode_frame(
                        seq,
                        next_time(f.t),
                        sid(f.s),
                        f.r,
                        f.direction,
                        f.index,
                        f.length,
                        counter,
                        sha256(&seq.to_le_bytes()),
                    )
                }
            };
            entries.push(entry);
            seq += 1;
        }
        let fields = Fields {
            key_id,
            log_id: LOG_ID,
            start: *start,
            end: *end,
            first_seq,
            count: entries.len() as u64,
            root: mth(&entries),
            prev: built.last().map_or([0; 32], |b| b.identity),
            scheme: 0,
            public_key: KEY.to_vec(),
            signature: Vec::new(),
        }
        .signed();
        let tlm = fields.seal();
        let identity: [u8; 32] = tlm[tlm.len() - 32..].try_into().unwrap();
        built.push(Built {
            fields,
            tdl: entries.concat(),
            tlm,
            entries,
            identity,
        });
    }
    built
}

impl Built {
    /// The checked manifest.
    pub fn checked(&self) -> CheckedManifest {
        checked(&self.tlm)
    }
    /// The verified log.
    pub fn verified(&self) -> VerifiedLog {
        VerifiedLog::check(&self.checked(), &self.tdl, &DecodeLimits::MESSAGE).unwrap()
    }
}

/// A period `P` and, when given, the period `P − 1` before it: the window a batch is checked
/// over. `P − 1` runs `[GENESIS_NS, GENESIS_NS + PERIOD_NS)` and `P` the period after it.
pub struct Scenario {
    pub previous: Option<VerifiedLog>,
    pub period: VerifiedLog,
}

impl Scenario {
    /// `P − 1` with `previous` and `P` with `current`.
    pub fn two(previous: Items, current: Items) -> Self {
        let built = chain(&[
            (GENESIS_NS, GENESIS_NS + PERIOD_NS, previous),
            (GENESIS_NS + PERIOD_NS, GENESIS_NS + 2 * PERIOD_NS, current),
        ]);
        Self {
            previous: Some(built[0].verified()),
            period: built[1].verified(),
        }
    }

    /// A log's first period, with no `P − 1`.
    pub fn one(current: Items) -> Self {
        let built = chain(&[(GENESIS_NS, GENESIS_NS + PERIOD_NS, current)]);
        Self {
            previous: None,
            period: built[0].verified(),
        }
    }

    /// The window, bound with no zero-length period between.
    pub fn window(&self) -> Window<'_> {
        Window::bind(&self.period, &[], self.previous.as_ref()).unwrap()
    }

    /// The statement of a batch anchored to `P`.
    pub fn statement(
        &self,
        requests: Vec<RequestRef>,
        predictor: PredictorSpec,
    ) -> PublicStatement {
        PublicStatement {
            claim_list_digest: [0; 32],
            weights_root: [0; 32],
            weights_epoch: 0,
            manifest_digest: *self.period.manifest_digest(),
            batch: BatchDescriptor { requests },
            predictor,
        }
    }

    /// The anchor checks of a greedy batch.
    pub fn greedy(&self, requests: Vec<RequestRef>) -> Result<(), tv_core::anchor::Failure> {
        tv_core::anchor::check_batch(
            &self.statement(requests, greedy()),
            &self.window(),
            &no_pins(),
        )
    }

    /// The anchor checks of a temperature batch under `pins`.
    pub fn sampled(
        &self,
        requests: Vec<RequestRef>,
        pins: &SamplerPins,
    ) -> Result<(), tv_core::anchor::Failure> {
        tv_core::anchor::check_batch(
            &self.statement(requests, temperature()),
            &self.window(),
            pins,
        )
    }
}

/// A request of the batch: session `s`, request `r`, with its lengths.
pub fn req(s: u8, r: u64, prompt_len: u32, output_len: u32) -> RequestRef {
    RequestRef {
        request_id: r,
        session_id: sid(s),
        prompt_len,
        output_len,
    }
}

/// Exact greedy decoding.
pub fn greedy() -> PredictorSpec {
    PredictorSpec::Exact {
        decoding: Decoding::Greedy,
    }
}

/// Exact decoding at temperature 1.
pub fn temperature() -> PredictorSpec {
    PredictorSpec::Exact {
        decoding: Decoding::Temperature {
            t: tv_codec::Rational::new(1, 1).unwrap(),
        },
    }
}

/// The default policy for the scenarios' period, and no pin.
pub fn no_pins() -> SamplerPins {
    SamplerPins::load(Vec::new(), SamplerPolicy::default_for(PERIOD_NS)).unwrap()
}
