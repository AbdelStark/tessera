// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Inputs of `audit commit` and `audit reveal`: the signed manifests and logs of
//! `spec/vectors/manifest/`, written by an oracle that shares no code with either workspace, as
//! files; the same two periods re-signed by a test recorder with two periods of zero length
//! between them, which the vectors do not have; a policy file; and the sample RFC-0016 derives
//! from them, computed here through `tv-core`'s own checks, for comparison with what the command
//! writes.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use tv_codec::DecodeLimits;
use tv_core::anchor::{LogManifest, RecorderKeys, VerifiedLog, Window};
use tv_core::audit::{AuditPolicy, AuditSample, CoinTossTranscript, RateBucket};
use tv_core::claims::ParamSetId;
use tv_hash::sha256::sha256;
use tv_hash::sha512::sha512;
use tv_testkit::{Json, hex, read_vectors};

/// The vectors' periods as files: the first, a log's first period, and the second, which
/// follows it; the recorder's public key; and a policy file at rate 1/2.
pub struct Inputs {
    pub dir: PathBuf,
    pub key: String,
    pub manifests: [PathBuf; 2],
    pub logs: [PathBuf; 2],
    pub identities: [[u8; 32]; 2],
    pub policy: PathBuf,
}

/// Writes the inputs into `dir`.
pub fn inputs(dir: &Path) -> Inputs {
    let doc = read_vectors("manifest/manifests.json");
    let text = |v: &Json, k: &str| v.get(k).and_then(Json::as_str).unwrap().to_string();
    let manifests = doc.get("manifests").and_then(Json::as_array).unwrap();
    let mut tlm = Vec::new();
    let mut tdl = Vec::new();
    let mut identities = Vec::new();
    for (i, m) in manifests.iter().take(2).enumerate() {
        let manifest = dir.join(format!("period-{i}.tlm"));
        fs::write(&manifest, hex(&text(m, "envelope"))).unwrap();
        let log: Vec<u8> = m
            .get("entries")
            .and_then(Json::as_array)
            .unwrap()
            .iter()
            .flat_map(|e| hex(e.as_str().unwrap()))
            .collect();
        let path = dir.join(format!("period-{i}.tdl"));
        fs::write(&path, log).unwrap();
        tlm.push(manifest);
        tdl.push(path);
        identities.push(<[u8; 32]>::try_from(hex(&text(m, "identity"))).unwrap());
    }
    let policy = dir.join("policy.json");
    fs::write(&policy, policy_json("0.5")).unwrap();
    Inputs {
        dir: dir.to_path_buf(),
        key: text(&doc, "public_key"),
        manifests: [tlm[0].clone(), tlm[1].clone()],
        logs: [tdl[0].clone(), tdl[1].clone()],
        identities: [identities[0], identities[1]],
        policy,
    }
}

/// A `tessera.audit-policy/1` file of one bucket at `rate`.
pub fn policy_json(rate: &str) -> String {
    format!(
        "{{\"schema\":\"tessera.audit-policy/1\",\"period_ns\":86400000000000,\"buckets\":[{{\"max_tokens\":4294967295,\"rate\":\"{rate}\"}}],\"min_batch_tokens\":16384,\"max_batch_tokens\":262144,\"proof_deadline_ns\":604800000000000,\"params\":\"live-64/1\"}}"
    )
}

/// The policy of [`policy_json`] at rate 1/2.
pub fn halves() -> AuditPolicy {
    AuditPolicy::new(
        86_400_000_000_000,
        vec![RateBucket {
            max_tokens: u32::MAX,
            threshold: 1 << 63,
        }],
        16_384,
        262_144,
        604_800_000_000_000,
        ParamSetId::new("live-64/1").unwrap(),
    )
    .unwrap()
}

/// The share of a sealed toss: its payload's last 32 bytes, before the envelope's digest
/// (RFC-0067 'The encoding').
pub fn share_of(tvt: &[u8]) -> [u8; 32] {
    tvt[tvt.len() - 64..tvt.len() - 32].try_into().unwrap()
}

/// The sample of the second period under the policy at rate 1/2, the toss `(c_V, s_P, s_V)`:
/// RFC-0016 'Sample derivation' over the logs checked against their manifests.
pub fn expected_sample(
    inputs: &Inputs,
    commitment: [u8; 32],
    prover: [u8; 32],
    share: [u8; 32],
) -> AuditSample {
    derive(
        &inputs.key,
        (&inputs.manifests[1], &inputs.logs[1]),
        (&inputs.manifests[0], &inputs.logs[0]),
        &[],
        CoinTossTranscript {
            verifier_commitment: commitment,
            prover_share: prover,
            verifier_share: share,
        },
    )
}

/// RFC-0016's sample of `period`, `(manifest, log)`, under the policy at rate 1/2: the manifests
/// checked under `key`, the logs against their manifests, the window bound from `previous`
/// through the zero-length manifests `between`, in chain order.
pub fn derive(
    key: &str,
    period: (&Path, &Path),
    previous: (&Path, &Path),
    between: &[&Path],
    toss: CoinTossTranscript,
) -> AuditSample {
    let keys = RecorderKeys::new(vec![<[u8; 32]>::try_from(hex(key)).unwrap()]);
    let checked = |path: &Path| {
        LogManifest::open(&fs::read(path).unwrap(), &DecodeLimits::MESSAGE)
            .unwrap()
            .check(&keys)
            .unwrap()
    };
    let verified = |(manifest, log): (&Path, &Path)| {
        VerifiedLog::check(
            &checked(manifest),
            &fs::read(log).unwrap(),
            &DecodeLimits::MESSAGE,
        )
        .unwrap()
    };
    let (period, previous) = (verified(period), verified(previous));
    let between: Vec<_> = between.iter().map(|p| checked(p)).collect();
    let window = Window::bind(&period, &between, Some(&previous)).unwrap();
    AuditSample::derive(&window, halves(), toss).unwrap()
}

/// `bytes` as lowercase hexadecimal.
pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A fresh directory of its own.
pub fn scratch(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "tessera-cli-toss-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    fs::canonicalize(dir).unwrap()
}

/// `path` as a string argument.
pub fn arg(path: &Path) -> String {
    path.to_str().unwrap().to_string()
}

// ----- A chain with periods of zero length ----------------------------------------------------

/// The vectors' two periods, re-signed by the test [`recorder`] with two periods of zero length
/// between them, `P − 1`, `Z1`, `Z2`, `P`, linked by RFC-0060's chain: each zero-length period
/// at the first period's end, with no entries, the first `seq` of the second period and the root
/// `SHA-256("")`. The logs are the vectors' own, unchanged.
pub struct ZeroLengthChain {
    /// The recorder's public key, as `--recorder-key` takes it.
    pub key: String,
    /// `P − 1`'s manifest, over the vectors' first log.
    pub previous: PathBuf,
    /// `Z1` and `Z2`, in chain order.
    pub zeros: [PathBuf; 2],
    /// `P`'s manifest, over the vectors' second log.
    pub period: PathBuf,
    /// `P`'s identity.
    pub identity: [u8; 32],
    /// A `Z1` whose signature has one byte flipped.
    pub forged: PathBuf,
}

/// Writes [`ZeroLengthChain`]'s manifests into `inputs.dir`.
pub fn zero_length_chain(inputs: &Inputs) -> ZeroLengthChain {
    let doc = read_vectors("manifest/manifests.json");
    let log_id: [u8; 16] = hex(doc.get("log_id").and_then(Json::as_str).unwrap())
        .try_into()
        .unwrap();
    let vectors = doc.get("manifests").and_then(Json::as_array).unwrap();
    let number = |m: &Json, k: &str| m.get(k).and_then(Json::as_u64).unwrap();
    let fields = |m: &Json, prev: [u8; 32]| Manifest {
        log_id,
        start: number(m, "period_start_ns"),
        end: number(m, "period_end_ns"),
        first_seq: number(m, "first_seq"),
        count: m.get("entries").and_then(Json::as_array).unwrap().len() as u64,
        root: hex(m.get("entries_root").and_then(Json::as_str).unwrap())
            .try_into()
            .unwrap(),
        prev,
    };
    let at = number(&vectors[0], "period_end_ns");
    let seq = number(&vectors[1], "first_seq");
    let zero = |prev: [u8; 32]| Manifest {
        log_id,
        start: at,
        end: at,
        first_seq: seq,
        count: 0,
        root: sha256(&[]),
        prev,
    };
    let write = |name: &str, tlm: &[u8]| {
        let path = inputs.dir.join(name);
        fs::write(&path, tlm).unwrap();
        path
    };
    let identity = |tlm: &[u8]| -> [u8; 32] { tlm[tlm.len() - 32..].try_into().unwrap() };
    let previous = fields(&vectors[0], [0; 32]).sealed();
    let z1 = zero(identity(&previous)).sealed();
    let z2 = zero(identity(&z1)).sealed();
    let period = fields(&vectors[1], identity(&z2)).sealed();
    let mut forged = zero(identity(&previous)).encode();
    // The signature's last byte, before the envelope.
    let last = forged.len() - 1;
    forged[last] ^= 1;
    ZeroLengthChain {
        key: hex_of(&recorder::KEY),
        previous: write("zero-previous.tlm", &previous),
        zeros: [write("zero-z1.tlm", &z1), write("zero-z2.tlm", &z2)],
        period: write("zero-period.tlm", &period),
        identity: identity(&period),
        forged: write("zero-forged.tlm", &envelope(0x0402, &forged)),
    }
}

fn hex_of(bytes: &[u8]) -> String {
    to_hex(bytes)
}

/// A manifest's fields (RFC-0015 'Log format', RFC-0060), signed by the test recorder.
struct Manifest {
    log_id: [u8; 16],
    start: u64,
    end: u64,
    first_seq: u64,
    count: u64,
    root: [u8; 32],
    prev: [u8; 32],
}

impl Manifest {
    /// Every field but the signature: 144 bytes.
    fn unsigned(&self) -> Vec<u8> {
        [
            &sha256(&recorder::KEY)[..],
            &self.log_id,
            &self.start.to_le_bytes(),
            &self.end.to_le_bytes(),
            &self.first_seq.to_le_bytes(),
            &self.count.to_le_bytes(),
            &self.root,
            &self.prev,
        ]
        .concat()
    }

    /// `TCE(LogManifest)`, signed: then the signature envelope, scheme 0 (Ed25519), the key and
    /// the signature over `"tessera-manifest/v1" ‖ unsigned`, each `Bytes`.
    fn encode(&self) -> Vec<u8> {
        let unsigned = self.unsigned();
        let signature = recorder::sign(&[&b"tessera-manifest/v1"[..], &unsigned].concat());
        [
            &unsigned[..],
            &0u16.to_le_bytes(),
            &32u64.to_le_bytes(),
            &recorder::KEY,
            &64u64.to_le_bytes(),
            &signature,
        ]
        .concat()
    }

    /// The `.tlm` file, schema `0x0402`.
    fn sealed(&self) -> Vec<u8> {
        envelope(0x0402, &self.encode())
    }
}

/// RFC-0003's envelope of `schema`, version 1, around `payload`, with the test kit's BLAKE3.
pub fn envelope(schema: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &schema.to_le_bytes(),
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

/// The test recorder of `tv-core`'s anchor tests (`tests/anchor_support`): a fixed secret
/// scalar and a fixed nonce, whose points were computed offline with RFC 8032 §5.1's arithmetic,
/// so that a test signs without point arithmetic. It signs nothing but tests.
pub mod recorder {
    use super::sha512;

    /// `A = encode([a]B)` for the secret scalar [`SECRET`].
    pub const KEY: [u8; 32] = [
        0xc8, 0xf4, 0xcf, 0xd7, 0xec, 0x9c, 0x81, 0xe1, 0x6c, 0x77, 0x14, 0x4a, 0x4d, 0x1c, 0xe0,
        0xdc, 0xa0, 0x33, 0x96, 0x61, 0xdf, 0x31, 0x3c, 0xce, 0xb9, 0x20, 0x31, 0x13, 0x46, 0x8f,
        0xb6, 0x08,
    ];
    /// The secret scalar `a`, `0x0707…07` little-endian, below `L`.
    const SECRET: [u8; 32] = [0x07; 32];
    /// The fixed nonce `r`, below `L`.
    const NONCE: [u8; 32] = {
        let mut r = [0x2a; 32];
        r[31] = 0x0a;
        r
    };
    /// `encode([r]B)`.
    const NONCE_POINT: [u8; 32] = [
        0x69, 0xe8, 0xec, 0x7e, 0xb4, 0x58, 0x6b, 0x63, 0xbf, 0x87, 0xa3, 0x9b, 0x21, 0x1e, 0xe4,
        0x7d, 0x66, 0x0d, 0xc5, 0xd3, 0x9c, 0x11, 0xb2, 0x98, 0x1d, 0xe1, 0x88, 0xf1, 0x33, 0x6a,
        0x08, 0xf6,
    ];

    /// `R ‖ S` on `message` (RFC 8032 §5.1.6 with a fixed nonce): `R = encode([r]B)`,
    /// `S = r + k·a mod L`, `k = SHA-512(R ‖ A ‖ message) mod L`.
    pub fn sign(message: &[u8]) -> [u8; 64] {
        let k = reduce(&sha512(&[&NONCE_POINT[..], &KEY, message].concat()));
        let s = add(&reduce(&NONCE), &mul(&k, &reduce(&SECRET)));
        let mut signature = [0; 64];
        signature[..32].copy_from_slice(&NONCE_POINT);
        for (i, limb) in s.iter().enumerate() {
            signature[32 + 8 * i..40 + 8 * i].copy_from_slice(&limb.to_le_bytes());
        }
        signature
    }

    /// An integer below `2^256`, least significant limb first.
    type Scalar = [u64; 4];

    /// The group order `L = 2^252 + 27742317777372353535851937790883648493`.
    const L: Scalar = [
        0x5812_631a_5cf5_d3ed,
        0x14de_f9de_a2f7_9cd6,
        0,
        0x1000_0000_0000_0000,
    ];

    /// `a + b mod L`, for `a, b < L`: their sum is below `2^254`.
    fn add(a: &Scalar, b: &Scalar) -> Scalar {
        let mut s = [0; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            let t = u128::from(a[i]) + u128::from(b[i]) + carry;
            s[i] = t as u64;
            carry = t >> 64;
        }
        let below = (0..4)
            .rev()
            .find(|&i| s[i] != L[i])
            .is_some_and(|i| s[i] < L[i]);
        if below {
            return s;
        }
        let mut out = [0; 4];
        let mut borrow = false;
        for i in 0..4 {
            let (t, b1) = s[i].overflowing_sub(L[i]);
            let (t, b2) = t.overflowing_sub(u64::from(borrow));
            out[i] = t;
            borrow = b1 || b2;
        }
        out
    }

    /// `a · b mod L`, for `a < L`: double and add over `b`'s bits.
    fn mul(a: &Scalar, b: &Scalar) -> Scalar {
        let mut x = [0; 4];
        for i in (0..256).rev() {
            x = add(&x, &x);
            if (b[i / 64] >> (i % 64)) & 1 == 1 {
                x = add(&x, a);
            }
        }
        x
    }

    /// A little-endian integer of any length, modulo `L`: Horner's rule over its bits.
    fn reduce(bytes: &[u8]) -> Scalar {
        let mut x = [0; 4];
        for byte in bytes.iter().rev() {
            for bit in (0..8).rev() {
                x = add(&x, &x);
                if (byte >> bit) & 1 == 1 {
                    x = add(&x, &[1, 0, 0, 0]);
                }
            }
        }
        x
    }
}
