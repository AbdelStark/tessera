// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Per-proof private BLAKE3 XOF streams under RFC-0004 domains (`tessera/v1/{advice-masks,
//! prover-share,ni-session}`, #567); only this reader touches the heap-resident per-proof seed.
//!
//! # Why `Stream` replaces `blake3::OutputReader` (OBS-2)
//!
//! `blake3` 1.8 offers no way to build or read an `OutputReader` without copying secret state onto
//! the stack:
//! - `blake3::Hasher::finalize_xof(&self) -> OutputReader` (`blake3 1.8.7`, `src/lib.rs:1413`)
//!   returns `OutputReader` by value, whose `inner.block[..32]` holds the 32-byte secret seed,
//!   before `Box::new` can move it to the heap, and `OutputReader::new` is private;
//! - `blake3::OutputReader::fill` for slices shorter than 64 bytes (`[u8; 8]`, `[u8; 32]`,
//!   `[u8; 16]`) calls `fill_one_block` (`src/lib.rs:1700`), which materializes the 64-byte root
//!   output block in a stack local `[u8; 64]` and never zeroizes the unread bytes;
//! - `blake3::OutputReader` derives `Clone`.
//!
//! Every per-proof stream here hashes a single 32-byte secret seed (`block_len = 32`, one chunk,
//! one block) under a public context key `ProverDomain::context_key`. [`Heap::zeroed`] allocates a
//! zeroed [`Stream`] on the heap first, [`Stream::init`] writes the public context key and copies
//! the 32-byte seed straight from [`HeapKey`] into `Stream::block` on the heap, and
//! [`Stream::refill`] runs BLAKE3's 7-round root compression in place inside `Stream`'s heap
//! buffers. Moving [`Seed`] or [`Masks`] moves a single pointer, neither implements `Clone` or
//! `Debug`, and dropping either zeroizes every buffer on the heap.

use core::borrow::{Borrow, BorrowMut};

use super::{ErrorCode, MaskRng, ProverError};
use tessera_codec::{HeapKey, Secret};
use tessera_field::{Goldilocks, PrimeField};
use tessera_hash::ProverDomain;
use zeroize::Zeroize;

const IV_0_3: [u32; 4] = [0x6A09_E667, 0xBB67_AE85, 0x3C6E_F372, 0xA54F_F53A];
const MSG_PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const MSG_SCHEDULE: [[usize; 16]; 7] = {
    let mut schedule = [[0; 16]; 7];
    let mut cur = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut round = 0;
    while round < 7 {
        schedule[round] = cur;
        let mut next = [0; 16];
        let mut i = 0;
        while i < 16 {
            next[i] = cur[MSG_PERM[i]];
            i += 1;
        }
        cur = next;
        round += 1;
    }
    schedule
};
/// Single-chunk, single-block root output in key-derivation-material mode (BLAKE3 §2.1, §2.6):
/// `CHUNK_START (1) | CHUNK_END (2) | ROOT (8) | DERIVE_KEY_MATERIAL (64)`.
const ROOT_DERIVE_FLAGS: u32 = 1 | 2 | 8 | 64;
const SEED_LEN: u32 = 32;
const BLOCK_WORDS: usize = 8;

/// Heap-resident single-block BLAKE3 extended-output state. Constructed zeroed in a `Box` and
/// initialized in place so no secret state or 64-byte output block ever passes through a stack
/// return value.
struct Stream {
    cv: [u32; 8],
    block: [u32; 16],
    work: [u32; 16],
    words: [u64; BLOCK_WORDS],
    counter: u64,
    pos: usize,
}

impl Zeroize for Stream {
    fn zeroize(&mut self) {
        let Self {
            cv,
            block,
            work,
            words,
            counter,
            pos,
        } = self;
        cv.zeroize();
        block.zeroize();
        work.zeroize();
        words.zeroize();
        counter.zeroize();
        pos.zeroize();
    }
}

struct Heap(Box<Stream>);

impl Heap {
    fn zeroed() -> Self {
        Self(Box::new(Stream {
            cv: [0; 8],
            block: [0; 16],
            work: [0; 16],
            words: [0; BLOCK_WORDS],
            counter: 0,
            pos: BLOCK_WORDS,
        }))
    }
}

impl Zeroize for Heap {
    fn zeroize(&mut self) {
        self.0.as_mut().zeroize();
    }
}

#[inline(always)]
fn g(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
    s[d] = (s[d] ^ s[a]).rotate_right(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(12);
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
    s[d] = (s[d] ^ s[a]).rotate_right(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(7);
}

impl Stream {
    fn init(&mut self, domain: ProverDomain, seed: &[u8; 32]) {
        let key = domain.context_key();
        for (dst, chunk) in self.cv.iter_mut().zip(key.chunks_exact(4)) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            *dst = u32::from_le_bytes(bytes);
        }
        for (dst, chunk) in self.block[..8].iter_mut().zip(seed.chunks_exact(4)) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            *dst = u32::from_le_bytes(bytes);
        }
        self.block[8..].fill(0);
        self.counter = 0;
        self.pos = BLOCK_WORDS;
    }

    fn refill(&mut self) {
        self.work[..8].copy_from_slice(&self.cv);
        self.work[8..12].copy_from_slice(&IV_0_3);
        self.work[12] = self.counter as u32;
        self.work[13] = (self.counter >> 32) as u32;
        self.work[14] = SEED_LEN;
        self.work[15] = ROOT_DERIVE_FLAGS;

        for s in &MSG_SCHEDULE {
            g(
                &mut self.work,
                0,
                4,
                8,
                12,
                self.block[s[0]],
                self.block[s[1]],
            );
            g(
                &mut self.work,
                1,
                5,
                9,
                13,
                self.block[s[2]],
                self.block[s[3]],
            );
            g(
                &mut self.work,
                2,
                6,
                10,
                14,
                self.block[s[4]],
                self.block[s[5]],
            );
            g(
                &mut self.work,
                3,
                7,
                11,
                15,
                self.block[s[6]],
                self.block[s[7]],
            );
            g(
                &mut self.work,
                0,
                5,
                10,
                15,
                self.block[s[8]],
                self.block[s[9]],
            );
            g(
                &mut self.work,
                1,
                6,
                11,
                12,
                self.block[s[10]],
                self.block[s[11]],
            );
            g(
                &mut self.work,
                2,
                7,
                8,
                13,
                self.block[s[12]],
                self.block[s[13]],
            );
            g(
                &mut self.work,
                3,
                4,
                9,
                14,
                self.block[s[14]],
                self.block[s[15]],
            );
        }
        for i in 0..8 {
            self.work[i] ^= self.work[i + 8];
            self.work[i + 8] ^= self.cv[i];
        }
        for (i, dst) in self.words.iter_mut().enumerate() {
            *dst = u64::from(self.work[2 * i]) | (u64::from(self.work[2 * i + 1]) << 32);
        }
        self.work.zeroize();
        self.counter += 1;
        self.pos = 0;
    }

    fn next_u64(&mut self) -> u64 {
        if self.pos == BLOCK_WORDS {
            self.refill();
        }
        let pos = self.pos;
        self.pos = pos + 1;
        let word = self.words[pos];
        self.words[pos].zeroize();
        word
    }
}

// Two simultaneous stream owners conservatively cover a mask reader and one
// temporary share/session reader beside the per-proof heap key (OBS-6).
pub(super) const fn heap_bytes() -> u64 {
    32 + 2 * size_of::<Stream>() as u64
}

pub(super) struct Seed(Secret<HeapKey<32>>);
pub(super) struct Masks(Secret<Heap>);

impl Seed {
    #[allow(clippy::disallowed_methods)]
    pub(super) fn new(mode: MaskRng) -> Result<Self, ProverError> {
        let mut seed = Secret::new(HeapKey::<32>::zeroed());
        let bytes: &mut [u8; 32] = seed.expose_secret_mut().borrow_mut();
        match mode {
            MaskRng::OsRng => {
                getrandom::fill(bytes).map_err(|_| ProverError::Resource(ErrorCode::ResIo))?
            }
            #[cfg(feature = "test-rng")]
            MaskRng::Seeded(mut state) => {
                for chunk in bytes.chunks_exact_mut(8) {
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
    fn stream(&self, domain: ProverDomain) -> Secret<Heap> {
        let mut reader = Secret::new(Heap::zeroed());
        reader
            .expose_secret_mut()
            .0
            .init(domain, self.0.expose_secret().borrow());
        reader
    }

    pub(super) fn masks(&self) -> Masks {
        Masks(self.stream(ProverDomain::AdviceMasks))
    }

    #[allow(clippy::disallowed_methods)]
    pub(super) fn share(&self) -> [u8; 32] {
        let mut out = [0; 32];
        let mut reader = self.stream(ProverDomain::ProverShare);
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&reader.expose_secret_mut().0.next_u64().to_le_bytes());
        }
        out
    }

    #[allow(clippy::disallowed_methods)]
    pub(super) fn session(&self) -> [u8; 16] {
        let mut out = [0; 16];
        let mut reader = self.stream(ProverDomain::NiSession);
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&reader.expose_secret_mut().0.next_u64().to_le_bytes());
        }
        out
    }
}

impl tessera_pcs::MaskSource for Masks {
    #[allow(clippy::disallowed_methods)]
    fn draw(&mut self) -> Goldilocks {
        // Allowed by INV-ARCH-2's RFC-0004 stream exception: fixed public test `w < P` on one
        // candidate word from `tessera/v1/advice-masks`, with rejected words discarded whole.
        field(|| self.0.expose_secret_mut().0.next_u64())
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
    use zeroize::Zeroizing;

    // Compile-time check that neither Seed, Masks, Heap nor Stream implements Clone or Debug:
    // if any gains Clone or Debug, `Check::<T>::ABSENT` becomes ambiguous with `Present::ABSENT`
    // and fails to compile (OBS-2).
    trait Absent {
        const ABSENT: bool = true;
    }
    impl<T> Absent for Check<T> {}
    struct Check<T>(core::marker::PhantomData<T>);
    impl<T: Clone> Check<T> {
        #[allow(dead_code)]
        const ABSENT: bool = false;
    }
    trait AbsentDebug {
        const NO_DEBUG: bool = true;
    }
    impl<T> AbsentDebug for CheckDebug<T> {}
    struct CheckDebug<T>(core::marker::PhantomData<T>);
    impl<T: core::fmt::Debug> CheckDebug<T> {
        #[allow(dead_code)]
        const NO_DEBUG: bool = false;
    }
    const _: () = {
        assert!(Check::<Seed>::ABSENT);
        assert!(Check::<Masks>::ABSENT);
        assert!(Check::<Heap>::ABSENT);
        assert!(Check::<Stream>::ABSENT);
        assert!(CheckDebug::<Seed>::NO_DEBUG);
        assert!(CheckDebug::<Masks>::NO_DEBUG);
        assert!(CheckDebug::<Heap>::NO_DEBUG);
        assert!(CheckDebug::<Stream>::NO_DEBUG);
    };

    #[test]
    fn rejects_noncanonical_words_instead_of_reducing_them() {
        let words = [
            u64::MAX,
            Goldilocks::MODULUS,
            Goldilocks::MODULUS + 1,
            Goldilocks::MODULUS - 1,
        ];
        let mut n = 0;
        let v = field(|| {
            let v = words[n];
            n += 1;
            v
        });
        assert_eq!(n, 4);
        assert_eq!(v.to_canonical_u64(), Goldilocks::MODULUS - 1);
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

    fn hex_bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn hex_u64(text: &str) -> u64 {
        u64::from_str_radix(text.trim_start_matches("0x"), 16).unwrap()
    }

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn streams_use_rfc0004_domains_and_match_oracle_including_real_rejection() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spec/vectors/hash/hash.json");
        let root: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let vectors = &root["prover_streams"];

        let raw_seed = hex_bytes(vectors["proof_seed"].as_str().unwrap());
        let mut seed = Seed::new(MaskRng::OsRng).unwrap();
        let dst: &mut [u8; 32] = seed.0.expose_secret_mut().borrow_mut();
        dst.copy_from_slice(&raw_seed);

        assert_eq!(
            seed.share().as_slice(),
            hex_bytes(vectors["prover_share"].as_str().unwrap()).as_slice()
        );
        assert_eq!(
            seed.session().as_slice(),
            hex_bytes(vectors["ni_session"].as_str().unwrap()).as_slice()
        );
        let expected_masks: Vec<u64> = vectors["advice_masks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| hex_u64(v.as_str().unwrap()))
            .collect();
        let mut masks = seed.masks();
        let drawn: SecretBuf<u64> = (0..expected_masks.len())
            .map(|_| masks.draw().to_canonical_u64())
            .collect();
        assert_eq!(&drawn[..], expected_masks.as_slice());

        let rej = &vectors["rejection"];
        let rej_seed = hex_bytes(rej["seed"].as_str().unwrap());
        let dst: &mut [u8; 32] = seed.0.expose_secret_mut().borrow_mut();
        dst.copy_from_slice(&rej_seed);
        let expected_rej: Vec<u64> = rej["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| hex_u64(v.as_str().unwrap()))
            .collect();
        let mut rej_masks = seed.masks();
        let drawn_rej: SecretBuf<u64> = (0..expected_rej.len())
            .map(|_| rej_masks.draw().to_canonical_u64())
            .collect();
        assert_eq!(&drawn_rej[..], expected_rej.as_slice());
    }

    #[test]
    fn heap_stream_matches_blake3_xof_across_blocks_and_domains() {
        let mut state = 0x5670_0004_u64;
        for _ in 0..32 {
            let mut key = [0u8; 32];
            for chunk in key.chunks_exact_mut(8) {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes());
            }
            for domain in ProverDomain::ALL {
                let mut heap = Heap::zeroed();
                heap.0.init(domain, &key);
                let mut got = [0u8; 512];
                for chunk in got.chunks_exact_mut(8) {
                    chunk.copy_from_slice(&heap.0.next_u64().to_le_bytes());
                }
                let mut expected = [0u8; 512];
                let mut b3 = blake3::Hasher::new_derive_key(domain.context());
                b3.update(&key);
                b3.finalize_xof().fill(&mut expected);
                assert_eq!(got, expected, "{domain:?}");
            }
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn reader_is_initialized_in_place_on_heap_and_zeroized_completely() {
        assert_eq!(size_of::<Seed>(), size_of::<usize>());
        assert_eq!(size_of::<Masks>(), size_of::<usize>());

        let mut reader = Secret::new(Heap::zeroed());
        let addr_before = (&raw const *reader.expose_secret().0) as usize;
        let key = [0xA5_u8; 32];
        reader
            .expose_secret_mut()
            .0
            .init(ProverDomain::AdviceMasks, &key);
        let addr_after_init = (&raw const *reader.expose_secret().0) as usize;
        assert_eq!(addr_before, addr_after_init);

        let moved = reader;
        let addr_after_move = (&raw const *moved.expose_secret().0) as usize;
        assert_eq!(addr_before, addr_after_move);

        let mut masks = Masks(moved);
        let _ = masks.draw();
        assert_eq!(
            (&raw const *masks.0.expose_secret().0) as usize,
            addr_before
        );
        // Temporary work buffer and consumed output words are cleared immediately on the heap.
        assert_eq!(masks.0.expose_secret().0.work, [0; 16]);
        assert_eq!(masks.0.expose_secret().0.words[0], 0);

        masks.0.expose_secret_mut().zeroize();
        let s = &*masks.0.expose_secret().0;
        assert_eq!(s.cv, [0; 8]);
        assert_eq!(s.block, [0; 16]);
        assert_eq!(s.work, [0; 16]);
        assert_eq!(s.words, [0; BLOCK_WORDS]);
        assert_eq!(s.counter, 0);
        assert_eq!(s.pos, 0);
    }
}
