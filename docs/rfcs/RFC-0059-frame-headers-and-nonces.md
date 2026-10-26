# RFC-0059: Frame Headers, Nonces and the Session Counter

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.4
- Supersedes in the listed sections:
  - [RFC-0015](RFC-0015-transcript-anchor.md#frames) 'Frames': the header's fields and size, the CTR counter blocks, `frame_seq`, `TokenFrame`'s encoding, the commitments' inputs;
  - [RFC-0015](RFC-0015-transcript-anchor.md#log-format) 'Log format': `DigestLogEntry::Frame` gains `frame_seq`;
  - [RFC-0015](RFC-0015-transcript-anchor.md#circuits) 'Circuits': the counter blocks and the plaintext's prefix are public inputs, and the sampler commitment is opened;
  - [RFC-0015](RFC-0015-transcript-anchor.md#verifier-checks) 'Verifier checks', checks 2 and 4: a session's counters are distinct, a frame's length is one a plaintext has, a session's open precedes its frames, and the counter blocks and plaintext prefixes are public inputs;
  - [RFC-0015](RFC-0015-transcript-anchor.md#prover-pre-check) 'Prover pre-check': it recomputes the sampler commitment too;
  - [RFC-0043](RFC-0043-rejection-payloads.md#anchor) 'Anchor': `Frames` covers a session's repeated counter and a session-open entry after its frames.
- Origin: [#137](https://github.com/AbdelStark/tessera/issues/137)
- Amended by: [RFC-0061](RFC-0061-sampling-randomness.md): a session commits no sampler key. The session-open frame is 52 bytes and its entry drops `sampler_commitment`; the circuits open the sampler key against the commitment the verifier pins.
- Amended by: [RFC-0062](RFC-0062-period-log-sessions.md): 'Session identity' and 'Verifier check 2, amended': the session rules over the period's log settle what this RFC left to #498, and fix which request and kind a break names.
- Amended by: draft [RFC-0074](RFC-0074-anchor-binding.md) ('The circuits': the zeros past a frame's length in its last AES block are public inputs; in effect when it is accepted)

## Summary

RFC-0015 encrypts each token frame under AES-128-CTR with the nonce `session_id[0..8] ‖ u32(frame_seq)`, where `frame_seq` is the frame's position in its session. That position is in no public record. Neither the frame's header nor its log entry carries it, so the verifier cannot build the counter blocks that are the AES circuit's public inputs. Reading the whole log would not help. A session's inbound and outbound frames cross the recorder on separate paths, in an order that need not be the order the serving stack sealed them in, so the log's order does not give the counter.

This RFC makes the counter public and fixes the encodings RFC-0015 left open:
- **The header** carries `frame_seq`. It is the 41-byte TCE encoding of its fields. RFC-0015 said 40 bytes, which its listed fields did not add up to (37).
- **The log entry** of a frame carries `frame_seq`, which the recorder copies from the header.
- **The counter blocks** are `session_id[0..8] ‖ u32_be(frame_seq) ‖ u32_be(i)`, with NIST SP 800-38A's standard incrementing function on the last 32 bits.
- **`TokenFrame`'s encoding** is fixed, and so are its agreement with the header and the commitments' exact inputs, with 16-byte salts.
- **The reference cipher** draws the session's identifier and secrets and keeps the counter itself, so it never encrypts two plaintexts under one counter (INV-ANC-3).
- **The sampler commitment** is opened by a circuit, which binds the sampler PRF's key to it. RFC-0015 committed `K_samp` and never checked it.

Two related gaps are left to their own issues, since closing them means rules over the whole log or a new PRF input:
- **What the log must show of a session.** Its opens, duplicate frames and periods are [#498](https://github.com/AbdelStark/tessera/issues/498), and the completeness of a request's frames is [#496](https://github.com/AbdelStark/tessera/issues/496).
- **The PRF's free inputs.** The serving stack chooses a prompt's `request_id`, and which of its open sessions, so which `K_samp`, serves it, after the prompt arrives. A prover can grind either ([#497](https://github.com/AbdelStark/tessera/issues/497)).

## Motivation

[#137](https://github.com/AbdelStark/tessera/issues/137) implements RFC-0015's frames. Implementing them found six gaps:

1. **An unknown nonce.** The AES-128-CTR circuit (RFC-0015 'Circuits', step 2) computes each frame's keystream from counter blocks, which must be public inputs fixed by the verifier. The log's order cannot give `frame_seq`, as the Summary says.
   - **Advice instead?** Taking the counter as the prover's advice would leave a circuit input to the prover. A wrong counter rarely gives a valid decryption: the plaintext's 22-byte prefix must equal the header's public fields, which a wrong keystream matches with probability about `2^{-176}`, or `2^{-144}` over all `2^32` counters. But a public input chosen by the prover is not a public input, and nothing would tie the circuit's counter to the one RFC-0015's invariant protects.
2. **The header's size.** "40 bytes" against fields of 4 + 16 + 8 + 1 + 4 + 4 = 37 bytes.
3. **Byte order.** `u32(frame_seq)` and the CTR counter's width and byte order were unspecified, as were the salts' lengths.
4. **Secret tokens.** `TokenFrame` holds tokens, which are secret-class (05 §3). Its TCE bytes, the plaintext, are therefore derived secret data.
5. **Session identity.** The session identifier was the caller's, with no rule that it be unique, and no check placed a session's open before its frames (INV-ANC-2 named the property without a check). Requests are keyed by session and request identifier (RFC-0016), and honest sessions sharing an identifier would share counters from 0.
6. **An unopened commitment.** RFC-0015 commits `K_samp` at session open, so that exact mode's sampling randomness is fixed before the prompts (RFC-0017 'Sampler PRF'). Its circuits hash the key commitment only. Nothing showed that the key the in-circuit PRF uses is the committed one.

## Goals

- Every input of a frame's AES and SHA-256 circuits is either public, read from the frame's signed log entry, or committed.
- One encoding per header, frame and entry (INV-TCE-1).
- INV-ANC-3 holds by construction in the reference implementation.

## Non-Goals

- The circuits (#142, #143), the recorder (#139), the manifests (#138) and the verifier's anchor checks (#150).
- Hiding request boundaries or lengths, which RFC-0015 already makes public.

## Proposed Design

### The header

```rust
pub struct FrameHeader {        // cleartext, 41 bytes: its TCE encoding
    pub magic: [u8; 4],         // b"TSFR"
    pub session_id: [u8; 16],
    pub request_id: u64,
    pub direction: u8,          // 0 = In, 1 = Out; any other value is refused
    pub frame_index: u32,
    pub length: u32,            // the ciphertext's length in bytes, = |TCE(TokenFrame)|
    pub frame_seq: u32,         // the session counter of 'The session counter'
}
```

All integers are little-endian, as TCE encodes them, and the header is exactly these 41 bytes. A decoder refuses:
- another magic;
- a direction other than 0 or 1;
- a `length` that is not `22 + 4n` for any `n`, which no plaintext has;
- a frame whose ciphertext is not `length` bytes.

### The session counter

- **What it counts.** `frame_seq` counts the frames a session's serving stack has sealed, in both directions, from 0, in the order it seals them.
- **Its range.** A session seals at most `2^32` frames. The frame after `frame_seq = 2^32 − 1` is refused, never wrapped.
- **Never reused.** No two frames of a session share a `frame_seq` (INV-ANC-3).
- **Not contiguous per request.** The values of one request's frames are increasing but not, in general, contiguous: the session interleaves requests and directions. `frame_index` remains each request's per-direction position, contiguous from 0 (RFC-0015 'Verifier checks', check 2).

### The keystream

For a frame with counter `s`, block `i` of the keystream, for `i` from 0, is

```text
K_i = AES-128_{k_s}( session_id[0..8] ‖ u32_be(s) ‖ u32_be(i) )
```

and `ciphertext = TCE(TokenFrame) ⊕ (K_0 ‖ K_1 ‖ …)[0..length]`.

- **The counter blocks** are NIST SP 800-38A's, with the standard incrementing function applied to the block's last 32 bits. Its test F.5.1, an initial block `f0f1…feff`, exercises exactly that function.
- **Big-endian here.** Elsewhere Tessera encodes integers little-endian. Big-endian here keeps the 16-byte block what a standard CTR implementation builds from the 12-byte nonce `session_id[0..8] ‖ u32_be(s)`.
- **The block counter's room.** A frame's ciphertext is at most `2^32 − 1` bytes, its `length`. That is at most `2^28` blocks, so the 32-bit block counter never wraps within a frame. For the same reason, the 32-, 64- and 128-bit incrementing functions agree on every frame. This RFC specifies the 32-bit one.
- **Distinct blocks.** Two frames of a session use disjoint counter blocks, since their `s` differ. Two sessions use independent keys.

### The plaintext

`TCE(TokenFrame)` is

```text
u64_le(request_id) ‖ u16_le(direction tag: 0 = In, 1 = Out) ‖ u32_le(frame_index) ‖ u64_le(n) ‖ u32_le(token_0) ‖ … ‖ u32_le(token_{n−1})
```

- **Shape.** Its length is `22 + 4n` bytes, and `n ≥ 0`.
- **Agreement with the header.** The plaintext's `request_id`, direction and `frame_index` equal the header's, and its count is `(length − 22)/4`. A decrypted frame that disagrees is refused. The circuits fix the plaintext's whole 22-byte prefix to these public values ('The circuits').
- **Secrecy.** The plaintext and the tokens are secret-class (05 §3). The ciphertext, the header and their digest are public.

### The session-open frame and the commitments

```text
SessionOpenFrame = b"TSOP" ‖ session_id ‖ key_commitment                          52 bytes, its TCE (RFC-0061)
key_commitment     = SHA-256( "tessera-anchor-key/v1"   ‖ k_s ‖ salt_k )     21 + 16 + 16 bytes hashed
sampler_commitment = SHA-256( "tessera-sampler-key/v1" ‖ K_p ‖ salt   )     22 + 16 + 16 bytes hashed, pinned by the verifier (RFC-0061)
```

- **The labels** are ASCII, without a length or a terminator. Every part has a fixed length, so the concatenation parses one way.
- **The salts** are 16 bytes each.
- **The secrets.** `k_s` and `salt_k` are drawn from the operating system's generator at session open, fresh for every session (A-ANCHOR-3). The sampler's base key `K_p` and its salt are no session's: the verifier pins their commitment ([RFC-0061](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins)).
- **Classes.** All four are secret-class, and the commitments are public.

### Session identity

- **The identifier.** A session's `session_id` is 16 bytes drawn from the operating system's generator at session open, with its secrets. The reference cipher takes none from its caller. Its first eight bytes, which the counter blocks carry, repeat across `N` sessions with probability below `N^2/2^65`.
- **What the verifier checks.** As this RFC had it, a session's `SessionOpen` entry in the batch has a lower `seq` than every frame of the session in the batch (INV-ANC-2, check 2 below), which is what a batch's own entries can show. [RFC-0062](RFC-0062-period-log-sessions.md#the-session-rules) reads it over the period's log, with carries.
- **What this RFC did not settle,** and [RFC-0062](RFC-0062-period-log-sessions.md#the-session-rules) does, over the period's log:
  - a session with no open, or with several;
  - two frames with one `frame_index`;
  - a session whose open is in an earlier period's manifest;
  - what the recorder does with any of these.

  Those need rules over the whole log, which RFC-0016's selection already gives the verifier. Neither RFC makes a recorder rule that drops anything: one that dropped frames would let them be served without being logged. RFC-0062's carry entries only add to the log.

### The circuits

To RFC-0015 'Circuits':
- **The counter blocks** of step 2 are public inputs, built by the verifier from the frame's entry: `session_id[0..8] ‖ u32_be(frame_seq) ‖ u32_be(i)`.
- **The plaintext's prefix**, its first 22 bytes, is a public input too: `request_id`, the direction tag, `frame_index` and the count `(length − 22)/4`, from the entry. Only the token bytes after it are the prover's, and draft [RFC-0074](RFC-0074-anchor-binding.md#the-token-byte-binding) makes the zeros past `length` in the frame's last AES block public too.
- **The header** of step 5, which enters SHA-256 as constants, includes `frame_seq`.
- **The sampler commitment** is opened. SHA-256 over `"tessera-sampler-key/v1" ‖ K_p ‖ salt`, with the base key `K_p` and the salt committed, equals the batch's pinned `sampler_commitment` ([RFC-0061](RFC-0061-sampling-randomness.md#the-circuits)). The key the in-circuit PRF uses (RFC-0017 'Sampler PRF', #152) is derived from `K_p` and the pin's nonce, class bits and start.

Check 4's public inputs are, then, the entries' digests, the sessions' key commitments, the pin's commitment, nonce, class bits and start, the counter blocks and the plaintexts' prefixes, and the zeros past each frame's length in its last block (draft RFC-0074), all supplied by the verifier.

### The log entry

`DigestLogEntry::Frame` (schema `0x0401`, version 1) is

```rust
Frame { seq: u64, t_ns: u64, session_id: [u8; 16], request_id: u64, direction: Direction,
        frame_index: u32, length: u32, frame_seq: u32, sha256: [u8; 32] }
```

The fields shared with the header come in the header's order, `frame_seq` after `length`.

- **The digest.** `sha256 = SHA-256(header ‖ ciphertext)`, over the 41-byte header of 'The header'.
- **Where the fields come from.** The recorder copies every field but `seq`, `t_ns` and `sha256` from the header.
- **The other variants.** `SessionClose` is unchanged. `SessionOpen` loses its `sampler_commitment` ([RFC-0061](RFC-0061-sampling-randomness.md#sessions)).
- **Encoding.** As every TCE enum, the entry is a `u16` tag, then the variant's fields in order (RFC-0003 'Composite types').

### Verifier check 2, amended

To RFC-0015's check 2 add:
- the `frame_seq` values of a session's frames in the batch are distinct;
- each frame entry's `length` is `22 + 4n` for some `n`;
- a session's `SessionOpen` entry has a lower `seq` than each of the session's frames in the batch.

A failure is `Reject(Anchor { request_id, kind: Frames })` ([RFC-0043](RFC-0043-rejection-payloads.md#anchor)). The verifier checks requests in `statement.batch` order, as RFC-0043 says. It names the first request whose own frames break a rule, a counter counting as repeated when an earlier request of the session, or an earlier frame of this one, already used it. Of two requests sharing a counter, the later one in the batch is named. The amended check is #150's.

[RFC-0062](RFC-0062-period-log-sessions.md#the-session-rules) supersedes, in this section, the third item of the list (the open before the session's frames, now its rule 1 and `Session`) and this paragraph's naming sentences. Distinct counters hold over all the session's frames in the period, and a session that breaks a rule names the first request of the batch in it.

### The reference cipher

`tessera-anchor` holds the session key in a cipher that:
- draws the session's identifier, `k_s` and `salt_k` at session open;
- emits the session-open frame;
- seals frames with the next counter, which it keeps.

It has no operation that takes a counter, a key or an identifier from its caller, and it cannot be cloned.
- **What it guarantees.** It never encrypts two plaintexts under one counter of its session (INV-ANC-3), and its sessions never share a key.
- **What it does not.** It cannot stop its caller from misusing what it returns. `seal`'s caller knows the plaintext and gets the ciphertext, so it holds their XOR, the keystream of every spent counter up to that frame's length. It could seal its own frame under a spent counter by hand, which would leak the XOR of two plaintexts to the network. INV-ANC-3 binds the serving stack, and the reference cipher is that stack's one encrypting path.
- **No decryption.** It offers none, since the protocol needs no decryption in the clear: the circuits check the prefix. It decrypts in its own tests only.

**Other code that holds the keys** must keep what it derives in clearing storage and never emit it:
- **The pre-check** (RFC-0015 'Prover pre-check', #146) recomputes a stored session's digests and key commitment, and the pinned sampler commitment. Its replay takes the session's keys and its recorded frames, re-seals each plaintext at the frame's recorded counter, and returns only whether each digest matches, never a ciphertext or a keystream.
- **The witness generators** of the AES-CTR circuit (#142) and of the sampler PRF (#152) evaluate AES under `k_s` at public counter blocks and under the sampler key `K` at the blocks of RFC-0061 'The stream'. The keystream they compute is witness data, secret-class (05 §3, OBS-6), and leaves them only committed.

Storing `SessionKeys` is #146's: it is a secret-class artifact, the `SessionTranscript` of [RFC-0015](RFC-0015-transcript-anchor.md#prover-pre-check) 'Prover pre-check'.

## Security Argument

- **The nonce is bound.** `frame_seq` is in the header, which the frame's digest covers, and in its log entry, which the recorder signs into the manifest (RFC-0015 'Signatures'). A prover cannot claim a counter other than the one recorded for the bytes it sent.
- **No choice of keystream.** With the counter blocks and the plaintext's prefix public, the AES circuit's keystream for a logged ciphertext is fixed by `k_s`, which `key_commitment` binds. The tokens are then `ciphertext[22..] ⊕ keystream[22..]`.
- **The sampler key, as this RFC had it, was bound to its session.** That did not fix the sampling randomness before the prompt: the serving stack picks the prompt's `request_id`, and the session that serves it, after the prompt arrives (#497). [RFC-0061](RFC-0061-sampling-randomness.md) replaces both with a keyed class of the prompt, of `b` public bits, under a key whose commitment the verifier pins.
- **Confidentiality.**
  - Within a session, distinct counters give disjoint counter blocks: no keystream is reused, which is the property INV-ANC-3 protects.
  - A recorder that sees two frames with one counter in a session has seen a misbehaving serving stack. Check 2 refuses such a pair in a batch.
- **Encodings.** The header, the plaintext and the entry each have one encoding: every field has a fixed width or a length prefix, and the decoders refuse every value outside its domain.

## Alternatives Considered

1. **The recorder counts frames.** It cannot. A session's two directions reach it on separate paths, in an order unrelated to the serving stack's sealing. (RFC-0063 later required it to sequence inbound frames before the cluster reads them, which orders a request's inbound frames before its outbound ones but not the sealing counters.)
2. **A private counter**, proven by the circuit. That gives `2^32` decryptions per logged ciphertext at the prover's choice, in exchange for nothing.
3. **A nonce derived from `(request_id, direction, frame_index)`**, which the entry already holds. That is 13 bytes, more than a 96-bit CTR nonce even without `session_id[0..8]`, which per-session keys make unnecessary there. Hashing it would leave standard CTR and add SHA-256 work to the circuit.
4. **Keeping 40 bytes, with three reserved bytes after `direction`.** That carries no counter, and adds a field whose only value is zero.

## Drawbacks

- **Size.** Four more bytes per header and per entry.
- **Byte order.** The keystream uses big-endian counters, where the rest of Tessera is little-endian.

## Migration / Rollout

Nothing implemented RFC-0015's frames before this RFC.
- **#137** implements the header, the plaintext, the session-open frame, the entry, the commitments and the reference cipher.
- **#138, #139, #142, #143, #146 and #150** build on them.

## Testing Strategy

- **The keystream.** NIST SP 800-38A F.5.1, CTR-AES128.Encrypt, with its initial counter block.
- **Shared vectors** from an independent oracle in standard-library Python, AES-128 from FIPS-197 and SHA-256 from `hashlib`, for several sessions:
  - the commitments;
  - each frame's header, plaintext, ciphertext and digest;
  - each entry's encoding.

  The sessions include counters of `0x01020304` and `2^32 − 1`, so that every byte of the block's counter word is exercised.
- **INV-ANC-3.** A session's sealed frames take counters 0, 1, 2, …, in sealing order across both directions. The frame after `2^32 − 1` is refused.
- **Decoders.** Each rejection has a test:
  - the wrong magic;
  - a direction of 2, or a tag whose high byte is set;
  - a truncated or extended header, frame or plaintext;
  - a header `length` that is not `22 + 4n`;
  - a header and plaintext that disagree;
  - an entry with a bad tag.
- **Identity.** Two sessions differ in identifier, key and key commitment.

## Open Questions

- **OQ-0059-1.** RFC-0015's check 2 requires every frame of a sampled request. Inclusion proofs show that an entry is in the log, not that no later frame of the request is. A request's last frame, or its session's close, would have to bound it. Owner: @AbdelStark. Tracked in #496. It must be resolved before the recorder (#139), the circuits (#142, #143), the pre-check (#146) and the verifier (#150) are built. Two of its options change shipped encodings: a request's end marker in the header or the public prefix, or frame counts in the session's close entry.

## References

- [RFC-0015](RFC-0015-transcript-anchor.md) 'Frames', 'Log format', 'Circuits', 'Verifier checks', 'Invariants'.
- [RFC-0003](RFC-0003-canonical-encoding.md) 'Composite types'.
- [RFC-0016](RFC-0016-audit-sampling.md) (requests keyed by session and request), [RFC-0043](RFC-0043-rejection-payloads.md#anchor) 'Anchor'.
- [RFC-0017](RFC-0017-scoring.md#sampler-prf) 'Sampler PRF'.
- NIST SP 800-38A, 'Recommendation for Block Cipher Modes of Operation' (CTR mode, Appendix B.1 and F.5.1); FIPS 180-4 (SHA-256).
