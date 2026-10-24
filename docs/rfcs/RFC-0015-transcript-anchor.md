# RFC-0015: Transcript Anchor and Recorder Digest Log

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.4
- Amended by: [RFC-0059](RFC-0059-frame-headers-and-nonces.md):
  - 'Frames': the header carries `frame_seq` and is 41 bytes; the counter blocks, the plaintext's encoding, the commitments' inputs and a random session identity;
  - 'Log format': `Frame` entries carry `frame_seq`;
  - 'Circuits': public counter blocks and plaintext prefixes, and the opened sampler commitment;
  - 'Verifier checks': a session's open precedes its frames, and its counters are distinct.
- Amended by: [RFC-0060](RFC-0060-log-manifests.md):
  - 'Log format': a manifest's identity, `prev_manifest` and the chain, the tree over a period's entries, and inclusion proofs;
  - 'Signatures': the envelope's lengths, strict Ed25519 and the key identifier's check.
- Amended by: [RFC-0061](RFC-0061-sampling-randomness.md):
  - 'Deployment assumptions', A-ANCHOR-2: the recorder's clock;
  - 'Frames' and 'Log format': the session-open frame and its entry carry no sampler commitment;
  - 'Circuits', step 6: the sampler key is opened against the commitment the verifier pins;
  - 'Prover pre-check': the sampler commitment it recomputes is the pinned one;
  - 'Verifier checks', checks 2 and 4: each request of a batch that decodes with temperature falls within the batch's pin, whose commitment, nonce and class bits are public inputs of the circuits;
  - INV-ANC-2: a session commits its key only.
- Amended by: [RFC-0063](RFC-0063-request-boundaries.md):
  - 'Deployment assumptions', A-ANCHOR-2: the recorder sequences inbound frames before the cluster reads them, and logs request ends;
  - 'Frames' and 'Log format': the request-end frame, its `RequestEnd` entry, and the order of a session's close;
  - 'Verifier checks' 2 and 'Prover pre-check': a request's window across a boundary, its one end, and inbound frames before outbound ones.
- Amended by: [RFC-0062](RFC-0062-period-log-sessions.md):
  - 'Goals' and 'Deployment assumptions', A-ANCHOR-1 and A-ANCHOR-2: the period's log, who sends on the recorded path, and the recorder's carries;
  - 'Log format': `SessionCarry`, and a batch's entries read from its period's log;
  - 'Prover pre-check', 'Verifier checks' 2 and INV-ANC-2: the session rules over the period's log.

## Summary

The transcript anchor binds the token streams inside a proof to the streams a recorder saw at generation time. Token streams cross the inference cluster boundary as AES-128-CTR encrypted frames under a per-session key whose SHA-256 commitment is logged at session open. The recorder, a network tap or attested NIC that stores and forwards inbound frames ([RFC-0063](RFC-0063-request-boundaries.md#inbound-before-outbound)), never sees plaintext: it logs the SHA-256 digest of each encrypted frame, and at the end of each period it signs a manifest over a Merkle root of its entries. Inside the proof, AES-128 and SHA-256 circuits show that the committed tokens encrypt, under the committed key, to frames with the logged digests. This RFC fixes the frame format, log and manifest schemas, the signature envelope, the circuits and the reference recorder.

## Motivation

The first conjunct of the statement is "the token streams committed inside the proof equal those recorded at generation time by an independent recorder (network tap or attested NIC)" (source §2). The source keeps VerInf's "AES-CTR + SHA-256 digests with a key commitment", which "costs under 0.01% of the witness; binds proofs to recorder hardware" (source §4 table), recorded "by Lucid taps or attested NICs", with "the recorder's log frozen and signed before the verifier's sample is drawn" (source §7). Without the anchor, a prover could prove `D` on streams it never served.

## Goals

- A frame format and key commitment that recorders can digest at line rate without plaintext access.
- A log and manifest format whose period logs the verifier checks whole, reading the sampled requests' entries from them ([RFC-0062](RFC-0062-period-log-sessions.md)).
- AES-128-CTR and SHA-256 circuits whose cost stays below 0.01% of the witness.
- A software reference recorder for tests and demonstrations.

## Non-Goals

- Recorder hardware or firmware (NG-8).
- Proving tokenization of text; the anchor point is the token boundary of the inference cluster.
- Post-quantum recorder signatures (OQ-SEC-1).

## Proposed Design

### Deployment assumptions

- **A-ANCHOR-1.** The recorded path is the serving path: the recorder sits on the inference cluster's token-level ingress and egress, so the frames it digests are the frames that are served. Every frame, open and close on that path is the serving stack's to send ([RFC-0062](RFC-0062-period-log-sessions.md#the-session-rules)).
- **A-ANCHOR-2.** The recorder logs every frame faithfully, sequences each inbound frame before the cluster can read it and logs request ends ([RFC-0063](RFC-0063-request-boundaries.md#inbound-before-outbound)), carries open sessions into each period as [RFC-0062 'The carry entry'](RFC-0062-period-log-sessions.md#the-carry-entry) says, delivers each period's log to the verifiers on a path that does not pass through the prover, its clock is within `ε` of true time, with `2ε` below every pinned grace ([RFC-0061](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins)), and its signing key is not compromised.
- **A-ANCHOR-3.** Session keys are generated by the prover's serving stack, kept secret, and committed in the clear at session open before any token frame of that session.

### Frames

Plaintext token frame (TCE):

```rust
pub struct TokenFrame { pub request_id: u64, pub direction: Direction /* In | Out */,
                        pub frame_index: u32, pub tokens: Vec<u32> }
```

On the wire, each frame is `header ‖ ciphertext`:

```rust
pub struct FrameHeader {                // cleartext, 41 bytes: its TCE encoding (RFC-0059)
    pub magic: [u8; 4],                 // b"TSFR"
    pub session_id: [u8; 16],
    pub request_id: u64, pub direction: u8, pub frame_index: u32, pub length: u32,
    pub frame_seq: u32,
}
// ciphertext = TCE(TokenFrame) XOR AES-128-CTR keystream under k_s, with counter blocks
//              session_id[0..8] ‖ u32_be(frame_seq) ‖ u32_be(i), i from 0 (RFC-0059)
```

`frame_seq` is the frame's position within the session, in both directions (monotone, never reused). The header carries it, so the recorder and the verifier read it rather than infer it ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-session-counter)). Request boundaries and lengths are visible to the recorder; lengths are public in the statement anyway.

Session open (cleartext, before any frame of the session):

```rust
pub struct SessionOpenFrame { pub magic: [u8; 4] /* b"TSOP" */, pub session_id: [u8; 16],
                              pub key_commitment: [u8; 32] }     // SHA-256("tessera-anchor-key/v1" ‖ k_s ‖ salt_k)
```

The sampler key `K` that keys exact mode's decoding PRF ([RFC-0017](RFC-0017-scoring.md#sampler-prf)) is not a session's. The verifier pins the commitment of the prover's base key with a nonce it draws, and `K` is derived from both. The PRF's only per-request input is a keyed class of `b` public bits, and `b` is 0 by default ([RFC-0061](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins)). Which session serves a request then chooses no randomness.

Request end (cleartext, on the outbound path after the request's last frame; a session's close follows its last request end; [RFC-0063](RFC-0063-request-boundaries.md#the-requests-end)):

```rust
pub struct RequestEndFrame { pub magic: [u8; 4] /* b"TRQE" */, pub session_id: [u8; 16], pub request_id: u64 }   // 28 bytes
```

### Log format

```rust
pub enum DigestLogEntry {                                   // schema 0x0401
    SessionOpen { seq: u64, t_ns: u64, session_id: [u8; 16], key_commitment: [u8; 32] },
    Frame { seq: u64, t_ns: u64, session_id: [u8; 16], request_id: u64, direction: Direction,
            frame_index: u32, length: u32, frame_seq: u32,
            sha256: [u8; 32] },                                // SHA-256(header ‖ ciphertext)
    SessionClose { seq: u64, t_ns: u64, session_id: [u8; 16] },
    SessionCarry { seq: u64, t_ns: u64, session_id: [u8; 16], key_commitment: [u8; 32] },   // RFC-0062
    RequestEnd { seq: u64, t_ns: u64, session_id: [u8; 16], request_id: u64 },                // RFC-0063
}

pub struct LogManifest {                                     // schema 0x0402
    pub recorder_key_id: [u8; 32],                           // SHA-256 of the recorder public key
    pub log_id: [u8; 16], pub period_start_ns: u64, pub period_end_ns: u64,
    pub first_seq: u64, pub entry_count: u64,
    pub entries_root: [u8; 32],                              // SHA-256 Merkle root, RFC 6962 leaf/node prefixes
    pub prev_manifest: [u8; 32],                             // hash chain across periods
    pub signature: SignatureEnvelope,
}
```

Entry leaves are `SHA-256(0x00 ‖ TCE(entry))`; nodes are `SHA-256(0x01 ‖ left ‖ right)`. [RFC-0060](RFC-0060-log-manifests.md) fixes the rest:
- a manifest's identity, its envelope digest, which `prev_manifest` and a statement's `manifest_digest` hold;
- the chain;
- the tree, empty periods included;
- the inclusion proofs.

The verifier reads a sampled request's entries, and its session's open or carry, from the verified logs of its end's period and the one before ([RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log), [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from)). Inclusion proofs serve parties that hold one entry, such as a client checking its own request.

### Signatures

```rust
pub struct SignatureEnvelope { pub scheme: u16 /* 0 = Ed25519 (RFC 8032); 1 = ML-DSA-65, reserved */,
                               pub public_key: Bytes, pub signature: Bytes }
```

The signed message is `"tessera-manifest/v1" ‖ TCE(manifest without signature)`. [RFC-0060](RFC-0060-log-manifests.md#signatures) fixes the envelope's lengths (a 32-byte key, a 64-byte signature), strict Ed25519 verification and the check of `recorder_key_id`. Verifiers pin recorder public keys out of band (`--recorder-key`). Scheme 1 is reserved and rejected by v1 verifiers (OQ-SEC-1).

### Circuits

For each request in the batch and each direction:

These steps are amended by [RFC-0059 'The circuits'](RFC-0059-frame-headers-and-nonces.md#the-circuits): the counter blocks and each plaintext's 22-byte prefix are public inputs, and the sampler commitment is opened too.

1. **Token bytes.** The committed token stream feeds both the forward pass (`OneHot`, RFC-0009) and the frame plaintext: each token's four little-endian bytes are committed advice, range-checked to 8 bits, with `token = Σ_b byte_b 2^{8b}` (linear).
2. **AES-128-CTR.** The key schedule runs once per session; each 16-byte counter block runs 10 rounds with S-box lookups (`AesSbox`), MixColumns through `Gf256Mul { 2 }`, `Gf256Mul { 3 }` and byte XORs (`Bitwise8 { xor }`). Ciphertext bytes are `plaintext ⊕ keystream`.
3. **SHA-256.** Over `header ‖ ciphertext` per frame and over `"tessera-anchor-key/v1" ‖ k_s ‖ salt_k` per session. Words are committed as 32 bits each, with booleanity in the batched zero-check; `Ch`, `Maj`, `Σ0`, `Σ1`, `σ0`, `σ1` are low-degree polynomials in bits; additions modulo 2^32 use committed carry bits.
4. **Binding.** The final SHA-256 state bits equal the public digest bits from the log entry, and the key-commitment digest equals the public `key_commitment`. Both are linear checks against public values.
5. **Header fields** (session, request, direction, index, length, counter) are public and enter the SHA-256 input as constants.
6. **The sampler key.** SHA-256 over `"tessera-sampler-key/v1" ‖ K_p ‖ salt`, with the base key `K_p` and the salt committed, equals the batch's pinned `sampler_commitment`. The in-circuit PRF of [RFC-0017](RFC-0017-scoring.md#sampler-prf) uses the key `K` derived from `K_p` and the pin's nonce, with each request's stream class ([RFC-0061](RFC-0061-sampling-randomness.md#the-circuits)).

Cost per token, both directions, is about 250 lookups and 1,000 committed bits, against about 1.7×10^7 commitments per token for 405B's linear layers: below 0.01%.

### Reference recorder

`tessera recorder ref --listen ADDR --upstream ADDR --log-dir DIR --key FILE --period SECONDS` is a TCP proxy that forwards bytes unchanged, parses frames, appends entries, and at each period boundary writes a signed manifest.
- **Store and forward.** In each direction it cuts frames out of the byte stream by their magic (`TSOP`, `TSFR`, `TRQE`). It forwards each frame once its entry is logged, so inbound frames are sequenced before the upstream reads them ([RFC-0063](RFC-0063-request-boundaries.md#inbound-before-outbound)). Bytes that start no frame are forwarded unlogged, and that direction is not parsed further.
- **Sessions.** When a connection ends, it logs the close of every session seen on it.
- **Periods.** At each boundary it writes `<start>.tdl` (the period's log, [RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log)) and `<start>.tlm` (the manifest's envelope) to `--log-dir`, then opens the next period with its carries.
  - **Names.** `<start>` is the period's `period_start_ns` in twenty decimal digits. Starts strictly increase along the chain (RFC-0060), so each period has its own files, an empty one included, and they sort in chain order. A period's `first_seq` would not do: an empty period shares it with the next ([#512](https://github.com/AbdelStark/tessera/issues/512)).
  - **Whole files.** Each file is written under a temporary name and renamed into place, the log before the manifest, so a `.tlm` that is visible has both files complete.
- **The key.** `--key` holds the recorder's 32-byte Ed25519 seed and nothing else. Each run starts a new log, with a random `log_id`. It exists for tests, demonstrations and integration development; it is not a production recorder and does not satisfy A-ANCHOR-2 by itself.

### Prover pre-check

Before `Round1`, the prover re-encrypts its stored transcripts, checks the session rules and the request's rules over its window ([RFC-0062](RFC-0062-period-log-sessions.md#the-pre-check), [RFC-0063](RFC-0063-request-boundaries.md#the-requests-rules)), recomputes every frame digest, every session's key commitment and the pinned sampler commitment ([RFC-0061](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins)), and compares them with the manifest's entries and the pin ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-reference-cipher) says what its replay may return); a mismatch is `ERR-WITNESS-ANCHOR` and the batch is not proven.

### Verifier checks

1. Manifest signature under a pinned recorder key; unknown scheme rejected.
2. A request's frames are all its frames in its end's period and the one before, read from those periods' verified logs, with no inclusion proof. Its session keeps the session rules in each of them ([RFC-0062](RFC-0062-period-log-sessions.md)). The request has one end, every frame before it, each position once, and its inbound frames before its outbound ones ([RFC-0063](RFC-0063-request-boundaries.md#the-requests-rules)). The statement's lengths count all its frames. Every frame of every sampled request present (`frame_index` contiguous from 0, matching the statement's lengths), the `frame_seq` values of a session's frames distinct, each frame's `length` one a plaintext has, and the session's open entry before its frames ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#verifier-check-2-amended)). For a batch that decodes with temperature, each request's frames lie within one pin's window and its grace, the pin of the batch's first request ([RFC-0061](RFC-0061-sampling-randomness.md#which-pin-applies)).
3. The batch's requests equal the audit sample's requests for this manifest ([RFC-0016](RFC-0016-audit-sampling.md#verifier-checks)).
4. The circuits' public inputs are exactly these digests and key commitments, the pinned sampler commitment with its nonce, class bits and start, the frames' counter blocks and the plaintexts' prefixes ([RFC-0059](RFC-0059-frame-headers-and-nonces.md#the-circuits), [RFC-0061](RFC-0061-sampling-randomness.md#the-circuits)).

Checks 1 and 2 fail as `Reject(Anchor { request_id, kind })`, with the kinds of [RFC-0043](RFC-0043-rejection-payloads.md#anchor) and no request for the manifest's signature. Check 3 is RFC-0016's. Check 4 cannot fail on its own: the verifier makes the entries' digests, the commitments, the pin's nonce, class bits and start, the counter blocks and the plaintexts' prefixes the circuits' public inputs itself.

### Invariants

- **INV-ANC-1.** Every token used by the forward pass of a request equals a token of that request's recorded frames, in order.
- **INV-ANC-2.** The key commitment of a session is logged before any frame of the session, and in every later period it is used in, by the session's open or a carry ([RFC-0062](RFC-0062-period-log-sessions.md#the-session-rules)). The sampler commitment is no session's: the verifier pins it ahead of the window it serves ([RFC-0061](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins)).
- **INV-ANC-3.** `frame_seq` values, and hence CTR nonces, are never reused within a session.

## Alternatives Considered

1. **A proof-friendly hash in the recorder** (source §10 alternative). Rejected: the in-circuit anchor is already below 0.01% of the witness, and SHA-256 is what NICs and taps accelerate.
2. **Proving the TLS 1.3 record layer directly (AES-GCM).** Considered so that the anchor could sit on ordinary client TLS. Rejected for v1: GHASH and the TLS key schedule (HKDF over SHA-256) add circuit complexity, and the anchor point at the token boundary is inside the cluster, where Tessera framing can be required.
3. **Plaintext recording.** Rejected: the recorder is shared infrastructure; it must not see prompts or outputs.
4. **Per-frame recorder signatures.** Rejected: signing at line rate is costly; a signed Merkle root per period gives the same non-repudiation.

## Drawbacks

- The anchor binds to the token boundary, not to end-user text; tokenization is outside the statement.
- A-ANCHOR-1 is a deployment property Tessera cannot verify.
- Ed25519 is not post-quantum.

## Migration / Rollout

v0.4: frames, log, manifest, signatures, circuits, reference recorder, prover pre-check and verifier checks. Production recorder integrations follow v1 as external work.

## Testing Strategy

- AES-128 circuit against FIPS-197 appendix vectors; CTR mode against NIST SP 800-38A F.5.1; SHA-256 circuit against NIST CAVP short and long message vectors; key commitment against a direct computation.
- Alternative witnesses rejected: one flipped ciphertext bit, one wrong carry bit in SHA-256, a key differing in one bit, a token whose bytes do not recompose.
- Log: inclusion proofs verify for random positions; a missing frame (gap in `frame_index`) and a reordered frame are rejected; a manifest signed by an unpinned key is rejected; `prev_manifest` chain breaks are detected.
- Reference recorder: forwarded bytes are unchanged (byte-for-byte comparison); digests match an offline recomputation; period rotation produces a valid chain.
- End to end: `e2e_anchor_mismatch` in [07 §6](../spec/07-testing-strategy.md#6-end-to-end-tests).
- Cost: the commitment report's anchor share is below 0.01% for 405B.

## Open Questions

- **OQ-0015-1.** In deployments where external clients reach the cluster through TLS, where exactly does the token-level recorder sit, and who terminates TLS? Owner: @AbdelStark. Resolve before the first bilateral pilot, with the parties' network architecture; the outcome is an integration guide, not a protocol change.

## References

- Source §2 (statement conjunct 1), §4 table (anchor, network taps), §7 ("Anchor, sampling, verifier"), §10 (recorder).
- FIPS 197 (AES); NIST SP 800-38A (CTR mode); FIPS 180-4 (SHA-256); RFC 8032 (Ed25519); RFC 6962 (Merkle tree hashing).
- Lucid Computing, "Building an adversarial testbed for AI verification in Europe" (source §11).
