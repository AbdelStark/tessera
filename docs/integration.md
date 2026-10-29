# Recorder and serving-engine integration

This guide connects a serving adapter to the recorder at the **token boundary**.
It implements the deployment guidance of [RFC-0015](rfcs/RFC-0015-transcript-anchor.md#deployment-assumptions),
including OQ-0015-1. Frame/nonces follow [RFC-0059](rfcs/RFC-0059-frame-headers-and-nonces.md),
period/session rules [RFC-0062](rfcs/RFC-0062-period-log-sessions.md), request
boundaries [RFC-0063](rfcs/RFC-0063-request-boundaries.md), and sampler pins
[RFC-0061](rfcs/RFC-0061-sampling-randomness.md). These accepted amendments take
precedence over earlier descriptions of the session-open or PRF inputs.

Tessera is pre-release. The reference recorder is a software proxy for integration
development; running it on a prover-controlled host does not establish A-ANCHOR-2.
The example below tests bytes, signed logs, stored transcripts and rejection
controls. It does not run an LLM, implement TLS or issue a trusted sampler pin.
The current full CPU CLI route is exact greedy, protocol/1, with no zero knowledge;
its verifier does not certify the anchor or ledger checks. Scoring/anchor library
components and their tests must not be presented as a complete deployed audit.
See the [verifier CLI boundary](verifier/reference-cli.md) and the
[sampled-route tests](testing/exact-cli.md#the-sampled-route).

## Agree on the trust boundary before connecting traffic

The parties record the chosen topology, its enforcement mechanism and the
recorder public keys outside the prover's control. Three assumptions remain
deployment duties, rather than conclusions of a proof:

| Rule | Required deployment evidence |
|---|---|
| A-ANCHOR-1 | Every served request crosses the recorded token ingress and egress. The adapter cannot send one token stream to the recorder and another to the client. Identify who enforces this mapping and prevents bypass. Text tokenization, chat templates and binding a prompt to a client's text are outside the proof. |
| A-ANCHOR-2 | An independent recorder logs every frame, assigns an inbound frame's sequence before the cluster reads it, logs request ends and carries, and delivers complete signed periods directly to verifiers. Its signing key and clock are protected. Clock error `ε` satisfies `2ε < G` for each applicable sampler grace. |
| A-ANCHOR-3 | The serving stack draws fresh session keys, keeps them secret, and sends their commitment before any token frame. The recorder does not receive the plaintext or encryption key. |

An independent engineer reviews the concrete mapping, termination point,
key-provisioning path, clock discipline and log-delivery path before a bilateral
pilot. The repository example supplies no attestation or hardware qualification.

## TLS placement (OQ-0015-1)

Because [TLS protects its record payloads](https://www.rfc-editor.org/rfc/rfc8446.html#section-5.2),
a passive tap outside its termination point cannot parse the inner `TSOP`,
`TSFR` and `TRQE` application frames. A capture of TLS records is not this
protocol's token-frame log. Choose a trusted termination/offload point:

```text
external client -- TLS --> trusted termination + token/frame adapter
                                     |
                          TSOP / TSFR / TRQE stream
                                     |
                          inline independent recorder
                                     |
                               serving engine
                     responses follow the recorded return path
```

One option is an independently controlled gateway that terminates client TLS,
maps the API to the agreed token stream and enforces the mapping on the return
path. This adapter can see application plaintext, so its confidentiality boundary
must also be agreed by the parties; it is additional deployment trust. Its frame
adapter and the engine provision the session secrets over an
authenticated confidential channel that excludes the recorder. The recorder sees
clear frame headers and application ciphertext, not token plaintext. If another
TLS hop encloses these frames, place its trusted termination/offload so the
recorder sees the frame stream; do not insert an opaque hop across the observation
point and then log a separate prover-supplied copy.

Another option combines trusted TLS termination, frame-path enforcement and
store-and-forward recording in an attested NIC/appliance. The parties must assess
its attestation, configuration and bypass controls. A hardware label or mTLS
connection alone does not prove that the adapter forwards the same token stream
to the client. A prover-controlled TLS terminator plus a software mirror remains
an integration test unless the deployment independently enforces A-ANCHOR-1/2.

The trusted inbound path must **store and forward**, rather than only mirror:
an asynchronous tap may observe a prompt after the cluster has already produced
an output (RFC-0063). Application framing still needs an authenticated transport;
the anchor's AES-CTR encryption is not a live message-authentication mechanism.
Hardware/firmware and vendor-specific TLS offload are outside this guide.

## Frame adapter contract

Use `tessera_anchor::Session::open`, `Session::seal` and `SessionTranscript::record`
as the reference path. There is no public decrypt operation in this reference
cipher. A production adapter's key sharing and peer-side decryption need their
own implementation and conformance review; the synthetic example already knows
the exchange it expects.

| Item | Wire or lifecycle rule |
|---|---|
| Session open | `TSOP`, 16-byte random session ID, 32-byte key commitment: 52 bytes. The commitment is `SHA-256("tessera-anchor-key/v1" ‖ k_s ‖ salt_k)`. No sampler commitment appears here. Send it before this session's token frames. |
| Token frame header | `TSFR`, session ID, `request_id: u64`, `direction: u8`, `frame_index: u32`, `length: u32`, `frame_seq: u32`: exactly 41 bytes. Header integers are little-endian; direction is 0 (`In`) or 1 (`Out`). |
| Encrypted plaintext | TCE of `request_id: u64`, direction tag **u16**, `frame_index: u32`, token count **u64**, then token IDs **u32**. Length is `22 + 4n` **bytes**, not `n` tokens. The header and plaintext's public fields agree. |
| Request end | Outbound `TRQE`, session ID, `request_id: u64`: 28 bytes. Send exactly once, after every frame of the request and before closing its session. A socket close does not replace it. |
| Per-direction index | For each `(session, request, direction)`, start `frame_index` at 0 and cover each position once without a gap. All of a request's inbound frames must be logged before its first outbound frame. |
| Session counter | One `frame_seq` allocator across **all requests and both directions** in a session. Start at 0, never reuse or wrap it, and open a new session after counter `2^32−1` is spent. A period boundary or sampler rotation does not reset it. |

The AES counter block is `session_id[0..8] ‖ u32_be(frame_seq) ‖ u32_be(block_index)`.
These two words are **big-endian**, unlike the header. Let the cipher assign
them; do not derive a nonce from `request_id`, `frame_index`, a direction or a
thread number. Serialize access to the session's counter when requests interleave.
TCP read/write or packet boundaries do not define frame boundaries.

After a restart or uncertain counter state, open a fresh session with fresh
keys; do not restore an old key with counter 0. Stored transcript replay is for
checking old digests, not for sending fresh ciphertext under spent counters.
TCP retransmission is below the byte-stream framing layer. An application-level
retry that repeats logged frame positions is a duplicate, not an invisible retry.
Route retries as new requests/sessions with the appropriate fresh counters.

The reference proxy forwards bytes that do not start a recognized frame unlogged
and stops parsing that direction. Consequently, putting JSON, HTTP or TLS bytes
on its listen socket does not turn them into recorded requests. A production
path must enforce the agreed frame-only mapping and account for bypass traffic.

## Periods, retention and handoff

The recorder signs each period's manifest and writes its whole `.tdl` before the
corresponding `.tlm` becomes visible. Filenames use the twenty-digit
`period_start_ns`, not `first_seq`; empty periods can share a sequence number.
Deliver both files, empty periods included, directly to verifiers without the
prover being able to suppress them. Verifiers pin recorder keys out of band,
check signatures and the manifest chain, then verify entry counts, contiguous
sequence numbers and Merkle roots before drawing any sample. A key carried in
the signature envelope is not its own trust anchor (RFC-0060/0062).

Requests may span the end's period and the previous nonzero-length period, not
an arbitrary number of periods. The verifier must hold both verified logs and
the manifests chaining them, including intervening zero-length periods. An
unterminated request, a second end, or a tail after the end is not silently
excluded from audit coverage (RFC-0063).

Open sessions receive deterministic carries at period starts. A session with
neither an open nor a frame in two consecutive nonzero-length periods lapses;
open a fresh session before sending again. Account for recorder clock error
and path delay at the boundary. A request's frame indices remain unique across
the two-period window, and a batch may not reuse a session counter in that window.

The serving adapter retains `.tst` artifacts: session ID, session secrets and
each stored plaintext at its assigned counter. Use `SessionTranscript::record`
and the secret-class envelope writer (0600 on Unix). Preserve clearing storage
when reading them. A prover reads `<session_id in lowercase hex>.tst` from its
`--transcripts` directory. The example's single `session.tst` is a test artifact,
not that multi-session directory layout.

Never send `.tst`, model weights, session keys, sampler base keys/salts, PRF draws
or token plaintexts with public log delivery. Retention and access controls follow
[06 §7](spec/06-security.md#7-secrets-handling); logs/manifests can go to verifiers,
while proving inputs remain on the serving/prover side. A recorder signing seed
belongs to the recorder's independent key boundary.

## Sampler and exact-mode engine conformance

Session encryption and sampling use separate keys and lifecycles. For temperature
decoding, integrate `tessera_scoring::{SamplerBaseKey, ClassBits, Sampler,
inverse_cdf}` or a port checked against the [independent scoring vectors](../spec/vectors/scoring/README.md).

1. Draw a sampler base key `K_p` and 16-byte salt using the OS generator in clearing
   secret storage. Announce `SHA-256("tessera-sampler-key/v1" ‖ K_p ‖ salt)`, class
   bits `b`, start and fixed window/grace to the authenticated verifier issuer.
2. The issuer checks its policy/list, then draws and returns its 32-byte nonce.
   Retain its pin; derive `K` using `SamplerBaseKey::derive(nonce, bits, from_ns)`.
   A local random nonce is not a verifier-issued pin. All recorders, sessions and
   requests of one prover use the same applicable pin list, with one issuer.
3. Compute the class from the inbound prompt tokens under that derived key.
   The reference `b = 0` gives one stream per key; other values require the
   verifier's explicit class-bit policy. The cap trades stream choice against
   shared randomness; it does not promise fresh independent draws per request.
4. Draw by **output position** `t = 0, 1, …`, with no request/session/worker ID
   added, no host RNG replacement and no skipped draw due to batching order.
   `Sampler::draw` implements the AES block with little-endian class and position
   words; do not reuse the anchor cipher's big-endian counter convention.
5. Select the least vocabulary index whose integer prefix satisfies
   `C(v)·2^32 > u_t·Z`. The comparison is strict. For exact temperature decoding,
   use the specified `Exp_T` integer weights from `D`, not host floating-point
   softmax weights or top-k/top-p filtering. Zero-mass distributions are errors.

A request's frames must fit one pin window with its allowed grace; do not select
a pin retrospectively after inspecting prompts/outputs. Follow RFC-0061's
non-overlap, load/arrival and clock rules. A greedy request needs no sampler pin.
Keep sampler bases/salts in the serving/prover secret store for replay; the
example discards its simulated sampler state and cannot be used to prove its
toy temperature exchange later.

For **exact mode**, the serving engine must implement the committed integer `D`
bit for bit: tokenizer/model identity, all integer operators, rounding, ranges,
public temperature parameters and decoding/tie rules. Compare its layer values
and logits with the [model oracle vectors](../spec/vectors/models/README.md)
and the [reference engine tests](../crates/tessera-witness/tests/engine.rs).
Then compare generated tokens, including the independent PRF/inverse-CDF cases.
An FP8 deployment or changed decoding policy is not exact merely because it
uses the same random draw. A mismatch is `ERR-WITNESS-EXACT-MISMATCH`; it must not
be hidden by changing the recorded prompt or replacing outputs. Bounded scoring
is a separate predictor and proof integration, not a fallback claim of equality.

## Run the reference integration

From the repository root, choose a directory that does not already exist:

```sh
cargo build -p tessera-cli --bin tessera --locked
cargo run -p tessera-cli --example recorder_integration --locked -- \
  "$PWD/target/debug/tessera" "$PWD/recorder-demo"
cargo test -p tessera-cli --example recorder_integration --locked
cargo test -p tessera-scoring --test sampler --test exact --locked
cargo test -p tessera-witness --test engine the_engine_computes_the_conformance_vectors --locked
```

The example starts the real `tessera recorder ref` on loopback and a synthetic
peer. It opens one session, uses a simulated issuer and toy sampler weights,
stores three sealed frames, fragments the byte stream across header boundaries,
forwards both directions and sends the request end. It waits for signed periods,
authenticates their chain/root, reloads its private transcript, checks replay and
asserts exact rejection controls for an unpinned recorder key, a modified log
digest, wrong token lengths, a changed wire frame and a missing wire frame.
The example refuses an existing output directory.

`PUBLIC-TEST-ONLY.seed` is a **public fixture key**, unsuitable for any deployment.
The generated `session.tst` contains actual random session secrets even though
the tokens are synthetic. Keep the example directory private. Its stdout reports
only check completion, never draws, tokens or keys. It intentionally provides
no production recorder, peer decryption, pin service, LLM or verifier verdict.

The automatic prover CI runs this exact example, its fresh-directory control and
the sampler/exact conformance suites and integer-model oracle cases. Changes to
this guide select that CI path.
The existing sampled-route suite separately exercises the tiny exact-greedy
record/sample/prove/verify flow and its explicit verifier limits. An engineer
outside the core team must review this guide and the chosen deployment before
the integration is treated as pilot-ready; CI does not provide that review.
