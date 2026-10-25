# RFC-0061: Sampling Randomness Under a Pinned Key

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.4
- Supersedes in the listed sections:
  - [RFC-0017](RFC-0017-scoring.md#sampler-prf) 'Summary', 'Semantics', 'Sampler PRF' and 'Testing Strategy': the PRF's input and key, and what `Û` bounds;
  - [RFC-0015](RFC-0015-transcript-anchor.md#frames):
    - 'Deployment assumptions', A-ANCHOR-2: the recorder's clock;
    - 'Frames', 'Log format', 'Circuits' step 6 and 'Prover pre-check': sessions commit no sampler key, and the verifier pins it;
    - 'Verifier checks' 2 and 4, and INV-ANC-2, for the same reason;
  - [RFC-0059](RFC-0059-frame-headers-and-nonces.md) 'The session-open frame and the commitments', 'The circuits', 'The log entry', 'The reference cipher', 'Security Argument' and 'Testing Strategy', for the same reason;
  - [RFC-0016](RFC-0016-audit-sampling.md#batching) 'Batching': a batch's temperature requests share one pin;
  - [RFC-0019](RFC-0019-verifier.md#inputs) 'Inputs' and 'Check order': the pins, the policy and their checks;
  - [RFC-0043](RFC-0043-rejection-payloads.md#anchor) 'Anchor' and [04 §2](../spec/04-error-model.md#2-verdicts): `AnchorFailure::SamplerKey`;
  - [06 §5](../spec/06-security.md#5-threats-and-mitigations) (the covert-channel row, and a row for learning the stream) and 06 §6 (the hash assumptions);
  - [RFC-0015](RFC-0015-transcript-anchor.md#verifier-checks) 'Verifier checks', the note on check 4.
- Origin: [#497](https://github.com/AbdelStark/tessera/issues/497)
- Amended by: [RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log) ('Which pin applies': a request's frames are its frames in the period's verified log)
- Amended by: [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from) ('Which pin applies': containment over the request's frames in its two periods), [RFC-0071](RFC-0071-request-slots.md) ('Per batch': packing by pin awaits the pins)

## Summary

Exact mode with temperature proves that each served token is `D`'s inverse-CDF sample under `u_t = PRF(K_samp, request_id, t)` (RFC-0017). The serving stack chooses `request_id`, and which session, so which `K_samp`, serves a prompt, after the prompt arrives. #137's reviews showed what that buys:
- **Grinding `request_id`.** `k` bits of the first draw cost `2^k` AES calls on average.
- **Choosing among open sessions.** With 4,096 open sessions, a given top byte of `u_0` is out of reach with probability `(255/256)^4096 ≈ 1.1·10^-7`.

The choice is best-of-`M` selection among `D`'s samples, and a covert channel, while every check passes and `Û = 0`.

A request can draw a stream of its own only through an input that differs between requests. Unless a party other than the prover supplies randomness per request, the serving stack produces every such input, the prompt included. Freshness across requests and no choice for the prover therefore exclude each other. This RFC fixes the PRF with that trade as one public parameter, and by default takes the end closest to #497's criterion:
- **The stream** is `u_t = u32_le(AES-128_K(u64_le(class) ‖ u32_le(t) ‖ 0^4)[0..4])`. The class fills RFC-0017's `request_id` slot: `b` bits of a keyed hash of the prompt, 0 when `b = 0`.
- **`b = 0` is the reference and the verifier's default.** No request has an input of its own. Every temperature request under a key draws one stream, a pad reused by the whole window, which an observer holding `D`'s distributions learns. A verifier may allow `b > 0`, up to its `b_max`. The prover's choice is then at most `b` bits per logged request under one pin, and the reuse shrinks only as classes outnumber requests.
- **Neither the prover nor its verifier chooses the key.** The prover commits a base key. The verifier pins the commitment with `b`, the start of a window of fixed length, and a nonce it draws on receipt. The PRF's key is derived from the base key, the nonce, `b` and the start. The prover cannot grind it, and it cannot keep a key it likes past its window.
- **A request falls within one pin**: all its frames lie in that pin's window, up to a short grace past its end. A batch's temperature requests share one pin.
- **Sessions commit no sampler key.** The session-open frame keeps its key commitment and is 52 bytes.

## Motivation

[#497](https://github.com/AbdelStark/tessera/issues/497). A prover that can draw its stream from `M` candidates after seeing a prompt picks the one whose sample it wants. That is about `log2 M` bits per request, at the cost of evaluating the candidates. Every input that the serving stack sets per request is such a choice:
- **the request's identifier** and **its session**, RFC-0017's two inputs;
- **the log or recorder** that records it, in a deployment with several;
- **its position in the log**, which the stack moves by reordering concurrent requests or adding traffic;
- **the prompt's tokens**, which the stack produces itself: it tokenizes, applies the chat template and truncates the history.

So is a key that the prover draws alone: it can grind the key against the window's stream before it announces it.

Earlier drafts of this RFC tried each part of the trade, and their reviews measured what each leaves. `research/tools/sampling_streams.py` reproduces every figure below from fixed seeds. The distributions are synthetic, softmaxes of Gaussian logits, and the observer holds them exactly.

**Every prompt its own stream** (`tokenizations`). Under a 32,000-entry SentencePiece vocabulary, with its byte-fallback pieces left out, the last two words of "Summarize the following email in one sentence." have 1,368 tokenizations that spell the same text.
- **Reach.** Hashing each tokenization into the PRF's input reached 256 and 253 of the 256 values of `u_0`'s top byte, under two keys.
- **Cost.** Each candidate costs a forward pass over the few positions that changed.

**One stream per key** (`shared`). Each observed (prompt, output) pair confines `u_t` to one slice of the distribution, and the slices intersect.
- **The test.** Distributions over 4,096 tokens with a mean entropy of 5.8 bits.
- **The result.** An observer predicted the next prompt's first token 91.7% of the time after 10 observed responses, and 99.5% after 100.

**Between the two: `2^8` keyed classes** (`classes`). Responses of one class have slices that intersect at every position, and responses of different classes almost never do, so an observer can cluster them.
- **The test.** Distributions over 1,024 tokens, 32 positions, 4.8 bits of entropy per position.
- **Clustering.** 4,096 observed responses fell into exactly 256 clusters.
- **Prediction.** The next token was predicted 92.2% of the time after 4 tokens of a response, and 98.8% after 8.
- **The baseline.** The first token stayed at the 32.5% of the best guess under fresh randomness.

## Goals

- **No per-request input.** At the reference setting, no input of the stream varies from one request to the next within a pin's window.
- **A public cap.** At any setting, the prover's choice of stream is at most `b` bits per logged request within one pin, `b` public and capped by the verifier.
- **A key neither party chooses.** The prover cannot choose the PRF's key, or keep one past its window, and the verifier that issues its pins does not learn it.
- **One pin at a time** for every request a verifier accepts from one prover, except near a rotation or after a hold (bounded in 'What remains the prover's').

## Non-Goals

- **Channels outside the sampling randomness**: what the prompt says, the output length, request boundaries and the temperature ('What remains the prover's').
- **Randomness from the client** for each request, the one way out of the trade (OQ-0061-2).
- **The PRF circuit** (#152) and the verifier's implementation of the pins (#150).

## Proposed Design

### The stream

For a request whose prompt is the tokens `x_0, …, x_{n−1}`, under its pin's key `K` with `b` class bits, `0 ≤ b ≤ 64`:

```text
h     = SHA-256( "tessera-stream-class/v1" ‖ K ‖ 0^25 ‖ u64_le(n) ‖ u32_le(x_0) ‖ … ‖ u32_le(x_{n−1}) )
class = u64_le(h[0..8]) >> (64 − b)      for 1 ≤ b ≤ 64;   class = 0 for b = 0
u_t   = u32_le( AES-128_K( u64_le(class) ‖ u32_le(t) ‖ 0^4 )[0..4] )
```

- **The prompt** is the request's inbound tokens: its `In` frames' tokens in `frame_index` order (RFC-0015 'Frames'). How `In` frames order against `Out` frames is #496's.
- **The hash.**
  - **The key block.** The 23-byte ASCII label and `K` fill SHA-256's first block, padded with zeros. That block depends on `K` alone, and its output state keys the rest of the hash.
  - **The count.** It starts the second block, so the inputs that follow are prefix-free.
- **The class** is the top `b` bits of the integer `u64_le(h[0..8])`.
- **The block** is RFC-0017's, with the class in the slot of `request_id`. At `b = 0` every request draws the stream RFC-0017 gives `request_id = 0`.
- **What `b` sets.** Every prompt maps to one of `2^b` classes. However the stack varies a prompt, it reaches at most `2^b` streams under one key. A window's requests spread over `2^b` streams.
- **The two ends.**
  - **`b = 0`, the reference.** No choice, and one stream per key: a pad reused by the whole window.
  - **Large `b`.** For `b` well above `log2` of a window's requests, classes are almost unique, so streams are almost never shared, except by identical prompts. The prover's choice is then bounded by its budget for evaluating prompt variants, not by `b`.
  - **In between.** A setting has some of each cost, as 'Motivation' measured at `b = 8`.
- **Secrecy.** The base key, `K`, the key block's state, `h`, the class and the stream are secret-class (05 §3). A public `K` would tell the client which slice of `D`'s distribution each output fell in.
- **The circuit** (#152).
  - **The hash.** For `b ≥ 1` it computes `h` from the key block's state and the prompt tokens it binds to the frames (RFC-0015 'Circuits', step 1). The state is computed once per pin. That is `⌈(17 + 4n)/64⌉` compressions per request: 1,025 at 16K prompt tokens.
  - **Zero knowledge.** It runs one AES per request and position. Sharing evaluations between requests of one class would show, in the proof's shape, which requests share a class.
  - **At `b = 0`** the blocks are public, and a batch's requests share their AES evaluations.

### The sampler key and its pins

- **The base key.** `K_p` is 16 bytes, drawn by the prover from the operating system's generator, with a 16-byte salt.
- **Its commitment.** `sampler_commitment = SHA-256("tessera-sampler-key/v1" ‖ K_p ‖ salt)`, the formula of RFC-0059 'The session-open frame and the commitments'.
- **Announcement.** The prover hands the verifier the commitment, `b`, the window's start, its length `W` and its grace `G`. It does so over the authenticated channel of TB-1 (06 §2), so no one else can take up a window with an announcement whose key no one knows.
- **The verifier's nonce.** On receipt, the verifier checks the announcement against the pins it holds ('Checked on arrival'), draws a 32-byte nonce from its operating system's generator, records the time, and returns the nonce with the pin.
- **One issuer.** One verifier issues a prover's pins. Further verifiers of that prover adopt the issuer's list, and so trust it for the nonce: an issuer colluding with the prover lets it choose `K`. A coin toss among verifiers, or a nonce the recorder logs or a beacon supplies, is OQ-0061-3.
- **The PRF's key.** For a pin with nonce `r` and start `from_ns`, `K = SHA-256("tessera-sampler-derive/v1" ‖ K_p ‖ 0^23 ‖ r ‖ u8(b) ‖ u64_le(from_ns))[0..16]`. As in the class hash, the 25-byte label and the key fill the first block alone, and the public inputs follow.
  - **Who chooses it.** Neither the prover nor an honest issuer. `K_p` is committed before `r` exists, so the prover cannot grind `K`. `r` is public and `K_p` hidden, so no verifier learns `K`.
  - **Distinct keys.** Two pins differ in their start, so one base key pinned twice gives two unrelated keys, whatever the nonces.
- **What the verifier pins.** It pins the pins beside the prover's recorder keys, under its policy:

  ```rust
  pub struct PinnedSamplerKey {
      pub commitment: [u8; 32],
      pub class_bits: u8,        // b
      pub from_ns: u64,          // from when it applies, on the recorders' clock
      pub window_ns: u64,        // W, the window's length
      pub grace_ns: u64,         // G, the grace past its end
      pub received_ns: u64,      // when the verifier received it, on its own clock
      pub nonce: [u8; 32],       // drawn by the verifier on receipt
  }
  pub struct SamplerPolicy {
      pub window_ns: u64,        // the W a new pin must have; by default the period of the verifier's own audit policy (RFC-0016)
      pub max_grace_ns: u64,     // the largest G a new pin may have; by default min(600 s, W/2)
      pub max_class_bits: u8,    // b_max, the largest b a new pin may have; by default 0
  }
  ```

  Every pin's window has its fixed length `W`: it runs from `from_ns` to `from_ns + W`. No pin applies between the end of one window and a later pin's start. Pins are told apart by their place in the list. A pin keeps the `W` and `G` it was accepted with, so a later change of policy cannot alter the windows of pins already held.
- **Why fixed windows.** A window without an end would let the prover rotate, look at each new key's stream, and keep the first it likes, best-of-`N` over `N` rotations. With fixed windows every key serves `W` and no longer.
- **Checked on arrival.** The verifier checks each announcement against the pins it already holds and against its policy, and refuses one that breaks a rule. It answers the prover with the refusal, draws no nonce and stores nothing, so one announcement cannot spoil the list. An announcement is refused when:
  - its `W` is not the policy's `window_ns`, its `G` exceeds `max_grace_ns`, or its `b` exceeds `max_class_bits`;
  - it breaks a rule of the list, below.
- **The policy applies to every verifier.** A verifier that adopts the issuer's list, or loads its own, applies its own `max_grace_ns` and `max_class_bits` to each pin. A pin outside them does not apply for that verifier: it treats the pin as absent, and a request under it is `SamplerKey`. A pin's `W` is not compared with the policy after arrival, so a later change of `window_ns` leaves held windows as they were.
- **Checked at load.** The verifier loads its pins and policy before step 1 of RFC-0019's check order. A failure is `Inconclusive(ERR-USAGE)`, a fault of the verifier's configuration, whatever the statement. The list's rules are checked on each pin's own values, so a change of policy does not invalidate the pins held. The load fails when:
  - a pin's `from_ns` is below its `received_ns`: a pin is never retroactive;
  - a pin starts before the previous pin's window ends, `from_ns < previous from_ns + W`, with the previous pin's `W`;
  - a pin's `W` is below twice the previous pin's `G`, so that no request can take three pins;
  - a pin has `W = 0`, `2·G > W` or `b > 64`;
  - the policy has `window_ns = 0`, `2·max_grace_ns > window_ns` or `max_class_bits > 64`.

  Sums such as `from_ns + W + G` are computed in checked arithmetic, and an overflow is the same error.
- **One pin list per prover.** Every recorder, log and session of the prover shares the pin that applies, and every verifier of the prover holds the issuer's list.
- **The clocks.** `from_ns` is compared with the recorders' `t_ns`. A-ANCHOR-2 now includes that a recorder's clock is within `ε` of true time, with `2ε` below every pinned `G`. That bounds what the prover gains by routing a request to a recorder whose clock suits it ('What remains the prover's'). It is one more recorder duty that no check enforces; OQ-0061-3 asks whether to log announcements instead.
- **Why the verifier pins it, and not the log.**
  - **Several logs.** A deployment with several recorders, every one of which must be pinned, would give the prover one key per log. Nothing makes a `log_id` unique.
  - **Timing.** A log's first manifest is signed at the end of its first period, after that period's prompts.

### Which pin applies

- **Per request.** Request `r`'s pin is the one whose window contains the earliest `t_ns` among `r`'s frames, `In` and `Out` alike: all its frames in its end's period and the one before ([RFC-0062](RFC-0062-period-log-sessions.md#the-periods-log), [RFC-0063](RFC-0063-request-boundaries.md#which-requests-a-period-draws-from)). Every one of them must have `t_ns < from_ns + W + G` for that pin.
- **Why the containment.** A request that took its pin from one early frame could have that frame logged long before the prompt, in any window: an empty `In` frame 0 opened in advance, with the prompt sent later as frame 1. Requiring every frame within the window and its grace leaves such a frame no use outside the grace.
- **Per batch.** A batch's temperature requests share one pin, since the circuit opens one key. The prover's batching (RFC-0016) will pack requests by pin, once pins are implemented, before it packs them by tokens.
- **The check.** It runs in RFC-0015 check 2, after a request's frames, when the statement's predictor decodes with temperature. The result is `Reject(Anchor { request_id: Some(r), kind: SamplerKey })` when any of these holds:
  - no pin's window contains `r`'s earliest frame;
  - a frame of `r` has `t_ns ≥ from_ns + W + G`;
  - `r`'s pin is not the batch's first request's.

  The verifier checks the requests in `statement.batch` order, so the first such request is named. A greedy statement needs no pin. The rejection mirrors an unpinned recorder key: the verifier never accepted the key the proof would need.
- **A request that falls within no pin** cannot be proven. A sampled one is the prover's failure in the audit report (RFC-0016): `Rejected` with `SamplerKey` if a proof includes it, `Unproven(DeadlineMissed)` if none does. The prover sets each pin's start, so a request that straddles a rotation by more than `G`, or is served in a gap between windows, is not an excuse.
- **Duplicates.** Containment counts all the request's frames in its two periods, and RFC-0062's rule 2 with RFC-0063's rule 5 refuses two frames at one position, so a prover holds no second copy to choose.
- **RFC-0043.** `AnchorFailure` gains `SamplerKey`, tag 3.

### Sessions

The session-open frame (RFC-0059) loses its sampler commitment and becomes `b"TSOP" ‖ session_id ‖ key_commitment`, 52 bytes. `DigestLogEntry::SessionOpen` loses its `sampler_commitment` field. A session draws `k_s` and `salt_k` only, and plays no part in sampling.

### The circuits

RFC-0015 'Circuits' step 6, as RFC-0059 added it, now reads:
- **The base key's opening.** SHA-256 over `"tessera-sampler-key/v1" ‖ K_p ‖ salt`, with `K_p` and the salt committed, equals the batch's pinned `sampler_commitment`.
- **The key.** `K` from `K_p` and the pin's public nonce, `b` and start.
- **The class.** For `b ≥ 1`, the key block's state from `K`, then `h` from it and the request's prompt tokens, and its top `b` bits.
- **The PRF.** The in-circuit PRF (#152) uses `K` on the blocks of 'The stream'.

The verifier supplies the commitment, the nonce, `b` and the start from its pin as public inputs (check 4). A proof computed under another key fails its checks.

### What remains the prover's

The stream bounds the choice of randomness. What `D` is given, and when a request ends, remain the serving stack's:
- **The prompt.** The stack tokenizes, templates and truncates. Varying the prompt changes `D`'s distribution, and for `b ≥ 1` it changes the class, at most `b` bits. A stack that writes into the prompt, such as an instruction in the system prompt, steers outputs without limit. Binding prompts to the client's text is outside the anchor (A-ANCHOR-1).
- **The output length.** Stopping early or late carries up to `log2(max_context − prompt_len)` bits per request, about 13 at 8K positions. Lengths are public, but no rule says where a request must end.
- **Request boundaries.** A stack can end a request and continue the response in a new one, whose prompt is the old prompt and the outputs so far.
  - **What a split buys.** It restarts `t`, and for `b ≥ 1` it draws a new class: at most `log2(2^b + 1)` bits per split, 1 at `b = 0`.
  - **How often.** At most one split per output token, so a response of `m` tokens can carry at most `m·log2(2^b + 1)` bits this way, each split a logged request.
  - **The trace.** At `b = 0` the public `prompt_len` of a split equals the previous request's `prompt_len + output_len`. At `b ≥ 1` re-tokenizing hides that. No rule checks for it.
- **The temperature** is the statement's `PredictorSpec`, a rational in `(0, 4]` not bound to the client's request. Its choice changes the distributions under one stream, and nothing caps it until a policy pins the allowed temperatures (OQ-0061-1).
- **Rotations.**
  - **Holding a prompt.** A prompt that the stack holds for a time `h` before serving it can be served under any of the pins whose windows begin during the hold. Windows are `W` long, so that is at most `1 + ⌈h/W⌉` pins: at most `log2(1 + ⌈h/W⌉)` bits.
  - **Near a rotation.** A request served in true time within `[from_ns − ε, from_ns + G' + ε)` of a window's start, `G'` the previous pin's grace, can take either that pin or the previous one. It can do so through a frame logged before the rotation, or through a recorder whose clock is off by up to `ε`. That is at most one bit, for those requests only: a share of about `(G' + 2ε)/W` of a window's requests, 0.7% at the defaults with a 24-hour period. Three pins would need `G' + 2ε ≥ W`, and `G' + 2ε < 2G' ≤ W` by the clock bound and the list's rule, so no request has three.
  - **When rotations fall.** The prover sets each pin's start, so it chooses which requests sit near a rotation, once per window, at the cost of a gap in which no temperature request is provable.
  - **A window it dislikes.** Once it has seen a window's key, the prover can serve that window's requests with greedy decoding, which needs no pin, since nothing binds the decoding to the client's request (OQ-0061-1). If it keeps temperature only for windows whose key passes a test of probability `q`, every temperature response is biased by up to `log2(1/q)` bits, 3 at `q = 1/8`. The test is over the key, so it is the same for every request of the window.
- **Several verifiers.** They trust the issuer for the nonce: an issuer that colludes with the prover lets it grind `K`, `k` bits for about `2^k` derivations per pin. If the verifiers of one prover hold different lists for one time, contrary to 'One pin list per prover', a prover that can route a request between their scopes chooses among their keys: `log2` of the scopes in bits.
- **Unaudited requests.** Every rule binds the requests that are proven. The others are bound only at the audit's sampling rate (RFC-0016).

## Security Argument

- **At most `b` bits per logged request under one pin.**
  - **The block.** It is the class, the position and constants.
  - **No other input.** No identifier, session, log or position in the log enters it, and the prompt enters only through the class.
  - **So** a prover that varies everything it controls reaches at most `2^b` streams for a request under one key.
  - **The cap is each verifier's.** The issuer refuses any `b` above its `b_max`, and every verifier treats a pin above its own `b_max` as absent. At the default `b_max = 0`, no request has an input of its own. What remains of #497's criterion, that no input be the prover's after the prompt, is the choice of window: the bit near a rotation and the hold bound.
- **Which keys a request can use.** Its pin is fixed by its frames: one pin's window, up to the grace. So outside `G + 2ε` of a rotation a request has exactly one pin, and near one at most two. A prover that holds a prompt adds the pins whose windows begin during the hold, at least `W` apart.
- **Neither party chooses the key.** `K_p` is fixed by its commitment before the nonce is drawn, and `K` is a hash of both and the pin's start. To bias a pin's stream, the prover would have to predict `r`, which an honest issuer draws uniformly. A window ends after `W`, so the prover cannot keep a key. To learn `K`, a verifier would have to invert the commitment or the derivation.
- **Binding.** The circuit opens `K_p` against the pinned commitment and derives `K` with the pinned nonce, `b` and start, all public inputs that the verifier supplies.
- **Unpredictability.**
  - **The derivation and the class** are SHA-256 used as a PRF keyed by secret input. The key block's output state is a pseudorandom key when SHA-256's compression function is a PRF keyed through its message block, the dual-PRF assumption of HMAC's analysis (Bellare 2006). The rest of the class hash is then a cascade on prefix-free inputs, a PRF under that key (Bellare, Canetti, Krawczyk 1996). The derivation's first block holds `K_p` alone, and its public inputs have one length.
  - **Keys used twice.** `K_p` is hashed twice, in the commitment and the derivation, and `K` is used twice, in the class's key block and as AES's key. Each use has its own label or primitive. This RFC takes those uses to be independent, as they are when the compression function is modeled as ideal (06 §6).
  - **What is revealed.** No output of the derivation or the class leaves the circuit or the sampler.
  - **The stream.** AES-128 under `K` is a pseudorandom permutation on distinct blocks. To anyone without `K`, the `u_t` of distinct positions or classes are uniform and independent, as RFC-0017 assumes.
  - **Within a class** the stream is shared.
- **What an observer learns.** An observer holding `D`'s distributions, or a close proxy of them, can cluster responses by class and learn each class's stream, as 'Motivation' measured. It then learns which slice of `D`'s distribution later tokens fell in. It cannot compute `K`, predict a class it has not observed, or choose a prompt's class in advance.

## Alternatives Considered

1. **The prompt's hash, unkeyed and whole, as input** (the first draft). Every variant the stack produces is a fresh draw, and an observer can compute any prompt's stream. Rejected for the keyed class, whose large-`b` settings keep the freshness without the computable index.
2. **A fixed `b > 0` as the reference** (the third draft, at `b = 8`). It gives back best-of-`2^b` selection, against #497's criterion, while a holder of `D` still clusters the responses. Rejected for `b = 0` by default, with larger `b` a policy choice.
3. **A key per log, committed in its genesis manifest** (the first draft). Several logs give several keys, `log_id`s are not unique, and a genesis is signed after its first period's prompts. Rejected for the verifier's pin.
4. **A key the prover draws alone** (the fourth draft). The prover grinds it against the window's stream before announcing it: biasing every request of the window, or carrying `log2` of its budget in bits per pin to whoever learns the stream. Rejected for the nonce.
5. **A nonce from several verifiers or a beacon** in place of the issuer's draw. A commit-then-reveal coin toss among verifiers needs abort handling: a contributor that reveals last can withhold a reveal it dislikes. A beacon round published after `received_ns` needs a beacon all verifiers trust. Both are OQ-0061-3.
6. **A pin chosen by a request's earliest frame alone** (the fourth draft). A frame logged in advance chooses among every window of the period. Rejected for containment.
7. **Windows that run until the next pin** (the fifth draft). The prover rotates every `W_min`, looks at each new stream, and keeps the key it likes: about `log2(T/W_min)` bits over a time `T`, for every later request. Rejected for fixed windows.
8. **A lead `Δ` between receipt and start.** With the nonce, the prover cannot grind the key, and a lead changes neither the hold bound nor the bit near a rotation. Dropped: a pin only needs to start after its receipt.
9. **A value the recorder assigns**, such as the `seq` of the request's first inbound frame or its ordinal in the log. The stack reorders concurrent requests within its latency and adds traffic, empty frames included. A busy deployment with `W` requests waiting at once gives about `log2 W` bits per request, unseen and uncapped. Rejected.
10. **A request's ordinal in its session**, or **a fresh key per session.** The stack chooses the session and opens new ones: RFC-0017's choice again. Rejected.
11. **Randomness from the client.** A nonce the client sends and checks would give each request its own stream with no choice for the prover. A nonce that no one checks is the prover's. It needs a client protocol: OQ-0061-2.
12. **A public `K`.** It reveals which slice of `D`'s distribution each output fell in. Rejected.
13. **The key in the first block beside the prompt** (the third draft). It puts prover-chosen bytes in the key's block, and BCK's cascade proof does not apply directly. Rejected for the key block.

## Drawbacks

- **At the reference `b = 0`, one pad per window.**
  - **What it leaks.** An observer holding `D`'s distributions, or a close proxy, learns the stream from a few responses. From then on it learns which slice of `D`'s distribution each output fell in. That is more about `D` than its samples alone say, and it matters most where `D` is close to a public model.
  - **Correlation.** Different users' outputs are correlated.
  - **Statistics.** Statistics over requests, such as the fidelity study's bootstrap, are not over independent draws. They must resample keys.
- **At `b ≥ 1`, best-of-`2^b`.** A property that `D`'s samples show with probability `p` can be shown in about `1 − (1 − p)^{2^b}` of the served responses: 63% at `p = 2^-8` and `b = 8`. The reuse leak shrinks only once classes far outnumber a window's requests.
- **Identical prompts give identical outputs**, at every `b`. A regenerated answer needs another prompt.
- **Cost.** At `b ≥ 1` the circuit hashes each prompt and runs AES per request. Every pin adds one derivation.
- **Pins need care.**
  - **The nonce.** The issuing verifier draws and returns a nonce for every pin, and other verifiers trust it.
  - **Rotation.** It is the only refresh of the key, and it is due every `W`; a missed one leaves a gap without provable temperature requests.
  - **Clocks.** The recorders' clocks must stay within `ε`.
  - **Straddlers.** A request in flight longer than `G` across a rotation cannot be proven.
- **Key compromise.** A leaked `K` makes the window's samples predictable to anyone who knows `D` and the prompt. Every serving replica holds it.

## Migration / Rollout

Nothing implemented RFC-0017's inputs beyond the reference sampler of #136.
- **#497, with this RFC.**
  - `tessera-anchor`: the 52-byte session-open frame and the `SessionOpen` entry.
  - `tessera-scoring`:
    - `SamplerBaseKey` and `SamplerBaseKey::derive`;
    - `ClassBits` and `StreamClass`;
    - `Sampler::new(key)`, `Sampler::class(prompt)` and `Sampler::draw(class, t)`. The class bits travel inside the `SamplerKey` they were derived with.
  - The vectors of `spec/vectors/scoring/`, `anchor/` and `manifest/`.
  - `research/tools/sampling_streams.py`.
- **#150**:
  - the pins, the nonce and the policy;
  - the checks on arrival and at load;
  - the containment rule and `AnchorFailure::SamplerKey`.
- **#152**: the derivation, the key block, the class and the PRF in the circuit.
- **#146**: the pre-check recomputes the pinned commitment.
- **#139**: the recorder keeps its clock within `ε`.

## Testing Strategy

- **The PRF.** Vectors from an independent Python oracle (FIPS-197 AES, `hashlib`):
  - **Derivation:** `(K_p, r, b, from_ns) → K`, at `b = 0, 8, 64`, with fixed and random keys, nonces and starts. Each row also chains through to the class and `u_0` of one prompt, so a port that keys the class or the AES with the wrong key fails it.
  - **Streams:** `(K, b, prompt, t) → class, u_t`, at `b = 0, 1, 8, 63, 64`. They include prompts that differ in one token or only in length, SHA-256's block boundaries after the key block, and positions at the edges of each byte of `t`.

  At `b = 0` the streams equal #136's at `request_id = 0`.
- **The session-open frame.** 52 bytes; 51 and 53 bytes are refused.
- **The pins (#150).** Each case alone:
  - **`SamplerKey`:**
    - a request before the first pin;
    - a batch whose requests have two pins;
    - a request whose earliest frame lies in one window and whose later frame has `t_ns = from_ns + W + G`;
    - a request in a gap between windows, and one after the last window's end;
  - **the edges:** an earliest frame exactly at a pin's `from_ns`, and a frame at `from_ns + W + G − 1`, both accepted;
  - **a greedy statement** with no pins, accepted;
  - **on arrival, each refused, and at load, each `ERR-USAGE`:**
    - a retroactive pin;
    - a pin that starts inside the previous window;
    - a pin with `W = 0`, `2·G > W`, or `W` other than the policy's (on arrival only);
    - `class_bits` above `b_max` (on arrival only);
    - `b_max = 65`;
    - an overflowing sum;
  - **a pin whose `W` is below twice the previous pin's `G`**, refused;
  - **a policy change**: pins held under the old policy still load and keep their windows, and a pin above the new `b_max` or `max_grace_ns` no longer applies;
  - **an adopted list** holding a pin above the adopter's `b_max`: that pin does not apply for the adopter.

## Open Questions

- **OQ-0061-1.** Should a request's temperature be bound to the client's request, or a policy pin the allowed temperatures? RFC-0017 allows four temperatures per batch, while 03's `PublicStatement` holds one `PredictorSpec`. Owner: @AbdelStark. Resolve with #155's mixture predictor, which reads the temperature too.
- **OQ-0061-2.** Should clients supply randomness, a nonce they check in the audit report, to remove the trade? Owner: @AbdelStark. Resolve with the v0.4 fidelity study, from the measured leak at `b = 0`.
- **OQ-0061-3.** Should the recorder log a pin's announcement and nonce, or the verifier sign the pin it returns?
  - **What logging would give.** It would put the pin's times on one clock and let several verifiers check one list. It would let a prover show which pin it was given, if the issuer later disputes it. And it would give a third party checking a non-interactive proof evidence of the pin.
  - **A nonce no single verifier chooses.** A coin toss among verifiers, with its aborts handled, for instance by spending the window of an incomplete toss. Or a beacon round after `received_ns`.
  - **Related.** Should the statement carry the pin's commitment, nonce and `b`, so that a proof names its key, and should the audit report record `b`?

  Owner: @AbdelStark. Resolve with #139's recorder and #150's verifier.

## References

- [RFC-0017](RFC-0017-scoring.md) 'Sampler PRF'; [RFC-0015](RFC-0015-transcript-anchor.md); [RFC-0059](RFC-0059-frame-headers-and-nonces.md); [RFC-0016](RFC-0016-audit-sampling.md); [RFC-0019](RFC-0019-verifier.md); [RFC-0043](RFC-0043-rejection-payloads.md); [06 §5](../spec/06-security.md#5-threats-and-mitigations).
- FIPS-197 (AES); FIPS 180-4 (SHA-256).
- Bellare, Canetti, Krawczyk, "Pseudorandom Functions Revisited: The Cascade Construction and its Concrete Security", FOCS 1996.
- Bellare, "New Proofs for NMAC and HMAC: Security without Collision-Resistance", CRYPTO 2006.
