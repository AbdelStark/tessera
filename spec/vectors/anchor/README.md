# Anchor frame vectors

`generate.py` writes `frames.json` from [RFC-0015 'Frames'](../../../docs/rfcs/RFC-0015-transcript-anchor.md#frames), [RFC-0059](../../../docs/rfcs/RFC-0059-frame-headers-and-nonces.md) and [RFC-0061 'Sessions'](../../../docs/rfcs/RFC-0061-sampling-randomness.md#sessions), in standard-library Python, importing nothing from either Rust workspace. Run `python3 -I spec/vectors/anchor/generate.py`; `cargo xtask gen-vectors` runs it too.

- **AES-128** is the FIPS-197 oracle of [`scoring/`](../scoring/README.md), checked against FIPS-197's examples.
- **CTR mode** is checked here against NIST SP 800-38A F.5.1 before anything is written.
- **SHA-256** is `hashlib`'s.

`frames.json` holds five sessions and fifteen frames.
- **The drawn bytes.** A session's 48 drawn bytes are its identifier, then `k_s` and `salt_k`: the bytes the reference cipher draws from the operating system at open. Here the identifier is fixed and the secrets are a SplitMix64 stream from the session's seed, each output little-endian; the Rust tests build the session from these bytes through a test-only constructor.
- **The counters.** Two sessions start at counters `0x01020304` and `2^32 − 1`, so that every byte of the counter blocks' counter word is exercised.

Every session gives:
- its key, its key commitment and its 52-byte session-open frame;
- for each frame, in sealing order: its counter `frame_seq`, its 41-byte header, its plaintext `TCE(TokenFrame)`, its ciphertext and `SHA-256(header ‖ ciphertext)`;
- the TCE encoding of every `DigestLogEntry`: the session's open, each frame and its close.

The frames interleave requests and directions, so a request's counters are not contiguous. They include:
- a frame without tokens;
- one of 40 tokens, 182 bytes in twelve keystream blocks;
- tokens `0` and `2^32 − 1`;
- request `2^64 − 1`.

`crates/tessera-anchor/tests/frames.rs` checks the reference cipher, the encoders and the decoders on every value.
