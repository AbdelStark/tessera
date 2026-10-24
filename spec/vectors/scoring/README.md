# Sampler PRF and inverse-CDF vectors

`generate.py` implements AES-128 from FIPS-197 alone, in standard-library Python, and takes SHA-256 from `hashlib`. It computes the S-box from inverses in `GF(2^8)` and the affine map of §5.1.1, and checks the cipher against the standard's Appendix B and C.1 examples before writing any vector. It imports nothing from either Rust workspace or the quantizer. Run `python3 -I spec/vectors/scoring/generate.py`; `cargo xtask gen-vectors` runs it too.

The contract is [RFC-0017 'Sampler PRF'](../../../docs/rfcs/RFC-0017-scoring.md#sampler-prf), as [RFC-0061 'The stream'](../../../docs/rfcs/RFC-0061-sampling-randomness.md#the-stream) amends it, and [RFC-0009 'InverseCdf'](../../../docs/rfcs/RFC-0009-gadgets.md#inversecdf-temperature-sampling), Lemma 10:

```text
h     = SHA-256( "tessera-stream-class/v1" ‖ K ‖ 0^25 ‖ u64_le(n) ‖ u32_le(x_0) ‖ … ‖ u32_le(x_{n−1}) )
class = u64_le(h[0..8]) >> (64 − b)   for 1 ≤ b ≤ 64;   class = 0 for b = 0
u_t   = u32_le( AES-128_K( u64_le(class) ‖ u32_le(t) ‖ 0^4 )[0..4] )
ŷ   = min{ v : C(v)·2^32 > u·Z },   C(v) = Σ_{w ≤ v} e_w,   Z = C(vocab − 1)
```

- **`prf.txt`** has 369 rows of seven whitespace-separated columns:
  - the sampler key `K` in hex;
  - the class bits `b`;
  - the prompt's tokens in decimal, separated by commas, or `-` for the empty prompt;
  - `t`;
  - the stream class;
  - the whole AES block in hex, for a port debugging its cipher;
  - `u_t`.

  Every number is in decimal. The rows cover:
  - **One stream per key.** Four fixed keys (zero, `000102…0f`, all `ff`, and FIPS-197's Appendix B key) at `b = 0` and `t = 0, 1, 255, 256, 2^16, 2^24, 2^32 − 1`, the edges of each byte of `u32_le(t)`.
  - **Classes.** The same keys at `b = 0, 1, 8, 63, 64` on six prompts, at `t = 0` and `2^32 − 1`. The prompts are the empty one, `[0]` and `[0, 0]`, which differ only in length, `[1, 2, 3]` and `[1, 2, 4]`, which differ in one token, and the token `2^18 − 1`.
  - **SHA-256's padding.** After the key block, prompts of 11 and 12 tokens, on either side of the point where the length field spills into another block, of 27 and 28 tokens at the next, and of 4,096.
  - **Random cases.** 32 random keys, `b` from 0 to 64, prompts of up to 39 tokens below `2^18`, and positions; then one request's positions 0 to 63 at `b = 8`.

  The block is RFC-0017's with the class in the slot of `request_id`, so the rows whose class equals a `request_id` of #136's file draw that file's values.
- **`derive.txt`** has 46 rows of seven columns (RFC-0061 'The sampler key and its pins'):
  - a base key `K_p`, a verifier nonce, `b` and the pin's start `from_ns`;
  - the pin's PRF key `K = SHA-256("tessera-sampler-derive/v1" ‖ K_p ‖ 0^23 ‖ nonce ‖ u8(b) ‖ u64_le(from_ns))[0..16]`;
  - the class and `u_0` of the prompt `1, 2, 3` under `K`, so that each row runs end to end.

  They cover three fixed base keys and nonces at `b = 0, 8, 64` and start 0, one key at three more starts, and 16 random rows. `prf.txt` is keyed by such a `K`.
- **`inverse_cdf.txt`** has 228 rows: `u`, then the token (`none` when every weight is zero), then the weights. The oracle finds the token by bisection and asserts RFC-0009's bracket `C_<·2^32 ≤ u·Z < C_≤·2^32` for each.
  - **Fixed distributions.** Zero weights inside and at both ends, and weights of `2^64 − 1`.
  - **Draws.** Each fixed distribution is drawn at `u = 0, 1, 2^32 − 2, 2^32 − 1` and on both sides of every prefix sum's boundary, where `u·Z` meets `C(v)·2^32`.
  - **Random distributions.** 160 of them, of 1 to 300 weights below `2^1` to `2^64`, with up to three quarters of the weights zero.

Each file draws its random cases from its own SplitMix64 stream, so a change to one file's cases leaves the other's rows alone. `crates/tessera-scoring/tests/sampler.rs` checks `Sampler::class`, `Sampler::draw` and `inverse_cdf` on every row, and a unit test of `sampler.rs` checks each class's value. The vocabulary's v1 limits, `1 ≤ vocab ≤ 2^18` (RFC-0007 'Verifier validation'), are the reference API's and are tested there, not here. The in-circuit PRF of #152 and the `InverseCdf` certificate of #144 are to read these same files.
