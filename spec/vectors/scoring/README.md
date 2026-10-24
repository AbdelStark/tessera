# Sampler PRF and inverse-CDF vectors

`generate.py` implements AES-128 from FIPS-197 alone, in standard-library Python. It computes the S-box from inverses in `GF(2^8)` and the affine map of §5.1.1, and checks the cipher against the standard's Appendix B and C.1 examples before writing any vector. It imports nothing from either Rust workspace or the quantizer. Run `python3 -I spec/vectors/scoring/generate.py`; `cargo xtask gen-vectors` runs it too.

The contract is [RFC-0017 'Sampler PRF'](../../../docs/rfcs/RFC-0017-scoring.md#sampler-prf) and [RFC-0009 'InverseCdf'](../../../docs/rfcs/RFC-0009-gadgets.md#inversecdf-temperature-sampling), Lemma 10:

```text
u_t = u32_le( AES-128_{K_samp}( u64_le(request_id) ‖ u32_le(t) ‖ 0x00000000 )[0..4] )
ŷ   = min{ v : C(v)·2^32 > u·Z },   C(v) = Σ_{w ≤ v} e_w,   Z = C(vocab − 1)
```

- **`prf.txt`** has 144 rows of five whitespace-separated columns:
  - the key `K_samp` in hex;
  - `request_id` and `t` in decimal;
  - the whole AES block in hex, for a port debugging its cipher;
  - `u_t` in decimal.

  The rows cover four fixed keys (zero, `000102…0f`, all `ff`, and FIPS-197's Appendix B key) at the edges of `request_id` and `t`. They add 32 random triples and one request's positions 0 to 63 under one random key.
- **`inverse_cdf.txt`** has 228 rows: `u`, then the token (`none` when every weight is zero), then the weights. The oracle finds the token by bisection and asserts RFC-0009's bracket `C_<·2^32 ≤ u·Z < C_≤·2^32` for each.
  - **Fixed distributions.** Zero weights inside and at both ends, and weights of `2^64 − 1`.
  - **Draws.** Each fixed distribution is drawn at `u = 0, 1, 2^32 − 2, 2^32 − 1` and on both sides of every prefix sum's boundary, where `u·Z` meets `C(v)·2^32`.
  - **Random distributions.** 160 of them, of 1 to 300 weights below `2^1` to `2^64`, with up to three quarters of the weights zero.

`crates/tessera-scoring/tests/sampler.rs` checks `Sampler::draw` and `inverse_cdf` on every row. The vocabulary's v1 limits, `1 ≤ vocab ≤ 2^18` (RFC-0007 'Verifier validation'), are the reference API's and are tested there, not here. The in-circuit PRF of #152 and the `InverseCdf` certificate of #144 are to read these same files.
