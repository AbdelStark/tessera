# Strict Ed25519 vectors

`generate.py` writes `strict.json`: signatures, keys and messages, each with the outcome of strict Ed25519 verification as [RFC-0060 'Signatures'](../../../docs/rfcs/RFC-0060-log-manifests.md#signatures) defines it. The oracle is standard-library Python written from two documents only, RFC 8032 Section 5.1 and RFC-0060, and imports nothing from either Rust workspace or from another oracle. Run `python3 -I -B spec/vectors/ed25519/generate.py` from the repository root; `cargo xtask gen-vectors` runs it too.

## The rule

`strict_verify(A, M, R ‖ S)` returns the first rule that fails, in the order of `check_order`, or `accept`:

1. `s-out-of-range`: `S`, all 256 bits of the signature's second half read little endian, is not below `L`.
2. `r-not-a-point`: `R` does not decode.
3. `key-not-a-point`: `A` does not decode.
4. `r-small-order`: `[8]R` is the identity.
5. `key-small-order`: `[8]A` is the identity.
6. `equation`: the canonical encoding (RFC 8032 5.1.2) of `[S]B − [k]A` differs from `R`'s 32 bytes. `k = SHA-512(R ‖ A ‖ M)` over the bytes as given, canonical or not, reduced modulo `L` before it multiplies `A`. This matters when `A` has a torsion component.

Decoding is RFC 8032 5.1.3 with one change, which RFC-0060 states: a `y` in `[p, 2^255)`, after bit 255 is cleared, is accepted and reduced modulo `p` instead of refused. Step 4 still holds as RFC-0060's text gives it: `x = 0` with sign bit 1 does not decode.

That last rule settles a reason, not an outcome. curve25519-dalek 4.1.3's `decompress` accepts `x = 0` with sign bit 1 and returns `x = 0`. A verifier built on it refuses the same eight cases, but reports `r-small-order` or `key-small-order` where this set expects `r-not-a-point` or `key-not-a-point`. Those cases are `r-noncanonical-y-p-plus-1-sign1`, `key-noncanonical-y-p-plus-1-sign1`, and the `y-p-plus-1-sign1` and `negative-zero-*` cases of `small-order-r` and `small-order-key`. Every one of them is refused under both readings.

## How it is checked

Before it writes anything, the oracle:
- runs RFC 8032 7.1's TEST 1, TEST 2, TEST 3 and TEST SHA(abc): key generation from the secret key, signing, and `strict_verify` returning `accept`;
- checks the base point against RFC 8032's Table 1 and the eight torsion points: a cyclic group, `(0, −1)` of order 2, `(±√−1, 0)` of order 4, and order-8 `y` values that solve `d·y⁴ + 2y² − 1 = 0`;
- asserts that each case's outcome is the one its construction intends. A second decoder, `classify`, predicts which bytes are points from Euler's criterion and the small-order `y` values, and `strict_verify` must agree with it.

Every pseudo-random value (seeds, messages, nonces, ground scalars, random bytes) is SHA-512 of a fixed label and a counter. Two runs write the same bytes.

## Format

```json
{"version": 1, "description": "...", "check_order": ["s-out-of-range", "r-not-a-point", "key-not-a-point", "r-small-order", "key-small-order", "equation"],
 "cases": [{"name": "...", "group": "...", "public_key": "<64 hex>", "message": "<hex>", "signature": "<128 hex>", "expected": "...", "cofactored": true, "comment": "..."}]}
```

- `name` is unique, in kebab case; `group` is one of the groups below.
- `public_key` is the 32 bytes of `A`, `message` the message (empty for none), `signature` the 64 bytes of `R ‖ S`, all lowercase hex.
- `expected` is `accept` or one entry of `check_order`.
- `cofactored` says whether `[8][S]B = [8]R + [8][k]A` holds with `S < L` and both points decoding as above. It is `false` when `S ≥ L` or a point does not decode. It marks the cases a lax, ZIP-215-style verifier would accept: 34 of the refused cases have it set.
- `comment` is one sentence on how the case is built.

## Cases

155 cases, by group:

| Group | Cases | Expected | Construction |
|---|---|---|---|
| `honest` | 16 | `accept` 16 | RFC 8032 signatures from four seeds on messages of 0, 1, 3, 32, 63, 64, 65, 111, 112, 127, 128, 129, 200, 255, 256 and 1000 bytes |
| `message` | 3 | `equation` 3 | an honest signature with the first, a middle or the last bit of its message flipped |
| `scalar` | 8 | `s-out-of-range` 6, `equation` 2 | an honest `R` with `S + L`, `S + 2L`, `S + 15L`, `S = L`, `S = 2^256 − 1` and `S` with bit 255 set (all refused), then `S = L − 1` and `S = 0` |
| `r-encoding` | 43 | `r-not-a-point` 20, `r-small-order` 3, `equation` 20 | `R` with each `y` in `[p, 2^255)` and both sign bits (38), three canonical `y` without an `x`, and two random strings that are not points. `S = k·a mod L` under an honest key |
| `key-encoding` | 41 | `key-not-a-point` 18, `key-small-order` 3, `equation` 20 | the same 38 non-canonical and 3 non-square encodings as `A`, with `R = [r]B` and `S = r` |
| `small-order-r` | 14 | `r-small-order` 11, `r-not-a-point` 3 | the 14 encodings of small-order points as `R`, with `S = k·a mod L` under an honest key |
| `small-order-key` | 14 | `key-small-order` 11, `key-not-a-point` 3 | the same 14 encodings as `A`, `R = [r]B` and `S = r` for the first `r` that makes `k` a multiple of the point's order, so `[S]B − [k]A = R` |
| `mixed-order-key` | 6 | `accept` 3, `equation` 3 | `A = [a]B + T`, `T` of order 2, 4 and 8, `S = r + k·a`; `r` ground so that `ord(T)` divides `k` (accept) or does not (equation) |
| `mixed-order-r` | 6 | `equation` 3, `accept` 3 | `R = [r]B + T` under an honest key (equation); and `R = [r]B + T_R`, of torsion order 2, 4 and 8, under `A = [a]B + T_A` with `T_A` of order 8, ground so that `T_R = −[k]T_A` (accept) |
| `wrong-r` | 2 | `equation` 2 | an honest signature with `R` replaced by `[r+1]B`, or with its sign bit flipped (`−[r]B`) |
| `swap` | 1 | `equation` 1 | an honest signature checked under another honest key |
| `field-carry` | 1 | `equation` 1 | `R` whose decoding meets the rarest carry of a product's reduction (below), with `S = k·a mod L` under an honest key |

The `field-carry` case targets an implementation, not a rule. A field element held in four 64-bit limbs multiplies into eight, `T = hi·2^256 + lo`, which reduces through `2^256 = 2p + 38`: `W = lo + 38·hi = r4·2^256 + r`, then `r + 38·r4`, which carries out of `2^256` again only when `r ≥ 2^256 − 38·r4`, for fewer than one random product in `2^247`. Decoding `R` multiplies `d` by `y² mod p`. The oracle picks `W` just below a multiple of `2^256`, solves for `hi` so that `d` divides `T = hi·(2^256 − 38) + W` and `hi` is `T`'s top half, and keeps the first draw whose `y² = T/d` has a root `y` that decodes to a point of large order, and for which a reduction that dropped that carry, computing `d·y² − 38`, would find no `x`: such a reduction refuses `R` as `r-not-a-point` where the case expects `equation`.

The 14 small-order encodings are the eight canonical ones and six non-canonical ones:
- the canonical ones: the identity, `(0, −1)`, the two points of order 4 and the four of order 8;
- `y = p` and `y = p + 1`, each with both sign bits;
- `y = 1` and `y = p − 1` with sign bit 1.

The three with `x = 0` and sign bit 1 do not decode. The canonical identity as `R` also satisfies the cofactorless equation, and so does each canonical small-order key. RFC 8032's lax verification accepts those nine; strict verification refuses them.

The 19 non-canonical `y` reduce to `y mod p` in `0..18`:
- `0`: both sign bits decode, to the points of order 4;
- `1`: the identity with sign bit 0; sign bit 1 is `x = 0` and does not decode;
- `3`, `5`, `6`, `10`, `14`, `16`: points of order 8L;
- `4`, `15`, `18`: points of order 4L;
- `9`: a point of order 2L;
- `2`, `7`, `8`, `11`, `12`, `13`, `17`: not points.

## RFC 8032 test vectors

`rfc8032-section-7.1.txt` is Section 7.1 of RFC 8032, "Test Vectors for Ed25519", copied unmodified from the RFC's text: lines 1295 to 1498 of [rfc8032.txt](https://www.rfc-editor.org/rfc/rfc8032.txt), page footers, form feeds and headers included. That file was downloaded from the RFC Editor on 2026-10-24 and its SHA-256 was `ed63657ff389301282b169b0abde9b5dd2c7e4d524fdfa5da6ff3094fc93c4c3`; the extract's SHA-256 is `c36090c85ac05e19c2b34a7b02a2225acd2e07631dcd66e74d02e88e0f79062c`. It holds the section's five tests, TEST 1, 2, 3, 1024 and SHA(abc): each a secret key, a public key, a message of 0, 1, 2, 1023 or 64 bytes, and a signature. The RFC is subject to the IETF Trust's Legal Provisions (BCP 78), under which code components such as these vectors may be reproduced.

`verifier/crates/tv-hash` reads the section's text as it is, and checks that each signature verifies under strict verification and that every one-bit change of its key, signature or message is refused. It checks every case of `strict.json` too, each with its exact outcome.
