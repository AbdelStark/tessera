# BLAKE3 test vectors

`test_vectors.json` is the official BLAKE3 test vector file. It is copied unmodified from `test_vectors/test_vectors.json` in the [BLAKE3 repository](https://github.com/BLAKE3-team/BLAKE3) at tag `1.8.7`, the release of the `blake3` crate the prover uses. Its SHA-256 is `dcb91ea8accc77e6d6e632af7cdc1a99a9f3ae78cf648da595c7d064db32f624`. Like the rest of that repository, it is licensed CC0-1.0, Apache-2.0, or Apache-2.0 WITH LLVM-exception.

Two things check against it:

- `crates/tessera-hash/tests/blake3.rs` checks the `blake3` crate in key-derivation mode, the mode every Tessera hash uses.
- [`../hash/generate.py`](../hash/generate.py) checks its own BLAKE3, in every mode, before it writes the hashing vectors of RFC-0004.
