# RFC 6962 Merkle inclusion proof vectors

`inclusion/` holds golden inclusion proofs for the SHA-256 Merkle trees of RFC 6962 (Certificate Transparency), which RFC-0015 uses for recorder logs. The 98 files are copied unmodified from `testdata/inclusion/` in the [transparency-dev/merkle](https://github.com/transparency-dev/merkle) repository at commit `bcba819ef46f73fd3aaa964f46778beab2e20d59`. That commit, of 2025-04-17, is the last one to touch the directory.

The copy's integrity can be checked in one line. In the repository, with paths written as `testdata/inclusion/…` and sorted, take the SHA-256 of each file. The SHA-256 of that `shasum -a 256` listing is `f437c8c3418985cb4e5e7299389ffa2088bb324c0b6612591b6c835ab4f65344`. The files are licensed Apache-2.0, as the copy of the repository's license beside this file states. The repository has no NOTICE file.

## Format

Each file is one case:

| Field | Meaning |
|---|---|
| `leafIdx` | the index of the leaf proven |
| `treeSize` | the number of leaves in the tree |
| `root` | the tree's root, base64 |
| `leafHash` | the leaf's hash, `SHA-256(0x00 ‖ leaf)`, base64 |
| `proof` | the inclusion proof's digests in RFC 9162's order, base64, or `null` for none |
| `desc` | what the case shows |
| `wantErr` | whether verification must fail |

Interior nodes are `SHA-256(0x01 ‖ left ‖ right)`, and a proof verifies as RFC 9162 section 2.1.3.2 describes.

There are 6 valid proofs: five happy paths and a tree of one leaf. The other 92 cases each change one thing that verification must catch:
- a leaf index changed by one either way or XORed;
- a tree size doubled or halved;
- a bit flipped in a proof digest;
- a digest inserted into the proof or removed from it;
- garbage, or the root, placed before or after the proof;
- a wrong, random or empty leaf, or a random or empty root.

The verifier's RFC 6962 tree in `verifier/crates/tv-hash` checks against every case.
