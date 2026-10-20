# Streaming commitment vectors

`streaming.json` fixes small canonical advice layouts and their commitments, from
[RFC-0011](../../../docs/rfcs/RFC-0011-advice-commitment.md),
[RFC-0046](../../../docs/rfcs/RFC-0046-advice-segments.md) and
[RFC-0047](../../../docs/rfcs/RFC-0047-advice-order.md). Every row and mask is public
synthetic test data. The small parameters do not certify soundness.

The independent Python oracle enumerates retained tensor coordinates in layout order,
aligns padded domains, and assigns layers to segments. It reuses the independent
PCS oracle's Lagrange interpolation and Horner evaluation, checked against a DFT,
then hashes whole columns and builds a dense Merkle tree. No Rust code or output
is an input. BLAKE3 is checked against all official vectors before generation.

The cases cover `S = 1, 2, 8`, full and partial blocks, alignment and padding gaps,
empty segments and entirely empty commitments, a single committed row, and two
padded tensor axes, retaining inner-axis length padding as RFC-0047 requires. Score tensors use `g = Q' = 1`, so their layout index is their
key index; causal and padded query blocks supply additional zero rows. The file
records the descriptor bytes, exact row indices, codewords, every segment's
digests, merged leaves and root. Both workspaces consume these independent data.

Regenerate with `python3 spec/vectors/pcs-streaming/generate.py`.
