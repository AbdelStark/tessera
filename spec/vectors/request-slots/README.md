# Request-slot batching vectors

`generate.py` writes `slots.json` from [RFC-0071](../../../docs/rfcs/RFC-0071-request-slots.md), [RFC-0044](../../../docs/rfcs/RFC-0044-lowering-nodes.md), [RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md), [RFC-0030](../../../docs/rfcs/RFC-0030-masked-exponentials.md), [RFC-0035](../../../docs/rfcs/RFC-0035-padded-rows.md), [RFC-0053](../../../docs/rfcs/RFC-0053-greedy-runtime-boundary.md) and [RFC-0017](../../../docs/rfcs/RFC-0017-scoring.md). It shares no code with `crates/`, `verifier/` or `python/`: multilinear values are summed directly over the Boolean hypercube in Python big integers over `Fp2 = Fp[X]/(X^2 - 7)` and cross-checked against the `O(R log S)` closed forms before writing.

| Section | What it pins |
|---|---|
| `lowering` | Slot width `S = 2^b`, token axis `N`, padded length `P`, real-row mask, public `positions` `[N]` and its `[0, n_max - 1]` bounds, and key lowered tensor shapes on `tiny-2-minimal.d` and `tiny-4-servable.d` for `R = 1`, `2` and `3` requests (including `R = 1` graph equivalence with `lowering/` and `tessera-protocol/2`'s `b >= 2` floor) |
| `predicates` | `ψ̃(r_token)`, `ψ̃(r_head, r_token)`, the intra-slot causal selector `C̃(r_key, r_token)` (`0 <= j <= p < n_r`) and the `SoftmaxRow` forest root claim `1 - ψ̃(r_head, r_token)` at SplitMix64 `Fp2` points and edge Boolean vertices |
| `greedy` | Active logit windows `r·S + prompt_r - 1 <= t < r·S + n_r - 1`, single `GreedyMaximum` presence, per-output-request key column entries and key-relation time coefficients `Σ_{u < outputs_r} eq(P_row, u)·eq(r_time, r·S + prompt_r - 1 + u)`, runtime native registration order, and per-request `Û_r` window sums |

Both workspaces check `slots.json` with their own implementations.
