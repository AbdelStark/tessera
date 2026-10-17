# 10 — Glossary

Canonical terms used across the corpus. When a document uses one of these words, it means exactly this.

| Term | Definition |
|---|---|
| Advice | Committed per-proof values: lookup outputs, per-row scalars, decomposition limbs, multiplicities, one-hot vectors, masking coefficients. Committed by the streaming advice commitment (RFC-0011). |
| Affine site | A rescale site whose key is the accumulator times a hidden 16-bit multiplier per output channel, shifted by one public shift. The claim list names it and its shift (RFC-0005, RFC-0025). A `pot` site shifts by its channel's class shift alone. |
| Amortized overhead | `s · μ · η`: proving cost relative to serving cost when a fraction `s` of requests is proven. |
| Anchor (transcript anchor) | The statement conjunct binding the token streams inside a proof to the recorder's digest log (RFC-0015). |
| Batch (proof batch) | The set of sampled requests proven together under one seed; at least 16,384 tokens in production. |
| Blinding row | A random row in a commitment added to each revealed combined row so that the row reveals nothing (RFC-0012, RFC-0013). |
| Bounded mode | Statement mode for an unmodified floating-point deployment; reports `Û` under a committed predictor. |
| Budget (hiding budget) | The number of weight columns and blinding rows that may still be revealed before the weight commitment must be refreshed (RFC-0012). |
| Claim | An assertion that a multilinear polynomial evaluates to a value at a point; sumchecks reduce claims to claims. |
| Claim list | The public, canonical description of the circuit from which the verifier recompiles every constraint (RFC-0007). |
| Column query | A verifier-chosen column index of a committed matrix, answered with the full column and its Merkle path. |
| Combined row | A linear combination of a committed matrix's rows, revealed during an opening or proximity test. |
| Completeness failure | The honest prover cannot produce a proof for the given inputs; reported with an `ERR-WITNESS-*` code. |
| Context length `S` | Number of tokens in the sequence a forward pass attends over. |
| `D` (integer model) | The committed integer model whose execution the proof certifies (RFC-0005). |
| Dense table | A table whose multilinear extension the verifier evaluates by enumerating at most 2^24 values. |
| Digest log | The recorder's append-only log of SHA-256 digests of encrypted transcript frames. |
| Epoch | Version number of the persistent weight commitment; increments on refresh. |
| Exact mode | Statement mode certifying token-for-token equality with `D`'s deterministic outputs (`Û = 0`). |
| `Fp`, `Fp2` | The Goldilocks field, `P = 2^64 − 2^32 + 1`, and its quadratic extension `Fp[X]/(X^2 − 7)`. |
| Folding pass (weight pass) | One pass over the weights computing `W · eq(r)` at the challenge point, which both feeds the sumcheck and forms the weight opening's combined row. |
| Gadget | A constraint pattern with a witness generator and a uniqueness argument (RFC-0009). |
| GKR | The Goldwasser–Kalai–Rothblum interactive proof for layered computations, built from sumchecks. |
| Grinding | Proof-of-work required before deriving a challenge in non-interactive mode, adding its bit count to the security level. |
| Inconclusive | Verdict meaning the verifier could not evaluate. Not certified. |
| Live mode | Interactive mode with a coin-tossed seed and verifier-drawn column queries. |
| LogUp-GKR | Logarithmic-derivative lookup argument proven with a GKR fraction tree. |
| Lookup | An assertion that a (key, value) pair belongs to a table. Keys are virtual; values are committed. |
| Manifest (log manifest) | The signed summary (count, Merkle root) of one period of the digest log, frozen before sampling. |
| Masked sumcheck | A sumcheck run on `f + ρ·G` for a committed random masking polynomial `G` so that round messages reveal nothing about `f`. |
| Max certificate | The softmax gadget proving that a committed `c_q` is the row maximum via a zero product of non-negative keys. |
| Multiplicity | The count of lookups hitting each table row; committed advice. |
| `μ` (proof multiplier) | Proving GPU-time divided by the GPU-time of an ideal teacher-forced INT8 forward pass over the same tokens. |
| Non-interactive mode | Fiat–Shamir mode with grinding; produces a transferable `NiProof`. |
| Opening | Proof that a committed polynomial evaluates to a claimed value at a point. |
| Padding columns | Random message coordinates in each committed row that make revealed columns hiding. |
| Parameter set | A named, versioned bundle of protocol parameters (`live-64`, `ni-100`, `test-small`). |
| Predictor | The committed distribution against which outputs are scored to compute `Û` (RFC-0017). |
| Product tree | A GKR circuit computing a product of many values; used to prove a zero product in the max certificate. |
| Profile | Compilation profile of `D`: `servable` (every weight matmul's operands are INT8; exact mode available) or `minimal` (fewest commitments; bounded mode only). |
| Proximity test | Check that a committed matrix is close to a codeword matrix, using a random combined row and column queries. |
| Rate `ρ` | Reed–Solomon code rate: 1/2 for advice, 1/4 for weights. |
| Recorder | Network tap or attested NIC that digests encrypted transcripts and signs the log. |
| Requantization (rescale) | Integer division by a power of two with rounding and clamping that maps a wide accumulator to a narrow integer type. |
| `Requant1`, `RequantSplit`, `RequantWide` | Single-lookup requantization; two-commitment requantization for keys wider than the table bound; four-commitment requantization for keys too wide for their shift, whose top part enters only through its saturation (RFC-0009, RFC-0024). |
| Round 1, 2, 3 | Commitments; seed and sumcheck responses; column queries and answers (RFC-0014). |
| Sampling rate `s` | Probability that a recorded request is selected for proof. |
| Seed | The single externally supplied random value from which every challenge of one proof derives. |
| Serving-path overhead | What production pays continuously for determinism and recording. |
| Stage | One batched sumcheck spanning all layers, heads and blocks of one kind of relation (RFC-0010). |
| Statement | The three conjuncts certified per request: anchor, forward pass, scoring. |
| Table | A public set of (key, value) rows defined by an integer-only generator with a pinned digest (RFC-0008). |
| TCE | Tessera Canonical Encoding (RFC-0003). |
| Tile | A shallow independent unit of the circuit (layer × op, or layer × head × query block). |
| True integer | A field element known by range analysis to equal the integer the honest computation produces, without wraparound. |
| Tuple table | A table over tuples that merges several range checks into one lookup. |
| `τ` | Number of opened columns per commitment per proof. |
| `Û` | Public upper bound, in bits, on information in the outputs not explained by the committed predictor. |
| Uniqueness | Property that exactly one witness value satisfies the constraints for each value upstream of the logits. |
| Unit costs `c_c`, `c_ℓ`, `c_π` | GPU time per committed element, per lookup and per product-tree leaf. |
| Verdict | `Accept`, `Reject(reason)` or `Inconclusive(error)`. |
| Virtual | A tensor that is never committed; claims on it reduce through sumchecks to committed tensors. |
| Witness pass | The prover's re-execution of `D` that produces and commits all advice. |
| `η` | Ratio of ideal prefill time to decode-dominated serving time per token; 0.3 by default. |
