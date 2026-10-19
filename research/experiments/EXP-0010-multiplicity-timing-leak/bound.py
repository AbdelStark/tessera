#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0010, parts (a) and (c): the E0 leak bound and the mitigation costs from the E1 probe.

Reads `results/run.json` (the harness's output) and prints: the per-size, per-arm derived
quantities of part (b) (`R_slow`, `s`, `g`, `t_typ/t_uniform`, coefficients of variation); the
leak bound of part (a) at the pre-registered jitter floors; the envelope and oblivious-counting
costs of part (c) under the `counted` profile; and the `--set` values for the cost-model runs.
Every constant below is the pre-registered one of README.md 'Success criterion', with its
provenance. Floating point is fine here: a planning computation, not part of the protocol.

Usage: python3 -I bound.py results/run.json
"""

import json
import math
import sys

# Cost-model `counted` profile (research/tools/cost_model.py --profile counted --json).
LOOKUPS_PER_TOKEN = 27_740_160.0  # K_lin: RISK-2's 220,160 per layer per token × 126 layers
COMMITS_PER_TOKEN = 27_740_160.0  # C_lin
SCORES_PER_TOKEN_PER_CTX = 8_064.0  # A
NATIVE_OPS_LIN = 811_700_000_000.0  # F_lin
NATIVE_OPS_ATT_PER_CTX = 4_128_768.0  # F_att
NATIVE_RATE = 1.1874e15  # R, 60% of the H100 SXM dense INT8 peak
COMMIT_PS = 94.0  # c_c under `counted`
LOOKUP_PS = 101.0  # c_ℓ under `counted`
GLUE = 1.15
BATCH_TOKENS = 16_384
MU_COUNTED = {1024: 17.198, 4096: 29.842, 16384: 78.644, 131072: 373.313}  # the tool's baseline
CONTEXTS = [1024, 4096, 16384, 131072]

# Hardware assumptions (E0): [NVIDIA-H100] 3.35 TB/s HBM3; one 32 B sector per random 4-byte
# access ([JMSS18]); utilization 60% as the knowledge base uses throughout ([VYTA25]).
HBM_BYTES_PER_S = 3.35e12
SECTOR_BYTES = 32.0
UTILIZATIONS = [0.4, 0.6, 1.0]
CENTRAL_UTILIZATION = 0.6
HBM_PER_NODE_BYTES = 8 * 80e9

# Jitter floors (fractions of P1's duration) and the epoch (RFC-0012 after #245).
JITTERS = [0.001, 0.01, 0.03]
EPOCH_PROOFS = 681

# Oblivious sort counts: bucket oblivious sort 6·N·log2 N memory accesses [ACNPRS20], one read and
# one write of a 4-byte key per access; Batcher's bitonic network (N/4)·log2 N·(log2 N + 1)
# compare-exchanges [Bat68], 16 B each.
BUCKET_BYTES_PER_ACCESS = 8.0
BITONIC_BYTES_PER_EXCHANGE = 16.0

# Verdict thresholds.
PREMISE_MAX = 1.25
PREMISE_REFUTE = 2.0
SPREAD_MIN = 0.5
BITS_MAX_01 = 8.0
BITS_MAX_3 = 3.0
BITS_REFUTE = 16.0
CV_LIMIT = 0.03


def native_s_per_token(ctx):
    return (NATIVE_OPS_LIN + NATIVE_OPS_ATT_PER_CTX * ctx) / NATIVE_RATE


def d_p1_s_per_token(ctx):
    """P1 per token: the forward pass plus the commitments it streams, counting excluded."""
    commits = COMMITS_PER_TOKEN + SCORES_PER_TOKEN_PER_CTX * ctx
    return native_s_per_token(ctx) + GLUE * commits * COMMIT_PS * 1e-12


def c_rand_s(utilization):
    return SECTOR_BYTES / (HBM_BYTES_PER_S * utilization)


def lookups_per_token(ctx, scope):
    if scope == "big":
        return LOOKUPS_PER_TOKEN
    return LOOKUPS_PER_TOKEN + SCORES_PER_TOKEN_PER_CTX * ctx


def bits_count(frac, z):
    return math.log2(1.0 + frac / z)


def bits_shannon(frac, z):
    return 0.5 * math.log2(1.0 + (frac / (2.0 * z)) ** 2)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def median(records, arm, log_t, dist):
    for r in records:
        if r["arm"] == arm and r["log_t"] == log_t and r["dist"] == dist:
            return r["median_s"]
    raise KeyError((arm, log_t, dist))


def record(records, arm, log_t, dist):
    for r in records:
        if r["arm"] == arm and r["log_t"] == log_t and r["dist"] == dist:
            return r
    raise KeyError((arm, log_t, dist))


DISTS = ["uniform", "zipf", "sorted", "sequential", "all_equal", "gauss_narrow", "gauss_wide"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/run.json"
    run = load(path)
    records = run["records"]
    n_keys = 1 << run["log_n"]
    log_ts = sorted({r["log_t"] for r in records})
    big_t = max(log_ts)
    core_t = 24 if 24 in log_ts else big_t

    print(f"EXP-0010 bound.py over {path}: N = 2^{run['log_n']} keys per stream, T ∈ {{{', '.join(f'2^{t}' for t in log_ts)}}}")

    # ---- Part (b): derived quantities per arm and size ----
    print("\n== part (b): the probe's derived quantities (medians; ns per key) ==")
    derived = {}
    for log_t in log_ts:
        for arm in ["count32", "count64", "gather"]:
            med = {d: median(records, arm, log_t, d) for d in DISTS}
            uni = med["uniform"]
            slowest = max(DISTS, key=lambda d: med[d])
            fastest = min(DISTS, key=lambda d: med[d])
            s = (med[slowest] - med[fastest]) / med[slowest]
            g = abs(med["gauss_wide"] - med["gauss_narrow"]) / uni
            typ = (med["gauss_wide"] + med["gauss_narrow"]) / 2.0 / uni
            cvs = {d: record(records, arm, log_t, d)["cv"] for d in DISTS}
            flagged = [d for d in DISTS if record(records, arm, log_t, d)["flagged"]]
            derived[(arm, log_t)] = dict(
                med=med, slowest=slowest, fastest=fastest, r_slow=med[slowest] / uni, s=s, g=g, typ=typ, cvs=cvs, flagged=flagged
            )
            row = "  ".join(f"{d} {med[d] * 1e9 / n_keys:.3f}" for d in DISTS)
            print(f"{arm:<8} T=2^{log_t:<2} {row}")
            print(
                f"{'':<8}        R_slow = {med[slowest] / uni:.3f} ({slowest}); s = {s:.3f}; g = {g:.4f}; "
                f"t_typ/t_uniform = {typ:.3f}; max CV {max(cvs.values()) * 100:.2f}%"
                + (f"; FLAGGED {flagged}" if flagged else "")
            )

    # Context arms.
    print("\n-- context arms --")
    for r in records:
        if r["arm"] in ("sortcount", "bitonic"):
            direct = median(records, "count32", r["log_t"], r["dist"]) * 1e9 / n_keys
            per_key = r["median_s"] * 1e9 / r["keys"]
            print(
                f"{r['arm']:<9} T=2^{r['log_t']} {r['dist']:<13} n=2^{int(math.log2(r['keys']))}: {per_key:.2f} ns/key, "
                f"{per_key / direct:.1f}× the direct count32 of the same stream ({direct:.3f} ns/key); CV {r['cv'] * 100:.2f}%; {r['note']}"
            )

    # The CPU reference backend's own spread (E1, reported).
    c32 = derived[("count32", big_t)]
    c_slow = c32["med"][c32["slowest"]] / n_keys
    c_fast = c32["med"][c32["fastest"]] / n_keys
    print("\n-- the CPU reference backend, one core (E1, reported, not judged) --")
    for ctx in CONTEXTS:
        n_all = lookups_per_token(ctx, "all") * BATCH_TOKENS
        print(
            f"  {ctx // 1024:>3}K context, {BATCH_TOKENS} tokens: N_all = {n_all:.3e}; per-proof spread "
            f"N_all × (c_{c32['slowest']} − c_{c32['fastest']}) = {n_all * (c_slow - c_fast):.0f} s "
            f"(slowest {n_all * c_slow:.0f} s, fastest {n_all * c_fast:.0f} s)"
        )

    # ---- Part (a): the E0 bound ----
    print("\n== part (a): the E0 leak bound ==")
    print(
        f"c_rand at 40 / 60 / 100% of 3.35 TB/s, 32 B per access: "
        + " / ".join(f"{c_rand_s(u) * 1e12:.1f}" for u in UTILIZATIONS)
        + " ps per random increment"
    )
    for scope, label in [("all", "N_all (linear + attention lookups, the pre-registered scope)"), ("big", "N_big (linear-layer lookups only, the queue item's 4.6×10^11)")]:
        print(f"\n-- scope: {label} --")
        for ctx in CONTEXTS:
            n_tok = lookups_per_token(ctx, scope)
            d_p1 = d_p1_s_per_token(ctx)
            line = f"  {ctx // 1024:>3}K: N = {n_tok * BATCH_TOKENS:.3e} per {BATCH_TOKENS}-token proof; D_P1 = {d_p1 * BATCH_TOKENS:.1f} GPU-s per proof ({d_p1 / native_s_per_token(ctx):.2f} native equivalents)"
            print(line)
            for u in UTILIZATIONS:
                delta = n_tok * c_rand_s(u)
                frac = delta / d_p1
                bits = [bits_count(frac, z) for z in JITTERS]
                shannon = [bits_shannon(frac, z) for z in JITTERS]
                mark = " <- central" if u == CENTRAL_UTILIZATION else ""
                print(
                    f"       u={u:.1f}: Δ_max = {delta * BATCH_TOKENS:.1f} GPU-s ({frac * 100:.2f}% of D_P1); "
                    f"B(0.1%/1%/3%) = {bits[0]:.2f} / {bits[1]:.2f} / {bits[2]:.2f} bits per proof "
                    f"(Shannon {shannon[0]:.2f} / {shannon[1]:.2f} / {shannon[2]:.2f}); "
                    f"per epoch {EPOCH_PROOFS} × B(0.1%) = {EPOCH_PROOFS * bits[0]:.0f} bits; "
                    f"same witness proven {EPOCH_PROOFS}×: {bits[0] + 0.5 * math.log2(EPOCH_PROOFS):.2f} bits{mark}"
                )

    # The two plausible witnesses (E1 ratio applied to the E0 all-miss duration).
    print("\n-- two plausible witnesses (gauss_narrow against gauss_wide), 16K context, scope N_all --")
    ctx = 16384
    frac = lookups_per_token(ctx, "all") * c_rand_s(CENTRAL_UTILIZATION) / d_p1_s_per_token(ctx)
    for (arm, log_t), d in sorted(derived.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        sep = d["g"] * frac
        levels = [sep / z for z in JITTERS]
        print(
            f"  {arm:<8} T=2^{log_t:<2}: g = {d['g']:.4f} → separation {sep * 100:.3f}% of D_P1 = "
            f"{levels[0]:.2f} / {levels[1]:.2f} / {levels[2]:.2f} jitter units at 0.1 / 1 / 3% "
            f"({'distinguishable' if levels[0] >= 1 else 'indistinguishable'} at 0.1%)"
        )

    # ---- Part (c): mitigations ----
    print("\n== part (c): mitigations under `counted` ==")
    r_typ = derived[("count32", big_t)]["typ"]
    print(f"envelope: t_typ/t_uniform (count32, T=2^{big_t}) = {r_typ:.3f} from E1; bracket r = 0 (typical case free)")
    for scope in ["all", "big"]:
        print(f"-- envelope, scope {scope} --")
        for ctx in CONTEXTS:
            nat = native_s_per_token(ctx)
            c_max = lookups_per_token(ctx, scope) * c_rand_s(CENTRAL_UTILIZATION)
            env = c_max * (1.0 - r_typ)
            margin = 3 * 0.03 * d_p1_s_per_token(ctx)
            print(
                f"  {ctx // 1024:>3}K: C_max = {c_max * BATCH_TOKENS:.1f} GPU-s per proof; envelope Δμ = {env / nat:.3f} "
                f"({env / nat / MU_COUNTED[ctx] * 100:.2f}% of μ = {MU_COUNTED[ctx]}); at r = 0: Δμ = {c_max / nat:.3f} "
                f"({c_max / nat / MU_COUNTED[ctx] * 100:.2f}%); 3σ margin at 3%: Δμ = {margin / nat:.3f} ({margin / nat / MU_COUNTED[ctx] * 100:.2f}%)"
                + (f"; --set regen_full_equivalents={4 + env / nat:.3f}" if scope == "all" else "")
            )

    print("-- oblivious counting --")
    for scope in ["big", "all"]:
        n_proof = lookups_per_token(16384, scope) * BATCH_TOKENS
        log_n = math.log2(n_proof)
        bucket_bytes = 6.0 * log_n * BUCKET_BYTES_PER_ACCESS
        bitonic_bytes = (log_n * (log_n + 1.0) / 4.0) * BITONIC_BYTES_PER_EXCHANGE
        bw = HBM_BYTES_PER_S * CENTRAL_UTILIZATION
        bucket_ps = bucket_bytes / bw * 1e12
        bitonic_ps = bitonic_bytes / bw * 1e12
        print(
            f"  scope {scope}: N = {n_proof:.3e} at 16K (log2 N = {log_n:.2f}); stream {n_proof * 4 / 1e12:.2f} TB against {HBM_PER_NODE_BYTES / 1e9:.0f} GB of HBM per node; "
            f"bucket sort {6 * log_n:.0f} accesses = {bucket_bytes:.0f} B per lookup = {bucket_ps:.0f} ps "
            f"({bucket_ps / 2:.0f} ps if an access is one 4-byte touch rather than a read and a write); "
            f"bitonic {log_n * (log_n + 1) / 4:.0f} exchanges = {bitonic_bytes:.0f} B per lookup = {bitonic_ps:.0f} ps (c_ℓ is {LOOKUP_PS:.0f} ps)"
        )
        for ctx in CONTEXTS:
            nat = native_s_per_token(ctx)
            per_tok = lookups_per_token(ctx, scope)
            d_bucket = per_tok * bucket_ps * 1e-12 / nat
            d_bitonic = per_tok * bitonic_ps * 1e-12 / nat
            print(
                f"    {ctx // 1024:>3}K: bucket Δμ = {d_bucket:.1f} ({d_bucket / MU_COUNTED[ctx] * 100:.0f}% of μ); "
                f"bitonic Δμ = {d_bitonic:.1f} ({d_bitonic / MU_COUNTED[ctx] * 100:.0f}%)"
                + (f"; --set lookup_ps={LOOKUP_PS + bucket_ps:.0f} (bucket) / {LOOKUP_PS + bitonic_ps:.0f} (bitonic)" if scope == "all" and ctx == 16384 else "")
            )

    # ---- Verdict helper ----
    print("\n== verdict helper (pre-registered rule) ==")
    frac_central = lookups_per_token(16384, "all") * c_rand_s(CENTRAL_UTILIZATION) / d_p1_s_per_token(16384)
    b01 = bits_count(frac_central, 0.001)
    b3 = bits_count(frac_central, 0.03)
    premise_arms = [("count32", big_t), ("count64", big_t), ("gather", core_t)]
    for arm, log_t in premise_arms:
        d = derived[(arm, log_t)]
        state = "ok" if d["r_slow"] <= PREMISE_MAX else ("REFUTE" if d["r_slow"] > PREMISE_REFUTE else "inconclusive")
        print(f"(i)  {arm} T=2^{log_t}: R_slow = {d['r_slow']:.3f} ({d['slowest']}) → {state}; max CV {max(d['cvs'].values()) * 100:.2f}% → {'ok' if max(d['cvs'].values()) <= CV_LIMIT else 'FAIL'}")
    s_big = derived[("count32", big_t)]["s"]
    print(f"(ii) s(2^{big_t}, count32) = {s_big:.3f} → {'ok' if s_big >= SPREAD_MIN else 'inconclusive'}")
    print(
        f"(iii) B(0.1%) = {b01:.2f} (≤ {BITS_MAX_01}: {'ok' if b01 <= BITS_MAX_01 else ('REFUTE' if b01 > BITS_REFUTE else 'inconclusive')}); "
        f"B(3%) = {b3:.2f} (≤ {BITS_MAX_3}: {'ok' if b3 <= BITS_MAX_3 else 'inconclusive'})"
    )


if __name__ == "__main__":
    main()
