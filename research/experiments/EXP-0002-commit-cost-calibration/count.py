#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0002: operation and byte counts per committed element of the advice commitment (E0).

The commitment as specified: a row of k = 2^22 values is encoded by an inverse NTT of size k
and a coset NTT of size n = 2k (RFC-0011 'Encoding'; crates/tessera-pcs/src/{ntt,encoding}.rs),
and its codeword is absorbed into B = 32-row chained BLAKE3 column digests (RFC-0004 'Column
digests'; crates/tessera-hash/src/columns.rs). "Per committed element" divides by k: one row
value, two codeword symbols.

Two ends are counted. The HIGH end is the radix-2 structure of the CPU code as written, with
each Goldilocks operation translated to 32-bit integer instructions line by line from
crates/tessera-field/src/goldilocks.rs. The LOW end is the radix-16 structure the byte model of
IDEA-0004 already assumes, with lazy reduction. The interval I = [lo, hi] of the report is
lo = max(low ops / int32 peak, low bytes / HBM peak), hi = max(high ops / (0.6 × peak),
high bytes / (0.8 × HBM)). Every figure here is E0 (own count, primary); the peak rates are
secondary (vendor documentation) and the utilizations are IDEA-0004's.

    python3 -I count.py
"""

from __future__ import annotations

LOG_K = 22
K = 1 << LOG_K
N = 2 * K
P_MASK = 200
B = 32
SYMBOLS_PER_ELEMENT = 2

# H100 SXM5: 132 SMs × 64 int32 lanes × 1.98 GHz (CUDA C++ Programming Guide, arithmetic
# instruction throughput, compute capability 9.0: 64 results per clock per SM for 32-bit integer
# add, multiply, shift and logic; NVIDIA H100 datasheet for the SM count and boost clock).
INT32_PEAK_OPS = 132 * 64 * 1.98e9
HBM_PEAK_BYTES = 3.35e12
INT_UTILIZATION = 0.60  # [VYTA25] via SUR-0002, measured for field-multiplication kernels
HBM_UTILIZATION = 0.80

# --- int32 translation of the field operations, from goldilocks.rs -------------------------
# Each entry: (low, high, where it comes from). A 64-bit add or subtract is two 32-bit
# instructions with a carry (IADD3 / IADD3.X); a predicate from the carry costs nothing extra;
# a 64-bit select is two instructions; "wide" 32×32→64 multiplication is one instruction.
FIELD_OPS = {
    "mul": (
        18,
        26,
        "a·b as 128 bits: 4 wide products + 5–6 carried adds (9–10); reduce128 (lines 55–70): "
        "t0 = lo − hi_hi (2), borrow fix (2), t1 = hi_lo·EPSILON (1 wide product), t2 = t0 + t1 (2), "
        "carry fix (2), canonicalize (4; skipped in the lazy low end): 9–16",
    ),
    "add": (
        4,
        8,
        "lines 35–40: overflowing_add (2), EPSILON on carry (2), canonicalize (4; skipped lazily)",
    ),
    "sub": (
        4,
        5,
        "lines 43–48: overflowing_sub (2), EPSILON on borrow (2), one more if the mask is materialized",
    ),
    "mul_pow2": (
        8,
        12,
        "multiplication by ω_16^j = 2^(12j) (8 has order 64 in Fp): a 128-bit shift (2–3) and a "
        "shortened reduce128 (6–9); only the radix-16 low end uses it",
    ),
}

# BLAKE3 compression: 7 rounds × 8 G × 14 ALU ops (6 adds, 4 xors, 4 rotates) = 784, plus the
# 8 output xors; the high end is the 15 ops per byte RFC-0004 Alternatives Considered 1 states.
COMPRESSION_OPS = (784 + 8, 15 * 64)


def ntt_structure():
    """Butterflies and unit butterflies of the two transforms, per committed element."""
    # cooley_tukey over k points on the subgroup (encoding.rs step 1): (k/2)·log2 k butterflies,
    # group 0 of every level unit (ct_unit_butterflies): k − 1 of them.
    intt_bf, intt_unit = (K // 2) * LOG_K, K - 1
    # gentleman_sande over n points on the subgroup (step 4; the coset is folded into the spread
    # table): (n/2)·log2 n butterflies, n − 1 unit.
    ntt_bf, ntt_unit = (N // 2) * (LOG_K + 1), N - 1
    spread = K  # step 2: one multiplication per row value
    mask = P_MASK * 4  # step 3: per coefficient 3 multiplications and a subtraction; 800 per row
    return {
        "intt_butterflies": intt_bf / K,
        "intt_unit": intt_unit / K,
        "ntt_butterflies": ntt_bf / K,
        "ntt_unit": ntt_unit / K,
        "spread_mults": spread / K,
        "mask_ops": mask / K,
        "point_levels": (K * LOG_K + N * (LOG_K + 1)) / K,
    }


def hash_structure():
    """Compressions per committed element of the B = 32 chain."""
    step_bytes = 32 + 8 * B  # chain ‖ B symbols of 8 bytes
    compressions_per_step = -(-step_bytes // 64)  # ceil
    per_symbol = compressions_per_step / B
    return {
        "step_bytes": step_bytes,
        "compressions_per_step": compressions_per_step,
        "compressions_per_symbol": per_symbol,
        "compressions_per_element": per_symbol * SYMBOLS_PER_ELEMENT,
        "padding_waste": compressions_per_step * 64 / step_bytes - 1,
    }


def ops_high(s, h):
    """Radix-2 as written: every non-unit butterfly one full multiplication."""
    mults = (s["intt_butterflies"] - s["intt_unit"]) + (s["ntt_butterflies"] - s["ntt_unit"]) + s["spread_mults"]
    adds = s["intt_butterflies"] + s["ntt_butterflies"]
    subs = adds
    mul, add, sub = FIELD_OPS["mul"][1], FIELD_OPS["add"][1], FIELD_OPS["sub"][1]
    ntt = mults * mul + adds * add + subs * sub + s["mask_ops"]
    hash_ops = h["compressions_per_element"] * COMPRESSION_OPS[1]
    return {"mults": mults, "cheap_mults": 0.0, "adds": adds, "subs": subs, "ntt_ops": ntt, "hash_ops": hash_ops, "total": ntt + hash_ops}


def ops_low(s, h):
    """Radix-16, lazy reduction: per 16 points and 4 levels, 15 full twiddle multiplications on
    entry and 17 multiplications by powers of ω_16 inside (levels of 2: 0, of 4: 4, of 8: 6, of
    16: 7), 32 additions and 32 subtractions. The levels not divisible by 4 are treated the same."""
    point_levels = s["point_levels"]
    full = point_levels * 15 / 64 + s["spread_mults"]
    cheap = point_levels * 17 / 64
    adds = point_levels / 2
    subs = adds
    mul, add, sub, cheap_mul = FIELD_OPS["mul"][0], FIELD_OPS["add"][0], FIELD_OPS["sub"][0], FIELD_OPS["mul_pow2"][0]
    ntt = full * mul + cheap * cheap_mul + adds * add + subs * sub + s["mask_ops"]
    hash_ops = h["compressions_per_element"] * COMPRESSION_OPS[0]
    return {"mults": full, "cheap_mults": cheap, "adds": adds, "subs": subs, "ntt_ops": ntt, "hash_ops": hash_ops, "total": ntt + hash_ops}


def bytes_per_element():
    """HBM bytes per committed element: fused (NTTs on chip) to four passes per radix-16 NTT."""
    witness_write = 8
    fused = witness_write + 8 + 16 + 16  # witness read, codeword write, hash read
    four_pass = witness_write + 4 * 16 * 1 + 4 * 16 * 2 + 16  # intt passes, coset NTT passes, hash read
    return fused, four_pass


def ps(ops, rate):
    return ops / rate * 1e12


def main() -> int:
    s = ntt_structure()
    h = hash_structure()
    print("EXP-0002 counts per committed element (k = 2^%d, n = 2^%d, P = %d, B = %d)" % (LOG_K, LOG_K + 1, P_MASK, B))
    print()
    print("NTT structure (ntt.rs, encoding.rs):")
    print("  inverse NTT of size k:   %.3f butterflies, %.3f of them unit" % (s["intt_butterflies"], s["intt_unit"]))
    print("  coset NTT of size n:     %.3f butterflies, %.3f of them unit" % (s["ntt_butterflies"], s["ntt_unit"]))
    print("  spread:                  %.3f multiplications; mask: %.4f ops" % (s["spread_mults"], s["mask_ops"]))
    print("  total butterflies:       %.3f (%.0f point-levels)" % (s["intt_butterflies"] + s["ntt_butterflies"], s["point_levels"]))
    print()
    print("Hash structure (columns.rs): column-step input %d bytes -> %d compressions per %d symbols = %.5f per symbol, %.5f per element; %.1f%% of the compressed bytes are padding"
          % (h["step_bytes"], h["compressions_per_step"], B, h["compressions_per_symbol"], h["compressions_per_element"], 100 * h["padding_waste"]))
    print()
    print("int32 translation of goldilocks.rs (low, high):")
    for name, (lo, hi, why) in FIELD_OPS.items():
        print("  %-9s %3d  %3d   %s" % (name, lo, hi, why))
    print("  compression ops: %d (counted) to %d (RFC-0004's 15 per byte)" % COMPRESSION_OPS)
    print()
    low, high = ops_low(s, h), ops_high(s, h)
    print("%-26s %12s %12s" % ("ops per element", "low (r16)", "high (r2)"))
    for key in ("mults", "cheap_mults", "adds", "subs", "ntt_ops", "hash_ops", "total"):
        print("%-26s %12.1f %12.1f" % (key, low[key], high[key]))
    fused, four_pass = bytes_per_element()
    print("%-26s %12d %12d" % ("HBM bytes", fused, four_pass))
    print()
    peak_lo = ps(low["total"], INT32_PEAK_OPS)
    peak_hi = ps(high["total"], INT32_PEAK_OPS)
    util_lo = ps(low["total"], INT32_PEAK_OPS * INT_UTILIZATION)
    util_hi = ps(high["total"], INT32_PEAK_OPS * INT_UTILIZATION)
    mem_peak_lo, mem_peak_hi = ps(fused, HBM_PEAK_BYTES), ps(four_pass, HBM_PEAK_BYTES)
    mem_util_lo, mem_util_hi = ps(fused, HBM_PEAK_BYTES * HBM_UTILIZATION), ps(four_pass, HBM_PEAK_BYTES * HBM_UTILIZATION)
    print("H100 projection, ps per committed element (int32 peak %.3g op/s, HBM %.3g B/s):" % (INT32_PEAK_OPS, HBM_PEAK_BYTES))
    print("  compute at peak:          %6.1f  (low)   %6.1f  (high)" % (peak_lo, peak_hi))
    print("  compute at %.0f%%:           %6.1f  (low)   %6.1f  (high)" % (100 * INT_UTILIZATION, util_lo, util_hi))
    print("  memory at peak:           %6.1f  (fused) %6.1f  (four-pass)" % (mem_peak_lo, mem_peak_hi))
    print("  memory at %.0f%%:            %6.1f  (fused) %6.1f  (four-pass)" % (100 * HBM_UTILIZATION, mem_util_lo, mem_util_hi))
    lo = max(peak_lo, mem_peak_lo)
    hi = max(util_hi, mem_util_hi)
    print()
    print("interval I = [lo, hi] = [%.1f, %.1f] ps; rho_c = [%.2f, %.2f] against 30 ps" % (lo, hi, lo / 30, hi / 30))
    bands = []
    if lo <= 60:
        bands.append("proceed (rho <= 2)")
    if lo <= 120 and hi > 60:
        bands.append("re-plan (2 < rho <= 4)")
    if hi > 120:
        bands.append("stop and redesign (rho > 4)")
    print("RFC-0020 bands the interval touches: " + "; ".join(bands))
    print("SC-2 (cost_model.py, corrected profile): mu(16K) <= 45 needs c_c <= 70 ps -> %s at lo, %s at hi; mu(1K) <= 12 needs c_c <= 82 ps -> %s at lo, %s at hi"
          % ("holds" if lo <= 70 else "fails", "holds" if hi <= 70 else "fails", "holds" if lo <= 82 else "fails", "holds" if hi <= 82 else "fails"))
    print()
    print("Share of the high count: NTT %.0f%%, hash %.0f%%; of the low count: NTT %.0f%%, hash %.0f%%"
          % (100 * high["ntt_ops"] / high["total"], 100 * high["hash_ops"] / high["total"], 100 * low["ntt_ops"] / low["total"], 100 * low["hash_ops"] / low["total"]))
    print("Counts the Rust fit uses: intt 11 butterflies + 1 scaled bit-reversal point; coset_ntt 23 + 2 points; encode_row 34 + 1 spread multiplication; chain (call + %d compressions) / %d"
          % (h["compressions_per_step"], B // 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
