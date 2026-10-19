#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Proximity-test parameter calculator for the Ligero-family commitments (RFC-0011, RFC-0012)
under the soundness budget of RFC-0014.

The verifier opens `τ` columns of an interleaved Reed–Solomon codeword of length `n = k / rate`
whose rows carry `P` vanishing-mask coefficients, so the effective rate is `ρ' = (k + P) / n`.
A cheating prover whose combined row is `δ`-far from the code is caught by one query with
probability at least `δ`, so `τ` queries give `τ · b` bits with `b = −log2(1 − δ)`, and `δ` is
what the proximity-gap theorem allows:

  unique decoding (RFC-0014, the bound in force), [BCIKS20] Theorem 1.2 for δ ≤ (1 − ρ')/2:
      δ = (1 − ρ') / 2,  error ε_U = n / q'
  Johnson (list decoding) regime, [BCIKS20] Theorem 1.2 for (1 − ρ')/2 < δ < 1 − √ρ', with
  η := 1 − √ρ' − δ:
      ε_J = (k' + 1)^2 / ((2 · min(η, √ρ'/20))^7 · q')
  where k' = k + P − 1 is the code's degree bound, q' = |Fp2| ≈ 2^128 is the field the
  combination coefficients are drawn from (Remark 1.5: sampling from an extension replaces q by
  q' in the denominators), and the test combines the rows with uniformly random coefficients,
  which samples a uniform element of an affine space, so Theorem 1.7 applies with no factor in
  the number of rows (the parametrized-curve version, Theorem 1.6, pays a factor l).
  Read from the ECCC report TR20-083 (May 30, 2020) on 2026-10-16; the `proximity` survey
  tracks later revisions and the 2025–2026 results.
  Johnson regime with an error linear in n (EXP-0003), two conjecture-free statements:
    [BCHKS25] Theorem 1.5 (ECCC TR25-169, 7 November 2025, page 9), correlated agreement for
    lines: with ρ = D/n for the maximum degree D = k + P − 1, γ = 1 − √ρ − η,
    m = max(⌈√ρ/(2η)⌉, 3) and t = m + 1/2, at most
        a = ((2 t^5 + 3 t γ ρ) / (3 ρ^{3/2})) · n + t/√ρ
    coefficients z give a γ-close combination of a pair that is γ-far from C^2; error a/q'.
    Theorem 4.6 (page 28) is the list (mutual) form for degree-M curves with |E| ≤ M · a.
    [ArkLib26] `johnsonExceptionCount` (Lean, `HiddenDerivative/Parameters/Johnson/FiniteBounds.lean`,
    Verified-zkEVM/ArkLib commit 4dae3acff846125521b7b3998010885071746f16 of 2026-10-13, read
    2026-10-16), the finite form of [DKT26]'s Johnson mutual correlated agreement: with the
    same m and t, μ = ⌈t/√ρ⌉ − 1, h = ⌈t²/(3ρ)⌉ − 1, A = ⌈(√ρ + η) n⌉, θ = (n − D)/(A − D),
        E₀ = (2μ − 1) h + θ (h + μ + 4 D μ h) + (n − D − 1) μ,
    and the closed bound E₀ < (8/3) n t³/ρ (`johnsonExceptionCount_lt_closed`). The affine-line
    MCA error at radius γ is min(1, E₀/|F|) (`mcaError_affineLine_johnson_le`, no characteristic
    hypothesis) and the affine-space MCA error in every dimension is E₀/(|F| − 1)
    (`mcaError_affineSpace_le_of_exactAgreement`). Both statements are for lines or affine spaces
    with independent uniform coefficients, not for the eq(ξ*, row) coefficients RFC-0011 step 3
    draws today.
  capacity (conjecture): δ = 1 − ρ' − η. Conjectures up to capacity were reported false in
  general in 2025 (Crites, Stewart, ePrint 2025/2046; survey to confirm), and RFC-0014 admits
  no conjectured bound, so this mode exists only to show what the conjecture would have bought.

Usage:
    python3 research/tools/proximity.py                      # live-64 and ni-100 parameters under unique decoding
    python3 research/tools/proximity.py --johnson            # the Johnson regime at several m
    python3 research/tools/proximity.py --bchks25 --arklib   # the linear-error Johnson bounds (EXP-0003)
    python3 research/tools/proximity.py --sweep-rates        # τ and opened bytes against the rate, unique decoding
    python3 research/tools/proximity.py --log2-k 22 --rate 1/4 --mask 160 --bits 64

A planning tool, non-normative. `tessera params check` (RFC-0014) is the calculator the protocol
trusts.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from fractions import Fraction

FIELD_BITS = 128  # Fp2 over Goldilocks: about 2^128 elements


@dataclass(frozen=True)
class Code:
    """One interleaved commitment: message width, rate, mask coefficients, rows."""

    name: str
    log2_k: int
    rate: Fraction
    mask: int
    rows: int  # rows combined into one opened row: the affine-space dimension
    target_bits: float
    grinding_bits: float = 0.0

    @property
    def k(self) -> int:
        return 1 << self.log2_k

    @property
    def n(self) -> int:
        return int(self.k / self.rate)

    @property
    def rho_eff(self) -> float:
        return (self.k + self.mask) / self.n

    @property
    def needed_bits(self) -> float:
        return self.target_bits - self.grinding_bits


# RFC-0014 'Parameter sets'. Rows: RFC-0011 'Sizes' gives about 5.8e5 non-zero advice rows at
# 16K tokens; the weight matrices have 2^20 / 2^... rows per type, taken as 2^12 here.
PARAMETER_SETS = {
    "live-64 advice": Code("live-64 advice", 22, Fraction(1, 2), 160, 580_000, 64),
    "live-64 weights": Code("live-64 weights", 20, Fraction(1, 4), 1 << 16, 4096, 64),
    "ni-100 advice": Code("ni-100 advice", 22, Fraction(1, 2), 200, 580_000, 100, 20),
    "ni-100 weights": Code("ni-100 weights", 20, Fraction(1, 4), 1 << 16, 4096, 100, 20),
}

# RFC-0014 table: τ_A 155 (64.3 bits), τ_W 97 (64.0), ni τ_A 193 (80.1 + 20), τ_W 122 (80.5 + 20).
EXPECTED_TAU = {"live-64 advice": 155, "live-64 weights": 97, "ni-100 advice": 193, "ni-100 weights": 122}


def bits_per_query(delta: float) -> float:
    return -math.log2(1 - delta)


def unique_decoding(code: Code) -> dict:
    delta = (1 - code.rho_eff) / 2
    b = bits_per_query(delta)
    tau = math.ceil(code.needed_bits / b)
    field_term_bits = FIELD_BITS - math.log2(code.n)
    return {"delta": delta, "bits_per_query": b, "tau": tau, "achieved": tau * b, "field_bits": field_term_bits}


def johnson(code: Code, eta: float, curve_degree: int = 0) -> dict | None:
    """[BCIKS20] Theorem 1.2, Johnson regime: `δ = 1 − √ρ' − η`, `ε_J = (k'+1)^2 / ((2 min(η, √ρ'/20))^7 q')`.

    `curve_degree = 0` is the affine-space test of Theorem 1.7 (uniform combination coefficients,
    no factor); a positive value is the parametrized-curve test of Theorem 1.6, which pays that
    factor.
    """
    r = code.rho_eff
    delta = 1 - math.sqrt(r) - eta
    if delta <= (1 - r) / 2:
        return None  # below the unique-decoding radius: Theorem 1.2's first case applies instead
    b = bits_per_query(delta)
    tau = math.ceil(code.needed_bits / b)
    degree_bound = code.k + code.mask - 1
    cap = min(eta, math.sqrt(r) / 20)
    log2_eps = 2 * math.log2(degree_bound + 1) - 7 * math.log2(2 * cap) - FIELD_BITS
    if curve_degree:
        log2_eps += math.log2(curve_degree)
    return {"eta": eta, "delta": delta, "bits_per_query": b, "tau": tau, "field_bits": -log2_eps}


LINEAR_ETAS = (0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005)


def johnson_recipe(code: Code, eta: float) -> dict | None:
    """The shared recipe of [BCHKS25] Theorem 1.5 and [ArkLib26] `johnsonM`/`johnsonT`.

    `D = k + P − 1` is the maximum degree (the code's dimension is `D + 1`), `ρ = D/n`,
    `γ = 1 − √ρ − η`, `m = max(⌈√ρ/(2η)⌉, 3)`, `t = m + 1/2`. Returns None when `γ` is at or
    below the unique-decoding radius `(1 − ρ)/2`, where the unique-decoding case is the better
    statement ([BCHKS25] Corollary 1.4 gives `a = γn + 1` there).
    """
    if eta <= 0:
        return None
    n = code.n
    degree = code.k + code.mask - 1
    rho = degree / n
    sqrt_rho = math.sqrt(rho)
    gamma = 1 - sqrt_rho - eta
    if gamma <= (1 - rho) / 2:
        return None
    m = max(math.ceil(sqrt_rho / (2 * eta)), 3)
    t = m + 0.5
    b = bits_per_query(gamma)
    return {
        "eta": eta, "n": n, "degree": degree, "rho": rho, "sqrt_rho": sqrt_rho, "gamma": gamma,
        "m": m, "t": t, "bits_per_query": b, "tau": math.ceil(code.needed_bits / b),
    }


def bchks25(code: Code, eta: float) -> dict | None:
    """[BCHKS25] Theorem 1.5: the number `a` of bad coefficients, linear in `n`; error `a/q'`.

    `field_bits` is the line error `−log2(a/q')`; `affine_bits` the affine-space error with the
    generic line-to-affine transfer `a/(q' − 1)` (the same transfer [ArkLib26] proves for the
    exact-agreement form). For `q' = 2^128` the two differ by less than 2^-127.
    """
    r = johnson_recipe(code, eta)
    if r is None:
        return None
    t, rho, gamma, n = r["t"], r["rho"], r["gamma"], r["n"]
    a = ((2 * t**5 + 3 * t * gamma * rho) / (3 * rho**1.5)) * n + t / math.sqrt(rho)
    log2_a = math.log2(a)
    r.update({"a": a, "field_bits": FIELD_BITS - log2_a, "affine_bits": -(log2_a - math.log2(2**FIELD_BITS - 1))})
    return r


def arklib_johnson(code: Code, eta: float) -> dict | None:
    """[ArkLib26] `johnsonExceptionCount n D A η` at `A = ⌈(√ρ + η) n⌉`, exactly, plus its closed bound.

    Hypotheses of `johnsonExceptionCount_lt_closed` and `mcaError_affineLine_johnson_le`:
    `1 ≤ D ≤ n − 2`, `η > 0`, `√ρ + η ≤ 1`, `(√ρ + η) n ≤ A`. `field_bits` is the affine-line
    MCA error `E₀/|F|`; `affine_bits` the affine-space MCA error `E₀/(|F| − 1)` in every
    dimension; `closed_bits` uses `(8/3) n t³/ρ` instead of `E₀`.
    """
    r = johnson_recipe(code, eta)
    if r is None:
        return None
    n, degree, t, sqrt_rho, rho = r["n"], r["degree"], r["t"], r["sqrt_rho"], r["rho"]
    if not (1 <= degree <= n - 2) or sqrt_rho + eta > 1:
        return None
    agreement_threshold = math.ceil((sqrt_rho + eta) * n)
    mu = math.ceil(t / sqrt_rho) - 1
    h = math.ceil(t * t / (3 * rho)) - 1
    theta = Fraction(n - degree, agreement_threshold - degree)
    e0 = (2 * mu - 1) * h + theta * (h + mu + 4 * degree * mu * h) + (n - degree - 1) * mu
    log2_e0 = math.log2(e0.numerator) - math.log2(e0.denominator)
    closed = Fraction(8, 3) * n * Fraction(t) ** 3 / Fraction(rho)
    log2_closed = math.log2(closed.numerator) - math.log2(closed.denominator)
    # Machine-checked list bound at agreement A: ⌊n (A − D) / (A² − n D)⌋ ([ArkLib26]
    # `closePolynomialSet_finite_and_ncard_le_pairwiseJohnson`, hypotheses D + 1 ≤ A, n D < A²).
    list_bound = (n * (agreement_threshold - degree)) // (agreement_threshold**2 - n * degree)
    r.update({
        "mu": mu, "h": h, "theta": float(theta), "A": agreement_threshold, "e0": float(e0),
        "list_bound": list_bound,
        "field_bits": FIELD_BITS - log2_e0,
        "affine_bits": -(log2_e0 - math.log2(2**FIELD_BITS - 1)),
        "closed_bits": FIELD_BITS - log2_closed,
    })
    return r


def print_linear_johnson(code: Code, use_bchks: bool, use_arklib: bool) -> None:
    """EXP-0003: `τ` and field-size terms under the linear-error Johnson bounds, per `η`."""
    u = unique_decoding(code)
    base_gb = opened_bytes(code, u["tau"]) / 1e9
    print(
        f"linear-error Johnson regime for {code.name} (needed {code.needed_bits:.0f} one-shot bits; "
        f"unique decoding gives τ = {u['tau']}, {base_gb:.3f} GB opened)"
    )
    head = f"{'η':>7} {'γ':>7} {'bits/q':>7} {'τ':>5} {'Δτ':>7} {'GB':>6}"
    if use_bchks:
        head += f" {'[BCHKS25] a bits':>17}"
    if use_arklib:
        head += f" {'[ArkLib] E₀ bits':>17} {'closed':>7}"
    print(head + "  admissible")
    best: dict[str, tuple[int, float]] = {}
    for eta in LINEAR_ETAS:
        r = johnson_recipe(code, eta)
        if r is None:
            print(f"{eta:>7.4f}  γ at or below the unique-decoding radius: Corollary 1.4 / unique decoding applies")
            continue
        gb = opened_bytes(code, r["tau"]) / 1e9
        line = f"{eta:>7.4f} {r['gamma']:>7.4f} {r['bits_per_query']:>7.4f} {r['tau']:>5} {100 * (r['tau'] - u['tau']) / u['tau']:>+6.1f}% {gb:>6.3f}"
        verdict = []
        if use_bchks:
            b = bchks25(code, eta)
            line += f" {b['field_bits']:>17.1f}"
            if b["field_bits"] >= code.needed_bits:
                verdict.append("BCHKS25")
                if "BCHKS25" not in best or r["tau"] < best["BCHKS25"][0]:
                    best["BCHKS25"] = (r["tau"], b["field_bits"])
        if use_arklib:
            k = arklib_johnson(code, eta)
            if k is None:
                line += f" {'n/a':>17} {'n/a':>7}"
            else:
                line += f" {k['field_bits']:>17.1f} {k['closed_bits']:>7.1f}"
                if k["field_bits"] >= code.needed_bits:
                    verdict.append("ArkLib")
                    if "ArkLib" not in best or r["tau"] < best["ArkLib"][0]:
                        best["ArkLib"] = (r["tau"], k["field_bits"])
        print(line + "  " + (", ".join(verdict) or "none"))
    for name, (tau, bits) in best.items():
        gb = opened_bytes(code, tau) / 1e9
        print(
            f"best admissible under {name}: τ = {tau} ({100 * (tau - u['tau']) / u['tau']:+.1f}% vs unique decoding), "
            f"field term {bits:.1f} bits, {gb:.3f} GB opened ({100 * (gb - base_gb) / base_gb:+.1f}%)"
        )
    if not best:
        print("no admissible η under the requested bounds")
    print()


def capacity(code: Code, eta: float) -> dict | None:
    delta = 1 - code.rho_eff - eta
    if delta <= 0:
        return None
    b = bits_per_query(delta)
    return {"eta": eta, "delta": delta, "bits_per_query": b, "tau": math.ceil(code.needed_bits / b)}


def opened_bytes(code: Code, tau: int) -> float:
    """Opened columns: τ columns × rows × 8 bytes (RFC-0011 'Sizes')."""
    return tau * code.rows * 8


def print_unique(codes: dict[str, Code]) -> bool:
    ok = True
    print("unique decoding (RFC-0014 'Soundness budget'): δ = (1 − ρ')/2")
    print(f"{'set':<18} {'n':>9} {'ρ′':>9} {'δ':>7} {'bits/q':>7} {'τ':>5} {'bits':>6} {'RFC τ':>6} {'field':>6}")
    for name, code in codes.items():
        u = unique_decoding(code)
        expected = EXPECTED_TAU.get(name)
        match = expected is None or expected == u["tau"]
        ok &= match
        print(
            f"{name:<18} {code.n:>9} {code.rho_eff:>9.6f} {u['delta']:>7.4f} {u['bits_per_query']:>7.4f} "
            f"{u['tau']:>5} {u['achieved']:>6.1f} {str(expected or '-'):>6} {u['field_bits']:>6.0f}"
            + ("" if match else "   MISMATCH")
        )
    print()
    return ok


def print_johnson(code: Code) -> None:
    u = unique_decoding(code)
    print(f"Johnson regime for {code.name} (needed {code.needed_bits:.0f} bits; unique decoding gives τ = {u['tau']})")
    print("   [BCIKS20] Theorem 1.2 with Theorem 1.7 (affine space, no row factor) and Remark 1.5 (q' = |Fp2|)")
    cap = math.sqrt(code.rho_eff) / 20
    print(f"   the error term stops improving at η = √ρ'/20 = {cap:.4f}; smaller η buys distance, never field bits")
    print(f"{'η':>8} {'δ':>8} {'bits/q':>8} {'τ':>6} {'field bits':>11}  verdict")
    for eta in (0.10, 0.05, cap, 0.02, 0.01, 0.005, 0.001):
        j = johnson(code, eta)
        if j is None:
            print(f"{eta:>8.4f}  δ at or below the unique-decoding radius: the first case of Theorem 1.2 applies")
            continue
        verdict = []
        if j["tau"] >= u["tau"]:
            verdict.append("no fewer queries than unique decoding")
        if j["field_bits"] < code.target_bits:
            verdict.append(f"field term below {code.target_bits:.0f} bits")
        print(f"{eta:>8.4f} {j['delta']:>8.4f} {j['bits_per_query']:>8.4f} {j['tau']:>6} {j['field_bits']:>11.1f}  {'; '.join(verdict) or 'usable'}")
    c = capacity(code, 0.01)
    if c:
        print(f"capacity conjecture (η = 0.01, not admissible, reported false in general in 2025): δ = {c['delta']:.4f}, τ = {c['tau']}")
    print()


def print_sweep(code: Code, rates: list[Fraction]) -> None:
    print(f"rate sweep for {code.name} under unique decoding; encode and hash volume scale with n/k = 1/rate")
    print(f"{'rate':>6} {'n':>9} {'ρ′':>9} {'bits/q':>7} {'τ':>5} {'opened GB':>10} {'volume ×':>9}")
    base = None
    for rate in rates:
        c = Code(code.name, code.log2_k, rate, code.mask, code.rows, code.target_bits, code.grinding_bits)
        u = unique_decoding(c)
        gb = opened_bytes(c, u["tau"]) / 1e9
        base = base or gb
        print(f"{str(rate):>6} {c.n:>9} {c.rho_eff:>9.6f} {u['bits_per_query']:>7.4f} {u['tau']:>5} {gb:>10.3f} {float(1 / rate):>9.1f}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log2-k", type=int, help="message width log2 k (custom code)")
    ap.add_argument("--rate", type=Fraction, default=Fraction(1, 2), help="code rate, e.g. 1/2")
    ap.add_argument("--mask", type=int, default=160, help="vanishing-mask coefficients P")
    ap.add_argument("--rows", type=int, default=580_000, help="rows combined into one opened row (sets opened bytes)")
    ap.add_argument("--bits", type=float, default=64, help="target bits")
    ap.add_argument("--grinding", type=float, default=0, help="grinding bits counted toward the target")
    ap.add_argument("--johnson", action="store_true", help="print the Johnson-regime table")
    ap.add_argument("--bchks25", action="store_true", help="[BCHKS25] Theorem 1.5, error linear in n (EXP-0003)")
    ap.add_argument("--arklib", action="store_true", help="[ArkLib26] johnsonExceptionCount, exact and closed (EXP-0003)")
    ap.add_argument("--sweep-rates", action="store_true", help="τ and opened bytes against the rate")
    args = ap.parse_args(argv)

    if args.log2_k is not None:
        codes = {"custom": Code("custom", args.log2_k, args.rate, args.mask, args.rows, args.bits, args.grinding)}
    else:
        codes = PARAMETER_SETS
    ok = print_unique(codes)
    if args.johnson:
        for code in codes.values():
            print_johnson(code)
    if args.bchks25 or args.arklib:
        for code in codes.values():
            print_linear_johnson(code, args.bchks25, args.arklib)
    if args.sweep_rates:
        for code in codes.values():
            print_sweep(code, [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 16)])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
