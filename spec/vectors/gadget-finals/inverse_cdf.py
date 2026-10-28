# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0009 Lemma 10 and RFC-0058 InverseCdf: independent Python big-integer and cube-sum oracle.

No Rust or quantizer implementation is imported or called. Token selection uses prefix bisection,
normalizations and borrows come from whole-integer prefixes, and every sumcheck round sums literal
Boolean cube vertices.
"""
import functools
import importlib.util
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cdf_params(vocab, sigma, bounded):
    assert 1 <= vocab <= (1 << 18) and 0 <= sigma <= 32
    w_bits = (vocab * (1 << sigma)).bit_length()
    assert 1 <= w_bits <= 51
    k = (w_bits + 15) // 16
    m = k + 2
    top_z_bits = w_bits - 16 * (k - 1)
    assert 1 <= top_z_bits <= 16
    box_u = [65535, 65535]
    box_z = [65535] * (k - 1) + [(1 << top_z_bits) - 1]
    box_p = [
        sum(box_u[i] * box_z[h] for i in range(2) for h in range(k) if i + h == j)
        for j in range(m)
    ]
    lower_left = [0, 0] + list(box_z)
    lower_right = list(box_p)
    upper_left = [box_p[j] + (1 if j == 0 else 0) for j in range(m)]
    upper_right = [
        (box_z[j - 2] if j >= 2 else 0) + ((1 << sigma) if j == 2 else 0)
        for j in range(m)
    ]
    return dict(
        vocab=vocab,
        sigma=sigma,
        bounded=bounded,
        w_bits=w_bits,
        k=k,
        m=m,
        top_z_bits=top_z_bits,
        lower_bounds=[lower_left, lower_right],
        upper_bounds=[upper_left, upper_right],
    )


def sample_bisect(weights, u):
    prefix, total = [], 0
    for w in weights:
        total += w
        prefix.append(total)
    assert total > 0
    lo, hi = 0, len(prefix) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix[mid] * (1 << 32) > u * total:
            hi = mid
        else:
            lo = mid + 1
    below = prefix[lo - 1] if lo else 0
    assert below * (1 << 32) <= u * total < prefix[lo] * (1 << 32)
    return lo


def row_witness(oracle, params, psi, u, z_val, b_val, s_val, alternative=False):
    k, m = params["k"], params["m"]
    u_limbs = [u & 0xFFFF, u >> 16]
    z_limbs = [(z_val >> (16 * h)) & 0xFFFF for h in range(k)]
    b_limbs = [(b_val >> (16 * h)) & 0xFFFF for h in range(k)]
    p_coeffs = [
        sum(u_limbs[i] * z_limbs[h] for i in range(2) for h in range(k) if i + h == j)
        for j in range(m)
    ]
    lower_left = [0, 0] + b_limbs
    lower_right = list(p_coeffs)
    upper_left = [p_coeffs[j] + (psi if j == 0 else 0) for j in range(m)]
    upper_right = [
        (b_limbs[j - 2] if j >= 2 else 0) + (s_val if j == 2 else 0)
        for j in range(m)
    ]
    lower_arrays = oracle.witness(lower_left, lower_right, 16, alternative=alternative)
    upper_arrays = oracle.witness(upper_left, upper_right, 16, alternative=alternative)
    committed = (
        [u_limbs[0]]
        + z_limbs[: k - 1]
        + b_limbs[: k - 1]
        + [x for arr in lower_arrays for x in arr]
        + [x for arr in upper_arrays for x in arr]
    )
    range_keys = (
        u_limbs
        + z_limbs
        + b_limbs
        + [x for arr in lower_arrays for x in arr]
        + [x for arr in upper_arrays for x in arr]
    )
    assert len(committed) == 1 + 2 * (k - 1) + 12 * m
    assert len(range_keys) == 2 + 2 * k + 12 * m
    return dict(
        psi=psi,
        u=u,
        z=z_val,
        b=b_val,
        s=s_val,
        u_limbs=u_limbs,
        z_limbs=z_limbs,
        b_limbs=b_limbs,
        p_coeffs=p_coeffs,
        lower_arrays=lower_arrays,
        upper_arrays=upper_arrays,
        committed=committed,
        range_keys=range_keys,
    )


def range_widths(oracle, params):
    k, m, top_z = params["k"], params["m"], params["top_z_bits"]
    lc_l = oracle.carry_widths(params["lower_bounds"][0], 16)
    rc_l = oracle.carry_widths(params["lower_bounds"][1], 16)
    lc_u = oracle.carry_widths(params["upper_bounds"][0], 16)
    rc_u = oracle.carry_widths(params["upper_bounds"][1], 16)
    z_widths = [16] * (k - 1) + [top_z]
    cmp_l = [16] * m + lc_l + [16] * m + rc_l + [16] * m + [1] * m
    cmp_u = [16] * m + lc_u + [16] * m + rc_u + [16] * m + [1] * m
    return (
        [16, 16] + z_widths + z_widths + cmp_l + cmp_u,
        [lc_l, rc_l],
        [lc_u, rc_u],
    )


def schema_cases(oracle, q, rng):
    configs = [
        ("w1-exact", 1, 0, False),
        ("w1-bounded", 1, 0, True),
        ("w17-exact", 3, 15, False),
        ("rfc0058-128k-s16", 128256, 16, False),
        ("rfc0058-128k-s32", 128256, 32, True),
        ("rfc0058-max-2p18-s32", 1 << 18, 32, False),
    ]
    out = []
    for name, vocab, sigma, bounded in configs:
        p = cdf_params(vocab, sigma, bounded)
        widths, lower_cw, upper_cw = range_widths(oracle, p)
        max_e = 1 << sigma
        max_z = vocab * max_e
        # Four synthetic rows: u=0, u=2^32-1 at full box ceiling, interior u, and padded row.
        row_specs = [
            (1, 0, max_z, 0, max_e),
            (1, (1 << 32) - 1, max_z, max_z - max_e, max_e),
            (1, (1 << 31) + 7, max(1, max_z // 2 + 1), max_z // 4, max(1, max_z - max_z // 4)),
            (0, 0xDEADBEEF, 0, 0, 0),
        ]
        # Adjust interior row 2 so s_val <= 2^sigma and bracket holds tightly.
        if vocab == 1:
            row_specs[2] = (1, (1 << 31) + 7, max_e, 0, max_e)
        else:
            z2 = min(max_z, 2 * max_e)
            u2 = (1 << 31) - 1
            # With two equal weights max_e, prefix[0]*2^32 = max_e*2^32 > u2 * (2*max_e), so token 0 wins.
            row_specs[2] = (1, u2, z2, 0, max_e)
        rows = [row_witness(oracle, p, *spec) for spec in row_specs]
        for r in rows:
            assert all(0 <= v < (1 << w) for v, w in zip(r["range_keys"], widths))
        point = [rng.fp2(), rng.fp2()]
        out.append(
            dict(
                name=name,
                vocab=vocab,
                sigma=sigma,
                bounded=bounded,
                w_bits=p["w_bits"],
                k=p["k"],
                m=p["m"],
                top_z_bits=p["top_z_bits"],
                committed_row_count=1 + 2 * (p["k"] - 1) + 12 * p["m"],
                range_widths=widths,
                lower_bounds=p["lower_bounds"],
                lower_carry_widths=lower_cw,
                upper_bounds=p["upper_bounds"],
                upper_carry_widths=upper_cw,
                point=point,
                u_eval=q.mle([q.lift(r["u"]) for r in rows], point),
                z_eval=q.mle([q.lift(r["z"]) for r in rows], point),
                b_eval=q.mle([q.lift(r["b"]) for r in rows], point),
                s_eval=q.mle([q.lift(r["s"]) for r in rows], point),
                committed_evals=[
                    q.mle([q.lift(r["committed"][j]) for r in rows], point)
                    for j in range(len(rows[0]["committed"]))
                ],
                range_key_evals=[
                    q.mle([q.lift(r["range_keys"][j]) for r in rows], point)
                    for j in range(len(rows[0]["range_keys"]))
                ],
                rows=[
                    dict(
                        psi=r["psi"],
                        u=r["u"],
                        z=r["z"],
                        b=r["b"],
                        s=r["s"],
                        committed=r["committed"],
                        range_keys=r["range_keys"],
                    )
                    for r in rows
                ],
            )
        )
    return out


def cases(oracle=None):
    here = Path(__file__).resolve().parent
    if oracle is None:
        oracle = load(here / "generate.py", "cdf_base_oracle")
    q = load(here.parent / "sumcheck" / "generate.py", "cdf_sumcheck_oracle")
    q.hashing.check_official_vectors()
    rng = q.Inputs(0x16000580009)
    schemas = schema_cases(oracle, q, rng)

    honest_specs = [
        ("scalar-exact", 1, 1, 0, False),
        ("scalar-bounded", 1, 1, 3, True),
        ("two-vocab-exact", 2, 2, 4, False),
        ("padded-exact-k1", 3, 5, 4, False),
        ("padded-bounded-k2", 3, 5, 16, True),
        ("wide-bounded-k3", 5, 6, 31, True),
    ]
    alt_specs = [
        ("neighbour-high", 3, 5, 4, True),
        ("neighbour-low", 3, 5, 4, True),
        ("zero-weight-token", 3, 5, 4, False),
        ("draw-overflow", 3, 5, 4, False),
        ("real-row-zero-mass", 3, 5, 4, False),
        ("padded-z-nonzero", 3, 5, 4, True),
        ("bounded-nonboolean", 3, 5, 4, True),
        ("bounded-vocab-padding", 3, 5, 4, True),
        ("bounded-two-hot", 3, 5, 4, True),
        ("bounded-padded-h", 3, 5, 4, True),
        ("bad-suffix", 3, 5, 4, True),
    ]

    inv_pow16 = [pow(1 << (16 * j), -1, q.P) for j in range(5)]
    component_cases = []

    for idx, (name, rows, vocab, sigma, bounded) in enumerate(honest_specs + alt_specs):
        bad = name if idx >= len(honest_specs) else None
        p = cdf_params(vocab, sigma, bounded)
        widths, lower_cw, upper_cw = range_widths(oracle, p)
        k, m = p["k"], p["m"]
        vb, rb = (vocab - 1).bit_length(), (rows - 1).bit_length()
        V_pad, T_pad = 1 << vb, 1 << rb
        max_e = 1 << sigma

        exps = []
        draws = []
        tokens = []
        for t in range(T_pad):
            real = t < rows
            if not real:
                row_e = [0] * vocab
                u = (0x10203040 + t) & 0xFFFFFFFF
            elif vocab == 1:
                row_e = [max_e]
                u = 0 if t % 2 == 0 else 0xFFFFFFFF
            else:
                # Include a zero-weight interior entry on row 0 when vocab >= 3.
                row_e = [
                    0
                    if (t == 0 and v == 1 and vocab >= 3)
                    else max(1, ((((v + 2 * t) % vocab) + 1) * max_e) // (vocab + 1))
                    for v in range(vocab)
                ]
                row_e[t % vocab] = max_e
                if row_e[0] == 0:
                    row_e[0] = max(1, max_e // 2)
                # Place draws across prefix boundaries.
                total_z = sum(row_e)
                if t == 0:
                    # Target token 2 when vocab >= 3 (right past zero-weight token 1), else token 0.
                    prefix_0 = row_e[0]
                    u = (prefix_0 * (1 << 32)) // total_z + 1 if vocab >= 3 else 0
                elif t == 1:
                    u = 0
                else:
                    u = 0xFFFFFFFF
            chosen = sample_bisect(row_e, u) if real else 0
            exps.extend(row_e + [0] * (V_pad - vocab))
            draws.append(u)
            tokens.append(chosen)

        # Apply surgical alternative mutations on row 0 or padded row `rows`.
        alt_compare = False
        local_reject = None  # 'range', 'lower_borrow', 'upper_borrow'
        failed_stage = None  # (stage_index, instance_index)

        if bad == "neighbour-high":
            # Row 0 chose token 2; force token 3 (higher prefix -> lower comparison fails).
            tokens[0] = min(vocab - 1, tokens[0] + 1)
            alt_compare = True
            local_reject = "lower_borrow"
        elif bad == "neighbour-low":
            # Row 0 chose token >= 2; force token 0 (smaller prefix -> upper comparison fails).
            assert tokens[0] >= 1
            tokens[0] = 0
            alt_compare = True
            local_reject = "upper_borrow"
        elif bad == "zero-weight-token":
            # Row 0 has row_e[1] == 0, with C(0) == C(1) and u right above C(0)*2^32/Z -> upper fails.
            assert exps[1] == 0
            tokens[0] = 1
            alt_compare = True
            local_reject = "upper_borrow"
        elif bad == "draw-overflow":
            draws[0] = (1 << 32) + 5
            # Re-pick token if needed or keep valid comparison? Note u_1 >= 2^16 exceeds Range(16).
            local_reject = "range"
        elif bad == "real-row-zero-mass":
            exps[:V_pad] = [0] * V_pad
            draws[0] = 0
            tokens[0] = 0
            alt_compare = True
            local_reject = "upper_borrow"
        elif bad == "padded-z-nonzero":
            # Padded row `rows` has u = 0 and e[rows, 0] = 1 -> Z = 1, B = S = 0; comparisons 0 <= 0 pass!
            draws[rows] = 0
            exps[rows * V_pad] = 1
            failed_stage = (1, m)

        h_hat = [int(t < rows and v == tokens[t]) for t in range(T_pad) for v in range(V_pad)]

        if bad == "bounded-nonboolean":
            # h_hat on row 0 has [0, 1, 1, -1, 0, ...] with count 1!
            # Make e on row 0 [1, 4, 4, 4, 1], u = 0 so Z=14, S = 4+4-4 = 4, L = [1, 0, -1, 0, 0],
            # wait: keep L >= 0 so B >= 0! With h_hat = [1, 1, -1, 0, ...]:
            # sum(h_hat) = 1!
            # L_0 = 1 + (-1) = 0, L_1 = -1 -> wait: if e_1 = 0, then e_1 * L_1 = 0, so B = sum e_v L_v = 0!
            # Check that! h_hat = [1, 1, -1, 0, ...], e = [4, 0, 2, 1, 1]:
            # sum(h_hat) = 1 (passes row count!)
            # h_hat[v >= vocab] = 0 (passes vocab!)
            # Z = 4 + 0 + 2 + 1 + 1 = 8
            # L = [0, -1, 0, 0, ...] -> B = 4*0 + 0*(-1) + 2*0 + ... = 0!
            # S = 1*4 + 1*0 + (-1)*2 = 2!
            # With u = 0: lower is 0 <= 0 (passes!), upper is 1 <= 2*2^32 (passes!)!
            exps[:V_pad] = [4, 0, 2, 1, 1] + [0] * (V_pad - 5)
            draws[0] = 0
            h_hat[:V_pad] = [1, 1, -1] + [0] * (V_pad - 3)
            failed_stage = (2, 4)
        elif bad == "bounded-vocab-padding":
            # Z = 0 on real row 0 for v < vocab, but e[0, vocab] = 1 and h_hat[0, vocab] = 1, u = 0:
            # booleanity passes, count = 1 passes, Z = 0, B = 0, S = 1, lower 0 <= 0 and upper 1 <= 2^32 pass!
            exps[:V_pad] = [0] * vocab + [1] + [0] * (V_pad - vocab - 1)
            draws[0] = 0
            h_hat[:V_pad] = [int(v == vocab) for v in range(V_pad)]
            failed_stage = (2, 5)
        elif bad == "bounded-two-hot":
            # Two ones on row 0: h_hat = [1, 1, 0, ...], e = [2, 2, 4, 1, 1], u = 2^31:
            # Z = 10, L = [1, 0, ...], B = 2*1 = 2, S = 2 + 2 = 4.
            # Lower: 2 * 2^32 <= 2^31 * 10 = 5 * 2^32 (passes!)
            # Upper: 5 * 2^32 + 1 <= (2 + 4) * 2^32 = 6 * 2^32 (passes!)
            exps[:V_pad] = [2, 2, 4, 1, 1] + [0] * (V_pad - 5)
            draws[0] = 1 << 31
            h_hat[:V_pad] = [1, 1] + [0] * (V_pad - 2)
            failed_stage = (2, 6)
        elif bad == "bounded-padded-h":
            # On padded row `rows`, h_hat[rows, 0] = 1 while e[rows, *] = 0 -> Z = B = S = 0,
            # booleanity & vocab pass, all row comparisons pass, only row count fails!
            h_hat[rows * V_pad] = 1
            failed_stage = (2, 6)

        actual_suffix = [
            sum(h_hat[t * V_pad + w] for w in range(v + 1, V_pad))
            for t in range(T_pad)
            for v in range(V_pad)
        ]
        suffix = list(actual_suffix)
        if bad == "bad-suffix":
            # u = 0, e = [4, 4, 1, 1, 1], honest token is 0, prover claims token 1 (h_hat has 1 at v=1)
            # and sets suffix L' = 0 on row 0 so B = 0 -> Stage 1 & 2 pass, Stage 3 suffix fails!
            exps[:V_pad] = [4, 4, 1, 1, 1] + [0] * (V_pad - 5)
            draws[0] = 0
            tokens[0] = 1
            h_hat[:V_pad] = [0, 1] + [0] * (V_pad - 2)
            actual_suffix = [
                sum(h_hat[t * V_pad + w] for w in range(v + 1, V_pad))
                for t in range(T_pad)
                for v in range(V_pad)
            ]
            suffix = list(actual_suffix)
            suffix[:V_pad] = [0] * V_pad
            failed_stage = (3, 0)

        row_records = []
        for t in range(T_pad):
            psi = int(t < rows)
            z_val = sum(exps[t * V_pad + v] for v in range(vocab))
            b_val = sum(exps[t * V_pad + v] * suffix[t * V_pad + v] for v in range(V_pad))
            s_val = sum(exps[t * V_pad + v] * h_hat[t * V_pad + v] for v in range(V_pad))
            if bad == "draw-overflow" and t == 0:
                # Build witness with unmasked u_1 = draws[0] >> 16 (= 65536, out of Range(16))
                # while keeping comparison arithmetic consistent for u * Z:
                u_val = draws[0]
                u_limbs = [u_val & 0xFFFF, u_val >> 16]
                z_limbs = [(z_val >> (16 * h)) & 0xFFFF for h in range(k)]
                b_limbs = [(b_val >> (16 * h)) & 0xFFFF for h in range(k)]
                p_coeffs = [
                    sum(u_limbs[i] * z_limbs[h] for i in range(2) for h in range(k) if i + h == j)
                    for j in range(m)
                ]
                # Make token vocab-1 and adjust dummy arrays if needed; local range check rejects u_1 = 65536 first.
                lower_arrays = oracle.witness([0, 0] + b_limbs, p_coeffs, 16, alternative=True)
                upper_arrays = oracle.witness(
                    [p_coeffs[j] + (psi if j == 0 else 0) for j in range(m)],
                    [(b_limbs[j - 2] if j >= 2 else 0) + (s_val if j == 2 else 0) for j in range(m)],
                    16,
                    alternative=True,
                )
                committed = (
                    [u_limbs[0]]
                    + z_limbs[: k - 1]
                    + b_limbs[: k - 1]
                    + [x for arr in lower_arrays for x in arr]
                    + [x for arr in upper_arrays for x in arr]
                )
                range_keys = (
                    u_limbs
                    + z_limbs
                    + b_limbs
                    + [x for arr in lower_arrays for x in arr]
                    + [x for arr in upper_arrays for x in arr]
                )
                row_records.append(
                    dict(
                        psi=psi,
                        u=u_val,
                        z=z_val,
                        b=b_val,
                        s=s_val,
                        p_coeffs=p_coeffs,
                        committed=committed,
                        range_keys=range_keys,
                    )
                )
            else:
                row_records.append(
                    row_witness(
                        oracle,
                        p,
                        psi,
                        draws[t],
                        z_val,
                        b_val,
                        s_val,
                        alternative=(alt_compare and t == 0),
                    )
                )

        if local_reject != "range":
            for r in row_records:
                assert all(0 <= v < (1 << w) for v, w in zip(r["range_keys"], widths))
        else:
            assert any(
                not (0 <= v < (1 << w))
                for r in row_records
                for v, w in zip(r["range_keys"], widths)
            )

        row_target = tuple(rng.fp2() for _ in range(rb))
        u_col = [q.lift(r["u"]) for r in row_records]
        z_col = [q.lift(r["z"]) for r in row_records]
        b_col = [q.lift(r["b"]) for r in row_records]
        s_col = [q.lift(r["s"]) for r in row_records]
        u0_col = [q.lift(r["committed"][0]) for r in row_records]
        z_low_cols = [
            [q.lift(r["committed"][1 + h]) for r in row_records] for h in range(k - 1)
        ]
        psi_col = [q.lift(r["psi"]) for r in row_records]
        e_col = [q.lift(v) for v in exps]
        h_col = [q.lift(v) for v in h_hat]
        l_col = [q.lift(v) for v in suffix]
        valid_vocab_col = [q.lift(int(v < vocab)) for v in range(V_pad)]

        u_eval = q.mle(u_col, row_target)
        z_eval = q.mle(z_col, row_target)
        b_eval = q.mle(b_col, row_target)
        s_eval = q.mle(s_col, row_target)
        range_key_evals = [
            q.mle([q.lift(r["range_keys"][j]) for r in row_records], row_target)
            for j in range(len(widths))
        ]
        p_targets = [
            q.mle([q.lift(r["p_coeffs"][j]) for r in row_records], row_target)
            for j in range(m)
        ]

        context = rng.context()

        def prove_chain(omit=False):
            t = q.transcript.Transcript(q.transcript.session_context(context))
            t.absorb(
                "stage/0/split",
                q.tce([u_eval, z_eval, b_eval, s_eval] + range_key_evals),
            )
            zero_row = tuple(q.challenge(t, f"stage/1/zero/{j}") for j in range(rb))

            @functools.lru_cache(None)
            def stage1_terms(pt):
                u_at = q.mle(u_col, pt)
                u0_at = q.mle(u0_col, pt)
                u1_at = q.mul(q.sub(u_at, u0_at), q.lift(inv_pow16[1]))
                u_l = [u0_at, u1_at]
                z_at = q.mle(z_col, pt)
                z_low = [q.mle(col, pt) for col in z_low_cols]
                rem = z_at
                for h, zh in enumerate(z_low):
                    rem = q.sub(rem, q.mul(zh, q.lift(1 << (16 * h))))
                z_top = q.mul(rem, q.lift(inv_pow16[k - 1]))
                z_l = z_low + [z_top]
                eq_rt = q.eq_points(row_target, pt)
                out = []
                for j in range(m):
                    pj = q.ZERO
                    for i in range(2):
                        for h in range(k):
                            if i + h == j:
                                pj = q.add(pj, q.mul(u_l[i], z_l[h]))
                    out.append(q.mul(eq_rt, pj))
                psi_at = q.mle(psi_col, pt)
                out.append(q.mul(q.eq_points(zero_row, pt), q.mul(q.sub(q.ONE, psi_at), z_at)))
                return out

            stage1_claims = p_targets + [q.ZERO]

            def stage1_evals(pt):
                per_p = [
                    q.mle(u_col, pt),
                    q.mle(u0_col, pt),
                    q.mle(z_col, pt),
                ] + [q.mle(col, pt) for col in z_low_cols]
                return per_p * m + [q.mle(z_col, pt)]

            def run_stage(stage_no, n_vars, degree, claims_list, terms_fn, evals_fn, omit_idx):
                gamma = q.challenge(t, f"stage/{stage_no}/gamma")
                weights = [q.power(gamma, i) for i in range(len(claims_list))]

                def current(pt):
                    vals = list(terms_fn(tuple(pt)))
                    if omit_idx is not None:
                        delta0 = q.mle([q.ONE] + [q.ZERO] * ((1 << n_vars) - 1), pt)
                        vals[omit_idx] = q.mul(claims_list[omit_idx], delta0)
                    return vals

                def at(prefix):
                    rem_vars = n_vars - len(prefix)
                    return q.total(
                        q.mul(w, val)
                        for rest in range(1 << rem_vars)
                        for w, val in zip(
                            weights, current(prefix + q.boolean_point(rest, rem_vars))
                        )
                    )

                pt, rounds = [], []
                claim = q.total(q.mul(w, c) for w, c in zip(weights, claims_list))
                for r_idx in range(n_vars):
                    values = [at(pt + [q.lift(i)]) for i in range(degree + 1)]
                    if omit_idx is not None or failed_stage is None or failed_stage[0] != stage_no:
                        assert q.add(values[0], values[1]) == claim
                    values[1] = q.sub(claim, values[0])
                    msg = [values[0]] + values[2:]
                    t.absorb(f"stage/{stage_no}/round/{r_idx}", q.tce(msg))
                    ch = q.challenge(t, f"stage/{stage_no}/r/{r_idx}")
                    pt.append(ch)
                    claim = q.lagrange(values, ch)
                    rounds.append(msg)
                final = evals_fn(pt)
                residual = q.total(q.mul(w, v) for w, v in zip(weights, terms_fn(tuple(pt))))
                valid = claim == residual
                if omit_idx is not None:
                    assert claim == q.total(q.mul(w, v) for w, v in zip(weights, current(pt)))
                t.absorb(f"stage/{stage_no}/evals", q.tce(final))
                return dict(
                    gamma=gamma,
                    degree=degree,
                    rounds=rounds,
                    point=pt,
                    final_evals=final,
                    final_claim=claim,
                    final_relations=terms_fn(tuple(pt)),
                    valid=valid,
                    state=t.state.hex(),
                    counter=t.counter,
                )

            s1_omit = failed_stage[1] if (omit and failed_stage and failed_stage[0] == 1) else None
            stage1 = run_stage(1, rb, 3, stage1_claims, stage1_terms, stage1_evals, s1_omit)
            stage1["zero_row"] = list(zero_row)

            r1 = tuple(stage1["point"])
            z_r1 = stage1["final_evals"][-1]
            zero_vt = (
                tuple(q.challenge(t, f"stage/2/zero/{j}") for j in range(vb + rb))
                if bounded
                else ()
            )

            @functools.lru_cache(None)
            def stage2_terms(pt):
                v_pt, row_pt = pt[:vb], pt[vb:]
                cv = q.mle(valid_vocab_col, v_pt)
                ev = q.mle(e_col, pt)
                lv = q.mle(l_col, pt)
                hv = q.mle(h_col, pt)
                out = [
                    q.mul(q.eq_points(row_target, row_pt), q.mul(cv, ev)),
                    q.mul(q.eq_points(r1, row_pt), q.mul(cv, ev)),
                    q.mul(q.eq_points(row_target, row_pt), q.mul(ev, lv)),
                    q.mul(q.eq_points(row_target, row_pt), q.mul(hv, ev)),
                ]
                if bounded:
                    eq_vt = q.eq_points(zero_vt, pt)
                    eq_zrow = q.eq_points(zero_vt[vb:], row_pt)
                    out.extend(
                        [
                            q.mul(eq_vt, q.mul(hv, q.sub(q.ONE, hv))),
                            q.mul(eq_vt, q.mul(q.sub(q.ONE, cv), hv)),
                            q.mul(eq_zrow, hv),
                        ]
                    )
                return out

            stage2_claims = [z_eval, z_r1, b_eval, s_eval] + (
                [q.ZERO, q.ZERO, q.mle(psi_col, zero_vt[vb:])] if bounded else []
            )

            def stage2_evals(pt):
                ev = q.mle(e_col, pt)
                lv = q.mle(l_col, pt)
                hv = q.mle(h_col, pt)
                out = [ev, ev, ev, lv, hv, ev]
                if bounded:
                    out.extend([hv, hv, hv])
                return out

            s2_omit = failed_stage[1] if (omit and failed_stage and failed_stage[0] == 2) else None
            stage2 = run_stage(2, vb + rb, 3, stage2_claims, stage2_terms, stage2_evals, s2_omit)
            if bounded:
                stage2["zero_vt"] = list(zero_vt)

            r2 = tuple(stage2["point"])
            l_r2 = stage2["final_evals"][3]

            @functools.lru_cache(None)
            def stage3_terms(pt):
                v_pt, row_pt = pt[:vb], pt[vb:]
                kernel = q.total(
                    q.mul(q.eq_at(r2[:vb], i), q.eq_at(v_pt, w))
                    for i in range(V_pad)
                    for w in range(i + 1, V_pad)
                )
                hv = q.mle(h_col, pt)
                return [q.mul(q.eq_points(r2[vb:], row_pt), q.mul(kernel, hv))]

            s3_omit = failed_stage[1] if (omit and failed_stage and failed_stage[0] == 3) else None
            stage3 = run_stage(
                3,
                vb + rb,
                2,
                [l_r2],
                stage3_terms,
                lambda pt: [q.mle(h_col, pt)],
                s3_omit,
            )

            if local_reject is None:
                assert stage1["valid"] == (failed_stage is None or failed_stage[0] != 1)
                assert stage2["valid"] == (failed_stage is None or failed_stage[0] != 2)
                assert stage3["valid"] == (failed_stage is None or failed_stage[0] != 3)
            return dict(stage1=stage1, stage2=stage2, stage3=stage3)

        record = dict(
            name=name,
            gadget=200 + idx,
            rows=rows,
            vocab=vocab,
            sigma=sigma,
            bounded=bounded,
            w_bits=p["w_bits"],
            k=k,
            m=m,
            top_z_bits=p["top_z_bits"],
            range_widths=widths,
            lower_bounds=p["lower_bounds"],
            lower_carry_widths=lower_cw,
            upper_bounds=p["upper_bounds"],
            upper_carry_widths=upper_cw,
            exps=exps,
            draws=draws,
            tokens=tokens,
            h_hat=h_hat,
            suffix=suffix,
            actual_suffix=actual_suffix,
            row_records=[
                dict(
                    psi=r["psi"],
                    u=r["u"],
                    z=r["z"],
                    b=r["b"],
                    s=r["s"],
                    p_coeffs=r["p_coeffs"],
                    committed=r["committed"],
                    range_keys=r["range_keys"],
                )
                for r in row_records
            ],
            row_target=list(row_target),
            u_eval=u_eval,
            z_eval=z_eval,
            b_eval=b_eval,
            s_eval=s_eval,
            range_key_evals=range_key_evals,
            p_targets=p_targets,
            context=context,
            local_reject=local_reject,
            failed_stage=list(failed_stage) if failed_stage else None,
            proof=prove_chain(False),
        )
        if failed_stage is not None:
            record["omission"] = prove_chain(True)
        component_cases.append(record)

    return dict(
        schema_version=1,
        description="RFC-0009 Lemma 10 and RFC-0058 InverseCdf: whole-integer witnesses, bounds and three-stage cube-sum component proofs",
        schema_cases=schemas,
        cases=component_cases,
    )
