# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0017 VocabMax, OneSidedLog2 and runtime table interning: independent Python oracle.

Uses arbitrary-precision Python integers, exact fixed-point table generators from ../tables,
and literal Boolean cube sums. Zero floating-point operations and zero Rust imports.
"""
import functools
import importlib.util
import math
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def c_match_exact(k):
    """RFC-0017 item 2: ceil(-log2(1 - 2^{-k}) * 256) by pure integer powers, 0 for k = 0."""
    assert 0 <= k <= 16
    if k == 0:
        return 0
    lhs = 1 << (256 * k)
    base = ((1 << k) - 1) ** 256
    c = 0
    while base * (1 << c) < lhs:
        c += 1
    assert c >= 1 and base * (1 << (c - 1)) < lhs <= base * (1 << c)
    return c


def vocab_exp_scale_exact(logits_exponent, temp_num, temp_den):
    """RFC-0017 'D's decoded token' and RFC-0027: exact reduced Rational for 2^{logits_exponent} / T."""
    assert temp_num > 0 and temp_den > 0 and temp_num <= 4 * temp_den
    g0 = math.gcd(temp_num, temp_den)
    temp_num //= g0
    temp_den //= g0
    if logits_exponent >= 0:
        num = temp_den << logits_exponent
        den = temp_num
    else:
        num = temp_den
        den = temp_num << (-logits_exponent)
    g = math.gcd(num, den)
    num //= g
    den //= g
    assert 0 < num < (1 << 63) and 0 < den < (1 << 64)
    return [num, den]


def log2_params(vocab, sigma, eps_log2):
    """RFC-0017 'In-circuit scoring (mixture)' items 4-5."""
    assert 1 <= vocab <= (1 << 18) and 0 <= sigma <= 32 and 0 <= eps_log2 <= 16
    w_z = (vocab * (1 << sigma)).bit_length()
    k_z = (w_z + 15) // 16
    top_z_bits = w_z - 16 * (k_z - 1)
    w_q = (((1 << sigma) + 1) * ((1 << 17) - 1)).bit_length()
    k_q = (w_q + 15) // 16
    top_q_bits = w_q - 16 * (k_q - 1)
    m_lhs = (((1 << w_z) - 1 + vocab) << 16).bit_length()
    m_rhs = (((1 << w_q) - 1) * ((1 << 63) - 1)).bit_length()
    m = (max(m_lhs, m_rhs) + 15) // 16
    assert m == (w_q + 63 + 15) // 16
    box_z = [65535] * (k_z - 1) + [(1 << top_z_bits) - 1]
    box_q = [65535] * (k_q - 1) + [(1 << top_q_bits) - 1]
    box_p = [65535, 65535, 65535, 32767]
    v_limbs = [vocab & 0xFFFF, vocab >> 16]
    a_left = [0] + [
        (box_z[j - 1] if j - 1 < k_z else 0) + (v_limbs[j - 1] if j - 1 < 2 else 0)
        for j in range(1, m)
    ]
    a_right = [
        sum(box_q[a] * box_p[b] for a in range(k_q) for b in range(4) if a + b == j)
        for j in range(m)
    ]
    return dict(
        vocab=vocab,
        sigma=sigma,
        eps_log2=eps_log2,
        c_match=c_match_exact(eps_log2),
        w_z=w_z,
        k_z=k_z,
        top_z_bits=top_z_bits,
        w_q=w_q,
        k_q=k_q,
        top_q_bits=top_q_bits,
        m=m,
        v_limbs=v_limbs,
        bounds=[a_left, a_right],
    )


def log2_lookup_specs(oracle, params):
    k_z, top_z = params["k_z"], params["top_z_bits"]
    k_q, top_q = params["k_q"], params["top_q_bits"]
    m = params["m"]
    lc_w = oracle.carry_widths(params["bounds"][0], 16)
    rc_w = oracle.carry_widths(params["bounds"][1], 16)
    specs = [
        {"Pow2": {"max_exp": 63}},
        {"Exp2Frac": {"frac_bits": 8, "out_bits": 16}},
    ]
    for bits in [16] * (k_z - 1) + [top_z]:
        specs.append({"Range": {"bits": bits}})
    for bits in [16] * (k_q - 1) + [top_q]:
        specs.append({"Range": {"bits": bits}})
    for bits in [16, 16, 16, 15]:
        specs.append({"Range": {"bits": bits}})
    for bits in [16] * m + lc_w + [16] * m + rc_w + [16] * m + [1] * m:
        specs.append({"Range": {"bits": bits}})
    assert len(specs) == 2 + k_z + k_q + 4 + 6 * m
    return specs, [lc_w, rc_w]


def tight_mismatch_if(e2_table, vocab, z_val, y_val):
    """Least (I_t, f_t) in [0..62] x [0..255] with (z_val + vocab)*2^16 <= (y_val + 1)*E2[f_t]*2^{I_t}."""
    lhs = (z_val + vocab) << 16
    num_base = y_val + 1
    # Ratio needed: E2[f] * 2^I >= ceil(lhs / num_base).
    needed = (lhs + num_base - 1) // num_base
    # Since E2[0] = 2^16 and E2[255] < 2^17, I_t is around bit_length - 17.
    start_i = max(0, needed.bit_length() - 18)
    for i_exp in range(start_i, 63):
        shifted = num_base << i_exp
        if shifted * e2_table[255] < lhs:
            continue
        lo, hi = 0, 255
        while lo < hi:
            mid = (lo + hi) // 2
            if shifted * e2_table[mid] >= lhs:
                hi = mid
            else:
                lo = mid + 1
        assert shifted * e2_table[lo] >= lhs
        if lo > 0:
            assert shifted * e2_table[lo - 1] < lhs
        elif i_exp > 0:
            assert (num_base << (i_exp - 1)) * e2_table[255] < lhs
        return i_exp, lo
    raise AssertionError("unreachable: max I_t=62 covers (2^50 + 2^18)*2^16 <= 1 * 2^16 * 2^62")


def check_gibbs_bound(vocab, eps_log2, z_val, y_val, matched, b_fixed):
    """Exact big-integer check that b_t = b_fixed / 256 >= -log2 q_t(y_t), i.e. den^256 <= num^256 * 2^{b_fixed}."""
    den = (1 << eps_log2) * (z_val + vocab)
    num = ((1 << eps_log2) - 1) * int(matched) * (z_val + vocab) + (y_val + 1)
    assert 0 < num <= den
    assert (den ** 256) <= (num ** 256) * (1 << b_fixed)


def log2_row_witness(
    oracle,
    e2_table,
    params,
    psi,
    z_val,
    y_val,
    m_bit,
    i_exp,
    f_frac,
    q_override=None,
    alternative=False,
):
    vocab = params["vocab"]
    k_z, k_q, m = params["k_z"], params["k_q"], params["m"]
    v_limbs = params["v_limbs"]
    assert 0 <= i_exp < 63 and 0 <= f_frac < 256
    p_val = 1 << i_exp
    e_val = e2_table[f_frac]
    q_val = (y_val + psi) * e_val if q_override is None else q_override
    z_limbs = [(z_val >> (16 * h)) & 0xFFFF for h in range(k_z)]
    q_limbs = [(q_val >> (16 * a)) & 0xFFFF for a in range(k_q)]
    p_limbs = [(p_val >> (16 * b)) & 0xFFFF for b in range(4)]
    x_coeffs = [0] + [
        psi * ((z_limbs[j - 1] if j - 1 < k_z else 0) + (v_limbs[j - 1] if j - 1 < 2 else 0))
        for j in range(1, m)
    ]
    y_coeffs = [
        sum(q_limbs[a] * p_limbs[b] for a in range(k_q) for b in range(4) if a + b == j)
        for j in range(m)
    ]
    cmp_arrays = oracle.witness(x_coeffs, y_coeffs, 16, alternative=alternative)
    committed = (
        [m_bit, i_exp, p_val, f_frac, e_val, q_val]
        + z_limbs[: k_z - 1]
        + q_limbs[: k_q - 1]
        + p_limbs[:3]
        + [x for arr in cmp_arrays for x in arr]
    )
    lookup_keys = (
        [i_exp, f_frac]
        + z_limbs
        + q_limbs
        + p_limbs
        + [x for arr in cmp_arrays for x in arr]
    )
    lookup_values = [p_val, e_val] + [0] * (k_z + k_q + 4 + 6 * m)
    eps_k = params["eps_log2"]
    b_t = psi * (
        m_bit * params["c_match"] + (1 - m_bit) * (256 * eps_k + 256 * i_exp + f_frac)
    )
    assert len(committed) == 6 + (k_z - 1) + (k_q - 1) + 3 + 6 * m
    assert len(lookup_keys) == 2 + k_z + k_q + 4 + 6 * m
    return dict(
        psi=psi,
        z=z_val,
        y=y_val,
        m_bit=m_bit,
        i_exp=i_exp,
        p_val=p_val,
        f_frac=f_frac,
        e_val=e_val,
        q_val=q_val,
        b_t=b_t,
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        committed=committed,
        lookup_keys=lookup_keys,
        lookup_values=lookup_values,
    )


def vocab_max_cases(tables_mod, q, rng):
    """VocabMax: Exp_T scale/width validation, forest root 1 - psi, masked key/leaf, and padded maximum."""
    configs = [
        ("scalar", 1, 1, 0, [1, 1], 3, -7, 7, 32, None),
        ("padded", 3, 5, -2, [3, 4], 8, -31, 31, 32, None),
        ("wide-sigma24", 3, 6, 1, [2, 1], 24, -127, 127, 32, None),
        ("masked-lookup", 3, 5, -2, [3, 4], 8, -31, 31, 32, 0),
        ("masked-tree", 3, 5, -2, [3, 4], 8, -31, 31, 32, 1),
        ("padded-maximum", 3, 5, -2, [3, 4], 8, -31, 31, 32, 2),
    ]
    out = []
    for idx, (name, rows, vocab, logits_exp, temp, sigma, lo, hi, max_log2, failed) in enumerate(
        configs
    ):
        scale = vocab_exp_scale_exact(logits_exp, temp[0], temp[1])
        spread = hi - lo
        min_kb = max(1, spread.bit_length())
        chosen_kb = None
        for kb in range(min_kb, max_log2 + 1):
            kind = {"Exp": {"key_bits": kb, "sigma": sigma, "scale": scale}}
            cnt, fn, _, _ = tables_mod.table(kind)
            if fn(cnt - 1) == 0:
                chosen_kb = kb
                exp_fn = fn
                k0 = cnt - 1
                break
        assert chosen_kb is not None
        vb, rb = (vocab - 1).bit_length(), (rows - 1).bit_length()
        V_pad, T_pad = 1 << vb, 1 << rb
        psi = [int(t < rows) for t in range(T_pad)]
        selector = [int(t < rows and v < vocab) for t in range(T_pad) for v in range(V_pad)]
        logits = [
            (lo + ((7 * t + 11 * v) % (spread + 1))) if (t < rows and v < vocab) else 0
            for t in range(T_pad)
            for v in range(V_pad)
        ]
        maxima = [
            max(logits[t * V_pad : t * V_pad + vocab]) if t < rows else 0
            for t in range(T_pad)
        ]
        raw = [maxima[t] - logits[t * V_pad + v] for t in range(T_pad) for v in range(V_pad)]
        keys = [d if c else k0 for d, c in zip(raw, selector)]
        leaves = [d if c else 1 for d, c in zip(raw, selector)]
        exps = [exp_fn(k) for k in keys]

        if failed == 0:
            # Masked vocab padding entry (t=0, v=vocab) uses k0 - 1, which ALSO maps to 0 in Exp_T!
            alt_k = k0 - 1
            assert exp_fn(alt_k) == 0
            keys[vocab] = alt_k
            exps[vocab] = exp_fn(alt_k)
        elif failed == 1:
            # Masked vocab padding entry (t=0, v=vocab) uses leaf 2 instead of 1 (row product stays 0).
            leaves[vocab] = 2
        elif failed == 2:
            # Padded row `rows` sets maxima[rows] = 1 while keeping masked keys k0 and leaves 1.
            maxima[rows] = 1

        # Forest root check: product over v < V_pad of leaves[t, v] equals 1 - psi[t] on every row!
        for t in range(T_pad):
            prod = 1
            for v in range(V_pad):
                prod = (prod * leaves[t * V_pad + v]) % q.P
            assert prod == (1 - psi[t])

        lookup_target = tuple(rng.fp2() for _ in range(vb + rb))
        tree_target = tuple(rng.fp2() for _ in range(vb + rb))
        forest_root_target = tuple(rng.fp2() for _ in range(rb))
        key_claim = q.mle([q.lift(x) for x in keys], lookup_target)
        val_claim = q.mle([q.lift(x) for x in exps], lookup_target)
        leaf_claim = q.mle([q.lift(x) for x in leaves], tree_target)
        forest_root_claim = q.mle([q.lift(1 - x) for x in psi], forest_root_target)
        c_lookup = q.mle([q.lift(x) for x in selector], lookup_target)
        c_tree = q.mle([q.lift(x) for x in selector], tree_target)
        claims = [
            q.sub(key_claim, q.mul(q.lift(k0), q.sub(q.ONE, c_lookup))),
            q.sub(leaf_claim, q.sub(q.ONE, c_tree)),
            q.ZERO,
        ]
        scales = [1, 1, V_pad]
        context = rng.context()

        def before():
            tr = q.transcript.Transcript(q.transcript.session_context(context))
            tr.absorb("stage/0/split", q.tce([key_claim, val_claim, leaf_claim]))
            zr = tuple(q.challenge(tr, f"stage/1/zero/{j}") for j in range(rb))
            return tr, zr

        _, zero_row = before()

        @functools.lru_cache(None)
        def terms(pt):
            row_pt = pt[vb:]
            c_at = q.mle([q.lift(x) for x in maxima], row_pt)
            ell_at = q.mle([q.lift(x) for x in logits], pt)
            sel_at = q.mle([q.lift(x) for x in selector], pt)
            psi_at = q.mle([q.lift(x) for x in psi], row_pt)
            diff = q.mul(sel_at, q.sub(c_at, ell_at))
            return [
                q.mul(q.eq_points(lookup_target, pt), diff),
                q.mul(q.eq_points(tree_target, pt), diff),
                q.mul(q.eq_points(zero_row, row_pt), q.mul(q.sub(q.ONE, psi_at), c_at)),
            ]

        n_vars = vb + rb
        actual = [q.ZERO] * 3
        for i in range(1 << n_vars):
            actual = [
                q.add(a, b) for a, b in zip(actual, terms(tuple(q.boolean_point(i, n_vars))))
            ]
        assert [
            i
            for i, (a, cl, sc) in enumerate(zip(actual, claims, scales))
            if a != q.mul(cl, q.lift(sc))
        ] == ([] if failed is None else [failed])

        def prove_vmax(omit=False):
            tr, _ = before()
            gamma = q.challenge(tr, "stage/1/gamma")
            weights = [q.power(gamma, j) for j in range(3)]

            def current(pt):
                vals = list(terms(tuple(pt)))
                if omit:
                    dom = pt if failed < 2 else pt[vb:]
                    delta0 = q.mle([q.ONE] + [q.ZERO] * ((1 << len(dom)) - 1), dom)
                    vals[failed] = q.mul(claims[failed], delta0)
                return vals

            def at(prefix):
                rem = n_vars - len(prefix)
                return q.total(
                    q.mul(w, v)
                    for rest in range(1 << rem)
                    for w, v in zip(weights, current(prefix + q.boolean_point(rest, rem)))
                )

            pt, rounds = [], []
            claim = q.total(
                q.mul(w, q.mul(cl, q.lift(sc)))
                for w, cl, sc in zip(weights, claims, scales)
            )
            for j in range(n_vars):
                values = [at(pt + [q.lift(i)]) for i in range(4)]
                if omit or failed is None:
                    assert q.add(values[0], values[1]) == claim
                values[1] = q.sub(claim, values[0])
                msg = [values[0], values[2], values[3]]
                tr.absorb(f"stage/1/round/{j}", q.tce(msg))
                ch = q.challenge(tr, f"stage/1/r/{j}")
                pt.append(ch)
                claim = q.lagrange(values, ch)
                rounds.append(msg)
            row_pt = pt[vb:]
            ce = q.mle([q.lift(x) for x in maxima], row_pt)
            le = q.mle([q.lift(x) for x in logits], pt)
            evals = [ce, le, ce, le, ce]
            valid = claim == q.total(q.mul(w, v) for w, v in zip(weights, terms(tuple(pt))))
            assert valid == (failed is None)
            if omit:
                assert claim == q.total(q.mul(w, v) for w, v in zip(weights, current(pt)))
            tr.absorb("stage/1/evals", q.tce(evals))
            return dict(
                zero_row=list(zero_row),
                gamma=gamma,
                degree=3,
                rounds=rounds,
                point=pt,
                final_evals=evals,
                final_claim=claim,
                final_relations=terms(tuple(pt)),
                valid=valid,
                state=tr.state.hex(),
                counter=tr.counter,
            )

        rec = dict(
            name=name,
            gadget=300 + idx,
            rows=rows,
            vocab=vocab,
            logits_exponent=logits_exp,
            temperature=temp,
            scale=scale,
            sigma=sigma,
            logit_lo=lo,
            logit_hi=hi,
            min_key_bits=min_kb,
            max_log2_rows=max_log2,
            key_bits=chosen_kb,
            k0=k0,
            logits=logits,
            maxima=maxima,
            keys=keys,
            leaves=leaves,
            exps=exps,
            psi=psi,
            selector=selector,
            lookup_target=list(lookup_target),
            tree_target=list(tree_target),
            forest_root_target=list(forest_root_target),
            lookup_key=key_claim,
            lookup_value=val_claim,
            tree_leaf=leaf_claim,
            forest_root_claim=forest_root_claim,
            claims=claims,
            context=context,
            failed=failed,
            proof=prove_vmax(False),
        )
        if failed is not None:
            rec["omission"] = prove_vmax(True)
        out.append(rec)
    return out


def table_interning_cases(oracle, tables_mod, icdf_mod):
    """RFC-0017 item 6: deterministic runtime TableId interning and deduplication."""
    out = []

    def derive_interning(
        base_tables,
        vocab,
        sigma,
        logits_exp,
        logit_lo,
        logit_hi,
        max_log2,
        mode,
        eps_log2,
        requests,
    ):
        catalog = list(base_tables)

        def intern(kind):
            for idx, existing in enumerate(catalog):
                if existing == kind:
                    return idx
            catalog.append(kind)
            return len(catalog) - 1

        min_kb = max(1, (logit_hi - logit_lo).bit_length())
        # Step 1: distinct effective temperatures in first-seen output-bearing request order.
        seen_temps = []
        exp_ids = []
        for req in requests:
            if req["outputs"] == 0:
                continue
            if req["decoding"] == "Greedy":
                if mode == "Exact":
                    continue
                eff_temp = [1, 1]
            else:
                eff_temp = req["temperature"]
            if eff_temp in seen_temps:
                continue
            seen_temps.append(eff_temp)
            scale = vocab_exp_scale_exact(logits_exp, eff_temp[0], eff_temp[1])
            chosen = None
            for kb in range(min_kb, max_log2 + 1):
                kind = {"Exp": {"key_bits": kb, "sigma": sigma, "scale": scale}}
                cnt, fn, _, _ = tables_mod.table(kind)
                if fn(cnt - 1) == 0:
                    chosen = kind
                    break
            assert chosen is not None
            exp_ids.append(dict(temperature=eff_temp, table_id=intern(chosen), kind=chosen))
        assert len(seen_temps) <= 4

        # Step 2: per output-bearing request in batch order: decoding tables, then OneSidedLog2.
        req_tables = []
        cdf_p = icdf_mod.cdf_params(vocab, sigma, mode == "Mixture")
        cdf_widths, _, _ = icdf_mod.range_widths(oracle, cdf_p)
        log2_specs = (
            log2_lookup_specs(oracle, log2_params(vocab, sigma, eps_log2))[0]
            if mode == "Mixture"
            else []
        )
        for req in requests:
            if req["outputs"] == 0:
                req_tables.append(dict(decoding_ids=[], log2_ids=[]))
                continue
            if req["decoding"] == "Greedy":
                dec_ids = [intern({"Range": {"bits": min_kb}})]
            else:
                dec_ids = [intern({"Range": {"bits": w}}) for w in cdf_widths]
            l2_ids = [intern(spec) for spec in log2_specs]
            req_tables.append(dict(decoding_ids=dec_ids, log2_ids=l2_ids))
        return dict(catalog=catalog, exp_tables=exp_ids, request_tables=req_tables)

    scenarios = [
        (
            "single-greedy-mixture",
            [{"Range": {"bits": 16}}, {"Range": {"bits": 8}}],
            5,
            8,
            -2,
            -31,
            31,
            32,
            "Mixture",
            6,
            [{"outputs": 3, "decoding": "Greedy"}],
        ),
        (
            "multi-temp-and-greedy-mixture-with-empty",
            [{"Range": {"bits": 16}}, {"Pow2": {"max_exp": 63}}],
            128,
            16,
            -3,
            -255,
            255,
            32,
            "Mixture",
            6,
            [
                {"outputs": 2, "decoding": "Greedy"},
                {"outputs": 0, "decoding": "Temperature", "temperature": [1, 2]},
                {"outputs": 4, "decoding": "Temperature", "temperature": [3, 4]},
                {"outputs": 1, "decoding": "Temperature", "temperature": [1, 1]},
                {"outputs": 2, "decoding": "Temperature", "temperature": [2, 1]},
            ],
        ),
        (
            "exact-temperature-two-temps",
            [{"Range": {"bits": 16}}],
            64,
            12,
            0,
            -63,
            63,
            32,
            "Exact",
            0,
            [
                {"outputs": 3, "decoding": "Temperature", "temperature": [1, 1]},
                {"outputs": 2, "decoding": "Temperature", "temperature": [5, 4]},
            ],
        ),
    ]
    for (
        name,
        base,
        vocab,
        sigma,
        logits_exp,
        lo,
        hi,
        max_log2,
        mode,
        eps_log2,
        reqs,
    ) in scenarios:
        res = derive_interning(
            base, vocab, sigma, logits_exp, lo, hi, max_log2, mode, eps_log2, reqs
        )
        out.append(
            dict(
                name=name,
                base_tables=base,
                vocab=vocab,
                sigma=sigma,
                logits_exponent=logits_exp,
                logit_lo=lo,
                logit_hi=hi,
                max_log2_rows=max_log2,
                mode=mode,
                eps_log2=eps_log2,
                requests=reqs,
                **res,
            )
        )
    return out


def schema_cases(oracle, e2_table, q, rng):
    configs = [
        ("v1-s0-k0", 1, 0, 0),
        ("v1-s0-k1", 1, 0, 1),
        ("v5-s3-k6", 5, 3, 6),
        ("v128k-s16-k6", 128256, 16, 6),
        ("v128k-s32-k16", 128256, 32, 16),
        ("vmax-s32-k0", 1 << 18, 32, 0),
        ("vmax-s32-k6", 1 << 18, 32, 6),
    ]
    out = []
    for name, vocab, sigma, eps_k in configs:
        p = log2_params(vocab, sigma, eps_k)
        specs, cw = log2_lookup_specs(oracle, p)
        max_e = 1 << sigma
        max_z = vocab * max_e
        # Four rows: match (when eps_k > 0 else mismatch), interior mismatch, zero-tail e=0 mismatch, padded row.
        rows = []
        if eps_k > 0:
            i0, f0 = tight_mismatch_if(e2_table, vocab, max_z, max_e)
            rows.append(log2_row_witness(oracle, e2_table, p, 1, max_z, max_e, 1, i0, f0))
            check_gibbs_bound(vocab, eps_k, max_z, max_e, True, rows[-1]["b_t"])
        else:
            i0, f0 = tight_mismatch_if(e2_table, vocab, max_z, max_e)
            rows.append(log2_row_witness(oracle, e2_table, p, 1, max_z, max_e, 0, i0, f0))
            check_gibbs_bound(vocab, eps_k, max_z, max_e, True, rows[-1]["b_t"])
        z1, y1 = max(1, max_z // 2), max(0, max_e // 2)
        i1, f1 = tight_mismatch_if(e2_table, vocab, z1, y1)
        rows.append(log2_row_witness(oracle, e2_table, p, 1, z1, y1, 0, i1, f1))
        check_gibbs_bound(vocab, eps_k, z1, y1, False, rows[-1]["b_t"])
        # Zero-tail mismatch y=0 at max_z (tests large I_t up to 50).
        i2, f2 = tight_mismatch_if(e2_table, vocab, max_z, 0)
        rows.append(log2_row_witness(oracle, e2_table, p, 1, max_z, 0, 0, i2, f2))
        check_gibbs_bound(vocab, eps_k, max_z, 0, False, rows[-1]["b_t"])
        # Padded row: psi=0, Z=0, Y=0, m=0, I=0 (P=1), f=0 (E=65536), q=0, B_t=0.
        rows.append(log2_row_witness(oracle, e2_table, p, 0, 0, 0, 0, 0, 0))
        assert rows[-1]["b_t"] == 0 and rows[-1]["p_val"] == 1 and rows[-1]["e_val"] == 65536

        point = [rng.fp2(), rng.fp2()]
        out.append(
            dict(
                name=name,
                vocab=vocab,
                sigma=sigma,
                eps_log2=eps_k,
                c_match=p["c_match"],
                w_z=p["w_z"],
                k_z=p["k_z"],
                top_z_bits=p["top_z_bits"],
                w_q=p["w_q"],
                k_q=p["k_q"],
                top_q_bits=p["top_q_bits"],
                m=p["m"],
                committed_count=6 + (p["k_z"] - 1) + (p["k_q"] - 1) + 3 + 6 * p["m"],
                lookup_specs=specs,
                bounds=p["bounds"],
                carry_widths=cw,
                point=point,
                z_eval=q.mle([q.lift(r["z"]) for r in rows], point),
                q_eval=q.mle([q.lift(r["q_val"]) for r in rows], point),
                lookup_key_evals=[
                    q.mle([q.lift(r["lookup_keys"][j]) for r in rows], point)
                    for j in range(len(specs))
                ],
                lookup_value_evals=[
                    q.mle([q.lift(r["lookup_values"][j]) for r in rows], point)
                    for j in range(len(specs))
                ],
                rows=[
                    dict(
                        psi=r["psi"],
                        z=r["z"],
                        y=r["y"],
                        m_bit=r["m_bit"],
                        i_exp=r["i_exp"],
                        p_val=r["p_val"],
                        f_frac=r["f_frac"],
                        e_val=r["e_val"],
                        q_val=r["q_val"],
                        b_t=r["b_t"],
                        committed=r["committed"],
                        lookup_keys=r["lookup_keys"],
                        lookup_values=r["lookup_values"],
                    )
                    for r in rows
                ],
            )
        )
    return out


def cases(oracle=None):
    here = Path(__file__).resolve().parent
    if oracle is None:
        oracle = load(here / "generate.py", "log2_base_oracle")
    tables_mod = load(here.parent / "tables" / "generate.py", "log2_tables_oracle")
    icdf_mod = load(here / "inverse_cdf.py", "log2_icdf_oracle")
    q = load(here.parent / "sumcheck" / "generate.py", "log2_sumcheck_oracle")
    q.hashing.check_official_vectors()
    rng = q.Inputs(0x16000170058)

    _, e2_fn, _, _ = tables_mod.table({"Exp2Frac": {"frac_bits": 8, "out_bits": 16}})
    e2_table = [e2_fn(f) for f in range(256)]
    assert e2_table[0] == 65536 and all(65536 <= v < 131072 for v in e2_table)

    c_match_vec = [c_match_exact(k) for k in range(17)]
    assert c_match_vec == [0, 256, 107, 50, 24, 12, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]

    vmax = vocab_max_cases(tables_mod, q, rng)
    interning = table_interning_cases(oracle, tables_mod, icdf_mod)
    schemas = schema_cases(oracle, e2_table, q, rng)

    honest_specs = [
        ("scalar-match", 1, 1, 3, 6),
        ("greedy-k0-surprisal", 3, 5, 8, 0),
        ("mixed-match-and-mismatch", 3, 5, 16, 6),
        ("wide-sigma32", 5, 6, 32, 16),
        ("one-sided-slack", 2, 4, 8, 4),
    ]
    alt_specs = [
        ("false-match-on-mismatch", 3, 5, 8, 6),
        ("match-when-k-zero", 3, 5, 8, 0),
        ("nonboolean-match", 3, 5, 8, 6),
        ("deflated-exponent", 3, 5, 8, 6),
        ("deflated-fraction", 3, 5, 8, 6),
        ("inflated-qt", 3, 5, 8, 6),
        ("padded-z-nonzero", 3, 5, 8, 6),
        ("underclaimed-uhat", 3, 5, 8, 6),
        ("bad-q-decomposition", 3, 5, 8, 6),
        ("bad-p-decomposition", 3, 5, 8, 6),
    ]

    inv_pow16 = [pow(1 << (16 * j), -1, q.P) for j in range(5)]
    component_cases = []

    for idx, (name, rows, vocab, sigma, eps_k) in enumerate(honest_specs + alt_specs):
        bad = name if idx >= len(honest_specs) else None
        p = log2_params(vocab, sigma, eps_k)
        specs, cw = log2_lookup_specs(oracle, p)
        k_z, k_q, m = p["k_z"], p["k_q"], p["m"]
        v_limbs = p["v_limbs"]
        vb, rb = (vocab - 1).bit_length(), (rows - 1).bit_length()
        V_pad, T_pad = 1 << vb, 1 << rb
        max_e = 1 << sigma

        exps = []
        y_tokens = []
        yhat_tokens = []
        for t in range(T_pad):
            real = t < rows
            if not real:
                row_e = [0] * vocab
                yt, yht = 0, 0
            elif vocab == 1:
                row_e = [max_e]
                yt, yht = 0, 0
            else:
                row_e = [
                    0
                    if (t == 2 and v == 3)
                    else max(1, ((((v + 3 * t) % vocab) + 1) * max_e) // (vocab + 1))
                    for v in range(vocab)
                ]
                row_e[t % vocab] = max_e
                yht = t % vocab
                if t == 0:
                    yt = yht  # match on row 0
                elif t == 2 and vocab >= 4:
                    yt = 3  # zero-tail mismatch on row 2
                else:
                    yt = (yht + 1) % vocab  # positive-weight mismatch
            exps.extend(row_e + [0] * (V_pad - vocab))
            y_tokens.append(yt)
            yhat_tokens.append(yht)

        if bad == "false-match-on-mismatch":
            y_tokens[0] = (yhat_tokens[0] + 1) % vocab
        elif bad in ("deflated-exponent", "deflated-fraction", "inflated-qt"):
            y_tokens[0] = (yhat_tokens[0] + 1) % vocab
        elif bad == "padded-z-nonzero":
            exps[rows * V_pad] = 1

        h_y = [int(t < rows and v == y_tokens[t]) for t in range(T_pad) for v in range(V_pad)]
        h_yhat = [
            int(t < rows and v == yhat_tokens[t]) for t in range(T_pad) for v in range(V_pad)
        ]

        local_reject = None
        failed_stage = None
        y_active = min(m, k_q + 3)
        # Stage 1 instance indices:
        # 0: match booleanity
        # 1: q_t - (Y_t + psi_t) * E_t = 0
        # 2 .. 2 + k_z: X_1 .. X_{k_z}
        # 2 + k_z .. 2 + k_z + y_active: Y_0 .. Y_{y_active - 1}
        # 2 + k_z + y_active: (psi_t - 1) * Z_t = 0
        # 2 + k_z + y_active + 1: sum_t B_t = U_hat_r
        idx_pad_z = 2 + k_z + y_active
        idx_uhat = idx_pad_z + 1

        row_records = []
        for t in range(T_pad):
            psi = int(t < rows)
            z_val = sum(exps[t * V_pad + v] for v in range(vocab))
            y_val = sum(exps[t * V_pad + v] * h_y[t * V_pad + v] for v in range(V_pad))
            if not psi:
                row_records.append(log2_row_witness(oracle, e2_table, p, 0, z_val, 0, 0, 0, 0))
                continue
            matched = y_tokens[t] == yhat_tokens[t]
            i_exp, f_frac = tight_mismatch_if(e2_table, vocab, z_val, y_val)
            m_bit = 1 if (matched and eps_k > 0) else 0
            q_override = None
            alt_cmp = False

            if name == "one-sided-slack":
                if t == 0:
                    m_bit = 0  # claim mismatch on a matching token -> raises U_hat
                elif t == 1:
                    i_exp = min(62, i_exp + 1)  # looser exponent -> raises U_hat
            elif bad == "false-match-on-mismatch" and t == 0:
                m_bit = 1
                failed_stage = (2, 3)
            elif bad == "match-when-k-zero" and t == 0:
                m_bit = 1
                failed_stage = (1, 0)
            elif bad == "nonboolean-match" and t == 0:
                m_bit = 2
                failed_stage = (1, 0)
            elif bad == "deflated-exponent" and t == 0:
                assert i_exp >= 1
                i_exp -= 1
                alt_cmp = True
                local_reject = "compare_borrow"
            elif bad == "deflated-fraction" and t == 0:
                if f_frac == 0:
                    assert i_exp >= 1
                    i_exp -= 1
                    f_frac = 255
                else:
                    f_frac -= 1
                alt_cmp = True
                local_reject = "compare_borrow"
            elif bad == "inflated-qt" and t == 0:
                assert i_exp >= 1
                i_exp -= 1
                # Inflate q_0 so the smaller exponent passes BigIntCompare!
                needed_q = (((z_val + vocab) << 16) + (1 << i_exp) - 1) >> i_exp
                q_override = max(needed_q, (y_val + 1) * e2_table[f_frac] + 1)
                failed_stage = (1, 1)

            rec = log2_row_witness(
                oracle,
                e2_table,
                p,
                psi,
                z_val,
                y_val,
                m_bit,
                i_exp,
                f_frac,
                q_override=q_override,
                alternative=alt_cmp,
            )
            if bad is None:
                check_gibbs_bound(vocab, eps_k, z_val, y_val, matched, rec["b_t"])
            row_records.append(rec)

        if bad == "padded-z-nonzero":
            failed_stage = (1, idx_pad_z)
        elif bad == "underclaimed-uhat":
            failed_stage = (1, idx_uhat)
        elif bad == "bad-q-decomposition":
            local_reject = "q_decomposition"
        elif bad == "bad-p-decomposition":
            local_reject = "p_decomposition"

        actual_uhat = sum(r["b_t"] for r in row_records)
        claimed_uhat = actual_uhat - 1 if bad == "underclaimed-uhat" else actual_uhat
        assert claimed_uhat >= 0

        row_target = tuple(rng.fp2() for _ in range(rb))
        z_col = [q.lift(r["z"]) for r in row_records]
        y_col = [q.lift(r["y"]) for r in row_records]
        m_col = [q.lift(r["m_bit"]) for r in row_records]
        i_col = [q.lift(r["i_exp"]) for r in row_records]
        p_col = [q.lift(r["p_val"]) for r in row_records]
        f_col = [q.lift(r["f_frac"]) for r in row_records]
        e_val_col = [q.lift(r["e_val"]) for r in row_records]
        q_col = [q.lift(r["q_val"]) for r in row_records]
        psi_col = [q.lift(r["psi"]) for r in row_records]
        z_low_cols = [
            [q.lift(r["lookup_keys"][2 + h]) for r in row_records] for h in range(k_z - 1)
        ]
        q_low_cols = [
            [q.lift(r["lookup_keys"][2 + k_z + a]) for r in row_records]
            for a in range(k_q - 1)
        ]
        p_low_cols = [
            [q.lift(r["lookup_keys"][2 + k_z + k_q + b]) for r in row_records]
            for b in range(3)
        ]
        e_vocab_col = [q.lift(v) for v in exps]
        hy_col = [q.lift(v) for v in h_y]
        hyhat_col = [q.lift(v) for v in h_yhat]
        valid_vocab_col = [q.lift(int(v < vocab)) for v in range(V_pad)]

        z_eval = q.mle(z_col, row_target)
        q_eval = q.mle(q_col, row_target)
        lookup_key_evals = [
            q.mle([q.lift(r["lookup_keys"][j]) for r in row_records], row_target)
            for j in range(len(specs))
        ]
        lookup_value_evals = [
            q.mle([q.lift(r["lookup_values"][j]) for r in row_records], row_target)
            for j in range(len(specs))
        ]
        if bad == "bad-q-decomposition":
            # Perturb virtual top limb Q_{k_q - 1}(row_target) while keeping individual keys in range.
            lookup_key_evals[2 + k_z + k_q - 1] = q.add(
                lookup_key_evals[2 + k_z + k_q - 1], q.ONE
            )
        elif bad == "bad-p-decomposition":
            # Perturb virtual top limb P_3(row_target).
            lookup_key_evals[2 + k_z + k_q + 3] = q.add(
                lookup_key_evals[2 + k_z + k_q + 3], q.ONE
            )

        x_targets = [
            q.mle([q.lift(r["x_coeffs"][j]) for r in row_records], row_target)
            for j in range(m)
        ]
        y_targets = [
            q.mle([q.lift(r["y_coeffs"][j]) for r in row_records], row_target)
            for j in range(m)
        ]
        # Verify affine properties on inactive high indices:
        assert x_targets[0] == q.ZERO
        for j in range(k_z + 1, m):
            vj = v_limbs[j - 1] if j - 1 < 2 else 0
            assert x_targets[j] == q.mul(q.lift(vj), q.mle(psi_col, row_target))
        for j in range(y_active, m):
            assert y_targets[j] == q.ZERO

        context = rng.context()

        def prove_log2(omit=False):
            tr = q.transcript.Transcript(q.transcript.session_context(context))
            tr.absorb(
                "stage/0/split",
                q.tce(
                    [z_eval, q_eval, lookup_value_evals[0], lookup_value_evals[1]]
                    + lookup_key_evals
                ),
            )
            zero_row = tuple(q.challenge(tr, f"stage/1/zero/{j}") for j in range(rb))

            @functools.lru_cache(None)
            def stage1_terms(pt):
                m_at = q.mle(m_col, pt)
                q_at = q.mle(q_col, pt)
                y_at = q.mle(y_col, pt)
                e_at = q.mle(e_val_col, pt)
                psi_at = q.mle(psi_col, pt)
                z_at = q.mle(z_col, pt)
                i_at = q.mle(i_col, pt)
                f_at = q.mle(f_col, pt)
                p_at = q.mle(p_col, pt)

                z_low = [q.mle(col, pt) for col in z_low_cols]
                rem_z = z_at
                for h, zh in enumerate(z_low):
                    rem_z = q.sub(rem_z, q.mul(zh, q.lift(1 << (16 * h))))
                z_l = z_low + [q.mul(rem_z, q.lift(inv_pow16[k_z - 1]))]

                q_low = [q.mle(col, pt) for col in q_low_cols]
                rem_q = q_at
                for a, qa in enumerate(q_low):
                    rem_q = q.sub(rem_q, q.mul(qa, q.lift(1 << (16 * a))))
                q_l = q_low + [q.mul(rem_q, q.lift(inv_pow16[k_q - 1]))]

                p_low = [q.mle(col, pt) for col in p_low_cols]
                rem_p = p_at
                for b, pb in enumerate(p_low):
                    rem_p = q.sub(rem_p, q.mul(pb, q.lift(1 << (16 * b))))
                p_l = p_low + [q.mul(rem_p, q.lift(inv_pow16[3]))]

                eq_zr = q.eq_points(zero_row, pt)
                eq_rt = q.eq_points(row_target, pt)

                out = []
                # 0: match booleanity (m_t = 0 when eps_k == 0, else m_t - m_t^2 = 0)
                m_rel = m_at if eps_k == 0 else q.sub(m_at, q.mul(m_at, m_at))
                out.append(q.mul(eq_zr, m_rel))
                # 1: q_t - (Y_t + psi_t) * E_t = 0
                out.append(q.mul(eq_zr, q.sub(q_at, q.mul(q.add(y_at, psi_at), e_at))))
                # 2 .. 2 + k_z: X_1 .. X_{k_z}
                for j in range(1, k_z + 1):
                    vj = v_limbs[j - 1] if j - 1 < 2 else 0
                    out.append(q.mul(eq_rt, q.mul(psi_at, q.add(z_l[j - 1], q.lift(vj)))))
                # 2 + k_z .. 2 + k_z + y_active: Y_0 .. Y_{y_active - 1}
                for j in range(y_active):
                    yj = q.ZERO
                    for a in range(k_q):
                        for b in range(4):
                            if a + b == j:
                                yj = q.add(yj, q.mul(q_l[a], p_l[b]))
                    out.append(q.mul(eq_rt, yj))
                # idx_pad_z: (psi_t - 1) * Z_t = 0
                out.append(q.mul(eq_zr, q.mul(q.sub(psi_at, q.ONE), z_at)))
                # idx_uhat: B_t
                mismatch_cost = q.add(
                    q.lift(256 * eps_k), q.add(q.mul(q.lift(256), i_at), f_at)
                )
                bt = q.mul(
                    psi_at,
                    q.add(
                        q.mul(m_at, q.lift(p["c_match"])),
                        q.mul(q.sub(q.ONE, m_at), mismatch_cost),
                    ),
                )
                out.append(bt)
                return out

            stage1_claims = (
                [q.ZERO, q.ZERO]
                + [x_targets[j] for j in range(1, k_z + 1)]
                + [y_targets[j] for j in range(y_active)]
                + [q.ZERO, q.lift(claimed_uhat)]
            )

            def stage1_evals(pt):
                out = [q.mle(m_col, pt)]
                out.extend([q.mle(q_col, pt), q.mle(y_col, pt), q.mle(e_val_col, pt)])
                for j in range(1, k_z + 1):
                    if j - 1 < k_z - 1:
                        out.append(q.mle(z_low_cols[j - 1], pt))
                    else:
                        out.append(q.mle(z_col, pt))
                        out.extend(q.mle(col, pt) for col in z_low_cols)
                qp_evals = (
                    [q.mle(q_col, pt)]
                    + [q.mle(col, pt) for col in q_low_cols]
                    + [q.mle(p_col, pt)]
                    + [q.mle(col, pt) for col in p_low_cols]
                )
                out.extend(qp_evals * y_active)
                out.append(q.mle(z_col, pt))
                out.extend([q.mle(m_col, pt), q.mle(i_col, pt), q.mle(f_col, pt)])
                return out

            def run_stage(stage_no, n_vars, degree, claims_list, terms_fn, evals_fn, omit_idx):
                gamma = q.challenge(tr, f"stage/{stage_no}/gamma")
                weights = [q.power(gamma, i) for i in range(len(claims_list))]

                def current(pt):
                    vals = list(terms_fn(tuple(pt)))
                    if omit_idx is not None:
                        delta0 = q.mle([q.ONE] + [q.ZERO] * ((1 << n_vars) - 1), pt)
                        vals[omit_idx] = q.mul(claims_list[omit_idx], delta0)
                    elif bad == "match-when-k-zero" and stage_no == 1:
                        # Prove booleanity m_t - m_t^2 = 0 (which sums to 0 on {0,1}^rb when m_0 = 1)
                        # so a verifier missing the k == 0 -> m_t = 0 check accepts the raw proof.
                        m_at = q.mle(m_col, pt)
                        vals[0] = q.mul(
                            q.eq_points(zero_row, pt), q.sub(m_at, q.mul(m_at, m_at))
                        )
                    return vals

                def at(prefix):
                    rem = n_vars - len(prefix)
                    return q.total(
                        q.mul(w, val)
                        for rest in range(1 << rem)
                        for w, val in zip(
                            weights, current(prefix + q.boolean_point(rest, rem))
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
                    tr.absorb(f"stage/{stage_no}/round/{r_idx}", q.tce(msg))
                    ch = q.challenge(tr, f"stage/{stage_no}/r/{r_idx}")
                    pt.append(ch)
                    claim = q.lagrange(values, ch)
                    rounds.append(msg)
                final = evals_fn(pt)
                residual = q.total(q.mul(w, v) for w, v in zip(weights, terms_fn(tuple(pt))))
                valid = claim == residual
                if omit_idx is not None:
                    assert claim == q.total(q.mul(w, v) for w, v in zip(weights, current(pt)))
                tr.absorb(f"stage/{stage_no}/evals", q.tce(final))
                return dict(
                    gamma=gamma,
                    degree=degree,
                    rounds=rounds,
                    point=pt,
                    final_evals=final,
                    final_claim=claim,
                    final_relations=terms_fn(tuple(pt)),
                    valid=valid,
                    state=tr.state.hex(),
                    counter=tr.counter,
                )

            s1_omit = failed_stage[1] if (omit and failed_stage and failed_stage[0] == 1) else None
            stage1 = run_stage(1, rb, 3, stage1_claims, stage1_terms, stage1_evals, s1_omit)
            stage1["zero_row"] = list(zero_row)

            r1 = tuple(stage1["point"])
            y_r1 = stage1["final_evals"][2]
            z_r1 = stage1["final_evals"][-4]
            zero_vt = tuple(q.challenge(tr, f"stage/2/zero/{j}") for j in range(vb + rb))

            @functools.lru_cache(None)
            def stage2_terms(pt):
                v_pt, row_pt = pt[:vb], pt[vb:]
                cv = q.mle(valid_vocab_col, v_pt)
                ev = q.mle(e_vocab_col, pt)
                hy = q.mle(hy_col, pt)
                hyhat = q.mle(hyhat_col, pt)
                mt = q.mle(m_col, row_pt)
                return [
                    q.mul(q.eq_points(row_target, row_pt), q.mul(cv, ev)),
                    q.mul(q.eq_points(r1, row_pt), q.mul(cv, ev)),
                    q.mul(q.eq_points(r1, row_pt), q.mul(hy, ev)),
                    q.mul(q.eq_points(zero_vt, pt), q.mul(mt, q.sub(hy, hyhat))),
                ]

            stage2_claims = [z_eval, z_r1, y_r1, q.ZERO]

            def stage2_evals(pt):
                ev = q.mle(e_vocab_col, pt)
                hy = q.mle(hy_col, pt)
                hyhat = q.mle(hyhat_col, pt)
                mt = q.mle(m_col, pt[vb:])
                return [ev, ev, hy, ev, mt, hy, hyhat]

            s2_omit = failed_stage[1] if (omit and failed_stage and failed_stage[0] == 2) else None
            stage2 = run_stage(2, vb + rb, 3, stage2_claims, stage2_terms, stage2_evals, s2_omit)
            stage2["zero_vt"] = list(zero_vt)

            if local_reject is None:
                assert stage1["valid"] == (failed_stage is None or failed_stage[0] != 1)
                assert stage2["valid"] == (failed_stage is None or failed_stage[0] != 2)
            return dict(stage1=stage1, stage2=stage2)

        rec = dict(
            name=name,
            gadget=400 + idx,
            rows=rows,
            vocab=vocab,
            sigma=sigma,
            eps_log2=eps_k,
            c_match=p["c_match"],
            w_z=p["w_z"],
            k_z=k_z,
            top_z_bits=p["top_z_bits"],
            w_q=p["w_q"],
            k_q=k_q,
            top_q_bits=p["top_q_bits"],
            m=m,
            lookup_specs=specs,
            bounds=p["bounds"],
            carry_widths=cw,
            exps=exps,
            y_tokens=y_tokens,
            yhat_tokens=yhat_tokens,
            h_y=h_y,
            h_yhat=h_yhat,
            actual_uhat=actual_uhat,
            claimed_uhat=claimed_uhat,
            row_records=[
                dict(
                    psi=r["psi"],
                    z=r["z"],
                    y=r["y"],
                    m_bit=r["m_bit"],
                    i_exp=r["i_exp"],
                    p_val=r["p_val"],
                    f_frac=r["f_frac"],
                    e_val=r["e_val"],
                    q_val=r["q_val"],
                    b_t=r["b_t"],
                    x_coeffs=r["x_coeffs"],
                    y_coeffs=r["y_coeffs"],
                    committed=r["committed"],
                    lookup_keys=r["lookup_keys"],
                    lookup_values=r["lookup_values"],
                )
                for r in row_records
            ],
            row_target=list(row_target),
            z_eval=z_eval,
            q_eval=q_eval,
            lookup_key_evals=lookup_key_evals,
            lookup_value_evals=lookup_value_evals,
            x_targets=x_targets,
            y_targets=y_targets,
            context=context,
            local_reject=local_reject,
            failed_stage=list(failed_stage) if failed_stage else None,
            proof=prove_log2(False),
        )
        if failed_stage is not None:
            rec["omission"] = prove_log2(True)
        component_cases.append(rec)

    return dict(
        schema_version=1,
        description="RFC-0017 VocabMax, OneSidedLog2 and runtime table interning: big-integer witnesses, Gibbs bounds and two-stage cube-sum component proofs",
        c_match=c_match_vec,
        vocab_max_cases=vmax,
        table_interning_cases=interning,
        schema_cases=schemas,
        cases=component_cases,
    )
