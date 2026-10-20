# SPDX-License-Identifier: Apache-2.0 OR MIT
"""OneHot relations: integer indicators and literal cube sums, no Rust imports."""
import importlib.util
from pathlib import Path


def cases():
    path = Path(__file__).resolve().parent.parent / 'sumcheck/generate.py'
    spec = importlib.util.spec_from_file_location('onehot_sumcheck_oracle', path)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    q.hashing.check_official_vectors()
    rng = q.Inputs(0x1240009)
    out = []
    shapes = [(1, 1), (1, 2), (3, 3), (3, 5), (5, 6), (5, 8)]
    shapes += [(3, 5)] * 6
    alternatives = ['nonboolean', 'vocabulary', 'row-sum', 'index', 'padded-h', 'padded-token']
    for case_index, (n, vocab) in enumerate(shapes):
        alternative = alternatives[case_index-6] if case_index >= 6 else None
        vbits, tbits = (vocab-1).bit_length(), (n-1).bit_length()
        V, T = 1 << vbits, 1 << tbits
        tokens = [(7*t+vocab-1) % vocab if t < n else 0 for t in range(T)]
        h = [int(t < n and v == tokens[t]) for t in range(T) for v in range(V)]
        failed = None
        if alternative:
            row = 3 if alternative.startswith('padded-') else 0
            if alternative == 'nonboolean':
                h[:V], tokens[0], failed = [1, -1, 1]+[0]*(V-3), 1, 0
            elif alternative == 'vocabulary':
                h[:V], tokens[0], failed = [int(v == vocab) for v in range(V)], vocab, 1
            elif alternative == 'row-sum':
                h[:V], tokens[0], failed = [0]*V, 0, 2
            elif alternative == 'index':
                tokens[0], failed = (tokens[0]+1) % vocab, 3
            elif alternative == 'padded-h':
                h[row*V], failed = 1, 2
            else:
                tokens[row], failed = 1, 3
        context = rng.context()
        t = q.transcript.Transcript(q.transcript.session_context(context))
        t.absorb('stage/0/split', q.tce([]))
        layer = q.challenge(t, 'stage/1/layer/0')
        zero = [q.challenge(t, f'stage/1/zero/{j}') for j in range(vbits+tbits)]

        def relations(point):
            v, row = point[:vbits], point[vbits:]
            hv = q.mle([q.lift(value) for value in h], point)
            token = q.mle([q.lift(value) for value in tokens], row)
            # Direct evaluations of the public tables, not the verifier's closed forms.
            valid = q.mle([q.lift(int(i < vocab)) for i in range(V)], v)
            psi = q.mle([q.lift(int(i < n)) for i in range(T)], row)
            index = q.mle([q.lift(i) for i in range(V)], v)
            origin = q.mle([q.ONE]+[q.ZERO]*(V-1), v)
            full_eq, row_eq = q.eq_points(zero, point), q.eq_points(zero[vbits:], row)
            return [q.mul(full_eq, q.mul(hv, q.sub(q.ONE, hv))),
                    q.mul(full_eq, q.mul(q.sub(q.ONE, valid), hv)),
                    q.mul(row_eq, q.sub(hv, q.mul(origin, psi))),
                    q.mul(row_eq, q.sub(q.mul(index, hv), q.mul(origin, token)))]

        actual = [q.ZERO]*4
        for i in range(V*T):
            actual = [q.add(a, b) for a, b in zip(actual, relations(q.boolean_point(i, vbits+tbits)))]
        assert [i for i, value in enumerate(actual) if value != q.ZERO] == ([] if failed is None else [failed])
        gamma = q.challenge(t, 'stage/1/gamma')
        weights = [q.power(gamma, j) for j in range(4)]

        def at(prefix):
            return q.total(q.mul(weight, value)
                           for rest in range(1 << (vbits+tbits-len(prefix)))
                           for weight, value in zip(weights, relations(prefix+q.boolean_point(rest, vbits+tbits-len(prefix)))))

        point, rounds, claim = [], [], q.ZERO
        for j in range(vbits+tbits):
            values = [at(point+[q.lift(i)]) for i in range(4)]
            # A cheating witness keeps claiming zero; p(1) is reconstructed from
            # that claim, not its actual sum. Its final residual must then fail.
            values[1] = q.sub(claim, values[0])
            message = [values[0], values[2], values[3]]
            t.absorb(f'stage/1/round/{j}', q.tce(message))
            coordinate = q.challenge(t, f'stage/1/r/{j}')
            point.append(coordinate)
            claim = q.lagrange(values, coordinate)
            rounds.append(message)
        hv = q.mle([q.lift(value) for value in h], point)
        token = q.mle([q.lift(value) for value in tokens], point[vbits:])
        final_evals = [hv]*4+[token]
        residual = q.total(q.mul(weight, value) for weight, value in zip(weights, relations(point)))
        assert (claim == residual) == (failed is None)
        t.absorb('stage/1/evals', q.tce(final_evals))
        out.append(dict(name=alternative or f'n={n}-vocab={vocab}', n=n, vocab=vocab,
                        failed=failed, tokens=tokens, h=h, context=context, layer=layer,
                        zero=zero, actual_sums=actual, gamma=gamma, degree=3,
                        rounds=rounds, final_evals=final_evals, point=point,
                        final_claim=claim, final_relations=relations(point),
                        state=t.state.hex(), counter=t.counter))
    return {'schema_version': 1, 'cases': out}
