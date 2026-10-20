# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Model-derived root bindings: independent integer witnesses and direct cube sums."""
import copy
import importlib.util
import math
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(oracle):
    parent = Path(__file__).resolve().parent.parent
    opening = load(parent/'advice-opening/generate.py', 'root_opening_oracle')
    q = load(parent/'sumcheck/generate.py', 'root_binding_sumcheck_oracle')
    q.hashing.check_official_vectors()
    rng = q.Inputs(0x12400500009)
    base = opening.micro_public()
    base['arch'].update(d_model=2, n_heads=1, n_kv_heads=1, d_ff=6)
    out = []
    for name, kappa, epsilon, bad in [('zero-root', 0, 4, False), ('width-1', 0, 2, False),
            ('width-8', 7, 2, False), ('width-16', 15, 2, False),
            ('width-32', 31, 2, False), ('padded-root', 7, 2, True),
            ('constant-above-P', 31, 8, False)]:
        public = copy.deepcopy(base)
        if name == 'constant-above-P':
            public['arch'].update(d_model=8,d_head=8)
        public['quant']['public'].update(rsqrt_kappa=kappa, rms_eps_int=epsilon)
        graph, _ = opening.lowering.case_of(name, 'tiny-2-servable.d', opening.lowering.Public(public), 3, None)
        tables, choices = opening.advice.catalog(graph, 32, 16)
        roots = []
        for gid, choice in enumerate(choices):
            if choice['pattern'] != ['RsqrtRow']:
                continue
            nid, _, p, inputs, outputs, _ = graph['nodes'][choice['node']]
            dim = p['dim']
            constant = dim << (2*kappa)
            width = max(1, math.isqrt(constant//epsilon).bit_length())
            m, low = (2*width+62)//8, (width-1)//8
            residuals = [[0]*dim, [7 if i%2 else -3 for i in range(dim)],
                         [127 if i%2 else -128 for i in range(dim)], [0]*dim]
            squares = [sum(x*x for x in row) for row in residuals]
            u = [s+epsilon if i < 3 else 0 for i,s in enumerate(squares)]
            rho = [math.isqrt(constant//v) if v else 0 for v in u]
            if bad:
                rho[-1] = 1
            root_digits = [[(v >> (8*j)) & 255 for v in rho] for j in range(low+1)]
            u_digits = [[(v >> (16*j)) & 65535 for v in u] for j in range(4)]
            raw = [[oracle.product_coefficients(r,v,width,inc) for r,v in zip(rho,u)] for inc in (False,True)]
            arrays = []
            for inc in (False,True):
                per_row = []
                for row in range(4):
                    digits = [((constant+int(inc)) >> (8*j)) & 255 if row < 3 else 0 for j in range(m)]
                    left, right = (raw[0][row],digits) if not inc else (digits,raw[1][row])
                    per_row.append(oracle.witness(left,right,8))
                arrays.extend([[per_row[row][a][j] for row in range(4)] for a in range(6) for j in range(m)])
            columns = [squares,rho]+u_digits[:3]+root_digits[:-1]+arrays
            key_columns = [rho]+u_digits+root_digits+arrays
            roles, widths = opening.advice.root_schema(dim,kappa,epsilon)
            sources = [['tensor',inputs[0]],['tensor',outputs[0][0]]]
            sources += [['auxiliary',nid,role,j] for role,j in roles]
            point = [rng.fp2() for _ in range(2)]
            root = dict(gadget=gid,node=nid,dim=dim,kappa=kappa,epsilon=epsilon,width=width,
                        constant=str(constant),sources=sources,columns=columns,
                        point=point,values=[q.mle([q.lift(x) for x in c],point) for c in columns],
                        tables=choice['tables'],range_bits=widths,key_columns=key_columns,
                        keys=[q.mle([q.lift(x) for x in c],point) for c in key_columns],
                        product_targets=[q.mle([q.lift(row[j]) for row in raw[inc]],point) for inc in range(2) for j in range(m)],
                        root_digits=root_digits,u_digits=u_digits)
            assert len(sources)==5+low+12*m and len(key_columns)==6+low+12*m
            assert all(0 <= value < 2**bits for c,bits in zip(key_columns,widths) for value in c)
            roots.append(root)
        record = dict(name=name,kappa=kappa,epsilon=epsilon,dim=public['arch']['d_model'],n=3,bad_padding=bad,roots=roots)
        record['proof'] = prove(q,rng,roots[0],False)
        if bad:
            record['omission'] = prove(q,rng,roots[0],True,context=record['proof']['context'])
        out.append(record)
    return {'schema_version':1,'cases':out}


def prove(q,rng,root,omit_padding,context=None):
    context = context or rng.context()
    t = q.transcript.Transcript(q.transcript.session_context(context))
    t.absorb('stage/0/split',q.tce(root['values']))
    layer = q.challenge(t,'stage/1/layer/0')
    zero = [q.challenge(t,f'stage/1/zero/{i}') for i in range(2)]
    gamma = q.challenge(t,'stage/1/gamma')
    columns = root['root_digits']+root['u_digits']
    count, m = len(root['root_digits']), len(root['product_targets'])//2
    claims = root['product_targets']+[q.ZERO]
    weights = [q.power(gamma,j) for j in range(len(claims))]

    def terms(point):
        operands = [q.mle([q.lift(v) for v in col],point) for col in columns]
        out = []
        for inc in (False,True):
            for j in range(m):
                value = q.ZERO
                for a in range(count):
                    for b in range(count):
                        for k in range(4):
                            if a+b+2*k == j:
                                x,y = operands[a],operands[b]
                                if inc:
                                    x = q.add(x,q.ONE) if a==0 else x
                                    y = q.add(y,q.ONE) if b==0 else y
                                value = q.add(value,q.mul(q.mul(x,y),operands[count+k]))
                out.append(q.mul(q.eq_points(root['point'],point),value))
        rho = q.mle([q.lift(x) for x in root['columns'][1]],point)
        psi = q.mle([q.ONE,q.ONE,q.ONE,q.ZERO],point)
        padding = q.mul(q.eq_points(zero,point),q.mul(q.sub(q.ONE,psi),rho))
        out.append(q.ZERO if omit_padding else padding)
        return out

    def at(prefix):
        return q.total(q.mul(w,v) for rest in range(1 << (2-len(prefix)))
                       for w,v in zip(weights,terms(prefix+q.boolean_point(rest,2-len(prefix)))))

    point, rounds = [], []
    claim = q.total(q.mul(w,c) for w,c in zip(weights,claims))
    padding_bad = root['columns'][1][-1] != 0
    assert (at([])==claim) == (not padding_bad or omit_padding)
    for j in range(2):
        values = [at(point+[q.lift(i)]) for i in range(5)]
        values[1] = q.sub(claim,values[0])
        message = [values[0]]+values[2:]
        t.absorb(f'stage/1/round/{j}',q.tce(message))
        coordinate = q.challenge(t,f'stage/1/r/{j}')
        point.append(coordinate)
        claim = q.lagrange(values,coordinate)
        rounds.append(message)
    operands = [q.mle([q.lift(v) for v in col],point) for col in columns]
    final_evals = operands*(2*m)+[q.mle([q.lift(v) for v in root['columns'][1]],point)]
    assert (q.total(q.mul(w,v) for w,v in zip(weights,terms(point)))==claim) == (not padding_bad or omit_padding)
    t.absorb('stage/1/evals',q.tce(final_evals))
    return dict(context=context,layer=layer,zero=zero,gamma=gamma,degree=4,rounds=rounds,
                final_evals=final_evals,point=point,final_relations=terms(point),state=t.state.hex(),counter=t.counter)
