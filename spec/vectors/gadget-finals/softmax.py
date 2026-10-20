# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0009/0030/0035 Softmax: integer witnesses and direct cube sums."""
import copy
import functools
import importlib.util
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    parent = Path(__file__).resolve().parent.parent
    opening = load(parent/'advice-opening/generate.py', 'softmax_opening_oracle')
    q = load(parent/'sumcheck/generate.py', 'softmax_sumcheck_oracle')
    q.hashing.check_official_vectors()
    rng = q.Inputs(0x12400300035)
    out = []
    variants = [('scalar', 1, 1, 3, 32), ('padded', 3, 3, 3, 8),
                ('wide', 3, 3, 23, 16), ('bit-limbs', 3, 2, 3, 1)]
    for name, n, heads, sigma, limb in variants:
        public = opening.micro_public()
        public['arch'].update(d_model=2*heads, n_heads=heads, n_kv_heads=heads, d_head=2, d_ff=6)
        public['quant']['public']['exp_sigma'] = sigma
        for point, shifts in public['quant']['public']['shift_class_sets']:
            if 'Layer' in point and next(iter(point['Layer']['site'])) != 'Silu':
                shifts[:] = [20]
        graph, _ = opening.lowering.case_of(name, 'tiny-2-servable.d', opening.lowering.Public(public), n, None)
        try:
            tables, choices = opening.advice.catalog(graph, 32, limb)
        except ValueError as error:
            raise ValueError(name) from error
        for gid, choice in enumerate(choices):
            if choice['pattern'] != ['SoftmaxRow']:
                continue
            nid, _, params, inputs, outputs, _ = graph['nodes'][choice['node']]
            exp_id = choice['tables'][0]
            kind = tables[exp_id][1]
            count, exp, _, _ = opening.lowering.tables.table(kind)
            K0 = count-1
            assert exp(K0) == 0
            tb, hb = (n-1).bit_length(), (heads-1).bit_length()
            T, H = 1 << tb, 1 << hb
            psi = [int(h < heads and t < n) for h in range(H) for t in range(T)]
            C = [int(h < heads and i <= t < n) for h in range(H) for t in range(T) for i in range(T)]
            score = [(13*h+7*t-11*i) if h < heads and t < n and i < n else 0
                     for h in range(H) for t in range(T) for i in range(T)]
            maximum = [max(score[r*T:r*T+t+1]) if valid else 0 for r, valid in enumerate(psi) for t in [r % T]]
            raw = [maximum[r]-score[r*T+i] for r in range(H*T) for i in range(T)]
            keys = [k if c else K0 for k,c in zip(raw,C)]
            leaves = [k if c else 1 for k,c in zip(raw,C)]
            e = [exp(k) for k in keys]
            sums = [sum(e[r*T:(r+1)*T]) for r in range(H*T)]
            R = [2**(2*sigma)//s if s else 0 for s in sums]
            alpha = [valid*2**(2*sigma)-r*s for valid,r,s in zip(psi,R,sums)]
            beta = [s-valid-a for s,valid,a in zip(sums,psi,alpha)]
            variants_here = [None]
            if name == 'padded' and gid == next(i for i,c in enumerate(choices) if c['pattern'] == ['SoftmaxRow']):
                variants_here += ['masked-lookup', 'masked-tree', 'reciprocal', 'padded-maximum', 'padded-reciprocal', 'padded-head']
            for bad in variants_here:
                record = dict(name=name+(':'+bad if bad else ''), n=n, heads=heads, sigma=sigma, limb=limb,
                              gadget=gid, node=nid, scores=inputs[0], exponentials=outputs[0][0], reciprocal=outputs[1][0],
                              tables=choice['tables'], key_bits=kind['Exp']['key_bits'])
                c, rec, ke, le, ev = map(list, (maximum,R,keys,leaves,e))
                failed = None
                if bad == 'masked-lookup':
                    ke[1] = raw[1]; ev[1] = exp(ke[1]); failed = 0
                    assert ev == e  # The alternate key is also in the zero tail.
                elif bad == 'masked-tree':
                    le[1] = raw[1]; failed = 1
                elif bad == 'reciprocal':
                    rec[0] += 1; failed = 2
                elif bad == 'padded-maximum':
                    c[n] = 1; failed = 3
                elif bad == 'padded-reciprocal':
                    rec[n] = 1; failed = 4
                elif bad == 'padded-head':
                    c[heads*T] = 1; failed = 3
                widths = [min(limb,sigma+18-j) for j in range(0,sigma+18,limb)]
                digits = [[(a >> (limb*j)) & (2**w-1) for a in alpha] for j,w in enumerate(widths)]
                digits += [[(b >> (limb*j)) & (2**w-1) for b in beta] for j,w in enumerate(widths)]
                assert all(0 <= v < 2**w for side in (alpha,beta) for v,w in [(max(side), sigma+18)])
                row_target = tuple(rng.fp2() for _ in range(tb+hb))
                lookup_target = tuple(rng.fp2() for _ in range(2*tb+hb))
                tree_target = tuple(rng.fp2() for _ in range(2*tb+hb))
                key_claim = q.mle(list(map(q.lift,ke)),lookup_target)
                tree_claim = q.mle(list(map(q.lift,le)),tree_target)
                lookup_value = q.mle(list(map(q.lift,ev)),lookup_target)
                ae, be = (q.mle(list(map(q.lift,x)),row_target) for x in (alpha,beta))
                half = q.lift(pow(2,-1,q.P))
                source_columns = [rec,sums]+digits[:len(widths)-1]+digits[len(widths):-1]
                source_values = [q.mle(list(map(q.lift,x)),row_target) for x in source_columns]
                source_values[1] = q.mul(source_values[1],q.lift(pow(T,-1,q.P)))
                range_columns = [rec]+digits
                range_keys = [q.mle(list(map(q.lift,x)),row_target) for x in range_columns]
                assert len(range_keys)+1 == len(choice['tables'])
                lookup_C = q.mle(list(map(q.lift,C)),lookup_target)
                tree_C = q.mle(list(map(q.lift,C)),tree_target)
                claims = [q.sub(key_claim,q.mul(q.lift(K0),q.sub(q.ONE,lookup_C))),
                          q.sub(tree_claim,q.sub(q.ONE,tree_C)),
                          q.sub(q.mul(q.lift(2**(2*sigma)),q.mle(list(map(q.lift,psi)),row_target)),ae), q.ZERO,q.ZERO]
                context = rng.context()
                def before():
                    transcript = q.transcript.Transcript(q.transcript.session_context(context))
                    transcript.absorb('stage/0/split',q.tce(source_values+range_keys+[key_claim,lookup_value,tree_claim]))
                    layer = q.challenge(transcript,'stage/1/layer/0')
                    zero = tuple(q.challenge(transcript,f'stage/1/zero/{j}') for j in range(tb+hb))
                    return transcript,layer,zero
                _, _, zero = before()
                @functools.lru_cache(None)
                def terms(point):
                    key, row = point[:tb], point[tb:]
                    # Row-only instances are front-padded into the square batch.
                    small = point[tb:]
                    maximum_at = q.mle(list(map(q.lift,c)),row)
                    score_at = q.mle(list(map(q.lift,score)),point)
                    causal = q.mle(list(map(q.lift,C)),point)
                    rec_at = q.mle(list(map(q.lift,rec)),small)
                    sum_at = q.mle(list(map(q.lift,sums)),small)
                    pad = q.mul(q.eq_points(zero,small),q.sub(q.ONE,q.mle(list(map(q.lift,psi)),small)))
                    return [q.mul(q.eq_points(lookup_target,point),q.mul(causal,q.sub(maximum_at,score_at))),
                            q.mul(q.eq_points(tree_target,point),q.mul(causal,q.sub(maximum_at,score_at))),
                            q.mul(q.eq_points(row_target,small),q.mul(rec_at,sum_at)),
                            q.mul(pad,q.mle(list(map(q.lift,c)),small)),q.mul(pad,rec_at)]
                variables = 2*tb+hb
                scales = [1,1,T,T,T]
                actual = [q.ZERO]*5
                for i in range(1 << variables):
                    actual = [q.add(a,b) for a,b in zip(actual,terms(tuple(q.boolean_point(i,variables))))]
                assert [i for i,(a,cl,scale) in enumerate(zip(actual,claims,scales)) if a != q.mul(cl,q.lift(scale))] == ([] if failed is None else [failed])
                def prove(omit=False):
                    t,layer,_ = before()
                    gamma = q.challenge(t,'stage/1/gamma')
                    weights = [q.power(gamma,j) for j in range(5)]
                    def current(point):
                        v = list(terms(tuple(point)))
                        if omit:
                            domain = point if failed < 2 else point[tb:]
                            v[failed] = q.mul(claims[failed],q.mle([q.ONE]+[q.ZERO]*((1 << len(domain))-1),domain))
                        return v
                    def at(prefix):
                        return q.total(q.mul(w,v) for rest in range(1 << (variables-len(prefix)))
                                       for w,v in zip(weights,current(prefix+q.boolean_point(rest,variables-len(prefix)))))
                    point,rounds=[],[]
                    claim=q.total(q.mul(w,q.mul(cl,q.lift(scale))) for w,cl,scale in zip(weights,claims,scales))
                    for j in range(variables):
                        values=[at(point+[q.lift(i)]) for i in range(4)]
                        if omit or failed is None: assert q.add(values[0],values[1]) == claim
                        values[1]=q.sub(claim,values[0])
                        message=[values[0],values[2],values[3]]
                        t.absorb(f'stage/1/round/{j}',q.tce(message))
                        point.append(q.challenge(t,f'stage/1/r/{j}'))
                        claim=q.lagrange(values,point[-1]); rounds.append(message)
                    row=point[tb:]
                    ce=q.mle(list(map(q.lift,c)),row)
                    xe=q.mle(list(map(q.lift,score)),point)
                    re=q.mle(list(map(q.lift,rec)),row)
                    se=q.mul(q.mle(list(map(q.lift,sums)),row),q.lift(pow(T,-1,q.P)))
                    evaluations=[ce,xe,ce,xe,re,se,ce,re]
                    assert (claim == q.total(q.mul(w,v) for w,v in zip(weights,terms(tuple(point))))) == (failed is None)
                    if omit: assert claim == q.total(q.mul(w,v) for w,v in zip(weights,current(point)))
                    t.absorb('stage/1/evals',q.tce(evaluations))
                    return dict(layer=layer,zero=zero,gamma=gamma,rounds=rounds,point=point,final_evals=evaluations,
                                final_claim=claim,final_relations=terms(tuple(point)),state=t.state.hex(),counter=t.counter)
                record.update(context=context,row_target=row_target,lookup_target=lookup_target,tree_target=tree_target,
                              lookup_key=key_claim,lookup_value=lookup_value,tree_leaf=tree_claim,source_values=source_values,
                              range_keys=range_keys,widths=widths,source_columns=source_columns,range_columns=range_columns,
                              score_column=score,maximum_column=c,exp_column=ev,bracket_exp_column=e,tree_column=le,
                              psi=psi,causal=C,claims=claims,failed=failed,proof=prove())
                if bad: record['omission']=prove(True)
                out.append(record)
    return dict(schema_version=1,cases=out)
