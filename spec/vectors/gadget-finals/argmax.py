# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Argmax: integer lowest-index ties and independent two-stage cube-sum proofs."""
import functools
import importlib.util
from pathlib import Path


def cases():
    path = Path(__file__).resolve().parent.parent/'sumcheck/generate.py'
    spec = importlib.util.spec_from_file_location('argmax_sumcheck_oracle', path)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    q.hashing.check_official_vectors()
    rng = q.Inputs(0x12400090017)
    shapes = [(1,1),(1,2),(3,3),(3,5),(5,6),(5,8)]
    alternatives = ['key-tie', 'suffix-tie', 'raised-maximum', 'vocabulary-mask', 'padded-maximum', 'suffix-direction']
    shapes += [(3,5)]*len(alternatives)
    out = []
    for index,(rows,vocab) in enumerate(shapes):
        bad = alternatives[index-6] if index >= 6 else None
        vb,rb = (vocab-1).bit_length(),(rows-1).bit_length()
        V,T = 1 << vb,1 << rb
        logits=[]
        tokens=[]
        maxima=[]
        for t in range(T):
            real = t < rows
            values = ([-7]*vocab if t%3 == 0 else
                      [-(2**28)+1]+[2**28-1]*(vocab-1) if t%3 == 1 else
                      [-23+((v+t)%4) for v in range(vocab)]) if real else [0]*vocab
            chosen = values.index(max(values)) if real else 0
            logits.extend(values+[0]*(V-vocab))
            tokens.append(chosen)
            maxima.append(values[chosen] if real else 0)
        if bad in ('key-tie','suffix-tie'):
            tokens[0]=1
        if bad == 'suffix-direction':
            logits[:V]=[7]+[1]*(vocab-1)+[0]*(V-vocab)
            maxima[0],tokens[0]=7,0
        h=[int(t < rows and v == tokens[t]) for t in range(T) for v in range(V)]
        actual_suffix=[sum(h[t*V+v+1:(t+1)*V]) for t in range(T) for v in range(V)]
        suffix=list(actual_suffix)
        if bad == 'suffix-tie': suffix[:V]=[0]*V
        if bad == 'suffix-direction': suffix[:V]=[sum(h[:v]) for v in range(V)]
        if bad == 'raised-maximum': maxima[0]+=1
        if bad == 'padded-maximum': maxima[rows]=1
        keys=[maxima[t]-logits[t*V+v]-suffix[t*V+v] if v < vocab else 0 for t in range(T) for v in range(V)]
        if bad == 'key-tie': keys[:V]=[0]*V
        if bad == 'vocabulary-mask': keys[vocab]=1
        assert all(0 <= key < 2**29 for key in keys)
        assert all(sum(h[t*V:(t+1)*V]) == int(t < rows) for t in range(T))
        assert all(h[t*V+v] == 0 for t in range(T) for v in range(vocab,V))
        row_target=tuple(rng.fp2() for _ in range(rb))
        key_target=tuple(rng.fp2() for _ in range(vb+rb))
        claims=[q.mle(list(map(q.lift,keys)),key_target),q.mle(list(map(q.lift,maxima)),row_target)]
        context=rng.context()
        @functools.lru_cache(None)
        def first(point):
            v,t=point[:vb],point[vb:]
            valid=q.mle([q.lift(int(i < vocab)) for i in range(V)],v)
            m=q.mle(list(map(q.lift,maxima)),t)
            ell=q.mle(list(map(q.lift,logits)),point)
            lt=q.mle(list(map(q.lift,suffix)),point)
            hv=q.mle(list(map(q.lift,h)),point)
            return [q.mul(q.eq_points(key_target,point),q.mul(valid,q.sub(q.sub(m,ell),lt))),
                    q.mul(q.eq_points(row_target,t),q.mul(hv,ell))]
        actual=[q.ZERO,q.ZERO]
        for i in range(V*T):actual=[q.add(a,b) for a,b in zip(actual,first(tuple(q.boolean_point(i,vb+rb))))]
        failed = 0 if bad in ('key-tie','vocabulary-mask') else 1 if bad in ('raised-maximum','padded-maximum') else 2 if bad else None
        assert [i for i,(a,b) in enumerate(zip(actual,claims)) if a != b] == ([] if failed in (None,2) else [failed])
        def prove_stage(t,stage,degree,stage_claims,terms,evaluations,omit=None):
            gamma=q.challenge(t,f'stage/{stage}/gamma')
            weights=[q.power(gamma,i) for i in range(len(stage_claims))]
            def current(point):
                result=list(terms(tuple(point)))
                if omit is not None:result[omit]=q.mul(stage_claims[omit],q.mle([q.ONE]+[q.ZERO]*(V*T-1),point))
                return result
            def at(prefix):
                return q.total(q.mul(w,value) for rest in range(1 << (vb+rb-len(prefix)))
                               for w,value in zip(weights,current(prefix+q.boolean_point(rest,vb+rb-len(prefix)))))
            point,rounds=[],[]
            claim=q.total(q.mul(w,c) for w,c in zip(weights,stage_claims))
            for j in range(vb+rb):
                values=[at(point+[q.lift(k)]) for k in range(degree+1)]
                if omit is not None or (failed is None or (stage==1 and failed==2) or (stage==2 and failed!=2)):
                    assert q.add(values[0],values[1]) == claim
                values[1]=q.sub(claim,values[0])
                message=[values[0]]+values[2:]
                t.absorb(f'stage/{stage}/round/{j}',q.tce(message))
                point.append(q.challenge(t,f'stage/{stage}/r/{j}'))
                claim=q.lagrange(values,point[-1]);rounds.append(message)
            final=evaluations(point)
            residual=q.total(q.mul(w,v) for w,v in zip(weights,terms(tuple(point))))
            valid=claim==residual
            if omit is not None:assert claim==q.total(q.mul(w,v) for w,v in zip(weights,current(point)))
            t.absorb(f'stage/{stage}/evals',q.tce(final))
            return dict(gamma=gamma,degree=degree,rounds=rounds,point=point,final_evals=final,
                        final_claim=claim,final_relations=terms(tuple(point)),valid=valid,state=t.state.hex(),counter=t.counter)
        def prove(omit=False):
            t=q.transcript.Transcript(q.transcript.session_context(context))
            t.absorb('stage/0/split',q.tce(claims))
            def first_evals(point):
                row=point[vb:]
                return [q.mle(list(map(q.lift,maxima)),row),q.mle(list(map(q.lift,logits)),point),
                        q.mle(list(map(q.lift,suffix)),point),q.mle(list(map(q.lift,h)),point),
                        q.mle(list(map(q.lift,logits)),point)]
            first_record=prove_stage(t,1,3,claims,first,first_evals,failed if omit and failed in (0,1) else None)
            target=first_record['point']
            def second(point):
                v,row=point[:vb],point[vb:]
                # Literal sum of both equality tables, independent of suffix_kernel.
                kernel=q.total(q.mul(q.eq_at(target[:vb],i),q.eq_at(v,w)) for i in range(V) for w in range(i+1,V))
                hv=q.mle(list(map(q.lift,h)),point)
                return [q.mul(q.eq_points(target[vb:],row),q.mul(kernel,hv))]
            second_record=prove_stage(t,2,2,[first_record['final_evals'][2]],second,
                                       lambda point:[q.mle(list(map(q.lift,h)),point)],0 if omit and failed==2 else None)
            assert first_record['valid'] == (failed in (None,2))
            assert second_record['valid'] == (failed != 2)
            return dict(first=first_record,second=second_record)
        record=dict(name=bad or f'rows={rows}-vocab={vocab}',gadget=100+index,rows=rows,vocab=vocab,
                    logits=logits,h=h,maxima=maxima,suffix=suffix,actual_suffix=actual_suffix,keys=keys,tokens=tokens,
                    row_target=row_target,key_target=key_target,claims=claims,context=context,failed=failed,proof=prove())
        if bad:record['omission']=prove(True)
        out.append(record)
    return dict(schema_version=1,cases=out)
