# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Canonical teacher-forcing views: literal row slices and shifted equality sums."""
import importlib.util
from pathlib import Path


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def cases():
    parent=Path(__file__).resolve().parent.parent
    opening=load(parent/'advice-opening/generate.py','argmax_view_opening')
    q=load(parent/'sumcheck/generate.py','argmax_view_sumcheck')
    q.hashing.check_official_vectors();rng=q.Inputs(0x12400170009)
    out=[]
    for n,prompt,vocab in [(2,1,1),(4,1,3),(5,2,5),(7,3,6),(8,1,8),(8,4,5)]:
        public=opening.micro_public();public['arch'].update(d_model=2,n_heads=1,n_kv_heads=1,d_head=2,d_ff=6,vocab=vocab)
        graph,_=opening.lowering.case_of('output-view','tiny-2-servable.d',opening.lowering.Public(public),n,None)
        h_id=graph['nodes'][0][4][0][0];logits_id=graph['nodes'][-1][4][0][0]
        outputs=n-prompt;vb=(vocab-1).bit_length();rb=(outputs-1).bit_length();sb=(n-1).bit_length();V=1<<vb;N=1<<sb;O=1<<rb
        tokens=[(3*t+1)%vocab for t in range(n)]+[0]*(N-n)
        h=[int(t<n and v==tokens[t]) for t in range(N) for v in range(V)]
        ell=[((31+t+v) if v==tokens[t+1] else -7-v) if t+1<n and v<vocab else
             (-13-v if t<n and v<vocab else 0) for t in range(N) for v in range(V)]
        views=[]
        for role,source,offset,table in [('Indicator',h_id,prompt,h),('Logits',logits_id,prompt-1,ell)]:
            target=[rng.fp2() for _ in range(vb+rb)]
            sliced=[table[(offset+u)*V+v] if u<outputs else 0 for u in range(O) for v in range(V)]
            value=q.mle(list(map(q.lift,sliced)),target)
            row_values=[q.mle(list(map(q.lift,table[t*V:(t+1)*V])),target[:vb]) for t in range(N)]
            assert value==q.total(q.mul(q.eq_at(target[vb:],u),row_values[offset+u]) for u in range(outputs))
            views.append(dict(role=role,source=source,offset=offset,target=target,column=table,sliced=sliced,value=value,row_values=row_values))
        context=rng.context();t=q.transcript.Transcript(q.transcript.session_context(context));t.absorb('stage/1/evals',q.tce([v['value'] for v in views]));gamma=q.challenge(t,'stage/2/gamma');weights=[q.ONE,gamma]
        def terms(point):
            result=[]
            for view in views:
                kernel=q.total(q.mul(q.eq_at(view['target'][vb:],u),q.eq_at(point,view['offset']+u)) for u in range(outputs))
                result.append(q.mul(kernel,q.mle(view['row_values'],point)))
            return result
        def at(prefix):
            return q.total(q.mul(w,v) for rest in range(1<<(sb-len(prefix))) for w,v in zip(weights,terms(prefix+q.boolean_point(rest,sb-len(prefix)))))
        claim=q.total(q.mul(w,v['value']) for w,v in zip(weights,views));point,rounds=[],[]
        for j in range(sb):
            values=[at(point+[q.lift(k)]) for k in range(3)];assert q.add(values[0],values[1])==claim;message=[values[0],values[2]]
            t.absorb(f'stage/2/round/{j}',q.tce(message));point.append(q.challenge(t,f'stage/2/r/{j}'));claim=q.lagrange(values,point[-1]);rounds.append(message)
        final=[q.mle(v['row_values'],point) for v in views];assert q.total(q.mul(w,v) for w,v in zip(weights,terms(point)))==claim
        t.absorb('stage/2/evals',q.tce(final))
        for view in views:view['final_point']=view['target'][:vb]+point
        out.append(dict(n=n,prompt=prompt,outputs=outputs,vocab=vocab,gadget=200+len(out),context=context,views=views,proof=dict(gamma=gamma,point=point,rounds=rounds,final_evals=final,final_relations=terms(point),state=t.state.hex(),counter=t.counter)))
    return dict(schema_version=1,cases=out)
