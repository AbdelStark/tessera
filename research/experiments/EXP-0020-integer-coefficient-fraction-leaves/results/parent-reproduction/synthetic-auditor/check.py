#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent integer/Fp2 oracle and saved-record audit, no prover imports."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=2**64-2**32+1

def add(a,b): return ((a[0]+b[0])%P,(a[1]+b[1])%P)
def neg(a): return ((-a[0])%P,(-a[1])%P)
def mul(a,b): return ((a[0]*b[0]+7*a[1]*b[1])%P,(a[0]*b[1]+a[1]*b[0])%P)
def scale(a,x): return (a[0]*x%P,a[1]*x%P)
def sub(a,b): return add(a,neg(b))
def weight(point,index):
    r=(1,0)
    for j,x in enumerate(point): r=mul(r,tuple(x) if (index>>j)&1 else sub((1,0),x))
    return r

def arithmetic(results):
    raw=json.loads((results/'arithmetic.json').read_text())
    assert raw['status']=='pass' and raw['cadence']==1024
    total=0
    for row in raw['weighted']:
        c=int(row['coefficient']);w=row['weight'];count=0;result=0
        for b in row['buckets']:
            expected=c*w*b['terms'];count+=b['terms'];words=b['words']
            value=sum(v<<(64*i) for i,v in enumerate(words));value-=2**192*(value>=2**191)
            assert -(2**191)<=expected<2**191
            assert value==expected,('carry/sign/cadence',value,expected)
            assert b['reduced']==expected%P
            result=(result+b['reduced'])%P
        assert count==row['count'] and result==row['result']==c*w*count%P
        total+=count
    assert raw['hand_channels']==['3','1','2','3','-2','5','7','4']
    assert raw['hand_product']==-48
    for row in raw['direct_signed_reductions']:
        n=sum(w<<(64*i) for i,w in enumerate(row['words']));n-=2**192*(n>=2**191)
        assert row['result']==n%P
    for row in raw['gate_boundaries']:
        n=int(row['value']);expected={'coordinate':n<=2**63-1,'coefficient':n<2**127,'bucket':1024*(P-1)*n<2**191}[row['gate']]
        assert row['admitted']==expected
    return {'groups':len(raw['weighted']),'weighted_terms':total}

def grids(results):
    raw=json.loads((results/'grids.json').read_text());count=0
    for row in raw['vectors']:
        v=row['depth'];n=len(row['point']);k=row['k'];values=row['v'];ms=row['m'];a=tuple(row['alpha']);b=tuple(row['beta']);lam=tuple(row['lambda'])
        def interp(col,parity,rest,zs,default):
            if not col:return default
            out=0
            for bits in range(1<<v):
                factor=1
                for j,z in enumerate(zs):factor*=z if(bits>>j)&1 else 1-z
                out+=factor*col[2*((rest<<v)+bits)+parity]
            return out
        for cell,actual in enumerate(row['grid']):
            zs=[cell//3**j%3 for j in range(v)];expected=(0,0)
            for rest in range(1<<(n-v)):
                k0=interp(k,0,rest,zs,0);k1=interp(k,1,rest,zs,0)
                v0=interp(values,0,rest,zs,0);v1=interp(values,1,rest,zs,0)
                m0=interp(ms,0,rest,zs,0);m1=interp(ms,1,rest,zs,0)
                d0=sub(sub(a,(k0%P,0)),scale(b,v0));d1=sub(sub(a,(k1%P,0)),scale(b,v1))
                numerator=add(scale(d1,-m0),scale(d0,-m1)) if ms else add(d0,d1)
                relation=add(numerator,mul(lam,mul(d0,d1)))
                expected=add(expected,mul(weight(row['point'][v:],rest),relation))
            assert tuple(actual)==expected,('coefficient grid',row['case'],v,cell,actual,expected)
            count+=1
    # F65537 is an algebra control, no statistical soundness measurement.
    seed=0x00200005
    for i in range(1000):
        vals=[]
        for _ in range(9):seed=(1664525*seed+1013904223)&0xffffffff;vals.append(seed%131073-65536)
        k0,k1,v0,v1,m0,m1,a,b,lam=vals
        direct=-m0*(a-k1-b*v1)-m1*(a-k0-b*v0)+lam*(a-k0-b*v0)*(a-k1-b*v1)
        expanded=-a*(m0+m1)+(m0*k1+m1*k0)+b*(m0*v1+m1*v0)+lam*(a*a-a*(k0+k1)-a*b*(v0+v1)+k0*k1+b*(k0*v1+v0*k1)+b*b*v0*v1)
        assert direct==expanded and direct%65537==expanded%65537
    return {'vectors':len(raw['vectors']),'grid_cells':count,'F65537_identities':1000}

def bounds():
    raw=json.loads((HERE/'results/bounds.preregistration.json').read_text())
    for r in raw['records']:
        a=3**r['v'];k=r['K'];v=r['V_abs'];m=r['M'];channels=[1,2*a*k,2*a*v,a*a*k*k,2*a*a*k*v,a*a*v*v,2*a*m,2*a*a*m*k,2*a*a*m*v]
        assert list(map(int,r['channel_absolute_bounds'].values()))==channels
        t=max(channels);x=1024*(P-1)*t
        assert t==int(r['Tmax']) and x==int(r['weighted_bucket_bound']) and x<2**191
    stress=raw['cadence_mutation_scalar_stress'];t=18*2**112
    assert int(stress['Tmax'])==t and 1024*(P-1)*t<2**191<2048*(P-1)*t
    return len(raw['records'])

def frozen():
    pairs=[('README.preregistered.txt','a4f1a68e4b266c935084cb429626bcbc4a72c1da3b0b4c8f27d3f687d06ac2ff','217238e'),('README.amendment-preregistered.txt','defec9b2d3f55bc6a2e3e35a88abba93ab03a28d20330a33fd0c98d5de4a282c','f8c2d71'),('README.output-boundary-preregistered.txt','19bb3201c06afd228d63eadb2c7826b8c92b06be97cbb229e79d999d4aa8e0a6','f77974f')]
    root=HERE.parents[2]
    for name,digest,commit in pairs:
        f=HERE/'results'/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==digest
        saved=subprocess.check_output(['git','show',commit+':'+str(f.relative_to(root))],cwd=root)
        assert saved==f.read_bytes()
    pins=json.loads((HERE/'results/source-pins.json').read_text())
    for row in pins['files']:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    return len(pins['files'])

def stats(x):
    return {'median':statistics.median(x),'min':min(x),'max':max(x),'cv':statistics.stdev(x)/statistics.mean(x)}

def audit(raw):
    # All ratios and CVs reconstructed from integer elapsed/repetition records.
    assert raw['primary_v']==2
    groups={}
    previous_end=-1
    details=[]
    raw_groups={}
    for cohort in raw['cohorts']:
        assert len(cohort['samples'])==10 and len(cohort['warmups'])==3
        names=['P','P2','D','C']
        for i,block in enumerate(cohort['warmups']+cohort['samples']):
            assert [x['arm'] for x in block]==names[i%4:]+names[:i%4]
            for x in block: assert x['end_ns']-x['start_ns']==x['elapsed_ns'] and x['end_ns']>=x['start_ns'] and isinstance(x['repetitions'],int) and x['repetitions']>0
        flat=[x for block in cohort['warmups']+cohort['samples'] for x in block]
        assert all(a['end_ns']<=b['start_ns'] for a,b in zip(flat,flat[1:]))
        assert flat[0]['start_ns']>=previous_end
        previous_end=flat[-1]['end_ns']
        for name in names:assert len({x['repetitions']for x in flat if x['arm']==name})==1
        per={name:[] for name in ['P','P2','D','C']}
        for block in cohort['samples']:
            assert sorted(x['arm'] for x in block)==sorted(per)
            for x in block:
                assert isinstance(x['elapsed_ns'],int) and isinstance(x['repetitions'],int) and x['repetitions']>0
                per[x['arm']].append(x['elapsed_ns']/x['repetitions'])
        ratios={k:[per['C'][i]/per[k][i] for i in range(10)] for k in ['P','D']}
        sham=[per['P2'][i]/per['P'][i] for i in range(10)]
        aggregate={a:sum(x['elapsed_ns'] for block in cohort['samples'] for x in block if x['arm']==a)/sum(x['repetitions'] for block in cohort['samples'] for x in block if x['arm']==a) for a in per}
        clean=all(stats(x)['cv']<=.03 for x in per.values()) and all(stats(ratios[k])['cv']<=.03 for k in (['P','D'] if cohort['boundary']=='whole' else ['P'])) and stats(sham)['cv']<=.03 and .98<=statistics.median(sham)<=1.02 and .98<=aggregate['P2']/aggregate['P']<=1.02 and all(x['elapsed_ns']>=20_000_000 for block in cohort['samples'] for x in block)
        saved=cohort.get('summary')
        derived={'clean':clean,'paired_saving_P':1-statistics.median(ratios['P']),'paired_saving_D':1-statistics.median(ratios['D']),'aggregate_saving_P':1-aggregate['C']/aggregate['P'],'aggregate_saving_D':1-aggregate['C']/aggregate['D']}
        if saved is not None:
            for k,v in derived.items():assert saved[k]==v if isinstance(v,bool) else math.isclose(saved[k],v,rel_tol=1e-12,abs_tol=1e-12),(k,saved[k],v)
            extras={'arm_cv':[stats(per[k])['cv']for k in names],'ratio_cv':[stats(ratios['P'])['cv'],stats(ratios['D'])['cv'],stats(sham)['cv']],'sham_paired':statistics.median(sham),'sham_aggregate':aggregate['P2']/aggregate['P']}
            for k,v in extras.items():
                left=saved[k]if isinstance(v,list)else[saved[k]];right=v if isinstance(v,list)else[v]
                assert len(left)==len(right)and all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)for a,b in zip(left,right)),k
        key=(cohort['case'],cohort['boundary']);groups.setdefault(key,[]).append((cohort['attempt'],derived))
        raw_groups.setdefault(key,[]).append(cohort)
        height=max(int(cohort['case'].split('-')[1][1:]),int(cohort['case'].split('-')[2][1:]))
        assert cohort['prefix_events']==height*height+height+1
        details.append({'case':cohort['case'],'boundary':cohort['boundary'],'attempt':cohort['attempt'],'derived':derived,'arms':{k:stats(v)for k,v in per.items()},'ratios':{k:stats(v)for k,v in ratios.items()},'sham':stats(sham),'aggregate_ns_per_proof':aggregate})
    selected=[]
    for key,attempts in groups.items():
        attempts.sort();assert [a for a,_ in attempts] in [[0],[0,1]]
        if len(attempts)==2:assert not attempts[0][1]['clean']
        if not attempts[0][1]['clean']:assert len(attempts)==2
        if len(attempts)==2:
            first,second=sorted(raw_groups[key],key=lambda c:c['attempt'])
            for arm in ['P','P2','D','C']:
                a=[x for b in first['samples']for x in b if x['arm']==arm]
                b=[x for block in second['samples']for x in block if x['arm']==arm]
                minimum=min(x['elapsed_ns']for x in a)
                expected=a[0]['repetitions']*(math.ceil(30_000_000/minimum)if minimum<20_000_000 else 1)
                assert b[0]['repetitions']==expected
        selected.append((key,attempts[-1][1]))
    expected_cases={f'{p}-l{h}-t{t}' for p,t in [('E',22),('R',21),('Q',20)] for h in [20,22]}
    assert set(groups)=={(c,b) for c in expected_cases for b in ['whole','prefix-replay+bottom']}
    whole=[v for (case,boundary),v in selected if boundary=='whole'];bottom=[v for (case,boundary),v in selected if boundary=='prefix-replay+bottom']
    assert len(whole)==len(bottom)==6
    g=min(v[k] for v in whole for k in ['paired_saving_P','paired_saving_D','aggregate_saving_P','aggregate_saving_D'])
    h=min(v[k] for v in bottom for k in ['paired_saving_P','aggregate_saving_P'])
    clean=all(v['clean'] for _,v in selected)
    verdict='validated' if clean and g>=.05 and h>=.30 else 'refuted' if clean and(g<.02 or h<.15) else 'inconclusive'
    assert raw['verdict']==verdict
    for key,value in [('G2',g),('H2',h)]:assert math.isclose(raw[key],value,rel_tol=1e-12,abs_tol=1e-12),key
    assert raw['all_noise_clean']==clean
    assert len(raw['exact'])==6 and {x['case'] for x in raw['exact']}==expected_cases
    for x in raw['exact']:
        assert x['before_after_all_equal']and x['independent']and len(x['digest'])==64
        for when in ['before','after']:
            assert [a['arm']for a in x[when]]==['P','P2','D','C']
            assert all(a['independent']and a['digest']==x['digest']for a in x[when])
    return {'verdict':verdict,'G2':g,'H2':h,'all_noise_clean':clean,'cohorts':len(raw['cohorts']),'cohort_statistics':details}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--results',type=Path,default=HERE/'results');args=p.parse_args()
    results=args.results
    out={'source_pins':frozen(),'bound_records':bounds(),'arithmetic':arithmetic(results),'grids':grids(results)}
    check=json.loads((results/'check.json').read_text());assert check['status']=='pass' and len(check['cases'])==33
    for case in check['cases']:
        assert len(case['digests'])==4 and len({x['digest'] for x in case['digests']})==1
        assert all(x['independent'] for x in case['digests'])
    extra=json.loads((results/'extras.json').read_text());assert extra['status']=='pass' and len(extra['controls'])==5
    assert len(check['diagnostics'])==30 and all(d['exact'] for d in check['diagnostics'])
    assert json.loads((results/'multiple.json').read_text())['independent']
    assert json.loads((results/'field-fallback.json').read_text())['independent']
    scripts=json.loads((results/'degenerate.json').read_text())['cases'];assert len(scripts)==4
    for case in scripts:
        assert len(case['arms'])==3
        for arm in case['arms']:
            assert arm['independent_arithmetic'] and arm['direct_round_evaluations']==30
            challenges={};messages={};index=0
            for event in arm['events']:
                if 'challenge'in event:
                    expected=[0,0]if case['mode']==0 else[1,0]if case['mode']==1 else[index%2,0]if case['mode']==2 else[[0,0],[1,0],[0,1],[P-1,0]][index%4]
                    assert event['challenge']==expected
                    challenges[event['label']]=event['challenge'];index+=1
                else:messages[event['label']]=event['message']
            rec=arm['record']
            labels=['logup/alpha','logup/beta','logup/roots']
            for i,level in enumerate(rec['levels']):
                labels.extend([f'tree/{i}/gamma',f'tree/{i}/lambda'])
                for j in range(i):labels.extend([f'tree/{i}/round/{j}',f'tree/{i}/r/{j}'])
                labels.extend([f'tree/{i}/evals',f'tree/{i}/line'])
            labels.append('logup/values')
            assert [e['label']for e in arm['events']]==labels and len(set(labels))==len(labels)
            assert rec['alpha']==challenges['logup/alpha']and rec['beta']==challenges['logup/beta']
            assert rec['roots']==messages['logup/roots']and rec['values']==messages['logup/values']
            for i,level in enumerate(rec['levels']):
                for key,label in [('gamma','gamma'),('lambda','lambda'),('rho','line')]:assert level[key]==challenges[f'tree/{i}/{label}']
                assert level['proof']['final_evals']==messages[f'tree/{i}/evals']
                for j,msg in enumerate(level['proof']['rounds']):
                    assert msg==messages[f'tree/{i}/round/{j}']
                    assert level['point'][j]==challenges[f'tree/{i}/r/{j}']
    assert len(json.loads((results/'rejections.json').read_text())['rejections'])==7
    if args.audit:out['timing']=audit(json.loads((results/'run.json').read_text()))
    (results/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
