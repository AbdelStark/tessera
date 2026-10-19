#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Secondary parent reconstruction of the already frozen EXP20 timing rules."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent

def median(values):
    v=sorted(values)
    return (v[4]+v[5])/2

def cv(values):
    mean=math.fsum(values)/len(values)
    return math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))/mean

def main():
    raw=json.loads((HERE/'run.json').read_text())
    groups={}
    arms=['P','P2','D','C']
    for row in raw['cohorts']:
        assert len(row['samples'])==10 and len(row['warmups'])==3
        per={a:[] for a in arms}
        for i,block in enumerate(row['warmups']+row['samples']):
            assert [x['arm'] for x in block]==arms[i%4:]+arms[:i%4]
            for sample in block:
                assert sample['elapsed_ns']==sample['end_ns']-sample['start_ns']
                assert sample['repetitions']>0
        flat=[s for b in row['warmups']+row['samples'] for s in b]
        assert all(a['end_ns']<=b['start_ns'] for a,b in zip(flat,flat[1:]))
        for block in row['samples']:
            for sample in block:per[sample['arm']].append(sample['elapsed_ns']/sample['repetitions'])
        ratios={a:[c/p for c,p in zip(per['C'],per[a])] for a in ['P','D']}
        sham=[p2/p for p2,p in zip(per['P2'],per['P'])]
        aggregate={a:sum(s['elapsed_ns'] for b in row['samples'] for s in b if s['arm']==a)/sum(s['repetitions'] for b in row['samples'] for s in b if s['arm']==a) for a in arms}
        relevant=['P','D'] if row['boundary']=='whole' else ['P']
        clean=(all(cv(v)<=.03 for v in per.values()) and all(cv(ratios[a])<=.03 for a in relevant)
               and cv(sham)<=.03 and .98<=median(sham)<=1.02
               and .98<=aggregate['P2']/aggregate['P']<=1.02
               and all(s['elapsed_ns']>=20_000_000 for b in row['samples'] for s in b))
        derived={'clean':clean,'paired_saving_P':1-median(ratios['P']),'paired_saving_D':1-median(ratios['D']),
                 'aggregate_saving_P':1-aggregate['C']/aggregate['P'],'aggregate_saving_D':1-aggregate['C']/aggregate['D']}
        for key,value in derived.items():
            prior=row['summary'][key]
            assert prior==value if isinstance(value,bool) else math.isclose(prior,value,rel_tol=1e-12,abs_tol=1e-12)
        key=(row['case'],row['boundary']);groups.setdefault(key,[]).append((row['attempt'],derived))
    expected={f'{p}-l{h}-t{t}' for p,t in [('E',22),('R',21),('Q',20)] for h in [20,22]}
    assert set(groups)=={(c,b) for c in expected for b in ['whole','prefix-replay+bottom']}
    selected={}
    for key,cohorts in groups.items():
        cohorts.sort(key=lambda c:c[0])
        assert [c[0] for c in cohorts] in ([0],[0,1])
        assert len(cohorts)==1 if cohorts[0][1]['clean'] else len(cohorts)==2
        selected[key]=cohorts[-1][1]
    G=min(v[k] for (c,b),v in selected.items() if b=='whole' for k in ['paired_saving_P','paired_saving_D','aggregate_saving_P','aggregate_saving_D'])
    H=min(v[k] for (c,b),v in selected.items() if b=='prefix-replay+bottom' for k in ['paired_saving_P','aggregate_saving_P'])
    clean=all(v['clean'] for v in selected.values())
    verdict='validated' if clean and G>=.05 and H>=.30 else 'refuted' if clean and (G<.02 or H<.15) else 'inconclusive'
    assert verdict==raw['verdict']
    assert {e['case'] for e in raw['exact']}==expected
    assert all(e['before_after_all_equal'] and e['independent'] for e in raw['exact'])
    result={'kind':'secondary independent parent reconstruction; frozen criteria unchanged',
            'verdict':verdict,'G2':G,'H2':H,'all_noise_clean':clean,'cohorts':len(raw['cohorts']),
            'measured_arm_samples':len(raw['cohorts'])*40,'warmup_arm_samples':len(raw['cohorts'])*12,
            'unclean_authoritative_boundaries':[list(k) for k,v in selected.items() if not v['clean']],
            'unchanged_source':'measured source manifest and binary verified separately'}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
