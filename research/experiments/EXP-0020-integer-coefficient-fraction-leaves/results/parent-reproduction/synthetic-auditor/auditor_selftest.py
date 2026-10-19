#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Synthetic timing-schema controls. These records are NOT benchmark evidence."""
from __future__ import annotations
import copy,importlib.util,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exp20audit',HERE/'check.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main():
    names=['P','P2','D','C'];tick=0;rows=[];exact=[]
    for p,t in [('E',22),('R',21),('Q',20)]:
        for h in[20,22]:
            case=f'{p}-l{h}-t{t}';exact.append({'case':case,'before_after_all_equal':True,'independent':True,'digest':'0'*64,'before':[{'arm':a,'digest':'0'*64,'independent':True}for a in names],'after':[{'arm':a,'digest':'0'*64,'independent':True}for a in names]})
            for boundary in['whole','prefix-replay+bottom']:
                blocks=[]
                for i in range(13):
                    block=[]
                    for j in range(4):
                        a=names[(i+j)%4];duration={'P':40,'P2':40,'D':35,'C':25}[a]*1_000_000
                        block.append({'arm':a,'start_ns':tick,'end_ns':tick+duration,'elapsed_ns':duration,'repetitions':1});tick+=duration+1
                    blocks.append(block)
                rows.append({'case':case,'boundary':boundary,'attempt':0,'prefix_events':max(h,t)**2+max(h,t)+1,'warmups':blocks[:3],'samples':blocks[3:]})
    raw={'fixture_only':True,'warning':'Synthetic schema/mutation control, not measured CPU evidence.','primary_v':2,'verdict':'validated','G2':1-25/35,'H2':1-25/40,'all_noise_clean':True,'cohorts':rows,'exact':exact}
    module.audit(raw)
    # A noisy first cohort is retained; a fresh complete second cohort is authoritative.
    retry=copy.deepcopy(raw);second=copy.deepcopy(retry['cohorts'][0]);second['attempt']=1
    retry['cohorts'].insert(1,second)
    retry['cohorts'][0]['samples'][0][0]['elapsed_ns']*=2
    cursor=0
    for c in retry['cohorts']:
        for b in c['warmups']+c['samples']:
            for x in b:
                x['start_ns']=cursor;x['end_ns']=cursor+x['elapsed_ns'];cursor=x['end_ns']+1
    module.audit(retry)
    controls=[]
    def kill(name,change):
        bad=copy.deepcopy(raw);change(bad)
        try:module.audit(bad)
        except(AssertionError,KeyError,ValueError):controls.append({'mutation':name,'outcome':'killed'})
        else:raise AssertionError(('auditor mutation survived',name))
    def summary_bad(x):
        c=x['cohorts'][0];c['summary']={'clean':True,'paired_saving_P':.9,'paired_saving_D':.9,'aggregate_saving_P':.9,'aggregate_saving_D':.9}
    kill('altered-saved-ratio',summary_bad)
    kill('altered-verdict',lambda x:x.update(verdict='refuted'))
    kill('altered-G2',lambda x:x.update(G2=.999))
    kill('missing-required-rerun',lambda x:x['cohorts'][0]['samples'][0][0].update(elapsed_ns=1,end_ns=x['cohorts'][0]['samples'][0][0]['start_ns']+1))
    kill('third-attempt',lambda x:x['cohorts'][0].update(attempt=2))
    kill('overlapping-cohorts',lambda x:x['cohorts'][1]['warmups'][0][0].update(start_ns=0,end_ns=40_000_000))
    kill('missing-shape',lambda x:x['cohorts'].pop())
    kill('wrong-order',lambda x:x['cohorts'][0]['warmups'][0].reverse())
    kill('short-sample',lambda x:x['cohorts'][0]['samples'][0][0].update(elapsed_ns=19_000_000,end_ns=x['cohorts'][0]['samples'][0][0]['start_ns']+19_000_000))
    out=HERE/'results';(out/'auditor-fixture.json').write_text(json.dumps(raw,indent=2)+'\n')
    (out/'auditor-selftest.json').write_text(json.dumps({'fixture_only':True,'noop':'pass','retained_noisy_first_then_fresh_authoritative_rerun':'pass','controls':controls},indent=2)+'\n')
    print('synthetic auditor noop passes;',len(controls),'corruptions rejected')
if __name__=='__main__':main()
