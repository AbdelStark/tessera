#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Compile each isolated source mutation; only a runtime/oracle failure kills it.

Shares dependency caches but uses a distinct package and binary name, preventing
stale original fingerprints from executing a mutant binary. Mutant outputs never overwrite canonical check records.
"""
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=Path('/tmp/tessera-exp20-parent-mutants-20261019T112818/mutations')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix='tessera-exp20-mutants-'))
    crate=scratch/'crate';crate.mkdir()
    shutil.copytree(HERE/'src',crate/'src')
    (crate/'Cargo.toml').write_text((HERE/'Cargo.toml').read_text().replace('../../../',str(ROOT)+'/').replace('name = "integer-fraction"','name = "integer-fraction-parent-mutant"'))
    (crate/'Cargo.lock').write_text((HERE/'Cargo.lock').read_text().replace('name = "integer-fraction"','name = "integer-fraction-parent-mutant"'))
    originals={p.name:p.read_text()for p in(HERE/'src').glob('*.rs')}
    mutants=[
        ('noop',None,None,None,'check'),
        ('table-sign','coeff.rs','weights[5] = -a;','weights[5] = a;','check'),
        ('mixed-channel','coeff.rs','c[3] = k0 * v1 + v0 * k1;','c[3] = k0 * v1;','check'),
        ('weighted-carry','arith.rs','((b >> 64) + (mid >> 64)) as u64','(b >> 64) as u64','check'),
        ('signed-correction','arith.rs','self.unsigned() - Fp::from_u64_reduced(self.0[2] >> 63)','self.unsigned()','check'),
        ('doubled-cadence','coeff.rs','pub const RESET: usize = 1024;','pub const RESET: usize = 2048;','check'),
        ('unit-tag','trees.rs','UNIT_TAG','LevelNumerators::Ext { even: &[], odd: &[] }','unit-tag-check'),
        ('lookup-padding','logup.rs','let pad_key = table.keys[0];','let pad_key = instance.lookups.keys[0];','padding-check'),
    ]
    records=[]
    env=dict(os.environ,CARGO_TARGET_DIR=str(HERE/'target'))
    for name,file,old,new,command in mutants:
        for f,s in originals.items():(crate/'src'/f).write_text(s)
        if file:
            s=originals[file]
            if old=='UNIT_TAG':
                anchor=s.index('let numerators = if let Some(r) = raw')
                pos=s.index('LevelNumerators::Unit',anchor)
                s=s[:pos]+new+s[pos+len('LevelNumerators::Unit'):]
            else:
                assert s.count(old)==1,(name,s.count(old))
                s=s.replace(old,new)
            (crate/'src'/file).write_text(s)
        results=scratch/name;results.mkdir();env['EXP20_RESULTS']=str(results)
        build=['cargo','build','--release','--locked','--manifest-path',str(crate/'Cargo.toml')]
        b=subprocess.run(build,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (OUT/(name+'-build.txt')).write_bytes(b.stdout)
        assert b.returncode==0,(name,'does not compile; not a caught mutant')
        cmd=[str(HERE/'target/release/integer-fraction-parent-mutant'),command]
        r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (OUT/(name+'-run.txt')).write_bytes(r.stdout)
        oracle=None
        if r.returncode==0 and command=='check':
            o=subprocess.run(['python3','-I',str(HERE/'check.py'),'--check','--results',str(results)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (OUT/(name+'-oracle.txt')).write_bytes(o.stdout);oracle=o.returncode
        if name=='noop':assert r.returncode==0 and oracle==0
        else:assert r.returncode!=0 or oracle not in (None,0),(name,'survived')
        if name=='unit-tag':assert b'FinalMismatch { round: 1 }' in r.stdout,r.stdout[-2000:]
        # Preserve compact arithmetic/grid evidence whenever reached; do not duplicate full proofs.
        for f in ['arithmetic.json','grids.json']:
            if (results/f).exists():shutil.copy2(results/f,OUT/(name+'-'+f))
        rec={'mutant':name,'source_file':file,'old':old,'new':new,'command':command,'compiled':True,'runtime_exit':r.returncode,'oracle_exit':oracle,'outcome':'control_pass'if name=='noop'else'killed','build_command':build,'execution_command':cmd,'run_log_sha256':sha(OUT/(name+'-run.txt')),'source_sha256':{f:sha(crate/'src'/f)for f in originals}}
        records.append(rec)
        (Path('/tmp/tessera-exp20-parent-mutants-20261019T112818/mutations.json')).write_text(json.dumps({'scratch':str(scratch),'records':records,'note':'Only compiled runtime/oracle failures count; target reused, original release binary must be rebuilt after this tool.'},indent=2)+'\n')
        print(name,rec['outcome'],r.returncode,oracle,flush=True)
    print('mutation gates pass',flush=True)
if __name__=='__main__':main()
