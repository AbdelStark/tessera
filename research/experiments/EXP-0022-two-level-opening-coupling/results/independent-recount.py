#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Read saved EXP22 evidence only. No audit import/execution, git or network.

This recount has NOT been executed. See companion instructions for evidence
limits. It reconstructs finite domains and balanced algebra independently.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

P = (1 << 64) - (1 << 32) + 1
FROZEN = "6676608fac008fc20b9bb7594dbe6463acb54f5bdc0fe2f9a1cb4a20fb22fd24"
ZERO, ONE, U = (0, 0), (1, 0), (0, 1)
ALPHAS = (ZERO, ONE, (2, 0), U)
NOTE = "research/ideas/IDEA-0035-combined-row-second-commitment.md"


def field(value):
    if type(value) is int:
        return (value % P, 0)
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError("noncanonical Fp2 JSON shape: " + repr(value)[:150])
    if any(x < 0 or x >= P for x in value):
        raise ValueError("noncanonical Fp2 component")
    return tuple(value)


def add(x, y):
    return ((x[0]+y[0]) % P, (x[1]+y[1]) % P)


def neg(x):
    return ((-x[0]) % P, (-x[1]) % P)


def sub(x, y):
    return add(x, neg(y))


def mul(x, y):
    # Standalone schoolbook oracle; never delegates to saved executable code.
    return ((x[0]*y[0]+7*x[1]*y[1]) % P,
            (x[0]*y[1]+x[1]*y[0]) % P)


def power(x, n):
    result = ONE
    for unused in range(n):
        result = mul(result, x)
    return result


def total(values):
    result = ZERO
    for value in values:
        result = add(result, value)
    return result


def scalar_product(left, right):
    if len(left) != len(right):
        raise ValueError("dot length")
    return total(mul(x,y) for x,y in zip(left,right))


def polynomial(coefficients, x):
    # Power-basis direct sum, independent of the audit's Horner implementation.
    return total(mul(c,power(x,i)) for i,c in enumerate(coefficients))


def equality(point, index):
    value = ONE
    for bit,r in enumerate(point):
        value = mul(value, r if (index >> bit)&1 else sub(ONE,r))
    return value


def multilinear(table, point):
    if len(table) != 1 << len(point):
        raise ValueError("MLE shape")
    return total(mul(v,equality(point,i)) for i,v in enumerate(table))


def next_power(n):
    return 1 << (max(n,1)-1).bit_length()


def nbits(n):
    if n < 1 or n & (n-1):
        raise ValueError("noncube length")
    return n.bit_length()-1


def point_catalog(cat, length, offset=0):
    if cat == 0:return (ZERO,)*length
    if cat == 1:return (ONE,)*length
    if cat == 2:return tuple(((bit+offset)%2,0) for bit in range(length))
    choices=(((P+1)//2,0),U,(1,1),(2,P-1))
    return tuple(choices[(bit+offset)%4] for bit in range(length))


def point_for_column(k, j):
    n=2*k
    omega=pow(7,(P-1)//n,P)
    return ((7*pow(omega,j,P)) % P,0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def tuple_key(value):
    if isinstance(value,list):return tuple(tuple_key(x) for x in value)
    if isinstance(value,dict):return tuple(sorted((k,tuple_key(v)) for k,v in value.items()))
    return value


def descriptor_key(d, family, catalog):
    return (d["k1"],d["D"],d["G"],d["mode"],d["P1"],family,catalog,d["k2"],d["P2"])


def expected_domains(mode):
    shapes=list(itertools.product((1,2,4),(1,3),(0,1,2),("plain","blinded-algebra"),(0,1)))
    if mode == "full":
        originals={shape+(family,cat) for shape in shapes for family in range(4) for cat in range(4)}
    else:
        originals={
            (1,1,0,"plain",0,0,0),
            (1,3,2,"blinded-algebra",1,3,3),
            (4,3,2,"blinded-algebra",1,3,3),
            (2,3,1,"plain",1,1,1),
        }
    cases={original+(k2,p2) for original in originals for k2 in (2,4) for p2 in (0,1)}
    return originals,cases


class Recount:
    def __init__(self):
        self.checks=0
        self.failures=[]
        self.gaps=[]
        self.limits=[]
        self.counts=Counter()
        self.hashes={}
        self.balanced=[]

    def check(self, condition, name, detail=None):
        self.checks+=1
        if not condition and len(self.failures)<200:
            self.failures.append({"name":name,"detail":detail})
        return condition

    def gap(self, name, detail=None, required=False):
        self.gaps.append({"name":name,"detail":detail,"required":required})


def canonical_equations(row, check):
    """Rederive all targets from saved source, not alleged Y/diagnostic residuals."""
    source=row["source"]
    identity=source["identity"]
    k,D,G,mode,p1,k2,p2=identity[:7]
    m=G+1
    cat=source["original_descriptor"][7]
    gamma=(ZERO,ONE,(2,0),(1,1))[cat]
    data=next_power(D)
    allocated=next_power(data+(2*m if mode=="blinded-algebra" else 0))
    physical=tuple(range(D))+(tuple(range(data,data+2*m)) if mode=="blinded-algebra" else ())
    check(tuple(source["physical"])==physical,"balanced/physical-map")
    draw=((0,),(0,0),(2*k-1,0,2*k-1),tuple(range(2*k)))[cat]
    check(tuple(source["draw"])==draw,"balanced/query-draw")
    check(tuple(j for j,_ in source["columns"])==tuple(sorted(set(draw))),"balanced/distinct-column-order")
    coeffs=[]
    for group in range(m):
        weights=[]
        for r in range(allocated):
            v=ZERO
            if r < data:
                if group<G:
                    v=total(mul(power(gamma,t+1),equality(point_catalog(cat,nbits(data),2*group+t),r)) for t in range(2))
                else:v=equality(point_catalog(cat,nbits(data)),r)
            if mode=="blinded-algebra":
                if r==data+2*group:v=ONE
                if r==data+2*group+1:v=U
            weights.append(v)
        coeffs.append(weights)
    equations=[]
    for group in range(G):
        point=point_catalog(cat,nbits(k),group+1)
        target=add(field(source["q"][group]),total(mul(power(gamma,t+1),field(v)) for t,v in enumerate(source["values"][group])))
        equations.append(("evaluation",group,-1,point,target))
    for j,symbols in source["columns"]:
        check(len(symbols)==len(physical),"balanced/column-symbol-length")
        x=point_for_column(k,j)
        check(field(source["column_points"][j])==x,"balanced/natural-column-point")
        for group in range(m):
            value=total(mul(coeffs[group][r],field(v)) for r,v in zip(physical,symbols))
            mask=[field(v) for v in source["M"][group]]
            check(len(mask)==p1,"balanced/combined-mask-length")
            target=sub(value,mul(power(x,k),polynomial(mask,x)))
            equations.append(("column",group,j,x,target))
    check(len(equations)==len(row["equations"]),"balanced/equation-count")
    for ordinal,(expected,saved) in enumerate(zip(equations,row["equations"])):
        kind,group,index,parameter,target=expected
        check((saved["kind"],saved["group"],saved["index"],saved["ordinal"])==(kind,group,index,ordinal),"balanced/canonical-equation-ordinal")
        check(tuple_key(saved["identity"])==tuple_key(identity),"balanced/equation-descriptor")
        saved_parameter=tuple(map(field,saved["parameter"])) if kind=="evaluation" else field(saved["parameter"])
        check(saved_parameter==parameter,"balanced/equation-parameter")
        check(field(saved["target"])==target,"balanced/target-from-source")
    return equations,(k,D,G,mode,p1,k2,p2),cat


def vector_for_equation(eq,k):
    kind,group,index,parameter,target=eq
    del group,index,target
    return tuple(equality(parameter,i) if kind=="evaluation" else power(parameter,i) for i in range(k))


def round_polynomial(a,b):
    coefficients=[ZERO,ZERO,ZERO]
    for index in range(0,len(a),2):
        ax=(a[index],sub(a[index+1],a[index]))
        bx=(b[index],sub(b[index+1],b[index]))
        # Generic degree-one polynomial multiplication, independent of audit helpers.
        for ai,av in enumerate(ax):
            for bi,bv in enumerate(bx):
                coefficients[ai+bi]=add(coefficients[ai+bi],mul(av,bv))
    return tuple(coefficients)


def replay_reduction(record,equations,Urows,shape,alpha,check):
    k,D,G,mode,p1,k2,p2=shape
    del D,mode,p1,p2
    cube=next_power(G+1)
    a=[v for values in Urows for v in values]+[ZERO]*((cube-(G+1))*k)
    b=[ZERO]*(cube*k)
    target=ZERO
    for ordinal,eq in enumerate(equations):
        lam=power(alpha,ordinal)
        target=add(target,mul(lam,eq[4]))
        for i,w in enumerate(vector_for_equation(eq,k)):
            at=eq[1]*k+i
            b[at]=add(b[at],mul(lam,w))
    proof=record["proof"]
    point=tuple(map(field,proof["point"]))
    check(len(point)==nbits(cube*k),"balanced/proof-point-length")
    check(tuple(map(field,proof["terminal_point"]))==point,"balanced/terminal-point")
    check(len(proof["rounds"])==len(point),"balanced/round-count")
    incoming=target
    predicted="accept"
    for index,(coefficients,r) in enumerate(zip(proof["rounds"],point)):
        saved=tuple(map(field,coefficients))
        expected=round_polynomial(a,b)
        check(len(saved)==3 and saved==expected,"balanced/direct-round-polynomial")
        if add(polynomial(saved,ZERO),polynomial(saved,ONE))!=incoming and predicted=="accept":
            predicted="batch/initial" if index==0 else "batch/round-identity"
        incoming=polynomial(saved,r)
        a=[add(a[i],mul(sub(a[i+1],a[i]),r)) for i in range(0,len(a),2)]
        b=[add(b[i],mul(sub(b[i+1],b[i]),r)) for i in range(0,len(b),2)]
    check(len(a)==len(b)==1,"balanced/fold-final-length")
    check(field(proof["terminal_u"])==a[0],"balanced/actual-committed-terminal-U")
    if incoming!=mul(a[0],b[0]) and predicted=="accept":predicted="batch/final"
    # Independently reconstruct V and verify both exact terminal claims. This
    # checks saved values, not any cryptographic opening or registry provenance.
    padded=Urows+tuple((ZERO,)*k for unused in range(cube-(G+1)))
    tensor=[(v[lane],0) for row in padded for lane in range(2) for v in row]
    tensor += [ZERO]*(max(len(tensor),k2)-len(tensor))
    si,sg=point[:nbits(k)],point[nbits(k):]
    extra=nbits(len(tensor))-nbits(2*cube*k)
    claims=record["claims"]
    check(len(claims)==2,"balanced/two-lane-claims")
    for lane,claim in enumerate(claims):
        expected=si+((lane,0),)+sg+(ZERO,)*extra
        check(claim["lane"]==lane and tuple(map(field,claim["point"]))==expected,"balanced/lane-point")
        check(field(claim["value"])==multilinear(tensor,expected),"balanced/exact-lane-value")
    if len(claims)==2:
        check(add(field(claims[0]["value"]),mul(U,field(claims[1]["value"])))==a[0],"balanced/lane-reconstruction")
    opening=record.get("opening")
    check(isinstance(opening,dict) and opening.get("consumed") is True,"balanced/completed-level2-opening")
    if isinstance(opening,dict):
        check(opening.get("claims")==claims,"balanced/level2-claim-inventory")
        for claim in claims:
            check(claim["root"]==opening["root"] and claim["descriptor"]==opening["descriptor"],"balanced/one-C_U")
    return predicted


def recount_balanced(row, audit):
    check=audit.check
    equations,shape,cat=canonical_equations(row,check)
    actual=tuple(tuple(map(field,values)) for values in row["committed_U"])
    k,D,G,mode,p1,k2,p2=shape
    check(len(actual)==G+1 and all(len(values)==k for values in actual),"balanced/committed-U-shape")
    residual=tuple(sub(scalar_product(actual[eq[1]],vector_for_equation(eq,k)),eq[4]) for eq in equations)
    check(residual==tuple(map(field,row["residual_coefficients"])),"balanced/independent-residual-coefficients")
    check(len(row["runs"])==4,"balanced/four-runs")
    seen=[];roots=0
    for run in row["runs"]:
        alpha=field(run["alpha"]);seen.append(alpha)
        Dalpha=polynomial(residual,alpha)
        check(Dalpha==field(run["residual_at_alpha"]),"balanced/independent-polynomial-at-alpha")
        record=run["full_record"]
        check(record["equations"]==row["equations"],"balanced/same-full-equation-list")
        expected_point=point_catalog(cat,nbits(next_power(G+1)*k),1)
        check(tuple(map(field,record["proof"]["point"]))==expected_point,"balanced/catalog-challenge-point")
        predicted=replay_reduction(record,equations,actual,shape,alpha,check)
        wanted="accept" if Dalpha==ZERO else ("batch/initial" if record["proof"]["rounds"] else "batch/final")
        check(predicted==wanted,"balanced/replayed-vs-polynomial-classification")
        check(run["actual"]==wanted and run["expected"]==wanted,"balanced/full-candidate-result")
        roots += Dalpha==ZERO
    check(tuple(seen)==ALPHAS,"balanced/exact-four-alpha-order")
    audit.balanced.append({"label":row["label"],"equations":len(equations),"nonzero_residual":any(v!=ZERO for v in residual),"roots":roots,"nonroots":4-roots})
    audit.counts["balanced_cases"]+=1
    audit.counts["balanced_root_acceptances"]+=roots
    audit.counts["balanced_nonroot_rejections"]+=4-roots


def replay_level2(record,tensor,shape,cat,check):
    k,D,G,mode,p1,width,p2=shape
    del D,mode,p1
    opening=record["opening"]
    claims=record["claims"]
    column_bits=nbits(width)
    row_count=len(tensor)//width
    check(tuple(opening["split"])==(column_bits,nbits(row_count)),"balanced/l2-public-split")
    keys=[];partitions=[]
    for claim in claims:
        key=tuple(map(field,claim["point"][:column_bits]))
        if key not in keys:
            keys.append(key);partitions.append([])
        partitions[keys.index(key)].append(claim)
    coefficients=[]
    for group in partitions:
        weights=[]
        for row in range(row_count):
            weights.append(total(mul(power((2,1),t+1),equality(tuple(map(field,claim["point"][column_bits:])),row))
                                 for t,claim in enumerate(group)))
        coefficients.append(weights)
    coefficients.append([equality(point_catalog(cat,nbits(row_count),2),row) for row in range(row_count)])
    check(len(opening["combined"])==len(coefficients),"balanced/l2-all-combined-including-proximity")
    check(len(opening["masks"])==len(coefficients),"balanced/l2-all-masks-including-proximity")
    combined=[tuple(map(field,row)) for row in opening["combined"]]
    masks=[tuple(map(field,row)) for row in opening["masks"]]
    for group,weights in enumerate(coefficients):
        expected=tuple(total(mul(weights[row],tensor[row*width+i]) for row in range(row_count)) for i in range(width))
        check(combined[group]==expected,"balanced/l2-complete-tensor-combined")
        check(len(masks[group])==p2,"balanced/l2-mask-width")
    for group,key in enumerate(keys):
        target=total(mul(power((2,1),t+1),field(claim["value"])) for t,claim in enumerate(partitions[group]))
        check(multilinear(combined[group],key)==target,"balanced/l2-evaluation-group")
    physical=tuple(row for row in range(row_count) if row*width < 2*(G+1)*k)
    check(tuple(opening["physical"])==physical,"balanced/l2-structural-row-map")
    draw=((0,),(0,0),(2*width-1,0,2*width-1),tuple(range(2*width)))[cat]
    check(tuple(opening["draw"])==draw,"balanced/l2-query-draw")
    check(tuple(j for j,_ in opening["columns"])==tuple(sorted(set(draw))),"balanced/l2-distinct-columns")
    for j,symbols in opening["columns"]:
        check(len(symbols)==len(physical),"balanced/l2-column-shape")
        x=point_for_column(width,j)
        for group,weights in enumerate(coefficients):
            left=total(mul(weights[row],field(symbol)) for row,symbol in zip(physical,symbols))
            right=add(polynomial(combined[group],x),mul(power(x,width),polynomial(masks[group],x)))
            check(left==right,"balanced/l2-column-equation")


def provenance(attempt,repo,audit,expected_source):
    summary=read_json(attempt/"summary.json")
    invocation=read_json(attempt/"invocation.json")
    saved=read_json(attempt/"provenance.json")
    pins=read_json(attempt/"source-pins.json")
    source_hash=sha(attempt/"audit.source.py")
    audit.hashes.update({name:sha(attempt/name) for name in (
        "audit.source.py","README.preregistered.txt","source-pins.json","events.jsonl","summary.json","controls.json",
    )})
    audit.check(source_hash==summary["source_sha256"]==invocation["source_sha256"]==saved["source_sha256"],"provenance/source-hash-agreement")
    if expected_source is not None:
        audit.check(source_hash==expected_source,"provenance/parent-expected-source")
    audit.check(sha(attempt/"README.preregistered.txt")==FROZEN,"provenance/frozen-bytes")
    audit.check(summary["frozen_sha256"]==saved["freeze_sha256"]==pins["preregistration_sha256"]==FROZEN,"provenance/freeze-hash-agreement")
    audit.check(saved["registration_commit"]==summary["registration_commit"]=="f344d76","provenance/registration-label")
    audit.check(saved["blocking_violations"]==[],"provenance/no-recorded-blockers")
    expected=pins["sources_sha256"]
    audit.check(len(expected)==12,"provenance/twelve-pins")
    reported={row["path"]:row for row in saved["pins"]}
    audit.check(len(saved["pins"])==len(reported)==12 and set(reported)==set(expected),"provenance/pin-inventory")
    for relative,wanted in expected.items():
        row=reported[relative]
        audit.check(row["expected_sha256"]==wanted,"provenance/expected-pin",relative)
        if relative==NOTE and row["actual_sha256"]!=wanted:
            audit.check(row["status"]=="idea35-note-only-drift","provenance/only-note-exception")
            audit.check(sha(attempt/"IDEA-0035.preregistered.txt")==wanted,"provenance/historical-note-bytes")
            audit.check(sha(attempt/"IDEA-0035.current.txt")==row["actual_sha256"],"provenance/run-note-bytes")
        else:
            audit.check(row["actual_sha256"]==wanted and row["status"]=="match","provenance/recorded-pin-match",relative)
        if repo is not None:
            path=repo/relative
            audit.check(path.is_file(),"provenance/live-source-present",relative)
            if path.is_file() and relative!=NOTE:
                audit.check(sha(path)==wanted,"provenance/live-normative/source-pin",relative)
    if repo is not None:
        trusted=repo/"research/experiments/EXP-0022-two-level-opening-coupling/results/source-pins.json"
        audit.check(trusted.read_bytes()==(attempt/"source-pins.json").read_bytes(),"provenance/trusted-manifest-bytes")
    else:
        audit.gap("live-pins-not-checked","Supply --repo for read-only current pinned-byte comparisons.")
    audit.limits.append("No git is used: committed-freeze membership relies on parent's prior commit verification; the saved freeze bytes are independently hashed here.")
    return summary,saved


def resource_checks(summary,audit):
    check=audit.check
    caps={"judgments":100_000_000,"metered_ops":3_000_000_000,"process_rss_bytes":256*1024*1024,"seconds_watchdog":900}
    check(summary["limits"]==caps,"caps/exact-registered-limits")
    check(type(summary["judgments"]) is int and 0<=summary["judgments"]<=caps["judgments"],"caps/reported-judgments")
    check(type(summary["metered_algebraic_operations"]) is int and 0<=summary["metered_algebraic_operations"]<=caps["metered_ops"],"caps/reported-metered-ops")
    check(0<=summary["maxima"].get("rss_bytes",caps["process_rss_bytes"]+1)<=caps["process_rss_bytes"],"caps/reported-RSS")
    check(summary["physical_judgments"]+summary["memoized_logical_judgments"]==summary["judgments"],"caps/physical-plus-cached")
    arm_sum=sum(v for k,v in summary["counts"].items() if k.startswith("judgments/"))
    check(arm_sum==summary["judgments"],"caps/arm-judgment-sum")
    check(summary["performance_claim"] is False and summary["evidence_level"]=="E0","caps/no-performance-claim")
    audit.limits.append("Operation/judgment totals, RSS and watchdog are meter reports with arithmetic consistency checks, not independently reconstructed primitive instructions or measured wall-time evidence. No operation-level trace exists.")


def recount(attempt,repo,mode,expected_source):
    audit=Recount()
    summary,saved_provenance=provenance(attempt,repo,audit,expected_source)
    expected_originals,expected_cases=expected_domains(mode)
    originals=Counter();original_finishes=Counter();starts=Counter();finishes=Counter();shapes=Counter()
    case_by_digest={};orig_digest={};control_rows=[];names=Counter();reductions=defaultdict(list)
    terminal_count=0;balanced_by_label={};unknown=Counter();stream_provenance=[]
    for line_number,line in enumerate((attempt/"events.jsonl").open(),1):
        try:
            event=json.loads(line);kind=event["kind"];row=event["data"]
        except (ValueError,KeyError) as error:
            audit.check(False,"stream/malformed-line",{"line":line_number,"error":str(error)});continue
        audit.counts["events/"+kind]+=1
        try:
            if kind=="provenance":
                stream_provenance.append(row)
            elif kind=="original-fixture":
                key=tuple(row["shape"])+(row["family"],row["catalog"])
                originals[key]+=1;orig_digest[key]=row["input_digest"]
            elif kind=="original-fixture-complete":
                key=tuple(row["shape"])+(row["family"],row["catalog"])
                original_finishes[key]+=1
                audit.check(row["input_digest"]==orig_digest.get(key),"domain/original-digest-continuity")
            elif kind=="two-level-fixture-start":
                key=descriptor_key(row["descriptor"],row["family"],row["catalog"])
                starts[key]+=1
                case_by_digest[row["input_digest"]]=key
                audit.check(key[:-2] in originals,"domain/start-after-original")
                audit.check(row["original_digest"]==orig_digest.get(key[:-2]),"domain/case-original-digest")
            elif kind=="two-level-fixture-complete":
                key=descriptor_key(row["descriptor"],row["family"],row["catalog"])
                finishes[key]+=1
                audit.check(starts[key]==1,"domain/one-start-before-complete",key)
                audit.check(case_by_digest.get(row["input_digest"])==key,"domain/case-digest-continuity")
                audit.check(tuple(map(field,row["alpha_catalog"]))==ALPHAS,"domain/declared-alpha-catalog")
                audit.check(row["reduction_count"]==4 and row["honest_acceptances"]==4,"domain/declared-four-reductions-acceptances")
                reference=row["independent_reference"]
                audit.check(reference.get("checked") is True,"domain/reference-completion-record")
                audit.counts["reference_completion_records"]+=reference.get("checked") is True
                audit.counts["fixture_declared_reductions"]+=row["reduction_count"]
                audit.counts["fixture_declared_acceptances"]+=row["honest_acceptances"]
            elif kind=="shape-complete":
                shapes[tuple(row["shape"])]+=1
            elif kind=="control":
                names[row["name"]]+=1
                audit.check(row["positive"]=="accept","controls/positive",row["name"])
                audit.check(row["actual"]==row["expected"] and row["expected"]!="accept","controls/exact-rejection",row["name"])
                if row.get("mutant") is not None:
                    audit.check(row["mutant"]=="accept","controls/sensitive-mutant",row["name"])
                    audit.counts["recorded_sensitive_mutants"]+=1
                control_rows.append({k:row.get(k) for k in ("name","positive","expected","actual","mutant")})
            elif kind=="balanced-control":
                recount_balanced(row,audit)
                audit.check(row["label"] not in balanced_by_label,"balanced/unique-label")
                balanced_by_label[row["label"]]=row
                # Also replay explicit level2 equations from actual committed V.
                shape=tuple(row["source"]["identity"][:7])
                k,D,G,unused,p1,width,p2=shape
                cube=next_power(G+1)
                actual=tuple(tuple(map(field,values)) for values in row["committed_U"])
                padded=actual+tuple((ZERO,)*k for unused in range(cube-(G+1)))
                tensor=[(v[lane],0) for values in padded for lane in range(2) for v in values]
                tensor += [ZERO]*(max(len(tensor),width)-len(tensor))
                for run in row["runs"]:
                    replay_level2(run["full_record"],tensor,shape,row["source"]["original_descriptor"][7],audit.check)
            elif kind=="naive-unweighted-mutant":
                prior=balanced_by_label.get(row["name"])
                audit.check(prior is not None,"balanced/unweighted-associated-case")
                if prior is not None:
                    audit.check(field(row["requested_alpha"])==(2,0),"balanced/unweighted-requested-alpha")
                    equations,shape,cat=canonical_equations(prior,audit.check)
                    actual=tuple(tuple(map(field,values)) for values in prior["committed_U"])
                    got=replay_reduction(row["record"],equations,actual,shape,ONE,audit.check)
                    audit.check(got=="accept","balanced/unweighted-mutant-replay")
                    audit.check(any(field(run["alpha"])==ONE and run["actual"]=="accept" for run in prior["runs"]),"balanced/unweighted-alpha1-actual-acceptance")
                    audit.counts["unweighted_mutant_records"]+=1
            elif kind in ("reduction-result","reduction","honest-reduction"):
                # Suggested lightweight final-source event schema. Never infer
                # an actual outcome merely from fixture totals/alpha_catalog.
                if row.get("label","honest")=="honest":
                    case_id=row.get("case_identity",row.get("input_digest",row.get("case")))
                    key=descriptor_key(row["descriptor"],row["family"],row["catalog"]) if "descriptor" in row else case_by_digest.get(case_id)
                    audit.check(key is not None,"domain/reduction-associated-case")
                    if key is not None:
                        reductions[key].append(row)
                        audit.check(field(row["alpha"]) in ALPHAS,"domain/raw-alpha")
                        audit.check(row["actual"]==row["expected"]=="accept","domain/raw-honest-acceptance")
                        if "D" in row:audit.check(field(row["D"])==ZERO,"domain/raw-zero-residual")
                        if "residual_at_alpha" in row:audit.check(field(row["residual_at_alpha"])==ZERO,"domain/raw-zero-residual")
            elif kind=="attempt-terminal":
                terminal_count+=1
                audit.check(row==summary,"stream/terminal-summary-exact-agreement")
            elif kind=="failure":
                audit.check(False,"stream/recorded-failure",row)
            elif kind not in ("attempt-start","original-gamma-degeneracy","diagnostic-corpus-complete","diagnostic-corpus-not-run"):
                unknown[kind]+=1
        except (ValueError,TypeError,KeyError,IndexError) as error:
            audit.check(False,"stream/unhandled-record-schema",{"line":line_number,"kind":kind,"error":str(error)})
    audit.check(terminal_count==1,"stream/one-terminal-record")
    audit.check(stream_provenance==[saved_provenance],"stream/provenance-file-agreement")
    audit.check(set(originals)==expected_originals and all(n==1 for n in originals.values()),"domain/exact-original-Cartesian-product",{"observed":len(originals),"expected":len(expected_originals),"missing":list(expected_originals-set(originals))[:5]})
    audit.check(original_finishes==originals,"domain/original-start-finish-multiset")
    audit.check(set(starts)==expected_cases and all(n==1 for n in starts.values()),"domain/exact-case-Cartesian-product",{"observed":len(starts),"expected":len(expected_cases),"missing":list(expected_cases-set(starts))[:5]})
    audit.check(finishes==starts,"domain/case-start-finish-multiset")
    if mode=="full":
        expected_shapes={key[:5] for key in expected_originals}
        audit.check(set(shapes)==expected_shapes and all(n==1 for n in shapes.values()),"domain/all-shape-completion-records")
    if reductions:
        audit.check(set(reductions)==expected_cases,"domain/raw-reductions-cover-all-cases")
        for key,rows in reductions.items():
            audit.check(tuple(field(row["alpha"]) for row in rows)==ALPHAS,"domain/raw-four-alphas-exactly-once",key)
        audit.counts["raw_honest_reduction_results"]=sum(map(len,reductions.values()))
    else:
        audit.gap("honest-per-alpha-records-absent","Fixture completion totals only assert four acceptances; they do not independently expose four observed candidate results.",required=True)
    audit.check(audit.counts["reference_completion_records"]==len(expected_cases),"domain/independent-reference-record-count")
    audit.check(audit.counts["fixture_declared_reductions"]==4*len(expected_cases),"domain/fixture-declared-reduction-sum")
    audit.check(audit.counts["fixture_declared_acceptances"]==4*len(expected_cases),"domain/fixture-declared-acceptance-sum")
    audit.check(all(n==1 for n in names.values()),"controls/unique-names")
    saved_controls=read_json(attempt/"controls.json")
    audit.check(saved_controls==control_rows,"controls/file-vs-stream-exact-records")
    audit.check(len(control_rows)==summary["controls_recorded"],"controls/summary-count-agreement")
    # These check the mandatory classes, not every possible semantic guard.
    required_prefixes=("descriptor/","inventory/drop/","original/","batch/","terminal/","ideal/","level2/","registry/","order/","blinding/","omission/false-source/","receipt/","binding/","weight/")
    for prefix in required_prefixes:
        audit.check(any(name.startswith(prefix) for name in names),"controls/required-family",prefix)
    audit.check({int(name.rsplit("/",1)[1]) for name in names if name.startswith("inventory/drop/")}==set(range(26)),"controls/all-26-equation-omissions")
    audit.check(audit.counts["balanced_cases"]>0,"balanced/nonempty-full-record-corpus")
    resource_checks(summary,audit)
    audit.limits.extend([
        "Recorded control outcomes are recounted, not regenerated by invoking their closures. Inputs and source review are needed to assess whether every intended check has a sensitive case.",
        "Honest level2 completion records establish event coverage. Full honest private tensors/row masks are not saved per fixture, so this script does not regenerate their algebra; balanced records permit direct committed-U/round/lane/level2 algebra replay.",
        "Ideal registry membership/root immutability and exact opening services are assumptions; saved records cannot prove cryptographic binding or replace a joint soundness/ZK argument.",
        "Balanced root/nonroot replay covers constructed honest-to-committed-U transcripts and finite catalogs, not arbitrary dishonest transcripts or all cryptographic bad events.",
    ])
    return {
        "attempt":str(attempt),"mode_expected_by_recounter":mode,
        "status":"failed" if audit.failures else ("incomplete-evidence" if any(g["required"] for g in audit.gaps) else "checked-records-pass"),
        "checks":audit.checks,"failures":audit.failures,"evidence_gaps":audit.gaps,"scope_limits":audit.limits,
        "event_counts":dict(audit.counts),"original_fixtures_recounted":len(originals),"two_level_cases_recounted":len(starts),
        "controls_recounted":len(control_rows),"control_mutants_with_saved_result":audit.counts["recorded_sensitive_mutants"],
        "balanced_cases":audit.balanced,"artifact_sha256":audit.hashes,"unknown_event_kinds":dict(unknown),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt",type=Path)
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--expect-mode",choices=("full","preflight"),default="full")
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--out",type=Path,required=True,help="new scratch JSON only; exclusive write")
    args=parser.parse_args()
    if not args.out.resolve().is_relative_to(Path("/tmp").resolve()):
        parser.error("--out must be a new file under /tmp; repository writes are forbidden")
    try:
        result=recount(args.attempt.resolve(),args.repo.resolve() if args.repo else None,args.expect_mode,args.expected_source_sha)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        result={"status":"incomplete-evidence","error":type(error).__name__,"message":str(error),"attempt":str(args.attempt)}
    with args.out.open("x") as handle:
        json.dump(result,handle,sort_keys=True,indent=2);handle.write("\n")
    print(json.dumps({"status":result["status"],"output":str(args.out)},sort_keys=True))
    return {"failed":1,"incomplete-evidence":2,"checked-records-pass":0}[result["status"]]


if __name__=="__main__":
    raise SystemExit(main())
