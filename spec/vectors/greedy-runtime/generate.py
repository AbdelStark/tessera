#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent draft RFC-0053 runtime geometry and native descriptors.

The initial catalog/layout fixtures preceded both runtime implementations.

Uses the existing independent Python graph/range/advice oracles only. Tail
locations are allocated individually; no verifier/prover/runtime imports.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('advice_oracle', HERE.parent / 'advice/generate.py')
advice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advice)


def variant_vocab(graph, vocab):
    g = copy.deepcopy(graph)
    old = g['nodes'][0][2]['vocab']
    def shape(s):
        return [vocab if d == old else d for d in s]
    for inp in g['inputs']:
        inp[2] = shape(inp[2])
    g['inputs'][0][4] = vocab - 1
    for node in g['nodes']:
        for output in node[4]:
            output[1] = shape(output[1])
    g['nodes'][0][2]['vocab'] = vocab
    return g


def runtime(graph, prompt, ordinal, log_k, query_block, segments, vocab=None, custom=None):
    g = variant_vocab(graph, vocab) if vocab is not None else graph
    old = advice.model_layout(g, log_k, query_block, segments)
    n = g['n']
    assert 1 <= prompt <= n
    tables, choices = advice.catalog(g)
    original_count = len(tables)
    lookup_counts = [0] * len(tables)
    nodes = {node[0]: node for node in g['nodes']}
    tensors = {t[0]: t[2] for t in g['inputs']}
    tensors.update({t[0]: t[1] for node in g['nodes'] for t in node[4]})
    for choice in choices:
        node = nodes[choice['node']]
        for j, tid in enumerate(choice['tables']):
            tensor = node[4][1][0] if node[1] == 'SoftmaxRow' and j else node[3][0]
            lookup_counts[tid] += 1 << sum(advice.log2_up(d) for d in tensors[tensor])
    model_tiles = [t for t in old['tiles'] if t['key'][0] != 'table']
    cursor = model_tiles[-1]['rows'][1]
    prefix_end = cursor
    tail = []
    certificate = None
    outputs = n - prompt
    if outputs:
        logit = g['nodes'][-1][4][0][0]
        lo, hi = advice.ranges.analyze(g)['ranges'][logit]
        assert -(1 << 28) < lo <= hi < 1 << 28
        bits = max(1, (hi-lo).bit_length())
        kind = {'Range': {'bits': bits}}
        tid = next((t[0] for t in tables if t[1] == kind), len(tables))
        if tid == len(tables):
            tables.append([tid, kind, bits])
            lookup_counts.append(0)
        vocabulary = g['nodes'][0][2]['vocab']
        entries = 1 << (advice.log2_up(outputs) + advice.log2_up(vocabulary))
        lookup_counts[tid] += entries
        gid = len(g['nodes']) + ordinal
        certificate = {'id': gid, 'range': tid, 'bits': bits, 'entries': entries,
                       'logit_bounds': [lo, hi], 'vocab': vocabulary}
        permutation, rows, zero = advice.geometry([n], log_k)
        base = (cursor + rows-1) // rows * rows
        tail.append({'key': ['greedy.maximum', gid], 'segment': segments-1,
                     'rows': [base, base+rows],
                     'values': [[['greedy.maximum', gid], [n], permutation, base, rows, zero]]})
        cursor = base+rows
    for tid, kind, log_rows in tables:
        if not lookup_counts[tid]:
            continue
        if 'Requant' in kind or 'Silu' in kind:
            q = kind.get('Requant', kind.get('Silu'))
            length = len(q['r_by_class']) << q['b']
        else:
            length = 1 << log_rows
        permutation, rows, zero = advice.geometry([length], log_k)
        base = (cursor + rows-1) // rows * rows
        tail.append({'key': ['table', tid], 'segment': segments-1, 'rows': [base, base+rows],
                     'values': [[['multiplicity', tid], [length], permutation, base, rows, zero]]})
        cursor = base+rows
    assert cursor <= 1 << 32
    all_tiles = model_tiles+tail
    edges = [0] + [next((t['rows'][0] for t in all_tiles if t['segment'] >= j), cursor)
                    for j in range(1, segments)] + [cursor]
    vocabulary = g['nodes'][0][2]['vocab']
    nt, nv = advice.log2_up(n), advice.log2_up(vocabulary)
    total = nt + nv
    h, token = g['nodes'][0][4][0][0], g['nodes'][0][3][0]
    lm, logit = g['nodes'][-1][0], g['nodes'][-1][4][0][0]
    native = [dict(relation='Tokens', owner=0, variables=total, degree=2,
                   evaluations=2, zero=True, outputs=[[h,0,total],[token,nv,nt]])]
    if outputs:
        native.extend(dict(relation=name, owner=lm, variables=total, degree=degree,
                           evaluations=nt+3, zero=zero, outputs=[[logit,0,total]])
                      for name,degree,zero in [('Key',2,False),('Selected',3,True)])
    else:
        native.append(dict(relation='ObserveLogits',owner=lm,variables=total,degree=1,
                           evaluations=1,zero=True,outputs=[[logit,0,total]]))
    return dict(file=g['file'], variant=g.get('variant'), n=n, vocab=vocab,
                prompt=prompt, ordinal=ordinal, custom=custom, log2_k=log_k, query_block=query_block,
                segments=segments, model_table_count=original_count, tables=tables, native=native,
                lookups=[[i, count] for i, count in enumerate(lookup_counts) if count],
                model_tiles=len(model_tiles), model_end=prefix_end, tail=tail,
                certificate=certificate, rows=cursor, nu=log_k+advice.log2_up(cursor),
                segment_rows=[[a,b] for a,b in zip(edges,edges[1:])])


def main():
    graphs = json.loads((HERE.parent/'lowering/lowering.json').read_text())['cases']
    cases = []
    for g in graphs:
        for k,q,s in [(10,32,3),(22,512,8)]:
            for p in sorted({1, max(1,g['n']-1),g['n']}):
                cases.append(runtime(g,p,3,k,q,s))
    for index in [1,2,13,14]:
        g = graphs[index]
        for v in [3,513]:
            cases.append(runtime(g,1,0,10,1,1,v))
    # Public metadata with small root/exponential bounds needs a genuinely new Range.
    low = advice.load_oracle('lowering')
    models = low.models
    spec = models.MODELS[0]
    model, sequences = models.build(spec)
    chooser = models.Run(model, spec['plan'])
    chooser.forward(sequences)
    base = models.integer_model(spec, model, chooser.c)
    for dimension in [16,32]:
        custom = dict(d_model=dimension,n_heads=dimension//2,n_kv_heads=1,d_head=2,
                      d_ff=3*dimension,rsqrt_kappa=0,exp_sigma=3,residual_bits=8,
                      residual_exponent=next(e for p,e in base['quant']['public']['act_exponents'] if 'Embed' in p))
        public = copy.deepcopy(base)
        public['arch'].update({k:custom[k] for k in ['d_model','n_heads','n_kv_heads','d_head','d_ff']})
        public['quant']['public'].update({k:custom[k] for k in ['rsqrt_kappa','exp_sigma','residual_bits','residual_exponent']})
        graph,_ = low.case_of('runtime-custom','tiny-2-servable.d',low.Public(public),5,None)
        for k in [6,10]:
            for prompt in [1,4,5]:
                cases.append(runtime(graph,prompt,2,k,1,3,custom=custom))
    cases.append(runtime(graphs[2],1,0,1,1,3))
    assert len(cases) == 141
    assert sum(c['certificate'] is not None and c['certificate']['range'] == c['model_table_count'] for c in cases) == 8
    result = dict(version=1, description='Draft RFC-0053 complete per-request catalog, maximum and multiplicity geometry; independent Python oracle.',cases=cases)
    # Keep large numeric/source rows compact for review.
    text = '{\n "version":1,\n "description":'+json.dumps(result['description'])+',\n "cases":[\n'
    text += ',\n'.join('  '+json.dumps(c,separators=(',',':')) for c in cases)
    text += '\n ]\n}\n'
    (HERE/'runtime.json').write_text(text)
    print('greedy runtime oracle:',len(cases),'cases')


if __name__ == '__main__':
    main()
