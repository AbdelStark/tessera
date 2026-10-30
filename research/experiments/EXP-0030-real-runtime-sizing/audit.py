#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent integer recount and saved-record corruption controls, never a verifier."""

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def power(n):
    assert isinstance(n, int) and n > 0, "positive domain"
    return 1 << (n - 1).bit_length()


def runs_union(runs):
    cursor = None
    count = 0
    for lo, hi in sorted(runs):
        assert 0 <= lo < hi, "row run"
        if cursor is None or lo > cursor:
            count += hi - lo
        elif hi > cursor:
            count += hi - cursor
        cursor = max(hi, cursor or 0)
    return count


def weight_charge(data):
    kw, rate = 1 << data['log2_k'], 1 << data['log2_inv_rate']
    width_sum = 0
    max_width = 0
    rows = 0
    folds = 0
    channels = 1
    for index, matrix in enumerate(data['matrices']):
        inputs, outputs = power(matrix['inputs']), power(matrix['out'])
        channels = max(channels, outputs)
        k = max(inputs, min(kw, inputs * outputs))
        width = k * rate
        layers = 1 if index in (0, 8) else data['layers']
        packed = k // inputs
        rows += ((matrix['out'] + packed - 1) // packed) * layers
        width_sum += width
        max_width = max(max_width, width)
        folds += k * power(layers)
    cp_cells = 16 * power(data['layers']) * channels
    cp_k = min(kw, cp_cells)
    width_sum += cp_k * rate
    max_width = max(max_width, cp_k * rate)
    rows += cp_cells // cp_k
    folds += cp_k
    terms = {
        'trees': width_sum * 64,
        'block': (data['row_block'] * 8 + 192) * max_width,
        'answers': rows * data['queries'] * 16,
        'folds': folds * 64,
        'metadata': data['layers'] * 4096 + 65536,
    }
    return sum(terms.values()), terms


def inspect(row):
    assert row['bytes_per_cell'] == 64, 'unchanged cell coefficient'
    tensors = sum(math.prod(power(axis) for axis in t['shape']) for t in row['tensors'])
    assert tensors == row['cells']['tensors'], 'complete padded tensor count'
    assert [t['id'] for t in row['tensors']] == list(range(len(row['tensors']))), 'all tensor identities'
    lookups = 0
    table_rows = 0
    ids = []
    for instance in row['instances']:
        cursor = 0
        for block in instance['blocks']:
            assert 1 << block['vars'] == math.prod(power(axis) for axis in block['domain']), 'per-axis lookup padding'
            assert block['offset'] == cursor, 'complete aligned lookup blocks'
            assert block['offset'] % (1 << block['vars']) == 0, 'lookup alignment'
            cursor += 1 << block['vars']
        assert cursor == instance['entries'], 'complete runtime lookup entries'
        assert power(cursor) == 1 << instance['height'], 'lookup tree tail'
        lookups += power(cursor)
        table_rows += 1 << instance['table_log2_rows']
        ids.append(instance['table'])
    assert ids == sorted(set(ids)), 'used table instances in canonical order'
    assert lookups == row['cells']['lookups'], 'all lookup trees padded separately'
    assert table_rows == row['cells']['table_rows'], 'all used table trees'
    advice = row['advice']
    placement_runs = [run for placement in advice['placements'] for run in placement['nonzero']]
    segment_runs = [run for segment in advice['segments'] for run in segment['nonzero']]
    assert runs_union(placement_runs) == runs_union(segment_runs) == advice['rows'], 'all advice nonzero rows'
    assert advice['rows'] * (1 << advice['log2_k']) == row['cells']['advice'], 'whole advice storage'
    weight, weight_terms = weight_charge(row['weight_inputs'])
    assert weight == row['weight_bytes'], 'all weight working-set terms'
    base = 64 * (tensors + lookups + table_rows + row['cells']['advice']) + weight
    floor = base - 64 * lookups
    assert floor == row['zero_lookup_base_floor'], 'grouping retains nonlookup cells'
    windows = [stage['windows'] for stage in row['stages']]
    assert all(stage[0]['cut'] == 0 and 'working_bytes' in stage[0] for stage in windows), 'resident stage charge'
    peak = base + max(stage[0]['working_bytes'] for stage in windows)
    assert peak == row['resident_bytes'], 'peak is base plus largest stage'
    for decision in row['budgets']:
        budget = decision['gib'] << 30
        choices = []
        if base <= budget:
            for stage in windows:
                fits = [p for p in stage if 'working_bytes' in p and p['working_bytes'] <= budget - base]
                if not fits:
                    break
                choices.append(fits[0])
        expected = len(choices) == len(windows)
        assert decision['fits'] == expected, 'original planner budget decision'
        if expected:
            assert decision['cuts'] == [p['cut'] for p in choices], 'first feasible cuts'
            assert decision['resident_bytes'] == base + max(p['working_bytes'] for p in choices), 'fitted peak'
    return {'positions':row['positions'], 'policy':row['policy'], 'case':row['case'], 'cells':row['cells'], 'base_bytes':base, 'zero_lookup_base_floor':floor, 'peak_bytes':peak, 'weight_terms':weight_terms, 'fits_gib':[d['gib'] for d in row['budgets'] if d['fits']]}


def controls(records):
    assert len(records) == 10, 'all frozen cases'
    assert sum(r['case'] == 'tiny-control' for r in records) == 1
    real = [r for r in records if r['case'] == 'real']
    assert {(r['policy'], r['positions']) for r in real} == {(b,n) for b in (30,31,32) for n in (2,512,2048)}
    return [inspect(row) for row in records]


def main():
    raw_path = ROOT / 'results/counts.json'
    raw = raw_path.read_bytes() if raw_path.exists() else gzip.decompress((ROOT / 'results/counts.json.gz').read_bytes())
    data = json.loads(raw)
    records = data['records']
    summaries = controls(records)
    source = next(row for row in records if row['case'] == 'real' and row['positions'] == 2 and row['policy'] == 32)
    faults = []

    def corrupt(name, mutate):
        changed = copy.deepcopy(source)
        mutate(changed)
        try:
            inspect(changed)
        except AssertionError as error:
            faults.append({'name':name, 'caught':True, 'assertion':str(error)})
        else:
            raise AssertionError('survived: ' + name)

    corrupt('missing-tensor', lambda r: r['tensors'].pop())
    def unpadded(r):
        raw = sum(i['entries'] for i in r['instances'])
        assert raw != r['cells']['lookups'], 'meaningful unpadded case'
        r['cells']['lookups'] = raw
    corrupt('unpadded-lookups', unpadded)
    def greedy(r):
        last_site = r['greedy_site']
        instance = next(i for i in r['instances'] if any(b['site'] == last_site for b in i['blocks']))
        instance['blocks'] = [b for b in instance['blocks'] if b['site'] != last_site]
    corrupt('missing-runtime-greedy-lookup', greedy)
    corrupt('omitted-table-rows', lambda r:r['instances'].pop())
    corrupt('short-advice-storage', lambda r:r['advice'].__setitem__('rows', r['advice']['rows']-1))
    def no_folds(r):
        _, terms = weight_charge(r['weight_inputs'])
        r['weight_bytes'] -= terms['folds']
    corrupt('omitted-weight-folds', no_folds)
    corrupt('grouping-drops-tables', lambda r:r.__setitem__('zero_lookup_base_floor', r['zero_lookup_base_floor']-64*r['cells']['table_rows']))
    corrupt('changed-cell-coefficient', lambda r:r.__setitem__('bytes_per_cell', 8))
    corrupt('false-64gib-decision', lambda r:next(d for d in r['budgets'] if d['gib']==64).__setitem__('fits', not next(d for d in r['budgets'] if d['gib']==64)['fits']))
    corrupt('off-by-one-stage-peak', lambda r:r.__setitem__('resident_bytes', r['resident_bytes']+1))
    assert controls(records) == summaries, 'restored controls'
    real = [r for r in summaries if r['case']=='real']
    verdict = 'validated' if all(r['zero_lookup_base_floor'] > 64 << 30 for r in real) else 'refuted'
    result = {'verdict':verdict, 'scope':'E0 exact public resource counts; WindowMemory internals source-pinned, no independent allocation/soundness proof', 'cases':summaries, 'corruptions_caught':len(faults), 'faults':faults, 'counts_sha256':hashlib.sha256(raw).hexdigest(), 'original_and_restored_controls_pass':True}
    (ROOT/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verdict':verdict,'cases':len(summaries),'corruptions_caught':len(faults)}))


if __name__ == '__main__':
    main()
