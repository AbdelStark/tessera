#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Literal tree-level and native-message counts from RFC-0010/0014/0053.

No Rust implementation or research import. Count the current parent rows of
each tree from its literal leaf population at every depth of the common tree.
"""
import json
from pathlib import Path


def oracle(trees):
    height = max((t['height'] for t in trees), default=0)
    levels = []
    for depth in range(height):
        parents = []
        evaluations = 0
        for tree in trees:
            leaf_rows = (1 << tree['height']) * (1 << depth)
            if leaf_rows < 1 << height:
                continue
            parents.append((leaf_rows // (1 << height)) << tree.get('trees_log', 0))
            evaluations += 2 if tree['kind'] == 'forest' else 4
        levels.append(dict(rounds=max(parents).bit_length()-1, degree=3,
                           evaluations=evaluations, instances=len(parents)))
    return dict(trees=trees, levels=levels,
                forests=[t['trees_log'] for t in trees if t['kind'] == 'forest'])


def main():
    fraction = lambda h: dict(kind='fraction', height=h)
    forest = lambda h, s: dict(kind='forest', height=h, trees_log=s)
    inputs = [[], [fraction(0)], [fraction(0), fraction(0)],
              [fraction(1)], [forest(1, 0)], [forest(1, 4)]]
    for h in [2, 3, 6]:
        inputs += [[fraction(h), fraction(0), forest(1, h)],
                   [fraction(h), fraction(1), forest(2, 3), forest(h, 0)],
                   [forest(h, 1), forest(1, h+1), fraction(h-1)]]
    doc = dict(version=1, description='Literal RFC-0010 common-level geometry; no proof verdict.',
               cases=[oracle(trees) for trees in inputs])
    Path(__file__).with_name('geometry.json').write_text(json.dumps(doc, indent=1)+'\n')
    print(f'runtime budget geometry: {len(inputs)} cases')


if __name__ == '__main__':
    main()
