"""
Count trees per file for selected datasets and print a summary table.

Usage:
    python count_trees.py
"""
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from plyutils import read_ply


DATASETS = [
    'CULS_CULS',
    'NIBIO_MLS',
    'NIBIO_NIBIO',
    'NIBIO2_NIBIO2',
    'RMIT_RMIT',
    'SCION_SCION',
    'TUWIEN_TUWIEN',
    'Yuchen_2023',
]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(base_dir, 'train_val_data'),
        os.path.join(base_dir, 'test_data'),
    ]

    # Collect all ply files grouped by dataset prefix
    stats = defaultdict(lambda: {'files': 0, 'trees': []})

    all_files = []
    for d in search_dirs:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.ply'):
                    all_files.append(os.path.join(d, f))

    for path in all_files:
        fname = os.path.basename(path)
        matched = next((ds for ds in DATASETS if fname.startswith(ds)), None)
        if matched is None:
            continue
        data = read_ply(path)
        n_trees = len(np.unique(data['treeID']))
        stats[matched]['files'] += 1
        stats[matched]['trees'].append(n_trees)
        print('  %s: %d trees' % (fname, n_trees))

    print()
    print('%-20s %6s %8s %6s %6s %6s' % ('Dataset', 'Files', 'Total', 'Min', 'Max', 'Avg'))
    print('-' * 60)
    for ds in DATASETS:
        s = stats[ds]
        if s['files'] == 0:
            print('%-20s  (no files found)' % ds)
            continue
        trees = s['trees']
        print('%-20s %6d %8d %6d %6d %6.1f' % (
            ds, s['files'], sum(trees), min(trees), max(trees), np.mean(trees)))


if __name__ == '__main__':
    main()
