"""
Split large BlueCat PLY files into smaller spatial subsets.

Each subset contains ~trees_per_subset trees grouped by geographic region
using KMeans clustering on tree centroids.

Usage:
    python split_bluecat.py
    python split_bluecat.py --trees_per_subset 60 --same_folder
"""
import argparse
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from plyutils import read_ply, write_ply


def compute_centroids(data):
    """Vectorized centroid computation — replaces per-tree loop."""
    sort_idx = np.argsort(data['treeID'])
    sorted_tids = data['treeID'][sort_idx]
    sorted_x = data['x'][sort_idx]
    sorted_y = data['y'][sort_idx]

    _, first_idx = np.unique(sorted_tids, return_index=True)
    counts = np.diff(np.append(first_idx, len(sorted_tids)))
    cx = np.add.reduceat(sorted_x, first_idx) / counts
    cy = np.add.reduceat(sorted_y, first_idx) / counts
    return np.stack([cx, cy], axis=1)


def save_subset(cluster_id, cluster_tree_ids, data, output_dir, base_name):
    mask = np.isin(data['treeID'], cluster_tree_ids)
    subset = data[mask]

    xyz = np.stack([subset['x'], subset['y'], subset['z']], axis=1).astype(np.float32)
    sem = subset['semantic_seg'].astype(np.int32).reshape(-1, 1)
    tid = subset['treeID'].astype(np.int32).reshape(-1, 1)

    out_path = os.path.join(output_dir, f'{base_name}_subset{cluster_id:03d}.ply')
    write_ply(out_path, [xyz, sem, tid], ['x', 'y', 'z', 'semantic_seg', 'treeID'])


def split_ply(input_path, output_dir, trees_per_subset=90, num_workers=8):
    if num_workers is None:
        num_workers = os.cpu_count()

    fname = os.path.basename(input_path)
    print(f'\n[1/4] Reading {fname} ...')
    data = read_ply(input_path)
    print(f'      {len(data):,} points loaded.')

    tree_ids = np.unique(data['treeID'])
    n_trees = len(tree_ids)
    n_subsets = max(1, round(n_trees / trees_per_subset))

    print(f'[2/4] Computing centroids for {n_trees} trees ...')
    centroids = compute_centroids(data)

    print(f'[3/4] Clustering into {n_subsets} subsets (~{n_trees/n_subsets:.0f} trees each) ...')
    kmeans = KMeans(n_clusters=n_subsets, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(centroids)

    print(f'[4/4] Saving {n_subsets} subsets using {num_workers} threads ...')
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(fname)[0]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                save_subset, cluster_id,
                tree_ids[cluster_labels == cluster_id],
                data, output_dir, base_name
            ): cluster_id
            for cluster_id in range(n_subsets)
        }
        for future in tqdm(as_completed(futures), total=n_subsets, desc='      saving', unit='subset'):
            future.result()

    print(f'  Done. {n_subsets} subsets saved to {output_dir}\n')


def main():
    parser = argparse.ArgumentParser(description='Split BlueCat PLY into spatial subsets')
    parser.add_argument('--trees_per_subset', type=int, default=100,
                        help='Target number of trees per subset (default: 90)')
    parser.add_argument('--same_folder', action='store_true',
                        help='Save subsets into the same folder as the source PLY files')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of threads for parallel saving (default: cpu count)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    files_and_dirs = [
        ('train_val_data/BlueCat_RN_merged_trees_train.ply', 'train_val_data'),
        ('train_val_data/BlueCat_RN_merged_trees_val.ply',   'train_val_data'),
        ('test_data/BlueCat_RN_merged_trees_test.ply',       'test_data'),
    ]

    for f, src_dir in files_and_dirs:
        path = os.path.join(base_dir, f)
        if not os.path.exists(path):
            print(f'Skipping (not found): {path}')
            continue
        out_dir = src_dir if args.same_folder else 'bluecat_subsets'
        split_ply(path, os.path.join(base_dir, out_dir), args.trees_per_subset, args.num_workers)


if __name__ == '__main__':
    main()
