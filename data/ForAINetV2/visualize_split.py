"""
Visualize the spatial KMeans clustering of BlueCat tree centroids.
Saves one PNG per PLY file so you can verify the split looks correct.

Usage:
    python visualize_split.py
    python visualize_split.py --trees_per_subset 60
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(__file__))
from plyutils import read_ply
from split_bluecat import compute_centroids


def visualize(input_path, trees_per_subset, output_dir):
    print(f'Reading {os.path.basename(input_path)} ...')
    data = read_ply(input_path)

    tree_ids = np.unique(data['treeID'])
    n_trees = len(tree_ids)
    n_subsets = max(1, round(n_trees / trees_per_subset))

    # Compute XY centroid per tree
    centroids = compute_centroids(data)

    # Cluster
    kmeans = KMeans(n_clusters=n_subsets, random_state=42, n_init=10)
    labels = kmeans.fit_predict(centroids)

    # Count trees per cluster
    counts = np.bincount(labels)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = cm.tab20(np.linspace(0, 1, n_subsets))

    for cid in range(n_subsets):
        mask = labels == cid
        ax.scatter(centroids[mask, 0], centroids[mask, 1],
                   color=colors[cid % len(colors)], s=20, alpha=0.8)
        # Label cluster center
        cx, cy = kmeans.cluster_centers_[cid]
        ax.text(cx, cy, str(counts[cid]), fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6))

    ax.set_title(f'{os.path.basename(input_path)}\n'
                 f'{n_trees} trees, {n_subsets} subsets '
                 f'(avg {n_trees/n_subsets:.0f}, min {counts.min()}, max {counts.max()} trees/subset)',
                 fontsize=11)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(input_path))[0] + '_split.png'
    out_path = os.path.join(output_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved -> {out_path}')
    print(f'  Subsets: {n_subsets}  |  trees/subset: avg={n_trees/n_subsets:.0f}, '
          f'min={counts.min()}, max={counts.max()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trees_per_subset', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='split_preview')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        'train_val_data/BlueCat_RN_merged_trees_train.ply',
        'train_val_data/BlueCat_RN_merged_trees_val.ply',
        'test_data/BlueCat_RN_merged_trees_test.ply',
    ]
    for f in files:
        path = os.path.join(base_dir, f)
        if not os.path.exists(path):
            print(f'Skipping (not found): {path}')
            continue
        visualize(path, args.trees_per_subset, os.path.join(base_dir, args.output_dir))


if __name__ == '__main__':
    main()
