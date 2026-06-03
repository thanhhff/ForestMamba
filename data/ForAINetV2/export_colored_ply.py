"""
Export a PLY file with RGB colors assigned per treeID for visualization in CloudCompare.

Usage:
    python export_colored_ply.py --input bluecat_subsets/BlueCat_RN_merged_trees_val_subset000.ply
    python export_colored_ply.py --input bluecat_subsets/BlueCat_RN_merged_trees_val_subset000.ply --output colored.ply
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plyutils import read_ply, write_ply


def generate_colors(n):
    """Generate n visually distinct RGB colors using golden ratio hue spacing."""
    colors = []
    golden_ratio = 0.618033988749895
    h = 0.0
    for _ in range(n):
        h = (h + golden_ratio) % 1.0
        # HSV to RGB (full saturation and value)
        i = int(h * 6)
        f = h * 6 - i
        q = 1 - f
        t = f
        if i % 6 == 0: r, g, b = 1, t, 0
        elif i % 6 == 1: r, g, b = q, 1, 0
        elif i % 6 == 2: r, g, b = 0, 1, t
        elif i % 6 == 3: r, g, b = 0, q, 1
        elif i % 6 == 4: r, g, b = t, 0, 1
        else:             r, g, b = 1, 0, q
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input PLY file')
    parser.add_argument('--output', default=None, help='Output PLY file (default: input_colored.ply)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input) if not os.path.isabs(args.input) else args.input

    if args.output is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + '_colored.ply'
    else:
        output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output

    print(f'Reading {os.path.basename(input_path)} ...')
    data = read_ply(input_path)
    print(f'  {len(data):,} points loaded.')

    tree_ids = np.unique(data['treeID'])
    n_trees = len(tree_ids)
    print(f'  {n_trees} unique treeIDs found.')

    # Map treeID -> color index
    colors = generate_colors(n_trees)
    id_to_color = {tid: colors[i] for i, tid in enumerate(tree_ids)}

    # Assign colors per point
    print('Assigning colors ...')
    rgb = np.array([id_to_color[tid] for tid in data['treeID']], dtype=np.uint8)

    # Write output PLY
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
    write_ply(output_path, [xyz, rgb], ['x', 'y', 'z', 'red', 'green', 'blue'])
    print(f'Saved -> {output_path}')


if __name__ == '__main__':
    main()
