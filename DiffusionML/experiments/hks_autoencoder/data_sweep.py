#!/usr/bin/env python3
"""
Build or update the sphere-CD cache for all meshes in small_meshes.

For each mesh, computes the Chamfer distance to a reference sphere of equal
surface area.  Results are written to sphere_cd_cache.json in the data
directory.  Meshes already in the cache are skipped (incremental).

Run from the repository root:

    python DiffusionML/experiments/hks_autoencoder/data_sweep.py \
        --data_path Data/small_meshes

Outputs:
    <data_path>/sphere_cd_cache.json    (created / updated)
    Text histogram and top/bottom-N summary printed to stdout.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# Make dataset helpers importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dataset import _sphere_cd_for_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_meshes(data_path: str) -> list[str]:
    """Return sorted list of *_mesh.obj paths that also have a *_data.npz."""
    paths = sorted(glob.glob(os.path.join(data_path, '*_mesh.obj')))
    return [p for p in paths
            if os.path.exists(p[:-len('_mesh.obj')] + '_data.npz')]


def text_histogram(values: np.ndarray, n_bins: int = 15,
                   bar_width: int = 40) -> str:
    counts, edges = np.histogram(values, bins=n_bins)
    max_count = counts.max() or 1
    lines = []
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = '█' * int(bar_width * c / max_count)
        lines.append(f"  [{lo:6.4f}, {hi:6.4f})  {bar:<{bar_width}}  {c:4d}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sphere-CD sweep for small_meshes")
    parser.add_argument('--data_path', default='Data/small_meshes',
                        help='Path to the small_meshes directory')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of extreme entries to print in the summary')
    args = parser.parse_args()

    data_path = args.data_path
    cache_path = os.path.join(data_path, 'sphere_cd_cache.json')

    # ------------------------------------------------------------------
    # Discover meshes
    # ------------------------------------------------------------------
    mesh_paths = discover_meshes(data_path)
    print(f"Found {len(mesh_paths)} paired meshes in {data_path}")

    # ------------------------------------------------------------------
    # Load existing cache
    # ------------------------------------------------------------------
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache)} existing entries  ({cache_path})")

    # ------------------------------------------------------------------
    # Compute missing entries
    # ------------------------------------------------------------------
    missing = [p for p in mesh_paths
               if os.path.basename(p) not in cache]

    if missing:
        print(f"Computing sphere-CD for {len(missing)} new mesh(es)...")
        for path in tqdm(missing, desc="Sphere-CD sweep", unit="mesh"):
            key = os.path.basename(path)
            cache[key] = _sphere_cd_for_mesh(path)

        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"Cache saved → {cache_path}")
    else:
        print("All meshes already cached — nothing to compute.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    # Only summarise entries that are still present on disk
    valid_keys = {os.path.basename(p) for p in mesh_paths}
    items = [(k, v) for k, v in cache.items() if k in valid_keys]
    items.sort(key=lambda x: x[1])

    values = np.array([v for _, v in items])
    print(f"\n{'─'*60}")
    print(f"sphere_cd summary  (n={len(values)})")
    print(f"  min  {values.min():.4f}   max  {values.max():.4f}   "
          f"mean  {values.mean():.4f}   median  {np.median(values):.4f}")
    print(f"\n{text_histogram(values)}")

    n = min(args.top_n, len(items))

    print(f"\n{'─'*60}")
    print(f"Top {n} most SPHERICAL  (low CD → most likely spheres):")
    for k, v in items[:n]:
        print(f"  {v:.4f}  {k}")

    print(f"\nTop {n} most NON-SPHERICAL  (high CD → most irregular):")
    for k, v in items[-n:][::-1]:
        print(f"  {v:.4f}  {k}")

    print(f"\nTo inspect interactively run:")
    print(f"  python DiffusionML/experiments/hks_autoencoder/inspect_sphere_cd.py "
          f"--data_path {data_path}")


if __name__ == '__main__':
    main()
