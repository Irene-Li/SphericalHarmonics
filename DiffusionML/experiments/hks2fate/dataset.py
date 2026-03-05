"""
Dataset for cross-prediction between HKS and cell fate markers.

Loads pre-processed small meshes from Data/small_meshes/. Each organoid has
a paired _mesh.obj (pre-scaled geometry) and _data.npz (pre-computed HKS
and cell-fate scalars). DiffusionNet operators are computed on-the-fly from
the mesh geometry (with optional disk caching).
"""

import os
import glob
import numpy as np
import torch
import igl
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from diffusion_net.geometry import get_operators
from diffusion_net.utils import sparse_np_to_torch

# ---------------------------------------------------------------------------
# Pre-computed HKS normalisation statistics (full small_meshes dataset,
# 397,732 vertices, 16 channels, after clipping at 99th percentile).
# ---------------------------------------------------------------------------
HKS_P99 = np.array([
    10.0092, 7.8560, 6.2617, 5.0711, 4.1665, 3.4682, 2.9043, 2.4324,
     2.0276, 1.6853, 1.4187, 1.2205, 1.0960, 1.0319, 1.0086, 1.0022,
], dtype=np.float32)

HKS_MEAN = np.array([
    8.3201, 6.2237, 4.6847, 3.5584, 2.7371, 2.1416, 1.7146, 1.4160,
    1.2191, 1.1025, 1.0430, 1.0163, 1.0053, 1.0011, 0.9997, 0.9992,
], dtype=np.float32)

HKS_STD = np.array([
    0.4195, 0.3874, 0.3579, 0.3307, 0.3043, 0.2771, 0.2461, 0.2088,
    0.1650, 0.1177, 0.0732, 0.0380, 0.0158, 0.0050, 0.0013, 0.0004,
], dtype=np.float32)


def normalise_hks(hks_np: np.ndarray) -> np.ndarray:
    """Clip-then-z-score normalise raw HKS. Values end up ~[-3, 3]."""
    hks_np = np.minimum(hks_np, HKS_P99)
    hks_np = (hks_np - HKS_MEAN) / (HKS_STD + 1e-8)
    return hks_np


class CrossPredictionDataset(Dataset):
    """
    PyTorch Dataset for cross-prediction between HKS and cell fate markers.

    Each item provides:
        - verts:       (V, 3)        vertex positions (pre-scaled)
        - faces:       (F, 3)        face indices
        - hks:         (V, n_hks)    HKS features (pre-computed, read from .npz)
        - fate_fields: (V, n_fates)  per-vertex fate marker intensities (pre-computed)
        - mass:        (V,)          mass vector
        - L:           (V, V)        sparse Laplacian
        - evals:       (K,)          eigenvalues
        - evecs:       (V, K)        eigenvectors
        - gradX:       (V, V)        sparse gradient operator (real)
        - gradY:       (V, V)        sparse gradient operator (imag)
        - meta:        dict with 'id', 'timepoint'
    """

    def __init__(self, data_path, k_eig=128, op_cache_dir=None):
        """
        Args:
            data_path:    path to the small_meshes directory (e.g. 'Data/small_meshes')
            k_eig:        number of eigenvalues for DiffusionNet operators
            op_cache_dir: directory to cache DiffusionNet operators (optional)
        """
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

        if op_cache_dir is not None:
            os.makedirs(op_cache_dir, exist_ok=True)

        # Discover all valid organoid entries
        self.entries = []
        self._discover_entries(data_path)

        # Infer n_hks and n_fates from the first entry
        self.n_hks = None
        self.n_fates = None
        self.fate_names = None
        if self.entries:
            npz = np.load(self.entries[0]['data_path'], allow_pickle=True)
            names = npz['names'].tolist()
            hks_names = [n for n in names if n.startswith('hks_')]
            fate_names = [n for n in names if not n.startswith('hks_')]
            self.n_hks = len(hks_names)
            self.n_fates = len(fate_names)
            self.fate_names = fate_names
            print(f"HKS features: {self.n_hks}")
            print(f"Fate markers: {self.n_fates} ({', '.join(self.fate_names)})")

        print(f"CrossPredictionDataset: {len(self.entries)} organoids loaded.")

    def _discover_entries(self, data_path):
        """Scan the small_meshes directory for paired _mesh.obj / _data.npz files."""
        mesh_files = sorted(glob.glob(os.path.join(data_path, '*_mesh.obj')))
        for mesh_path in mesh_files:
            stem = mesh_path[:-len('_mesh.obj')]
            data_path_npz = stem + '_data.npz'
            if not os.path.exists(data_path_npz):
                continue

            # Parse id from filename stem: {timepoint}_{well}_{label}
            basename = os.path.basename(stem)
            parts = basename.split('_')
            label = parts[-1]
            well = parts[-2]
            timepoint = '_'.join(parts[:-2])

            self.entries.append({
                'mesh_path': mesh_path,
                'data_path': data_path_npz,
                'id': basename,
                'timepoint': timepoint,
            })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load pre-scaled mesh geometry
        verts_np, faces_np = igl.read_triangle_mesh(entry['mesh_path'])
        verts_np = verts_np.astype(np.float32)
        faces_np = faces_np.astype(np.int64)

        verts = torch.tensor(verts_np, dtype=torch.float32)
        faces = torch.tensor(faces_np, dtype=torch.int64)

        # Load pre-computed scalar fields (HKS + fates)
        npz = np.load(entry['data_path'], allow_pickle=True)
        scalars = npz['scalars']          # (V, n_features)
        names = npz['names'].tolist()     # list of feature name strings

        # Split into HKS and fate columns
        hks_mask = [n.startswith('hks_') for n in names]
        fate_mask = [not n.startswith('hks_') for n in names]

        hks_np = scalars[:, hks_mask].astype(np.float32)
        hks_np = normalise_hks(hks_np)
        hks = torch.tensor(hks_np, dtype=torch.float32)
        fate_fields = torch.tensor(scalars[:, fate_mask].astype(np.float32), dtype=torch.float32)

        # Compute DiffusionNet operators (with optional caching)
        frames, mass, L, evals, evecs, gradX, gradY = get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        return {
            'verts': verts,
            'faces': faces,
            'hks': hks,
            'fate_fields': fate_fields,
            'mass': mass,
            'L': L,
            'evals': evals,
            'evecs': evecs,
            'gradX': gradX,
            'gradY': gradY,
            'meta': {
                'id': entry['id'],
                'timepoint': entry['timepoint'],
            }
        }


def collate_single(batch):
    """
    Custom collate function that does NOT stack into a batch tensor,
    since each mesh has a different number of vertices.
    Returns a list of dicts.
    """
    return batch
