"""
Dataset for loading organoid meshes for DiffusionNet training.

Loads pre-saved FateMarkers results (mesh + eigenbasis) and computes
DiffusionNet operators on-the-fly (with disk caching). Input features
are HKS computed from the eigendecomposition.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fatemarkers import FateMarkers
from diffusion_net.geometry import get_operators, compute_hks
from diffusion_net.utils import sparse_np_to_torch


class OrganoidDataset(Dataset):
    """
    PyTorch Dataset for organoid meshes.

    Each item provides:
        - verts:  (V, 3)  vertex positions
        - faces:  (F, 3)  face indices
        - hks:    (V, n_hks) HKS features (input to DiffusionNet)
        - mass:   (V,)    mass vector
        - L:      (V, V)  sparse Laplacian
        - evals:  (K,)    eigenvalues
        - evecs:  (V, K)  eigenvectors
        - gradX:  (V, V)  sparse gradient operator (real)
        - gradY:  (V, V)  sparse gradient operator (imag)
        - meta:   dict with 'id', 'timepoint', 'area'
    """

    def __init__(self, data_path, config_path, k_eig=128,
                 hks_scales_path='sim/vocab_new.npz',
                 op_cache_dir=None, recon_quality_threshold=None,
                 preload=True):
        """
        Args:
            data_path:   path to the dataset root (e.g. 'Data/20260224')
            config_path: path to config.json
            k_eig:       number of eigenvalues for DiffusionNet
            hks_scales_path: path to .npz containing 'ts' array of HKS time scales
            op_cache_dir: directory to cache DiffusionNet operators (optional)
            recon_quality_threshold: percentile threshold on recon quality
                                    (e.g. 95 to filter top 5% worst)
            preload:     if True, scan and validate all entries on init
        """
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

        # Load HKS time scales from the existing vocabulary file
        vocab_data = np.load(hks_scales_path, allow_pickle=True)
        self.hks_scales = torch.tensor(vocab_data['ts'], dtype=torch.float32)
        self.n_hks = len(self.hks_scales)
        print(f"HKS scales: {self.n_hks} values in [{self.hks_scales[0]:.2f}, {self.hks_scales[-1]:.2f}]")

        if op_cache_dir is not None:
            os.makedirs(op_cache_dir, exist_ok=True)

        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Discover all valid organoid entries
        self.entries = []
        self._discover_entries(data_path, cfg)

        # Optionally filter by reconstruction quality
        if recon_quality_threshold is not None and preload:
            self._filter_by_quality(recon_quality_threshold)

        print(f"OrganoidDataset: {len(self.entries)} organoids loaded.")

    def _discover_entries(self, data_path, cfg):
        """Walk the data directory and find all valid organoid entries."""
        timepoints = cfg['timepoints']
        zarr_names = cfg['zarr_names']
        wells = cfg['wells']
        rounds = cfg['rounds']

        for tp in timepoints:
            zarr_name = zarr_names[tp]
            for well_name in wells[tp]:
                round_name = rounds[tp]
                base = os.path.join(
                    data_path, 'fractal_output', tp, zarr_name,
                    well_name[0], well_name[1:], round_name)
                fm_dir = os.path.join(base, 'fm_data')
                labels_path = os.path.join(fm_dir, 'good_labels.npy')

                if not os.path.exists(labels_path):
                    continue

                labels = np.load(labels_path).astype(int)
                for label in labels:
                    save_path = os.path.join(fm_dir, str(label))
                    coeffs_path = save_path + '_coeffs.npz'
                    mesh_path = save_path + '_transformed_mesh.obj'

                    if os.path.exists(coeffs_path) and os.path.exists(mesh_path):
                        self.entries.append({
                            'save_path': save_path,
                            'mesh_path': mesh_path,
                            'id': f"{tp}_{well_name}_{label}",
                            'timepoint': tp,
                        })

    def _filter_by_quality(self, percentile_threshold):
        """Filter out organoids with poor reconstruction quality."""
        fracs = []
        valid_entries = []

        for entry in tqdm(self.entries, desc="Checking reconstruction quality"):
            try:
                m = FateMarkers()
                m.load_results(entry['save_path'])
                frac = m.compute_recon_quality()
                fracs.append(frac)
                valid_entries.append(entry)
            except Exception:
                continue

        fracs = np.array(fracs)
        threshold = np.percentile(fracs, percentile_threshold)
        mask = fracs < threshold

        self.entries = [e for e, m in zip(valid_entries, mask) if m]
        print(f"  Filtered to {len(self.entries)} organoids "
              f"(threshold={threshold:.4f} at {percentile_threshold}th percentile)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load mesh
        m = FateMarkers()
        m.load_results(entry['save_path'])
        verts_np = m.v.astype(np.float32)
        faces_np = m.f.astype(np.int64)
        area = m.mass_matrix.diagonal().sum() if hasattr(m, 'mass_matrix') and m.mass_matrix is not None else 0.

        verts = torch.tensor(verts_np, dtype=torch.float32)
        faces = torch.tensor(faces_np, dtype=torch.int64)

        # Compute DiffusionNet operators (with caching)
        frames, mass, L, evals, evecs, gradX, gradY = get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # Compute HKS using the existing time scales
        hks = compute_hks(evals, evecs, self.hks_scales.to(evals.device))  # (V, n_hks)

        return {
            'verts': verts,
            'faces': faces,
            'hks': hks,
            'mass': mass,
            'L': L,
            'evals': evals,
            'evecs': evecs,
            'gradX': gradX,
            'gradY': gradY,
            'area': area,
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
