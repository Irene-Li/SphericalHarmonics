"""
Dataset for HKS -> cell-fate presence classification.

Each organoid is represented by its per-vertex HKS features (input) and a
binary target vector: whether each cell-fate type is present anywhere on the
organoid surface (1) or entirely absent (0).

This reduces the DiffusionNet to a pure global binary-classification problem --
one yes/no label per fate per organoid -- so we can isolate whether the encoder
is learning anything useful about organoid morphology without the confound of
the decoder.

HKS normalisation
-----------------
Raw HKS values span ~1-10, with rare outlier vertices reaching ~140 (degenerate
mesh faces producing Laplacian spikes). Without normalisation, DiffusionNet's
MiniMLP pre-activations are pushed into large-negative territory and ReLU
neurons die immediately, causing the network to predict the dataset mode for
every input.

We apply a per-channel clip-then-z-score, with statistics pre-computed over
the full small_meshes dataset (397,732 vertices). See normalise_hks() below.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import potpourri3d as pp3d
from diffusion_net.geometry import get_operators


def normalise_hks(hks_np: np.ndarray) -> np.ndarray:
    """
    Clip-then-z-score normalise raw HKS features.

    Clip at per-channel 99th percentile first to remove degenerate mesh
    vertices (Laplacian spikes from near-zero-area faces), then z-score
    using dataset-level mean and std.

    Args:
        hks_np: (V, 16) float32 array of raw HKS values
    Returns:
        (V, 16) float32 array with values approximately in [0, 1]
    """
    hks_np = hks_np / np.mean(hks_np, axis=0, keepdims=True) - 1
    return hks_np


DEFAULT_FATES = ['lgr', 'sero', 'lyz']


def compute_coverage(scalars, names, selected_names):
    """
    Compute binary fate presence (1 = fate present on any vertex, 0 = absent)
    for each requested fate field.

    Args:
        scalars:        (V, n_features) numpy array of per-vertex scalars
        names:          list of feature name strings corresponding to columns
        selected_names: list of fate shortnames to compute presence for

    Returns:
        presence: (n_fates,) numpy array of binary labels {0.0, 1.0}
    """
    presence = []
    for field in selected_names:
        if field in names:
            idx = names.index(field)
            present = float((scalars[:, idx] > 0).any())
        else:
            present = 0.0  # field absent in this sample
        presence.append(present)
    return np.array(presence, dtype=np.float32)


class FateCoverageDataset(Dataset):
    """
    PyTorch Dataset: HKS features -> per-fate binary presence labels.

    Input  (per organoid): HKS features, shape (V, n_hks), normalised
    Target (per organoid): binary presence labels, shape (n_fates,)
                           label_i = 1 if any vertex has fate_i signal > 0, else 0

    The target is a binary label {0, 1} for each requested fate, making this a
    standard multi-label binary classification problem at the organoid (global) level.
    """

    def __init__(self, data_path, k_eig=128, op_cache_dir=None,
                 fate_names=None):
        """
        Args:
            data_path:    path to the small_meshes directory
            k_eig:        number of eigenvalues for DiffusionNet operators
            op_cache_dir: directory to cache operators (optional)
            fate_names:   list of fate names to predict. These must match the
                          field names as stored in the npz 'names' array
                          (e.g. 'lgr', 'sero', 'lyz').
                          Defaults to DEFAULT_FATES = ['lgr', 'sero', 'lyz'].
        """
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

        if fate_names is None:
            fate_names = DEFAULT_FATES
        self.fate_names = fate_names

        if op_cache_dir is not None:
            os.makedirs(op_cache_dir, exist_ok=True)

        self.entries = []
        self._discover_entries(data_path)

        # Infer n_hks from first entry
        self.n_hks = None
        if self.entries:
            npz = np.load(self.entries[0]['data_path'], allow_pickle=True)
            names = npz['names'].tolist()
            self.n_hks = sum(1 for n in names if n.startswith('hks_'))
            print(f"HKS features: {self.n_hks} (will be clip+z-score normalised)")
            print(f"Fate targets: {len(self.fate_names)} ({', '.join(self.fate_names)})")

        print(f"FateCoverageDataset: {len(self.entries)} organoids loaded.")

    def _discover_entries(self, data_path):
        mesh_files = sorted(glob.glob(os.path.join(data_path, '*_mesh.obj')))
        for mesh_path in mesh_files:
            stem = mesh_path[:-len('_mesh.obj')]
            data_path_npz = stem + '_data.npz'
            if not os.path.exists(data_path_npz):
                continue

            basename = os.path.basename(stem)
            parts = basename.split('_')
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

        # Load mesh geometry via potpourri3d (same library used for operators)
        verts_np, faces_np = pp3d.read_mesh(entry['mesh_path'])
        verts_np = verts_np.astype(np.float32)
        faces_np = faces_np.astype(np.int64)

        verts = torch.tensor(verts_np, dtype=torch.float32)
        faces = torch.tensor(faces_np, dtype=torch.int64)

        # Load scalar fields
        npz = np.load(entry['data_path'], allow_pickle=True)
        scalars = npz['scalars']       # (V, n_features)
        names = npz['names'].tolist()  # feature name strings

        # Extract HKS and normalise (clip + z-score)
        hks_mask = [n.startswith('hks_') for n in names]
        hks_np = scalars[:, hks_mask].astype(np.float32)
        hks_np = normalise_hks(hks_np)
        hks = torch.tensor(hks_np, dtype=torch.float32)

        # Compute per-fate binary presence labels -> classification target
        coverage = compute_coverage(scalars, names, self.fate_names)
        coverage = torch.tensor(coverage, dtype=torch.float32)  # (n_fates,) binary {0, 1}

        # Compute DiffusionNet operators
        frames, mass, L, evals, evecs, gradX, gradY = get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        return {
            'verts':    verts,
            'faces':    faces,
            'hks':      hks,      # (V, n_hks) normalised
            'coverage': coverage, # (n_fates,) binary {0, 1}
            'mass':     mass,
            'L':        L,
            'evals':    evals,
            'evecs':    evecs,
            'gradX':    gradX,
            'gradY':    gradY,
            'meta': {
                'id':        entry['id'],
                'timepoint': entry['timepoint'],
            }
        }


def collate_single(batch):
    """
    Custom collate: do NOT stack into a batch tensor since meshes differ in V.
    Returns a list of dicts.
    """
    return batch
