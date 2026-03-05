"""
Dataset for loading organoid meshes for DiffusionNet training.

Loads pre-processed small meshes from Data/small_meshes/. Each organoid has
a paired _mesh.obj (pre-scaled geometry) and _data.npz (pre-computed HKS
and cell-fate scalars). DiffusionNet operators are computed on-the-fly from
the mesh geometry (with optional disk caching).

HKS normalisation
-----------------
Raw HKS values span ~1–10 across 16 log-spaced timescales, with rare degenerate
vertices reaching ~140 (Laplacian spikes from near-zero-area faces). Without
normalisation, DiffusionNet MiniMLP pre-activations are pushed into large-negative
territory and ReLU neurons die immediately.

We divide each channel by its per-vertex column mean and subtract 1, giving
values centred near 0 with no hardcoded dataset-level statistics required.

Sphere-bias mitigation
----------------------
When the dataset contains many near-spherical shapes, the decoder can achieve
a low Chamfer loss by learning to output a sphere for every input.
``OrganoidDataset`` computes the Chamfer distance between each mesh and a
reference sphere of equal surface area (results cached to
``sphere_cd_cache.json`` in the data directory).  ``BalancedBinSampler`` then
buckets entries by this score and, each epoch, draws an equal number of
samples from every bucket so that spherical and non-spherical shapes are seen
at equal rates.
"""

import os
import glob
import math
import json
import numpy as np
import torch
import potpourri3d as pp3d
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from diffusion_net.geometry import get_operators

try:
    from scipy.spatial import cKDTree as _CKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# HKS normalisation
# ---------------------------------------------------------------------------

def normalise_hks(hks_np: np.ndarray) -> np.ndarray:
    """
    Per-channel mean normalisation of raw HKS features.

    Divides each channel by its column mean and subtracts 1, giving values
    centred near 0. Requires no pre-computed dataset statistics.

    Args:
        hks_np: (V, 16) float32 array of raw HKS values
    Returns:
        (V, 16) float32 array centred near 0
    """
    hks_np = hks_np / np.mean(hks_np, axis=0, keepdims=True) - 1
    return hks_np


# ---------------------------------------------------------------------------
# Sphere-likeness helpers
# ---------------------------------------------------------------------------

_REF_SPHERE_VERTS: np.ndarray = None   # module-level cache, built once


def _icosphere_unit_verts(subdivisions: int = 3) -> np.ndarray:
    """Unit icosphere vertices via recursive midpoint subdivision.

    subdivisions=3 → 642 vertices, a good balance between accuracy and speed
    for the Chamfer sweep.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    for _ in range(subdivisions):
        new_f, mid = [], {}

        def _mid(a, b):
            nonlocal v
            key = (min(a, b), max(a, b))
            if key not in mid:
                m = (v[a] + v[b]) / 2.0
                mid[key] = len(v)
                v = np.vstack([v, m / np.linalg.norm(m)])
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = np.array(new_f, dtype=np.int64)

    return v.astype(np.float32)


def _get_ref_sphere_verts() -> np.ndarray:
    """Return cached unit icosphere vertices (built once per process)."""
    global _REF_SPHERE_VERTS
    if _REF_SPHERE_VERTS is None:
        _REF_SPHERE_VERTS = _icosphere_unit_verts(subdivisions=3)
    return _REF_SPHERE_VERTS


def _mesh_surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    """Compute total surface area of a triangle mesh."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _chamfer_distance_np(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Chamfer distance between two (N,3) and (M,3) arrays."""
    if _HAS_SCIPY:
        d_AB = _CKDTree(B).query(A, k=1)[0].mean()
        d_BA = _CKDTree(A).query(B, k=1)[0].mean()
    else:
        # Fallback O(N·M) — acceptable for the small meshes used here
        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        d_AB = D.min(axis=1).mean()
        d_BA = D.min(axis=0).mean()
    return float(d_AB + d_BA)


def _sphere_cd_for_mesh(mesh_path: str) -> float:
    """Chamfer distance between a mesh and a reference sphere of equal area.

    The reference sphere is an icosphere scaled so its surface area matches
    the mesh, centred at the mesh centroid.  A perfect sphere returns 0; more
    irregular shapes return larger values.
    """
    verts, faces = pp3d.read_mesh(mesh_path)
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)

    area = _mesh_surface_area(verts, faces)
    r = math.sqrt(max(area, 1e-12) / (4.0 * math.pi))

    ref_verts = _get_ref_sphere_verts() * r          # scale to same area
    mesh_verts_c = verts - verts.mean(axis=0)         # centre at origin

    return _chamfer_distance_np(mesh_verts_c, ref_verts)


# ---------------------------------------------------------------------------
# Balanced bin sampler
# ---------------------------------------------------------------------------

class BalancedBinSampler:
    """Two-bucket sampler split by a sphere-CD threshold.

    Entries with ``sphere_cd < threshold`` are "spherical" (bin 0); entries
    with ``sphere_cd >= threshold`` are "irregular" (bin 1).  Each call to
    ``sample_epoch`` draws an equal number of samples from both buckets, so
    irregular shapes are oversampled relative to their frequency in the
    dataset, counteracting sphere-heavy bias.

    Usage::

        sampler = BalancedBinSampler(dataset.entries, threshold=0.05)
        for epoch in range(n_epochs):
            epoch_indices = sampler.sample_epoch(train_indices)
            train_one_epoch(model, dataset, epoch_indices, ...)
    """

    def __init__(self, entries: list, threshold: float = 0.05):
        scores = np.array([e['sphere_cd'] for e in entries], dtype=np.float64)
        self.n_bins = 2
        self.threshold = threshold

        # bin 0 = spherical (CD < threshold), bin 1 = irregular (CD >= threshold)
        self.bin_labels = (scores >= threshold).astype(int)

        n_spherical  = (self.bin_labels == 0).sum()
        n_irregular  = (self.bin_labels == 1).sum()
        print(f"BalancedBinSampler: threshold={threshold}  "
              f"spherical={n_spherical}  irregular={n_irregular}")

    def sample_epoch(self, indices: np.ndarray) -> np.ndarray:
        """Draw a balanced list of indices from the given subset.

        Returns a shuffled array of length ``n_active_bins * per_bin`` where
        each active bin (non-empty intersection with ``indices``) contributes
        exactly ``per_bin = len(indices) // n_active_bins`` samples.
        """
        # Partition supplied indices by bin
        bin_idx = [[] for _ in range(self.n_bins)]
        for i in indices:
            bin_idx[self.bin_labels[i]].append(i)

        active = [np.array(b) for b in bin_idx if len(b) > 0]
        if not active:
            return indices.copy()

        per_bin = max(1, len(indices) // len(active))

        result = []
        for b_arr in active:
            replace = len(b_arr) < per_bin
            chosen = np.random.choice(b_arr, size=per_bin, replace=replace)
            result.append(chosen)

        out = np.concatenate(result)
        np.random.shuffle(out)
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrganoidDataset(Dataset):
    """
    PyTorch Dataset for organoid meshes loaded from small_meshes directory.

    Each item provides:
        - verts:  (V, 3)     vertex positions (pre-scaled)
        - faces:  (F, 3)     face indices
        - hks:    (V, n_hks) HKS features (pre-computed, read from .npz)
        - mass:   (V,)       mass vector
        - L:      (V, V)     sparse Laplacian
        - evals:  (K,)       eigenvalues
        - evecs:  (V, K)     eigenvectors
        - gradX:  (V, V)     sparse gradient operator (real)
        - gradY:  (V, V)     sparse gradient operator (imag)
        - meta:   dict with 'id', 'timepoint'

    Args:
        data_path:      path to the small_meshes directory
        k_eig:          number of eigenvalues for DiffusionNet operators
        op_cache_dir:   directory to cache DiffusionNet operators (optional)
        sphere_cd_threshold: sphere-CD threshold for balanced epoch sampling.
                        Meshes with CD < threshold are "spherical"; those with
                        CD >= threshold are "irregular".  Each epoch draws
                        equally from both buckets.  Set to 0 to disable
                        (``sphere_sampler`` will be ``None``).  Default: 0.05.
        corrected_only: if True, only load meshes whose filename starts with
                        ``'N'`` (hand-corrected meshes).  Default: False.
    """

    def __init__(self, data_path, k_eig=128, op_cache_dir=None,
                 sphere_cd_threshold=0.05, corrected_only=False):
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.corrected_only = corrected_only

        if op_cache_dir is not None:
            os.makedirs(op_cache_dir, exist_ok=True)

        # Discover all valid organoid entries
        self.entries = []
        self._discover_entries(data_path)
        if corrected_only:
            print(f"OrganoidDataset: corrected_only=True — keeping 'N*' meshes only.")

        # Infer n_hks from the first entry
        self.n_hks = None
        if self.entries:
            npz = np.load(self.entries[0]['data_path'], allow_pickle=True)
            names = npz['names'].tolist()
            self.n_hks = sum(1 for n in names if n.startswith('hks_'))
            print(f"HKS features: {self.n_hks}")

        print(f"OrganoidDataset: {len(self.entries)} organoids loaded.")

        # Sphere-bias mitigation: compute/load CDs and build sampler
        self.sphere_sampler = None
        if sphere_cd_threshold > 0 and self.entries:
            cache_path = os.path.join(data_path, 'sphere_cd_cache.json')
            self._compute_sphere_cds(cache_path)
            self.sphere_sampler = BalancedBinSampler(self.entries,
                                                     threshold=sphere_cd_threshold)

    def _discover_entries(self, data_path):
        """Scan the small_meshes directory for paired _mesh.obj / _data.npz files."""
        mesh_files = sorted(glob.glob(os.path.join(data_path, '*_mesh.obj')))
        for mesh_path in mesh_files:
            stem = mesh_path[:-len('_mesh.obj')]
            data_path_npz = stem + '_data.npz'
            if not os.path.exists(data_path_npz):
                continue

            basename = os.path.basename(stem)

            # Optional filter: only hand-corrected meshes (filenames starting with 'N')
            if self.corrected_only and not basename.startswith('N'):
                continue

            # Parse id from filename stem: {timepoint}_{well}_{label}
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

    def _compute_sphere_cds(self, cache_path: str):
        """Compute (or load from cache) Chamfer-to-sphere distances for all entries.

        Results are stored in ``entry['sphere_cd']`` and persisted to
        ``cache_path`` (JSON keyed by mesh basename) so the sweep only runs
        once per dataset.  New meshes added later are computed incrementally.
        """
        # Load existing cache
        cache: dict = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cache = json.load(f)

        # Find entries that need computation
        missing = [e for e in self.entries
                   if os.path.basename(e['mesh_path']) not in cache]

        if missing:
            print(f"Computing sphere-CD for {len(missing)} mesh(es) "
                  f"(cached: {len(cache)})...")
            for entry in tqdm(missing, desc="Sphere-CD sweep", unit="mesh"):
                key = os.path.basename(entry['mesh_path'])
                cache[key] = _sphere_cd_for_mesh(entry['mesh_path'])

            with open(cache_path, 'w') as f:
                json.dump(cache, f)
            print(f"Sphere-CD cache updated: {cache_path}")
        else:
            print(f"Sphere-CD cache: {len(cache)} entries loaded from {cache_path}")

        # Attach scores to entries
        for entry in self.entries:
            key = os.path.basename(entry['mesh_path'])
            entry['sphere_cd'] = cache[key]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load pre-scaled mesh geometry
        verts_np, faces_np = pp3d.read_mesh(entry['mesh_path'])
        verts_np = verts_np.astype(np.float32)
        faces_np = faces_np.astype(np.int64)

        verts = torch.tensor(verts_np, dtype=torch.float32)
        faces = torch.tensor(faces_np, dtype=torch.int64)

        # Load pre-computed scalar fields (HKS + fates)
        npz = np.load(entry['data_path'], allow_pickle=True)
        scalars = npz['scalars']          # (V, n_features)
        names = npz['names'].tolist()     # list of feature name strings

        # Extract HKS columns (names starting with 'hks_') and normalise
        hks_mask = [n.startswith('hks_') for n in names]
        hks_np = scalars[:, hks_mask].astype(np.float32)
        hks_np = normalise_hks(hks_np)
        hks = torch.tensor(hks_np, dtype=torch.float32)

        # Compute DiffusionNet operators (with optional caching)
        frames, mass, L, evals, evecs, gradX, gradY = get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

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
