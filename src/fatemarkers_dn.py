"""
FateMarkersDN: subclass of FateMarkers that adds DiffusionNet-compatible
gradient operator preprocessing.

This computes and stores the tangent frames, gradient operators (gradX, gradY),
and DiffusionNet-compatible eigendecomposition alongside the existing
FateMarkers results.
"""

import numpy as np
import scipy.sparse as sp
import torch
import igl

from src.fatemarkers import FateMarkers

# Import DiffusionNet geometry (for tangent frames and gradient operators)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ML'))
from diffusion_net.geometry import (
    build_tangent_frames, edge_tangent_vectors, build_grad,
    vertex_normals_from_mesh
)
from diffusion_net.utils import toNP, sparse_np_to_torch, sparse_torch_to_np


class FateMarkersDN(FateMarkers):
    """
    Extends FateMarkers with DiffusionNet gradient operators.

    After calling precompute_eigens(), call precompute_diffusionnet_ops()
    to compute the tangent frames and gradient matrices needed by DiffusionNet.
    """

    def __init__(self):
        super().__init__()
        self.frames = None
        self.gradX = None
        self.gradY = None
        self.dn_massvec = None  # DiffusionNet-style mass vector (with eps regularization)
        self.dn_evals = None
        self.dn_evecs = None

    def precompute_diffusionnet_ops(self, k_eig=128):
        """
        Compute the DiffusionNet operators (tangent frames, gradient matrices,
        eigendecomposition) from the already-loaded mesh.

        This uses the same igl cotangent Laplacian as FateMarkers, but computes
        additional geometric quantities needed by DiffusionNet:
        - tangent frames at each vertex
        - sparse gradient operators gradX, gradY

        Args:
            k_eig: number of eigenvectors for DiffusionNet (default 128).
                   This can differ from self.lmax**2 used by FateMarkers.
        """
        if self.v is None or self.f is None:
            raise ValueError("Mesh not loaded. Call load_mesh_from_file() first.")

        verts = torch.tensor(self.v, dtype=torch.float64)
        faces = torch.tensor(self.f, dtype=torch.int64)

        eps = 1e-8

        # --- Tangent frames ---
        self.frames = build_tangent_frames(verts, faces)

        # --- Laplacian and mass (matching igl convention) ---
        L_np = igl.cotmatrix(self.v.astype(np.float64), self.f.astype(np.int64))
        M_np = igl.massmatrix(
            self.v.astype(np.float64), self.f.astype(np.int64),
            igl.MASSMATRIX_TYPE_VORONOI)
        massvec_np = np.array(M_np.diagonal()).flatten()
        massvec_np += eps * np.mean(massvec_np)
        self.dn_massvec = massvec_np

        # --- Eigendecomposition (positive semi-definite convention) ---
        import scipy.sparse.linalg as sla
        L_eigsh = (-L_np + sp.identity(L_np.shape[0]) * eps).tocsc()
        Mmat = sp.diags(massvec_np)

        failcount = 0
        while True:
            try:
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eps)
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))
                break
            except Exception as e:
                if failcount > 3:
                    raise ValueError("failed to compute eigendecomp") from e
                failcount += 1
                L_eigsh = L_eigsh + sp.identity(L_np.shape[0]) * (eps * 10 ** failcount)

        self.dn_evals = evals_np
        self.dn_evecs = evecs_np

        # --- Gradient operators ---
        L_coo = L_np.tocoo()
        edges = torch.tensor(
            np.stack((L_coo.row, L_coo.col), axis=0),
            dtype=torch.int64)
        edge_vecs = edge_tangent_vectors(verts, self.frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)

        self.gradX = np.real(grad_mat_np).astype(np.float32)
        self.gradY = np.imag(grad_mat_np).astype(np.float32)

    def save_diffusionnet_ops(self, path):
        """
        Save DiffusionNet operators to disk.
        Saves as {path}_dn_ops.npz.
        """
        if self.frames is None:
            raise ValueError("DiffusionNet ops not computed. Call precompute_diffusionnet_ops() first.")

        frames_np = toNP(self.frames).astype(np.float32)

        # Store sparse gradX and gradY in CSC format
        gradX_csc = sp.csc_matrix(self.gradX)
        gradY_csc = sp.csc_matrix(self.gradY)

        np.savez(
            path + '_dn_ops.npz',
            frames=frames_np,
            massvec=self.dn_massvec.astype(np.float32),
            evals=self.dn_evals.astype(np.float32),
            evecs=self.dn_evecs.astype(np.float32),
            gradX_data=gradX_csc.data,
            gradX_indices=gradX_csc.indices,
            gradX_indptr=gradX_csc.indptr,
            gradX_shape=gradX_csc.shape,
            gradY_data=gradY_csc.data,
            gradY_indices=gradY_csc.indices,
            gradY_indptr=gradY_csc.indptr,
            gradY_shape=gradY_csc.shape,
        )

    def load_diffusionnet_ops(self, path):
        """
        Load previously saved DiffusionNet operators.
        """
        data = np.load(path + '_dn_ops.npz', allow_pickle=True)
        self.frames = torch.from_numpy(data['frames'])
        self.dn_massvec = data['massvec']
        self.dn_evals = data['evals']
        self.dn_evecs = data['evecs']

        self.gradX = sp.csc_matrix((
            data['gradX_data'],
            data['gradX_indices'],
            data['gradX_indptr']),
            shape=tuple(data['gradX_shape']))

        self.gradY = sp.csc_matrix((
            data['gradY_data'],
            data['gradY_indices'],
            data['gradY_indptr']),
            shape=tuple(data['gradY_shape']))
