"""
Geometric operators for DiffusionNet.
Adapted from https://github.com/nmwsharp/diffusion-net (Sharp et al., 2022).

Modified to use igl for Laplacian/mass computation (matching the existing
pipeline in this repo) instead of robust_laplacian/potpourri3d.
"""

import scipy
import scipy.sparse.linalg as sla

import os.path
import sys
import random

import numpy as np
import scipy.spatial
import torch
import sklearn.neighbors

from .utils import toNP, sparse_np_to_torch, sparse_torch_to_np, hash_arrays, ensure_dir_exists


def _import_igl():
    """Lazy import of igl (only needed for compute_operators)."""
    import igl
    return igl


# ============================================================
#  Vector operations
# ============================================================

def norm(x, highdim=False):
    return torch.norm(x, dim=len(x.shape) - 1)

def norm2(x, highdim=False):
    return dot(x, x)

def normalize(x, divide_eps=1e-6, highdim=False):
    if len(x.shape) == 1:
        raise ValueError("called normalize() on single vector of dim " + str(x.shape))
    if not highdim and x.shape[-1] > 4:
        raise ValueError("called normalize() with large last dimension " + str(x.shape))
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)

def face_coords(verts, faces):
    return verts[faces]

def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)

def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


# ============================================================
#  Surface geometry
# ============================================================

def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    raw_normal = cross(vec_A, vec_B)
    if normalized:
        return normalize(raw_normal)
    return raw_normal

def mesh_vertex_normals(verts, faces):
    """Numpy in / numpy out."""
    face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces)))
    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    return vertex_normals

def vertex_normals_from_mesh(verts, faces):
    """Compute vertex normals for a mesh. Torch in / torch out."""
    verts_np = toNP(verts)
    faces_np = toNP(faces)
    normals = mesh_vertex_normals(verts_np, faces_np)

    # Fix any NaN normals
    bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
    if bad_normals_mask.any():
        bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
        scale = np.linalg.norm(bbox) * 1e-4
        wiggle = (np.random.RandomState(seed=777).rand(*verts_np.shape) - 0.5) * scale
        wiggle_verts = verts_np + bad_normals_mask * wiggle
        normals = mesh_vertex_normals(wiggle_verts, faces_np)

    bad_normals_mask = np.isnan(normals).any(axis=1)
    if bad_normals_mask.any():
        normals[bad_normals_mask, :] = (np.random.RandomState(seed=777).rand(*verts_np.shape) - 0.5)[bad_normals_mask, :]
        normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)
    if torch.any(torch.isnan(normals)):
        raise ValueError("NaN normals")
    return normals


# ============================================================
#  Tangent frames and gradient operators
# ============================================================

def build_tangent_frames(verts, faces, normals=None):
    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals is None:
        vert_normals = vertex_normals_from_mesh(verts, faces)
    else:
        vert_normals = normals

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where(
        (torch.abs(dot(vert_normals, basis_cand1)) < 0.9).unsqueeze(-1),
        basis_cand1, basis_cand2
    )
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame")
    return frames


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]
    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)
    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator.
    Given real inputs at vertices, produces a complex (vector value) at vertices
    giving the gradient. All values pointwise.
    edges: (2, E)
    """
    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    N = verts.shape[0]
    vert_edge_outgoing = [[] for _ in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5

    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])
        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]

        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)
            edge_vec = edge_tangent_vectors_np[iE][:]
            w_e = 1.
            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T
        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]
            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(N, N)
    ).tocsc()
    return mat


# ============================================================
#  Main operator computation (using igl)
# ============================================================

def compute_operators(verts, faces, k_eig, normals=None):
    """
    Builds spectral operators for a mesh using igl for the Laplacian and mass matrix.

    This replaces the robust_laplacian/potpourri3d dependency in the original DiffusionNet
    with igl.cotmatrix and igl.massmatrix, matching the existing pipeline in this repo.

    Arguments:
      - verts: (V,3) torch tensor of vertex positions
      - faces: (F,3) torch tensor of face indices
      - k_eig: number of eigenvectors to use

    Returns:
      - frames: (V,3,3) tangent coordinate frames
      - massvec: (V) diagonal of lumped mass matrix
      - L: (VxV) sparse Laplacian
      - evals: (k) eigenvalues
      - evecs: (V,k) eigenvectors
      - gradX: (VxV) sparse gradient operator (real part)
      - gradY: (VxV) sparse gradient operator (imaginary part)
    """
    device = verts.device
    dtype = verts.dtype
    V = verts.shape[0]
    eps = 1e-8

    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces).astype(np.int64)

    # Build tangent frames
    frames = build_tangent_frames(verts, faces, normals=normals)

    # Build Laplacian and mass matrix using igl
    igl = _import_igl()
    L_np = igl.cotmatrix(verts_np, faces_np)
    M_np = igl.massmatrix(verts_np, faces_np, igl.MASSMATRIX_TYPE_VORONOI)
    massvec_np = np.array(M_np.diagonal()).flatten()
    massvec_np += eps * np.mean(massvec_np)

    if np.isnan(L_np.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec_np).any():
        raise RuntimeError("NaN mass matrix")

    # Compute eigenbasis
    if k_eig > 0:
        # igl.cotmatrix returns a negative semi-definite matrix; negate for positive eigenvalues
        L_eigsh = (-L_np + scipy.sparse.identity(L_np.shape[0]) * eps).tocsc()
        Mmat = scipy.sparse.diags(massvec_np)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma)
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))
                break
            except Exception as e:
                print(e)
                if failcount > 3:
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L_np.shape[0]) * (eps * 10 ** failcount)
    else:
        evals_np = np.zeros((0))
        evecs_np = np.zeros((V, 0))

    # Build gradient matrices
    # Use the Laplacian sparsity pattern for edges
    L_coo = L_np.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col
    edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    grad_mat_np = build_grad(verts, edges, edge_vecs)

    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # Convert to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L_torch = sparse_np_to_torch(-L_np).to(device=device, dtype=dtype)  # negate back to positive semi-definite convention
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return frames, massvec, L_torch, evals, evecs, gradX, gradY


def get_operators(verts, faces, k_eig=128, op_cache_dir=None, normals=None, overwrite_cache=False):
    """
    Wrapper around compute_operators with disk caching.
    All arrays are computed in double precision, stored as float32, and returned
    matching the dtype/device of `verts`.
    """
    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)

    if np.isnan(verts_np).any():
        raise RuntimeError("tried to construct operators from NaN verts")

    found = False
    if op_cache_dir is not None:
        ensure_dir_exists(op_cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))

        i_cache_search = 0
        while True:
            search_path = os.path.join(
                op_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                if overwrite_cache:
                    os.remove(search_path)
                    break

                if cache_k_eig < k_eig:
                    os.remove(search_path)
                    break

                if "L_data" not in npzfile:
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    return scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)

                frames = torch.from_numpy(npzfile["frames"]).to(device=device, dtype=dtype)
                mass = torch.from_numpy(npzfile["mass"]).to(device=device, dtype=dtype)
                L = sparse_np_to_torch(read_sp_mat("L")).to(device=device, dtype=dtype)
                evals = torch.from_numpy(npzfile["evals"][:k_eig]).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(npzfile["evecs"][:, :k_eig]).to(device=device, dtype=dtype)
                gradX = sparse_np_to_torch(read_sp_mat("gradX")).to(device=device, dtype=dtype)
                gradY = sparse_np_to_torch(read_sp_mat("gradY")).to(device=device, dtype=dtype)

                found = True
                break

            except FileNotFoundError:
                break
            except Exception as E:
                print("unexpected error loading file: " + str(E))
                break

    if not found:
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, k_eig, normals=normals)

        dtype_np = np.float32

        if op_cache_dir is not None:
            L_np = sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)

            np.savez(search_path,
                     verts=verts_np.astype(dtype_np),
                     frames=toNP(frames).astype(dtype_np),
                     faces=faces_np,
                     k_eig=k_eig,
                     mass=toNP(mass).astype(dtype_np),
                     L_data=L_np.data.astype(dtype_np),
                     L_indices=L_np.indices,
                     L_indptr=L_np.indptr,
                     L_shape=L_np.shape,
                     evals=toNP(evals).astype(dtype_np),
                     evecs=toNP(evecs).astype(dtype_np),
                     gradX_data=gradX_np.data.astype(dtype_np),
                     gradX_indices=gradX_np.indices,
                     gradX_indptr=gradX_np.indptr,
                     gradX_shape=gradX_np.shape,
                     gradY_data=gradY_np.data.astype(dtype_np),
                     gradY_indices=gradY_np.indices,
                     gradY_indptr=gradY_np.indptr,
                     gradY_shape=gradY_np.shape,
                     )

    return frames, mass, L, evals, evecs, gradX, gradY


# ============================================================
#  Basis transformations
# ============================================================

def to_basis(values, basis, massvec):
    """
    Transform data into an orthonormal basis (orthonormal wrt massvec).
    Inputs:  values (B,V,D), basis (B,V,K), massvec (B,V)
    Outputs: (B,K,D)
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis.
    Inputs:  values (K,D), basis (V,K)
    Outputs: (V,D)
    """
    return torch.matmul(basis, values)


# ============================================================
#  Heat kernel signature
# ============================================================

def compute_hks(evals, evecs, scales):
    """
    Inputs:  evals (K), evecs (V,K), scales (S)
    Outputs: (V,S) hks values
    """
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1)  # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)
    out = torch.sum(terms, dim=-1)  # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    return out


def compute_hks_autoscale(evals, evecs, count):
    """Compute HKS with automatically chosen log-spaced scales."""
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)
    return compute_hks(evals, evecs, scales)
