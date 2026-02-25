"""
Utility functions for DiffusionNet.
Adapted from https://github.com/nmwsharp/diffusion-net (Sharp et al., 2022).
"""

import sys
import os
import time
import hashlib

import torch
import numpy as np
import scipy


def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array.
    """
    return x.detach().to(torch.device('cpu')).numpy()


def sparse_np_to_torch(A):
    """Convert a scipy sparse matrix to a pytorch sparse tensor."""
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(shape)
    ).coalesce()


def sparse_torch_to_np(A):
    """Convert a pytorch sparse tensor to a scipy csc matrix."""
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))
    indices = toNP(A.indices())
    values = toNP(A.values())
    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()
    return mat


def hash_arrays(arrs):
    """Hash a list of numpy arrays."""
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()


def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers.
    """
    if randgen is None:
        randgen = np.random.RandomState()

    theta, phi, z = tuple(randgen.rand(3).tolist())
    theta = theta * 2.0 * np.pi
    phi = phi * 2.0 * np.pi
    z = z * 2.0

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def random_rotate_points(pts, randgen=None):
    """Randomly rotate points. Torch in, torch out."""
    R = random_rotation_matrix(randgen)
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R)


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
