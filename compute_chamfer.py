"""
Compute a symmetric chamfer-distance matrix between high-complexity organoids.

Chamfer distance is used as *trustworthy-when-small* supervision for learning HKS
spectrum weights (see optimize_hks_weights.py). Because it is only meaningful for
already-aligned meshes, we restrict to high-complexity organoids and read the
PCA-transformed, cell-unit-rescaled meshes straight off disk (no re-alignment).

Scope (matches the weight-learning subset):
  - organoids with l_cross_value > L_CROSS_MIN
  - each mesh optionally AREA-NORMALISED to surface area 1 when NORMALISE_MESHES
    is True (OFF by default — the non-normalised chamfer is a better proxy for
    shape closeness here), then decimated by DECIMATE_FACTOR before chamfer
  - meshes igl.decimate cannot collapse (non-manifold) are SKIPPED, not kept at
    full resolution: chamfer is point-density dependent, so mixing a full-res
    cloud with decimated ones would bias that organoid's distances

Symmetric chamfer for two vertex point clouds A, B:
  cd(A, B) = mean_a min_b ||a - b||  +  mean_b min_a ||b - a||

Output: Data/npz/chamfer_highcomplexity.npz
  ids  (n,)    organoid ids, row/col order of C
  C    (n, n)  symmetric chamfer matrix, zero diagonal

Run with the scmpx env (has igl):
  /opt/homebrew/anaconda3/envs/scmpx/bin/python compute_chamfer.py
"""

import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import igl

from src import utils


MASTER          = "Data/npz/master.npz"
OUT             = "Data/npz/chamfer_highcomplexity.npz"
L_CROSS_MIN     = 5.3   # lmax scale (degree)
DECIMATE_FACTOR = 10
VTP_CFG         = {"vtp_dir": "vtp"}   # both datasets use layout 'vtp_flat', vtp_dir 'vtp'
NORMALISE_MESHES  = False


def obj_path(dataset, full_id):
    """Resolve the transformed-mesh .obj for an organoid id '{dataset}_{tp}_{well}_{label}'."""
    bare_uid = full_id[len(dataset) + 1:]          # strip '{dataset}_' prefix
    return utils.vtp_flat_obj_path(f"Data/{dataset}", VTP_CFG, bare_uid)


def _surface_area(v, f):
    """Total triangle-mesh surface area."""
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum()


def load_cloud(path):
    """Read a mesh, optionally area-normalise, decimate, return vertices (M, 3),
    or None if decimation fails.

    When NORMALISE_MESHES is True the vertices are scaled by 1/sqrt(area) so every
    mesh has surface area 1 (scale-invariant / pure-shape chamfer, matching the
    unit-area HKS). It is False by default here — the non-normalised chamfer is a
    better shape-closeness proxy for the weight learning.

    Returns None when igl.decimate fails (ok=False / no faces) — a few source
    meshes are non-manifold and can't be collapsed. Such a mesh must be SKIPPED,
    not kept at full resolution: chamfer is point-density dependent, so mixing a
    full ~18k-vertex cloud with everyone else's ~3.6k-vertex decimated clouds
    would bias its distances. The caller drops these organoids.
    """
    v, f = igl.read_triangle_mesh(path)
    if NORMALISE_MESHES:
        area = _surface_area(v, f)
        if area > 0:
            v = v / np.sqrt(area)                      # -> surface area == 1
    target_faces = max(4, f.shape[0] // DECIMATE_FACTOR)
    ok, vd, fd, _, _ = igl.decimate(v, f, target_faces)
    # igl.decimate may leave unreferenced verts; keep only those used by a face.
    used = np.unique(fd)
    if not ok or len(used) == 0:
        return None                                     # non-manifold / undecimatable -> skip
    return np.ascontiguousarray(vd[used])


def chamfer(pts_a, tree_a, pts_b, tree_b):
    """Symmetric chamfer between two clouds given their prebuilt KD-trees."""
    da, _ = tree_b.query(pts_a)
    db, _ = tree_a.query(pts_b)
    return float(da.mean() + db.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=MASTER)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=8, help="threads for the pair loop")
    ap.add_argument("--l_cross_min", type=float, default=L_CROSS_MIN,
                    help="keep organoids with l_cross strictly above this")
    ap.add_argument("--l_cross_max", type=float, default=np.inf,
                    help="keep organoids with l_cross at or below this "
                         "(e.g. a small value -> low-complexity / sphere-like organoids)")
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly sample this many organoids from the range (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = np.load(args.master, allow_pickle=True)
    ids      = m["ids"].astype(str)
    datasets = m["datasets"].astype(str)
    l_cross  = m["l_cross_values"]

    sel = np.where((l_cross > args.l_cross_min) & (l_cross <= args.l_cross_max))[0]
    if args.sample and len(sel) > args.sample:
        sel = np.sort(np.random.default_rng(args.seed).choice(sel, args.sample, replace=False))
    sub_ids, sub_ds = ids[sel], datasets[sel]
    n = len(sub_ids)
    print(f"Organoids with {args.l_cross_min} < l_cross <= {args.l_cross_max}"
          f"{f' (sampled {args.sample})' if args.sample else ''}: {n}")

    # --- load + decimate all meshes once, build KD-trees -------------------
    clouds, kept_ids, kept_ds = [], [], []
    for uid, ds in tqdm(list(zip(sub_ids, sub_ds)), desc="meshes"):
        path = obj_path(ds, uid)
        if not os.path.exists(path):
            print(f"  [skip] missing mesh: {path}")
            continue
        cloud = load_cloud(path)
        if cloud is None:
            print(f"  [skip] decimation failed (non-manifold mesh): {uid}")
            continue
        clouds.append(cloud)
        kept_ids.append(uid)
        kept_ds.append(ds)

    n = len(clouds)
    kept_ids = np.array(kept_ids)
    trees = [cKDTree(c) for c in clouds]
    print(f"Loaded {n} clouds "
          f"(verts: min {min(map(len, clouds))}, max {max(map(len, clouds))}, "
          f"mean {int(np.mean([len(c) for c in clouds]))})")

    # --- pairwise chamfer (threaded over rows; cKDTree.query releases GIL) --
    C = np.zeros((n, n), dtype=np.float64)

    def row(i):
        out = np.zeros(n)
        for j in range(i + 1, n):
            out[j] = chamfer(clouds[i], trees[i], clouds[j], trees[j])
        return i, out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, out in tqdm(ex.map(row, range(n)), total=n, desc="chamfer rows"):
            C[i, i + 1:] = out[i + 1:]

    C = C + C.T   # symmetrize (diagonal stays 0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # provenance: the parameters that produced this matrix, so the file is
    # self-describing (which master, l_cross window, area-normalisation, etc.)
    np.savez(args.out, ids=kept_ids, C=C,
             meta_master=str(args.master),
             meta_l_cross_min=float(args.l_cross_min),
             meta_l_cross_max=float(args.l_cross_max),
             meta_normalise_meshes=bool(NORMALISE_MESHES),
             meta_decimate_factor=int(DECIMATE_FACTOR),
             meta_sample=int(args.sample),
             meta_seed=int(args.seed),
             meta_n=int(n))
    iu = np.triu_indices(n, k=1)
    print(f"\nSaved -> {args.out}   C {C.shape}")
    print(f"chamfer off-diagonal: min {C[iu].min():.4f}  median {np.median(C[iu]):.4f}  "
          f"max {C[iu].max():.4f}")


if __name__ == "__main__":
    main()
