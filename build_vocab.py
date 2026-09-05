"""
Refit the HKS bag-of-features vocabularies on the vtp_flat datasets
(main_dataset + sup_dataset + pert2), so the codebook spans the perturbation
shapes too (the old vocab was fit only on Data/20260224 = WT main, no perts).

This mirrors the vocab-building logic in bag_of_features.ipynb exactly
(decimate->per-vertex HKS at 20 variable times -> per-organoid normalise ->
95th-pct recon filter -> complexity-balanced subsample -> Normalizer ->
curvature mask -> KMeans k=8 for the kmeans vocab, PCA-3 for the pca vocab).
Only the data source changes: it walks each dataset's config.json timepoints and
reads vtp/<tp>/fm_data/good_labels_<tp>.npy instead of the fractal_output tree.

Eigenvalues/eigenvectors are NOT recomputed — they are read from the saved
_coeffs.npz in fm_data (run_new_meshes.py output).

Outputs (same format/shape as before -> compute_master_npz reads them unchanged):
  sim/vocab_variable_time.npz       vocab (8,20), scaler (Normalizer), sigma
  sim/vocab_pca_variable_time.npz   components (3,20), mean (20,), scaler

Run in the scmpx env:
  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/anaconda3/envs/scmpx/bin/python build_vocab.py --workers 6
"""

import os
import json
import argparse
import concurrent.futures

import numpy as np
from tqdm import tqdm
import igl
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.fatemarkers import FateMarkers

DATASETS      = ["Data/main_dataset", "Data/sup_dataset", "Data/pert2"]
N_TIMES       = 20
DECIMATE_FACE = 200
KMEANS_K      = 8
PCA_COMPS     = 3
COMPLEXITY_THRESHOLD = 0.005                 # matches bag_of_features complexity()
COMPLEXITY_INTERVALS = [(0, 5.3), (5.3, np.inf)]
RECON_PCTL    = 95
CURV_Z        = 4
SEED          = 42


def _init_worker(n_threads):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)


def find_times(area, n=N_TIMES):
    final_time = np.sqrt(area)
    return np.exp(2 * np.linspace(0, np.log(final_time), n))


def complexity(m, threshold=COMPLEXITY_THRESHOLD):
    # max SH degree (lmax) at which recon error first drops below threshold;
    # lmax scale (0 = single degree-0 mode). Mirrors bag_of_features.complexity().
    for l in range(1, m.lmax + 1):
        if m.compute_recon_quality(lmax=l) < threshold:
            return l - 1
    return m.lmax - 1


def load_one(save_path):
    """Return (hks[n_decim_verts, N_TIMES], frac, complexity, area) for one organoid.
    Mirrors bag_of_features.ipynb load(): decimate to DECIMATE_FACE faces, compute
    per-vertex HKS at find_times(area) on the surviving vertices."""
    m = FateMarkers()
    m.load_results(save_path)

    _, _, _, _, new_indices = igl.decimate(m.v, m.f, DECIMATE_FACE)
    ts = find_times(m.area)
    eigvecs = m.eigvecs * np.sqrt(m.area)
    hks = np.array([
        np.einsum('i, ji->j', np.exp(-m.eigvals * t), eigvecs[new_indices, :] ** 2)
        for t in ts
    ]).T
    return hks, m.compute_recon_quality(), complexity(m), m.area


def _run_one(save_path):
    try:
        hks, frac, c, area = load_one(save_path)
        return save_path, hks, frac, c, area
    except Exception as e:
        print(f"Failed {save_path}: {e}")
        return None


def normalise(hks):
    # per-organoid: divide each time-column by its mean over vertices, minus 1
    return hks / np.mean(hks, axis=0, keepdims=True) - 1


def gather_save_paths():
    paths = []
    for ds in DATASETS:
        with open(f"{ds}/config.json") as f:
            cfg = json.load(f)
        vtp_dir = cfg.get("vtp_dir", "vtp")
        for tp in cfg["timepoints"]:
            fm_dir = os.path.join(ds, vtp_dir, tp, "fm_data")
            gl = os.path.join(fm_dir, f"good_labels_{tp}.npy")
            if not os.path.exists(gl):
                print(f"  [skip] no good_labels at {gl}")
                continue
            for label in np.load(gl, allow_pickle=True):
                paths.append(os.path.join(fm_dir, str(label)))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--suffix", default="",
                    help="appended to the output vocab filenames (e.g. '_reprocheck' to "
                         "write beside the live vocab without overwriting it)")
    args = ap.parse_args()

    save_paths = gather_save_paths()
    print(f"organoids to load: {len(save_paths)} across {len(DATASETS)} datasets")

    # ---- load per-organoid HKS (reads saved eigvecs; no eigendecomposition) ----
    results = []
    if args.workers == 1:
        for p in tqdm(save_paths, desc="load"):
            r = _run_one(p)
            if r is not None:
                results.append(r)
    else:
        tpw = max(1, 8 // args.workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker, initargs=(tpw,)) as ex:
            for r in tqdm(ex.map(_run_one, save_paths), total=len(save_paths), desc="load"):
                if r is not None:
                    results.append(r)

    hkss         = [r[1] for r in results]
    fracs        = np.array([r[2] for r in results])
    complexities = np.array([r[3] for r in results])
    print(f"loaded {len(hkss)} organoids")

    # ---- 95th-pct recon-quality filter ----
    recon_mask = fracs < np.percentile(fracs, RECON_PCTL)
    recon_idx  = np.where(recon_mask)[0]
    filtered_hkss         = [hkss[i] for i in recon_idx]
    filtered_complexities = complexities[recon_mask]
    print(f"recon filter: kept {len(filtered_hkss)}/{len(hkss)} "
          f"(dropped worst {100 - RECON_PCTL}% recon)")

    # ---- complexity-balanced subsample ----
    rng = np.random.default_rng(SEED)
    per_group = [int(np.sum((filtered_complexities >= lo) & (filtered_complexities <= hi)))
                 for lo, hi in COMPLEXITY_INTERVALS]
    samples_per_group = int(np.min(per_group))
    print(f"complexity groups {COMPLEXITY_INTERVALS} counts {per_group} "
          f"-> {samples_per_group} each")
    selected_indices = []
    for lo, hi in COMPLEXITY_INTERVALS:
        gi = np.where((filtered_complexities >= lo) & (filtered_complexities <= hi))[0]
        selected_indices.extend(rng.choice(gi, size=samples_per_group, replace=False))

    # ---- per-organoid normalise, concatenate all vertices ----
    normalised = [normalise(filtered_hkss[i]) for i in selected_indices]
    collected_hks = np.concatenate(normalised)
    print(f"collected per-vertex HKS: {collected_hks.shape}")

    # ---- Normalizer scale + curvature mask ----
    scaler = Normalizer()
    rescaled_hks = scaler.fit_transform(collected_hks)
    z0 = (rescaled_hks[:, 0] - rescaled_hks[:, 0].mean()) / rescaled_hks[:, 0].std()
    curvature_mask = np.abs(z0) < CURV_Z
    filtered_rescaled_hks = rescaled_hks[curvature_mask]
    print(f"curvature mask (|z(col0)| < {CURV_Z}): "
          f"{filtered_rescaled_hks.shape[0]}/{rescaled_hks.shape[0]} points")

    # ---- KMeans vocab ----
    km = KMeans(n_clusters=KMEANS_K, random_state=SEED)
    km.fit(filtered_rescaled_hks)
    centers, dist = [], np.zeros(filtered_rescaled_hks.shape[0])
    for i in range(KMEANS_K):
        sel = km.labels_ == i
        center = filtered_rescaled_hks[sel].mean(axis=0)
        centers.append(center)
        dist[sel] = np.linalg.norm(filtered_rescaled_hks[sel] - center.reshape(1, -1), axis=1)
    sigma = float(np.sqrt(np.mean(dist ** 2)))
    vocab = np.array(centers)
    print(f"KMeans vocab {vocab.shape}  sigma={sigma:.6f}")

    # provenance: the recipe that produced this codebook, so the file records
    # which datasets / knobs it was fit on (not just the fitted centres).
    meta = dict(
        meta_datasets=np.array(DATASETS),
        meta_kmeans_k=KMEANS_K,
        meta_pca_comps=PCA_COMPS,
        meta_n_times=N_TIMES,
        meta_decimate_face=DECIMATE_FACE,
        meta_complexity_threshold=COMPLEXITY_THRESHOLD,
        meta_complexity_intervals=np.array(COMPLEXITY_INTERVALS, dtype=float),
        meta_recon_pctl=RECON_PCTL,
        meta_curv_z=CURV_Z,
        meta_seed=SEED,
        meta_n_loaded=len(hkss),
        meta_n_after_recon=len(filtered_hkss),
        meta_samples_per_group=samples_per_group,
        meta_n_train_points=int(collected_hks.shape[0]),
        meta_n_curv_points=int(filtered_rescaled_hks.shape[0]),
    )
    km_out  = f"sim/vocab_variable_time{args.suffix}.npz"
    pca_out = f"sim/vocab_pca_variable_time{args.suffix}.npz"

    np.savez(km_out, vocab=vocab, scaler=scaler, sigma=sigma, **meta)
    print(f"saved {km_out}")

    # ---- PCA vocab ----
    pca = PCA(n_components=PCA_COMPS).fit(filtered_rescaled_hks)
    print(f"PCA vocab components {pca.components_.shape}  "
          f"explained variance ratio {np.round(pca.explained_variance_ratio_, 4)}")
    np.savez(pca_out, components=pca.components_, mean=pca.mean_, scaler=scaler, **meta)
    print(f"saved {pca_out}")


if __name__ == "__main__":
    main()
