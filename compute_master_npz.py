"""
Compute all per-organoid features in one pass and save to sim/master.npz.

Arrays saved (filtered to the 95th-percentile recon-quality threshold):

  ids                 (N,)              organoid identifier strings
  times               (N,)              timepoint label strings  e.g. '3p5'
  areas               (N,)              true mesh surface area   (m.area)
  mass_areas          (N,)              modes-mesh area          (mass_matrix diag sum)
  fracs               (N,)              reconstruction quality   (lower = better)
  complexity_errors   (N, 9)            recon error at lmax = 1..9
  l_cross_values      (N,)              interpolated l where error crosses 0.015
  fate_names          (n_fates,)        ordered fate marker names
  fm_coeffs           (N, n_modes², n_fates)   sph-harm coefficients of FM fields
  hks_coeffs_sparse   (N, n_modes², 4)          HKS coeffs at ts = [1, 4, 25, 100]
  hks_bof_coeffs      (N, n_modes², n_vocab)    BoF-encoded HKS  (only if vocab found)
  hks_bof_ts          (n_ts_vocab,)              time-scale values used for BoF

Usage:
  python compute_master_npz.py [data_path] [vocab_path] [out_path]

Defaults:
  data_path  = Data/20260224
  vocab_path = sim/vocab_new.npz
  out_path   = sim/master.npz
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

from src.fatemarkers import FateMarkers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

data_path  = sys.argv[1] if len(sys.argv) > 1 else "Data/20260224"
vocab_path = sys.argv[2] if len(sys.argv) > 2 else "sim/vocab_new.npz"
out_path   = sys.argv[3] if len(sys.argv) > 3 else f"{data_path}/master.npz"

SPARSE_TS           = [1, 4, 25, 100]
COMPLEXITY_LMAX     = 9
FRAC_PERCENTILE     = 95
L_CROSS_THRESHOLD   = 0.015

with open(f"{data_path}/config.json") as f:
    cfg = json.load(f)

timepoints       = cfg["timepoints"]
zarr_names       = cfg["zarr_names"]
wells            = cfg["wells"]
rounds           = cfg["rounds"]
annotation_names = cfg["annotation_names"]
correct_order    = list(annotation_names.keys())

# ---------------------------------------------------------------------------
# Optionally load HKS vocabulary
# ---------------------------------------------------------------------------

vocab = None
if os.path.exists(vocab_path):
    res        = np.load(vocab_path, allow_pickle=True)
    vocab_hks  = res["vocab"]
    sigma_hks  = res["sigma"]
    scaler_hks = res["scaler"].item()
    ts_vocab   = res["ts"]
    vocab      = True
    print(f"Loaded HKS vocab from {vocab_path}  (n_words={vocab_hks.shape[0]})")
else:
    print(f"Vocab not found at {vocab_path} — skipping BoF HKS encoding")

# ---------------------------------------------------------------------------
# Per-organoid feature extraction
# ---------------------------------------------------------------------------

def compute_complexity_errors(m):
    return np.array([m.compute_recon_quality(lmax=l) for l in range(1, COMPLEXITY_LMAX + 1)])


def compute_hks_bof(m):
    hks        = m.compute_hks_for_new_times(ts_vocab, coeffs=False)
    hks_scaled = scaler_hks.transform(hks / np.mean(hks, axis=0) - 1)
    dist       = np.linalg.norm(
        hks_scaled[:, np.newaxis, :] - vocab_hks[np.newaxis, :, :], axis=2
    )
    encoding   = np.exp(-dist**2 / (2 * sigma_hks**2))
    return m.modes.T @ (m.mass_matrix @ encoding)


# ---------------------------------------------------------------------------
# Main loop  — mirrors double_clustering / multi_shape_clustering / correlation_multi_new
# ---------------------------------------------------------------------------

ids_raw        = []
times_raw      = []
areas_raw      = []
mass_areas_raw = []
fracs_raw      = []
complexity_raw = []
fm_coeffs_raw  = []
hks_sparse_raw = []
hks_bof_raw    = []

for timepoint in timepoints:
    zarr_name  = zarr_names[timepoint]
    round_name = rounds[timepoint]
    for well_name in wells[timepoint]:
        file_path = f"{data_path}/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
        labels = np.load(f"{file_path}/fm_data/good_labels.npy")
        for label in tqdm(labels, desc=f"{timepoint} {well_name}"):
            save_path = f"{file_path}/fm_data/{label}"
            if os.path.exists(save_path + "_coeffs.npz"):
                try:
                    m = FateMarkers()
                    m.load_results(save_path)

                    fate_indices = [
                        m.field_names.index(annotation_names[name])
                        for name in correct_order
                    ]

                    ids_raw.append(f"{timepoint}_{well_name}_{label}")
                    times_raw.append(timepoint[3:])
                    areas_raw.append(m.area)
                    mass_areas_raw.append(m.mass_matrix.diagonal().sum())
                    fracs_raw.append(m.compute_recon_quality())
                    complexity_raw.append(compute_complexity_errors(m))
                    fm_coeffs_raw.append(m.coeffs_fm[:, fate_indices])
                    hks_sparse_raw.append(m.compute_hks_for_new_times(SPARSE_TS))
                    if vocab:
                        hks_bof_raw.append(compute_hks_bof(m))

                except Exception as e:
                    print(f"Failed {save_path}: {e}")
                    continue

# ---------------------------------------------------------------------------
# Compute l_cross_values from complexity errors
# ---------------------------------------------------------------------------

l_cross_values = []
for errs in complexity_raw:
    if errs[-1] > L_CROSS_THRESHOLD:
        l_cross_values.append(COMPLEXITY_LMAX)
    else:
        l_cross = np.interp(L_CROSS_THRESHOLD, errs[::-1], np.arange(COMPLEXITY_LMAX, 0, -1))
        l_cross_values.append(l_cross)
l_cross_values = np.array(l_cross_values)

# ---------------------------------------------------------------------------
# Filter by reconstruction quality
# ---------------------------------------------------------------------------

fracs_arr = np.array(fracs_raw)
threshold = np.percentile(fracs_arr, FRAC_PERCENTILE)
mask      = fracs_arr < threshold
print(f"\nRetaining {mask.sum()} / {len(mask)} organoids (frac < {threshold:.4f})")

save_kwargs = dict(
    ids               = np.array(ids_raw)[mask],
    times             = np.array(times_raw)[mask],
    areas             = np.array(areas_raw)[mask],
    mass_areas        = np.array(mass_areas_raw)[mask],
    fracs             = fracs_arr[mask],
    complexity_errors = np.array(complexity_raw)[mask],
    l_cross_values    = l_cross_values[mask],
    fate_names        = np.array(correct_order),
    fm_coeffs         = np.array(fm_coeffs_raw)[mask],
    hks_coeffs_sparse = np.array(hks_sparse_raw)[mask],
)

if vocab:
    save_kwargs["hks_bof_coeffs"] = np.array(hks_bof_raw)[mask]
    save_kwargs["hks_bof_ts"]     = ts_vocab

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
np.savez(out_path, **save_kwargs)
print(f"Saved → {out_path}")
for k, v in save_kwargs.items():
    print(f"  {k:25s}  {np.asarray(v).shape}")
