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

  For each configured vocabulary (see VOCABS below), one array:
  hks_bof_coeffs__<name>  (N, n_modes², n_features)   HKS encoded with that vocab
  bof_vocab_names         (n_vocabs,)                 ordered vocab names

Each vocabulary encodes the per-vertex heat-kernel signature (HKS) and projects
the encoding onto the spherical-harmonic modes. Two encoding types are supported:

  kmeans : soft bag-of-features assignment to KMeans cluster centres
           encoding[v, w] = exp(-||hks_v - centre_w||² / 2σ²)
  pca    : projection (coefficients) onto the leading PCA components
           encoding[v, c] = (hks_v - mean) · component_c

Time rescaling: 'variable'-time vocabs compute the HKS at organoid-specific
time scales ts = exp(2·linspace(0, log(sqrt(area)), 20)) with the LB eigvecs
rescaled to unit area (eigvecs·sqrt(area)) — matching how the vocab was trained
in bag_of_features.ipynb. 'fixed'-time vocabs use the stored `ts` array.

Usage:
  python compute_master_npz.py [data_path] [out_path]

Defaults:
  data_path  = Data/20260224
  out_path   = {data_path}/master.npz
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
out_path   = sys.argv[2] if len(sys.argv) > 2 else f"{data_path}/master.npz"

SPARSE_TS           = [1, 4, 25, 100]
COMPLEXITY_LMAX     = 9
FRAC_PERCENTILE     = 95
L_CROSS_THRESHOLD   = 0.015
N_VARIABLE_TS       = 20    # number of organoid-specific time scales

# Vocabularies to encode against. Each produces an `hks_bof_coeffs__<name>` array.
VOCABS = [
    {"name": "kmeans_variable", "path": "sim/vocab_variable_time.npz",
     "encoding": "kmeans", "time": "variable"},
    {"name": "pca_variable",    "path": "sim/vocab_pca_variable_time.npz",
     "encoding": "pca",    "time": "variable"},
]

with open(f"{data_path}/config.json") as f:
    cfg = json.load(f)

timepoints       = cfg["timepoints"]
zarr_names       = cfg["zarr_names"]
wells            = cfg["wells"]
rounds           = cfg["rounds"]
annotation_names = cfg["annotation_names"]
correct_order    = list(annotation_names.keys())

# ---------------------------------------------------------------------------
# Load HKS vocabularies
# ---------------------------------------------------------------------------

def find_times(area, n=N_VARIABLE_TS):
    """Organoid-specific HKS time scales (matches bag_of_features.ipynb)."""
    final_time = np.sqrt(area)
    return np.exp(2 * np.linspace(0, np.log(final_time), n))


def load_vocab(spec):
    res = np.load(spec["path"], allow_pickle=True)
    v = dict(spec)
    v["scaler"] = res["scaler"].item()
    if spec["encoding"] == "kmeans":
        v["centres"] = res["vocab"]            # (n_words, n_t)
        v["sigma"]   = float(res["sigma"])
        v["n_features"] = v["centres"].shape[0]
    elif spec["encoding"] == "pca":
        v["components"] = res["components"]    # (n_comp, n_t)
        v["mean"]       = res["mean"]          # (n_t,)
        v["n_features"] = v["components"].shape[0]
    else:
        raise ValueError(f"Unknown encoding {spec['encoding']!r}")
    if spec["time"] == "fixed":
        v["ts"] = res["ts"]
    return v


vocabs = []
for spec in VOCABS:
    if os.path.exists(spec["path"]):
        v = load_vocab(spec)
        vocabs.append(v)
        print(f"Loaded vocab '{v['name']}' ({v['encoding']}, {v['time']} time) "
              f"from {spec['path']}  n_features={v['n_features']}")
    else:
        print(f"Vocab '{spec['name']}' not found at {spec['path']} — skipping")

# ---------------------------------------------------------------------------
# Per-organoid feature extraction
# ---------------------------------------------------------------------------

def compute_complexity_errors(m):
    return np.array([m.compute_recon_quality(lmax=l) for l in range(1, COMPLEXITY_LMAX + 1)])


def hks_unit_area(m, ts):
    """Per-vertex HKS at the given times with unit-area eigvecs, then the
    per-time mean-removal used during vocab training."""
    eigvecs = m.eigvecs * np.sqrt(m.area)
    hks = np.array([
        np.einsum('i, ji->j', np.exp(-m.eigvals * t), eigvecs ** 2)
        for t in ts
    ]).T
    return hks / np.mean(hks, axis=0) - 1


def encode_vocab(hks_scaled, v):
    if v["encoding"] == "kmeans":
        dist = np.linalg.norm(
            hks_scaled[:, np.newaxis, :] - v["centres"][np.newaxis, :, :], axis=2
        )
        return np.exp(-dist**2 / (2 * v["sigma"]**2))
    else:  # pca — projection (coefficients) onto leading components
        return (hks_scaled - v["mean"]) @ v["components"].T


def compute_bof_coeffs(m):
    """Return {vocab_name: (n_modes, n_features)} sph-harm coeffs for every vocab.

    The raw HKS is cached per time-array so vocabs sharing a time scheme
    (e.g. all 'variable' vocabs of one organoid) only compute it once."""
    hks_cache = {}
    out = {}
    for v in vocabs:
        ts  = find_times(m.area) if v["time"] == "variable" else np.asarray(v["ts"])
        key = ts.tobytes()
        if key not in hks_cache:
            hks_cache[key] = hks_unit_area(m, ts)
        hks_scaled = v["scaler"].transform(hks_cache[key])
        encoding   = encode_vocab(hks_scaled, v)
        out[v["name"]] = m.modes.T @ (m.mass_matrix @ encoding)
    return out


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
bof_raw        = {v["name"]: [] for v in vocabs}

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
                    if vocabs:
                        for name, coeffs in compute_bof_coeffs(m).items():
                            bof_raw[name].append(coeffs)

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

if vocabs:
    save_kwargs["bof_vocab_names"] = np.array([v["name"] for v in vocabs])
    for v in vocabs:
        save_kwargs[f"hks_bof_coeffs__{v['name']}"] = np.array(bof_raw[v["name"]])[mask]

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
np.savez(out_path, **save_kwargs)
print(f"Saved → {out_path}")
for k, v in save_kwargs.items():
    print(f"  {k:25s}  {np.asarray(v).shape}")
