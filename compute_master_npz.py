"""
Compute all per-organoid features across one or more datasets and save a single
combined master .npz.

Arrays saved:

  ids                 (N,)              organoid identifier strings  '{dataset}_{tp}_{well}_{label}'
  datasets            (N,)              source dataset folder name    e.g. 'main_dataset'
  times               (N,)              timepoint label strings       e.g. '3p5'
  areas               (N,)              true mesh surface area        (m.area)
  mass_areas          (N,)              modes-mesh area               (mass_matrix diag sum)
  fracs               (N,)              reconstruction quality        (lower = better)
  complexity_errors   (N, 9)            recon error at lmax = 1..9
  l_cross_values      (N,)              interpolated l where error crosses 0.015

  (when --fate is set)
  fate_names          (n_fates,)        ordered fate marker names (intersection across datasets)
  fm_coeffs           (N, n_modes², n_fates)   sph-harm coefficients of FM fields

  hks_coeffs_sparse   (N, n_modes², 4)          HKS coeffs at ts = [1, 4, 25, 100]

  For each configured vocabulary (see VOCABS below), one array:
  hks_bof_coeffs__<name>  (N, n_modes², n_features)   HKS encoded with that vocab
  bof_vocab_names         (n_vocabs,)                 ordered vocab names

Usage:
  python compute_master_npz.py [--folders DIR [DIR ...]] [--out PATH] [--fate] [--workers N]

Defaults:
  --folders  Data/main_dataset Data/sup_dataset
  --out      Data/master.npz
  --workers  1
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import concurrent.futures

from src.fatemarkers import FateMarkers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_FOLDERS = ["Data/main_dataset", "Data/sup_dataset"]
DEFAULT_OUT     = "Data/master.npz"

SPARSE_TS           = [1, 4, 25, 100]
COMPLEXITY_LMAX     = 9
L_CROSS_THRESHOLD   = 0.015
N_VARIABLE_TS       = 20

VOCABS = [
    {"name": "kmeans_variable", "path": "sim/vocab_variable_time.npz",
     "encoding": "kmeans", "time": "variable"},
    {"name": "pca_variable",    "path": "sim/vocab_pca_variable_time.npz",
     "encoding": "pca",    "time": "variable"},
]


# ---------------------------------------------------------------------------
# Load HKS vocabularies
# ---------------------------------------------------------------------------

def find_times(area, n=N_VARIABLE_TS):
    final_time = np.sqrt(area)
    return np.exp(2 * np.linspace(0, np.log(final_time), n))


def load_vocab(spec):
    res = np.load(spec["path"], allow_pickle=True)
    v = dict(spec)
    v["scaler"] = res["scaler"].item()
    if spec["encoding"] == "kmeans":
        v["centres"] = res["vocab"]
        v["sigma"]   = float(res["sigma"])
        v["n_features"] = v["centres"].shape[0]
    elif spec["encoding"] == "pca":
        v["components"] = res["components"]
        v["mean"]       = res["mean"]
        v["n_features"] = v["components"].shape[0]
    else:
        raise ValueError(f"Unknown encoding {spec['encoding']!r}")
    if spec["time"] == "fixed":
        v["ts"] = res["ts"]
    return v


def load_vocabs():
    vocabs = []
    for spec in VOCABS:
        if os.path.exists(spec["path"]):
            v = load_vocab(spec)
            vocabs.append(v)
            print(f"Loaded vocab '{v['name']}' ({v['encoding']}, {v['time']} time) "
                  f"from {spec['path']}  n_features={v['n_features']}")
        else:
            print(f"Vocab '{spec['name']}' not found at {spec['path']} — skipping")
    return vocabs


# ---------------------------------------------------------------------------
# Per-organoid feature extraction
# ---------------------------------------------------------------------------

def compute_complexity_errors(m):
    """Incrementally reconstruct at lmax=1..9 by accumulating one shell at a time."""
    recon = np.zeros_like(m.v)
    errors = []
    for l in range(1, COMPLEXITY_LMAX + 1):
        lo, hi = (l - 1) ** 2, l ** 2
        recon += m.modes[:, lo:hi] @ m.coeffs_v[lo:hi, :]
        diff = m.v - recon
        errors.append(np.sqrt(np.einsum('ij,ij->', diff, diff) / m.v.shape[0]) / np.sqrt(m.area))
    return np.array(errors)


def hks_unit_area(m, ts):
    eigvecs = m.eigvecs * np.sqrt(m.area)
    hks = np.array([
        np.einsum('i,ji->j', np.exp(-m.eigvals * t), eigvecs ** 2)
        for t in ts
    ]).T
    return hks / np.mean(hks, axis=0) - 1


def encode_vocab(hks_scaled, v):
    if v["encoding"] == "kmeans":
        dist = np.linalg.norm(
            hks_scaled[:, np.newaxis, :] - v["centres"][np.newaxis, :, :], axis=2
        )
        return np.exp(-dist**2 / (2 * v["sigma"]**2))
    else:
        return (hks_scaled - v["mean"]) @ v["components"].T


def compute_bof_coeffs(m, vocabs):
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


def fate_field(annotation_names, name):
    fld = annotation_names[name]
    return name if isinstance(fld, list) else fld


# ---------------------------------------------------------------------------
# Per-organoid worker (top-level so it's picklable)
# ---------------------------------------------------------------------------

def _worker_init(n_threads):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)


def _process_one(task):
    """Load one organoid and compute all features. Returns a result dict or None on error."""
    save_path, uid, dataset_name, timepoint, fate_order, annotation_names, compute_fate, vocabs = task
    try:
        m = FateMarkers()
        m.load_results(save_path)
        result = {
            "uid":        uid,
            "dataset":    dataset_name,
            "time":       timepoint[3:],
            "area":       m.area,
            "mass_area":  m.mass_matrix.diagonal().sum(),
            "frac":       m.compute_recon_quality(),
            "complexity": compute_complexity_errors(m),
            "hks_sparse": m.compute_hks_for_new_times(SPARSE_TS),
        }
        if compute_fate:
            fate_indices = [
                m.field_names.index(fate_field(annotation_names, n))
                for n in fate_order
            ]
            result["fm_coeffs"] = m.coeffs_fm[:, fate_indices]
        if vocabs:
            result["bof"] = compute_bof_coeffs(m, vocabs)
        return result
    except Exception as e:
        print(f"Failed {save_path}: {e}")
        return None


def _run_tasks(tasks, workers, vocabs):
    """Run tasks in parallel (workers>1) or serial, return list of result dicts."""
    results = []

    def _collect(r):
        if r is not None:
            results.append(r)

    if workers == 1:
        for t in tqdm(tasks, desc="organoids"):
            _collect(_process_one(t))
    else:
        threads_per_worker = max(1, 8 // workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init, initargs=(threads_per_worker,)) as ex:
            futs = {ex.submit(_process_one, t): t for t in tasks}
            for f in tqdm(concurrent.futures.as_completed(futs),
                          total=len(futs), desc="organoids"):
                _collect(f.result())

    return results


# ---------------------------------------------------------------------------
# Per-dataset task builders
# ---------------------------------------------------------------------------

def load_config(data_path):
    with open(f"{data_path}/config.json") as f:
        return json.load(f)


def intersect_fate_order(configs):
    key_lists = [list(c["annotation_names"].keys()) for c in configs]
    common = set(key_lists[0]).intersection(*[set(k) for k in key_lists[1:]])
    return [k for k in key_lists[0] if k in common]


def _build_tasks_vtp_flat(data_path, cfg, fate_order, compute_fate, vocabs):
    dataset_name     = os.path.basename(data_path.rstrip("/"))
    annotation_names = cfg["annotation_names"]
    vtp_dir          = cfg.get("vtp_dir", "vtp")
    tasks = []
    for timepoint in cfg["timepoints"]:
        fm_dir  = os.path.join(data_path, vtp_dir, timepoint, "fm_data")
        gl_path = os.path.join(fm_dir, f"good_labels_{timepoint}.npy")
        if not os.path.exists(gl_path):
            print(f"  [skip] no good_labels at {gl_path}")
            continue
        for label_uid in np.load(gl_path, allow_pickle=True):
            save_path = os.path.join(fm_dir, str(label_uid))
            if not os.path.exists(save_path + "_coeffs.npz"):
                continue
            tasks.append((save_path, f"{dataset_name}_{label_uid}",
                          dataset_name, timepoint,
                          fate_order, annotation_names, compute_fate, vocabs))
    return tasks


def _build_tasks_fractal_output(data_path, cfg, fate_order, compute_fate, vocabs):
    dataset_name     = os.path.basename(data_path.rstrip("/"))
    annotation_names = cfg["annotation_names"]
    timepoints, zarr_names, wells, rounds = (
        cfg["timepoints"], cfg["zarr_names"], cfg["wells"], cfg["rounds"])
    tasks = []
    for timepoint in timepoints:
        zarr_name  = zarr_names[timepoint]
        round_name = rounds[timepoint]
        for well_name in wells[timepoint]:
            file_path = (f"{data_path}/fractal_output/{timepoint}/{zarr_name}/"
                         f"{well_name[0]}/{well_name[1:]}/{round_name}/")
            gl_path = f"{file_path}/fm_data/good_labels.npy"
            if not os.path.exists(gl_path):
                print(f"  [skip] no good_labels at {gl_path}")
                continue
            for label in np.load(gl_path):
                save_path = f"{file_path}/fm_data/{label}"
                if not os.path.exists(save_path + "_coeffs.npz"):
                    continue
                uid = f"{dataset_name}_{timepoint}_{well_name}_{label}"
                tasks.append((save_path, uid, dataset_name, timepoint,
                               fate_order, annotation_names, compute_fate, vocabs))
    return tasks


def build_tasks(data_path, cfg, fate_order, compute_fate, vocabs):
    if cfg.get("layout") == "vtp_flat":
        return _build_tasks_vtp_flat(data_path, cfg, fate_order, compute_fate, vocabs)
    else:
        return _build_tasks_fractal_output(data_path, cfg, fate_order, compute_fate, vocabs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS,
                    help="dataset roots, each with its own config.json")
    ap.add_argument("--out", default=DEFAULT_OUT, help="combined master .npz path")
    ap.add_argument("--fate", action="store_true",
                    help="include fate marker coefficients (requires compute_fate=True run)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (default 1). Each gets 8//workers BLAS threads.")
    args = ap.parse_args()

    vocabs  = load_vocabs()
    configs = [load_config(fp) for fp in args.folders]

    fate_order = intersect_fate_order(configs) if args.fate else []
    print(f"\nDatasets: {args.folders}")
    if args.fate:
        print(f"Combined fate order ({len(fate_order)}): {fate_order}")
    else:
        print("Fate coefficients: skipped (pass --fate to include)")
    print()

    # Build the full task list across all datasets, then run in parallel
    all_tasks = []
    for data_path, cfg in zip(args.folders, configs):
        tasks = build_tasks(data_path, cfg, fate_order, args.fate, vocabs)
        print(f"{os.path.basename(data_path)}: {len(tasks)} organoids")
        all_tasks.extend(tasks)
    print(f"Total: {len(all_tasks)} organoids\n")

    results = _run_tasks(all_tasks, args.workers, vocabs)

    if not results:
        raise SystemExit("No organoids processed — did you run run_new_meshes.py for each dataset?")

    # Assemble accumulator from result dicts
    acc = {k: [r[k] for r in results]
           for k in ("uid", "dataset", "time", "area", "mass_area", "frac",
                     "complexity", "hks_sparse")}
    if args.fate:
        acc["fm_coeffs"] = [r["fm_coeffs"] for r in results]
    if vocabs:
        acc["bof"] = {v["name"]: [r["bof"][v["name"]] for r in results] for v in vocabs}

    # -- l_cross from complexity errors --------------------------------------
    l_cross_values = []
    for errs in acc["complexity"]:
        if errs[-1] > L_CROSS_THRESHOLD:
            l_cross_values.append(COMPLEXITY_LMAX)
        else:
            l_cross = np.interp(L_CROSS_THRESHOLD, errs[::-1],
                                np.arange(COMPLEXITY_LMAX, 0, -1))
            l_cross_values.append(l_cross)

    N = len(results)
    print(f"\nSaving {N} organoids")

    save_kwargs = dict(
        ids               = np.array(acc["uid"]),
        datasets          = np.array(acc["dataset"]),
        times             = np.array(acc["time"]),
        areas             = np.array(acc["area"]),
        mass_areas        = np.array(acc["mass_area"]),
        fracs             = np.array(acc["frac"]),
        complexity_errors = np.array(acc["complexity"]),
        l_cross_values    = np.array(l_cross_values),
        hks_coeffs_sparse = np.array(acc["hks_sparse"]),
    )

    if args.fate:
        save_kwargs["fate_names"] = np.array(fate_order)
        save_kwargs["fm_coeffs"]  = np.array(acc["fm_coeffs"])

    if vocabs:
        save_kwargs["bof_vocab_names"] = np.array([v["name"] for v in vocabs])
        for v in vocabs:
            save_kwargs[f"hks_bof_coeffs__{v['name']}"] = np.array(acc["bof"][v["name"]])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **save_kwargs)
    print(f"Saved → {args.out}")
    for k, v in save_kwargs.items():
        print(f"  {k:25s}  {np.asarray(v).shape}")


if __name__ == "__main__":
    main()
