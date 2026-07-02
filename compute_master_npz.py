"""
Compute all per-organoid features across one or more datasets and save a single
combined master .npz.

Arrays saved:

  ids                 (N,)              organoid identifier strings  '{dataset}_{tp}_{well}_{label}'
  datasets            (N,)              source dataset folder name    e.g. 'main_dataset'
  times               (N,)              developmental-day label       e.g. '3p5' (config-driven)
  conditions          (N,)              perturbation/genotype label   'WT' | e.g. 'stem-ChirVpaD1'
  areas               (N,)              true mesh surface area        (m.area)
  mass_areas          (N,)              modes-mesh area               (mass_matrix diag sum)
  fracs               (N,)              reconstruction quality        (lower = better)
  complexity_errors   (N, 9)            recon error at lmax = 0..8 (col i = SH degrees 0..i)
  l_cross_values      (N,)              interpolated lmax (max SH degree) where error crosses 0.005

  (when --fate is set)
  fate_names          (n_fates,)        ordered fate marker names (intersection across datasets)
  fm_coeffs           (N, n_modes², n_fates)   sph-harm coefficients of FM fields

  hks_coeffs_sparse   (N, n_modes², 4)          HKS coeffs at ts = [1, 4, 25, 100]

  For each configured vocabulary (see VOCABS below), one array:
  hks_bof_coeffs__<name>  (N, n_modes², n_features)   HKS encoded with that vocab
  bof_vocab_names         (n_vocabs,)                 ordered vocab names

Usage:
  python compute_master_npz.py [--folders DIR [DIR ...]] [--out PATH] [--fate]
                               [--workers N] [--update]
                               [--l-cross-threshold T] [--recompute-l-cross]

Defaults:
  --folders  Data/main_dataset Data/sup_dataset
  --out      Data/npz/master.npz
  --workers  1

--update incrementally syncs an existing --out to the current good_labels
(i.e. after editing the discard lists): rows whose uid is no longer in
good_labels are dropped, and only uids that should be present but are missing
from the npz are computed; uids already present are left untouched. The run
must use the same --fate / vocab setup as the existing file. Without --update
(or if --out does not exist) every organoid is recomputed from scratch.

--recompute-l-cross only re-derives l_cross_values in an existing --out from its
stored complexity_errors at --l-cross-threshold; no meshes are reprocessed. Use
it to pick a complexity threshold (see complexity_analysis.ipynb) without
rebuilding the npz, e.g.
  python compute_master_npz.py --recompute-l-cross --l-cross-threshold 0.02
--l-cross-threshold also applies to full and --update runs.
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

DEFAULT_FOLDERS = ["Data/main_dataset", "Data/sup_dataset", "Data/pert"]
DEFAULT_OUT     = "Data/npz/master.npz"

SPARSE_TS           = [1, 4, 25, 100]
COMPLEXITY_LMAX     = 9
L_CROSS_THRESHOLD   = 0.005
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
    """Incrementally reconstruct at lmax=0..8 by accumulating one degree band at a time.

    Column i is the area-normalised RMSE of the reconstruction including SH degrees
    0..i (i.e. lmax=i); column 0 uses the single degree-0 mode."""
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
    save_path, uid, dataset_name, time, condition, fate_order, annotation_names, compute_fate, vocabs = task
    try:
        m = FateMarkers()
        m.load_results(save_path)
        result = {
            "uid":        uid,
            "dataset":    dataset_name,
            "time":       time,
            "condition":  condition,
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


def derive_time(cfg, timepoint):
    """Developmental-day label for a row. Config-driven so a dataset whose
    subfolders are not days (e.g. pert: normal/small) can pin a constant day:
      cfg['time']      -> constant for the whole dataset (e.g. pert: '4p5')
      cfg['time_map']  -> {subfolder: day}
      otherwise        -> strip a leading 'day' (main/sup: 'day4p5' -> '4p5')."""
    if "time" in cfg:
        return cfg["time"]
    if "time_map" in cfg:
        return cfg["time_map"][timepoint]
    return timepoint[3:] if timepoint.startswith("day") else timepoint


def derive_condition(cfg, uid, timepoint):
    """Perturbation/genotype label for a row. Config-driven:
      cfg['condition']           -> constant (main/sup: 'WT')
      cfg['condition_map']       -> {subfolder: condition}
      cfg['condition_uid_index'] -> uid.split('_')[i] (pert: 1 -> the drug token)
      otherwise                  -> 'WT'."""
    if "condition" in cfg:
        return cfg["condition"]
    if "condition_map" in cfg:
        return cfg["condition_map"][timepoint]
    if "condition_uid_index" in cfg:
        return uid.split("_")[cfg["condition_uid_index"]]
    return "WT"


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
            uid = f"{dataset_name}_{label_uid}"
            tasks.append((save_path, uid, dataset_name,
                          derive_time(cfg, timepoint), derive_condition(cfg, uid, timepoint),
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
                tasks.append((save_path, uid, dataset_name,
                              derive_time(cfg, timepoint), derive_condition(cfg, uid, timepoint),
                              fate_order, annotation_names, compute_fate, vocabs))
    return tasks


def build_tasks(data_path, cfg, fate_order, compute_fate, vocabs):
    if cfg.get("layout") == "vtp_flat":
        return _build_tasks_vtp_flat(data_path, cfg, fate_order, compute_fate, vocabs)
    else:
        return _build_tasks_fractal_output(data_path, cfg, fate_order, compute_fate, vocabs)


# ---------------------------------------------------------------------------
# Assemble / merge saved arrays
# ---------------------------------------------------------------------------

# Arrays that label the columns rather than the rows (one entry per fate /
# vocab, not per organoid). Everything else in the saved npz is per-organoid.
GLOBAL_KEYS = ("fate_names", "bof_vocab_names")


def compute_l_cross(complexity_errors, threshold=L_CROSS_THRESHOLD):
    """Interpolated lmax (max SH degree) where the recon error crosses `threshold`.

    complexity_errors[i] is the area-normalised RMSE of the reconstruction that
    includes SH degrees 0..i, i.e. lmax=i. So the crossing degree runs over
    0..(COMPLEXITY_LMAX - 1): lmax=0 means a single mode (the degree-0 constant).
    """
    lmax_max = COMPLEXITY_LMAX - 1
    out = []
    for errs in complexity_errors:
        errs = np.asarray(errs)
        if errs[-1] > threshold:
            out.append(float(lmax_max))
        else:
            out.append(float(np.interp(threshold, errs[::-1],
                                       np.arange(lmax_max, -1, -1))))
    return np.array(out)


def assemble_save_kwargs(results, fate_order, vocabs, compute_fate,
                         l_cross_threshold=L_CROSS_THRESHOLD):
    """Turn per-organoid result dicts into the dict of arrays saved to the npz."""
    acc = {k: [r[k] for r in results]
           for k in ("uid", "dataset", "time", "condition", "area", "mass_area",
                     "frac", "complexity", "hks_sparse")}

    out = dict(
        ids               = np.array(acc["uid"]),
        datasets          = np.array(acc["dataset"]),
        times             = np.array(acc["time"]),
        conditions        = np.array(acc["condition"]),
        areas             = np.array(acc["area"]),
        mass_areas        = np.array(acc["mass_area"]),
        fracs             = np.array(acc["frac"]),
        complexity_errors = np.array(acc["complexity"]),
        l_cross_values    = compute_l_cross(acc["complexity"], l_cross_threshold),
        hks_coeffs_sparse = np.array(acc["hks_sparse"]),
    )
    if compute_fate:
        out["fate_names"] = np.array(fate_order)
        out["fm_coeffs"]  = np.array([r["fm_coeffs"] for r in results])
    if vocabs:
        out["bof_vocab_names"] = np.array([v["name"] for v in vocabs])
        for v in vocabs:
            out[f"hks_bof_coeffs__{v['name']}"] = np.array([r["bof"][v["name"]] for r in results])
    return out


def merge_existing(existing, new_kwargs, keep_mask):
    """Filter existing rows by keep_mask and append the newly computed rows.

    Global label arrays must agree; per-row column sets must match exactly when
    there are new rows (otherwise the merge would misalign columns)."""
    per_row_existing = sorted(k for k in existing if k not in GLOBAL_KEYS)
    per_row_new      = sorted(k for k in new_kwargs if k not in GLOBAL_KEYS)
    if new_kwargs and per_row_existing != per_row_new:
        raise SystemExit(
            "--update: the existing npz and this run produce different columns "
            f"({per_row_existing} vs {per_row_new}). Match the --fate / vocab "
            "setup, or rebuild without --update.")

    merged = {}
    for k in GLOBAL_KEYS:
        ev, nv = existing.get(k), new_kwargs.get(k)
        if ev is not None and nv is not None and not np.array_equal(ev, nv):
            raise SystemExit(
                f"--update: '{k}' differs between the existing npz and this run "
                f"({ev} vs {nv}). Rebuild without --update.")
        if ev is not None or nv is not None:
            merged[k] = ev if ev is not None else nv

    for k in per_row_existing:
        kept = existing[k][keep_mask]
        merged[k] = np.concatenate([kept, new_kwargs[k]], axis=0) if new_kwargs else kept
    return merged


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
    ap.add_argument("--update", action="store_true",
                    help="incrementally sync an existing --out to the current good_labels: "
                         "drop discarded uids and compute only the newly-present ones.")
    ap.add_argument("--l-cross-threshold", type=float, default=L_CROSS_THRESHOLD,
                    help=f"recon-error threshold for l_cross (default {L_CROSS_THRESHOLD}).")
    ap.add_argument("--recompute-l-cross", action="store_true",
                    help="only recompute l_cross_values in an existing --out from its stored "
                         "complexity_errors at --l-cross-threshold; no meshes are reprocessed.")
    ap.add_argument("--recompute-meta", action="store_true",
                    help="only re-derive the metadata arrays (times, conditions) for every row "
                         "in an existing --out from the current configs; no meshes are reprocessed. "
                         "Use it to fix labels or backfill 'conditions' before an --update.")
    args = ap.parse_args()

    # Fast path: just re-derive l_cross from the already-stored per-degree errors
    # at a new threshold. No configs, vocabs, or mesh processing needed.
    if args.recompute_l_cross:
        if not os.path.exists(args.out):
            raise SystemExit(f"--recompute-l-cross: {args.out} does not exist.")
        existing = dict(np.load(args.out, allow_pickle=True))
        if "complexity_errors" not in existing:
            raise SystemExit("--recompute-l-cross: no complexity_errors in the npz "
                             "(rebuild it once so per-degree errors are stored).")
        old = existing.get("l_cross_values")
        existing["l_cross_values"] = compute_l_cross(existing["complexity_errors"],
                                                     args.l_cross_threshold)
        np.savez(args.out, **existing)
        new = existing["l_cross_values"]
        print(f"Recomputed l_cross_values for {len(new)} organoids at threshold "
              f"{args.l_cross_threshold} → {args.out}")
        print(f"  mean l_cross: {new.mean():.3f}"
              + (f" (was {np.asarray(old, float).mean():.3f})" if old is not None else ""))
        return

    vocabs  = load_vocabs()
    configs = [load_config(fp) for fp in args.folders]

    fate_order = intersect_fate_order(configs) if args.fate else []
    print(f"\nDatasets: {args.folders}")
    if args.fate:
        print(f"Combined fate order ({len(fate_order)}): {fate_order}")
    else:
        print("Fate coefficients: skipped (pass --fate to include)")
    print()

    # Fast path: re-derive only the metadata arrays (times, conditions) from the
    # current configs for every row already in the npz. No meshes are reprocessed.
    # Fixes mislabelled rows and backfills 'conditions' onto an older npz so a
    # later --update doesn't trip the "different columns" check.
    if args.recompute_meta:
        if not os.path.exists(args.out):
            raise SystemExit(f"--recompute-meta: {args.out} does not exist.")
        existing = dict(np.load(args.out, allow_pickle=True))
        ids = existing["ids"].astype(str)
        meta = {}
        for data_path, cfg in zip(args.folders, configs):
            for t in build_tasks(data_path, cfg, fate_order, args.fate, vocabs):
                meta[t[1]] = (t[3], t[4])          # uid -> (time, condition)
        n_missing = sum(u not in meta for u in ids)
        if n_missing:
            print(f"  [warn] {n_missing}/{len(ids)} rows not found in the current "
                  "folders — their times/conditions are left unchanged.")
        old_times = existing["times"].astype(str)
        old_conds = (existing["conditions"].astype(str) if "conditions" in existing
                     else np.array(["WT"] * len(ids)))
        existing["times"]      = np.array([meta[u][0] if u in meta else old_times[i]
                                           for i, u in enumerate(ids)])
        existing["conditions"] = np.array([meta[u][1] if u in meta else old_conds[i]
                                           for i, u in enumerate(ids)])
        np.savez(args.out, **existing)
        print(f"Recomputed times+conditions for {len(ids)} rows → {args.out}")
        for arr, label in ((existing["times"], "times"), (existing["conditions"], "conditions")):
            uniq, cnts = np.unique(arr, return_counts=True)
            print(f"  {label}: {dict(zip(uniq.tolist(), cnts.tolist()))}")
        return

    # Build the full task list across all datasets (the target set of uids that
    # should be in the npz, per the current good_labels).
    all_tasks = []
    for data_path, cfg in zip(args.folders, configs):
        tasks = build_tasks(data_path, cfg, fate_order, args.fate, vocabs)
        print(f"{os.path.basename(data_path)}: {len(tasks)} organoids")
        all_tasks.extend(tasks)
    print(f"Total: {len(all_tasks)} organoids\n")

    update = args.update and os.path.exists(args.out)
    if args.update and not update:
        print(f"--update: no existing {args.out} — computing all organoids from scratch.\n")

    if update:
        # Incremental sync: keep existing rows still in the target set, drop the
        # rest, and compute only the target uids missing from the npz.
        existing = dict(np.load(args.out, allow_pickle=True))
        existing_ids = existing["ids"].astype(str)
        existing_set = set(existing_ids)
        target_uids  = {t[1] for t in all_tasks}

        keep_mask     = np.array([u in target_uids for u in existing_ids], dtype=bool)
        missing_tasks = [t for t in all_tasks if t[1] not in existing_set]
        n_keep, n_drop = int(keep_mask.sum()), int((~keep_mask).sum())
        print(f"Update: existing {len(existing_ids)} | target {len(target_uids)} | "
              f"keep {n_keep} | drop {n_drop} | compute {len(missing_tasks)}\n")

        if not missing_tasks and n_drop == 0:
            print("Already up to date — nothing to do.")
            return

        results = _run_tasks(missing_tasks, args.workers, vocabs) if missing_tasks else []
        if missing_tasks and not results:
            raise SystemExit("All new organoids failed to process — "
                             "existing npz left unchanged.")
        new_kwargs = (assemble_save_kwargs(results, fate_order, vocabs, args.fate,
                                           args.l_cross_threshold)
                      if results else {})
        save_kwargs = merge_existing(existing, new_kwargs, keep_mask)
    else:
        results = _run_tasks(all_tasks, args.workers, vocabs)
        if not results:
            raise SystemExit("No organoids processed — did you run run_new_meshes.py for each dataset?")
        save_kwargs = assemble_save_kwargs(results, fate_order, vocabs, args.fate,
                                           args.l_cross_threshold)

    N = len(save_kwargs["ids"])
    print(f"\nSaving {N} organoids")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **save_kwargs)
    print(f"Saved → {args.out}")
    for k, v in save_kwargs.items():
        print(f"  {k:25s}  {np.asarray(v).shape}")


if __name__ == "__main__":
    main()
