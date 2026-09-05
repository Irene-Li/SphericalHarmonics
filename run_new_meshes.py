"""
Compute per-organoid shape coefficients from meshes, writing each dataset's
fm_data: <label>_coeffs.npz (PCA-aligned eigendecomposition + SH coeffs) and
<label>_transformed_mesh.obj, plus good_labels_<timepoint>.npy.

Dispatches on the dataset's config.json 'layout':
  vtp_flat        {folder}/{vtp_dir}/{timepoint}/{label}.vtp   (main/sup/pert2; shape
                  coefficients only, compute_fate=False)
  fractal_output  the original per-well tree                    (also computes fate coeffs)

Organoids listed in config['discard'].labels_to_discard_csv (built by
manage_discards.py) are excluded from good_labels. --workers N parallelises with
one BLAS thread per worker; --reprocess recomputes even if outputs already exist.

Run in the scmpx env, e.g.:
  KMP_DUPLICATE_LIB_OK=TRUE python run_new_meshes.py Data/pert2 --workers 6
"""
from tqdm import tqdm
from src.fatemarkers import FateMarkers
import numpy as np
import os
import csv
import glob
import json
import argparse
import concurrent.futures


def _init_worker(n_threads):
    """Set BLAS/OpenMP thread count in each worker process to avoid oversubscription."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)


def run_fatemarkers(mesh_path, save_path, compute_fate=True,
                    annotation_names=None, exclusion_rules=None):
    m = FateMarkers()
    m.load_mesh_from_file(mesh_path)
    if compute_fate:
        m._refine_markers(annotation_names, exclusion_rules)
    m.align_with_pca()
    m.precompute_eigens(lmax=15)
    m.compute_coefficients(fate=compute_fate)
    m.save_results(save_path, fate=compute_fate)


def _run_one(args):
    """Top-level wrapper for multiprocessing (must be picklable)."""
    mesh_path, save_path, compute_fate, annotation_names, exclusion_rules = args
    run_fatemarkers(mesh_path, save_path, compute_fate, annotation_names, exclusion_rules)
    return mesh_path


def _process_batch(tasks, workers, desc=""):
    """Run a list of (mesh_path, save_path, compute_fate, ann, excl) tuples.
    workers=1 → serial; workers>1 → ProcessPoolExecutor with 1 BLAS thread per worker."""
    if workers == 1:
        for t in tqdm(tasks, desc=desc):
            try:
                _run_one(t)
            except Exception as e:
                print(f"\nError processing {t[0]}: {e}")
    else:
        threads_per_worker = max(1, 8 // workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker, initargs=(threads_per_worker,)) as ex:
            futs = {ex.submit(_run_one, t): t for t in tasks}
            for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc=desc):
                try:
                    f.result()
                except Exception as e:
                    print(f"\nError processing {futs[f][0]}: {e}")


# ---------------------------------------------------------------------------
# Layout 1: 'vtp_flat'  —  {folder}/{vtp_dir}/{timepoint}/{label_uid}.vtp
# ---------------------------------------------------------------------------

def discard_set(cfg):
    """Set of label_uids to discard, read from the path in config['discard'].
    manage_discards.py owns building that file. Returns empty set when no 'discard' block."""
    d = cfg.get("discard")
    if not d:
        return set()
    path = d.get("labels_to_discard_csv")
    if not path or not os.path.exists(path):
        print(f"  [warn] discard csv not found: {path}")
        return set()
    with open(path) as f:
        return {r["label_uid"].strip() for r in csv.DictReader(f) if r.get("label_uid")}


def run_vtp_flat(folder_path, cfg, skip_existing=True, workers=1):
    """Process a 'vtp_flat' dataset: {folder}/{vtp_dir}/{timepoint}/{label_uid}.vtp.
    Shape coefficients only (compute_fate=False)."""
    discard = discard_set(cfg)
    print(f"discard set: {len(discard)} organoids")
    vtp_dir = cfg.get("vtp_dir", "vtp")

    def label_of(p):
        return os.path.splitext(os.path.basename(p))[0]

    for timepoint in cfg["timepoints"]:
        tp_dir = os.path.join(folder_path, vtp_dir, timepoint)
        if not os.path.isdir(tp_dir):
            print(f"  [skip] no vtp dir {tp_dir}")
            continue
        files = sorted(glob.glob(os.path.join(tp_dir, "*.vtp")))
        kept = [f for f in files if label_of(f) not in discard]

        save_dir = f'{tp_dir}/fm_data/'
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"good_labels_{timepoint}.npy"),
                np.array([label_of(f) for f in kept]))

        if skip_existing:
            todo = [f for f in kept
                    if not (os.path.exists(os.path.join(save_dir, f"{label_of(f)}_coeffs.npz")) and
                            os.path.exists(os.path.join(save_dir, f"{label_of(f)}_transformed_mesh.obj")))]
        else:
            todo = kept
        print(f"{timepoint}: {len(todo)}/{len(kept)} to process "
              f"({len(files) - len(kept)} discarded)")

        tasks = [(f, os.path.join(save_dir, label_of(f)), False, None, None) for f in todo]
        _process_batch(tasks, workers, desc=timepoint)


# ---------------------------------------------------------------------------
# Layout 2: 'fractal_output'  —  the original per-well directory tree
#   {folder}/fractal_output/{tp}/{zarr}/{w[0]}/{w[1:]}/{round}/meshes/{mesh_name}/{label}.vtp
# ---------------------------------------------------------------------------

def run_fractal_output(folder_path, cfg, skip_existing=False, workers=1):
    """Process an old 'fractal_output' dataset. Computes fate coefficients."""
    timepoints = cfg['timepoints']
    zarr_names = cfg['zarr_names']
    wells = cfg['wells']
    mesh_name = cfg['mesh_name']
    rounds = cfg['rounds']
    annotation_names = cfg['annotation_names']
    exclusion_rules = cfg['exclusion_rules']

    discard = discard_set(cfg)
    print(f"discard set: {len(discard)} organoids")

    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint]
            path = f"{folder_path}fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            mesh_path = f"{path}meshes/{mesh_name}/"
            all_labels = [int(l.split('.')[0]) for l in os.listdir(mesh_path)]

            good_labels = [l for l in all_labels
                           if f"{timepoint}_{well_name}_{l}" not in discard]

            fm_data = f"{path}fm_data/"
            if not os.path.exists(fm_data):
                os.mkdir(fm_data)

            np.save(f"{fm_data}good_labels.npy", np.array(good_labels))

            if skip_existing:
                todo = [l for l in good_labels
                        if not (os.path.exists(f"{fm_data}{l}_coeffs.npz") and
                                os.path.exists(f"{fm_data}{l}_transformed_mesh.obj"))]
            else:
                todo = good_labels
            print(f"{timepoint} {well_name}: {len(todo)}/{len(good_labels)} meshes to process")
            if not todo:
                continue

            tasks = [(f"{mesh_path}{l}.vtp", f"{fm_data}{l}", True, annotation_names, exclusion_rules)
                     for l in todo if os.path.exists(f"{mesh_path}{l}.vtp")]
            _process_batch(tasks, workers, desc=f"{timepoint} {well_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run FateMarkers on mesh data.")
    parser.add_argument("folder_path", type=str, help="Path to the data folder (e.g. Data/20260224/)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1 = serial). "
                             "Each worker uses 1 BLAS thread to avoid oversubscription.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Reprocess meshes even if _coeffs.npz and _transformed_mesh.obj already exist")
    args = parser.parse_args()

    folder_path = args.folder_path
    if not folder_path.endswith('/'):
        folder_path += '/'

    with open(f"{folder_path}config.json", 'r') as f:
        cfg = json.load(f)

    # dispatch on the dataset layout
    if cfg.get("layout") == "vtp_flat":
        run_vtp_flat(folder_path, cfg, skip_existing=not args.reprocess, workers=args.workers)
    else:
        run_fractal_output(folder_path, cfg, skip_existing=not args.reprocess, workers=args.workers)
