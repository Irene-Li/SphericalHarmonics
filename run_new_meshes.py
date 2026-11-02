"""
Per-organoid preprocessing of the meshes, writing each dataset's fm_data:
<label>_transformed_mesh.obj (the PCA-aligned true-geometry mesh) and
<label>_coeffs.npz (its true-geometry Laplace-Beltrami eigendecomposition, which the
HKS is built from, + PCA transform and area), plus good_labels_<timepoint>.npy.
The SPHARM spherical maps and every harmonic decomposition are computed (and cached)
by compute_master_npz.py; fate fields are not decomposed.

Dispatches on the dataset's config.json 'layout':
  vtp_flat        {folder}/{vtp_dir}/{timepoint}/{label}.{vtp,stl}  (main/sup/pert2/wnt_pert)
  fractal_output  the original per-well tree

Organoids listed in config['discard'].labels_to_discard_csv (built by
manage_discards.py) are excluded from good_labels. --workers N parallelises with
one BLAS thread per worker; --reprocess recomputes even if outputs already exist.

Run in the scmpx env, e.g.:
  KMP_DUPLICATE_LIB_OK=TRUE python run_new_meshes.py Data/pert2 --workers 6
"""
from tqdm import tqdm
from src.fatemarkers import FateMarkers
import igl
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


def run_fatemarkers(mesh_path, save_path, compute_fate=False,
                    annotation_names=None, exclusion_rules=None, hks=True, save_coeffs=True):
    """compute_fate / annotation_names / exclusion_rules are accepted for the task-tuple
    layout but unused: fate fields are not decomposed. Lean output (config "save_coeffs":
    false, needs "hks": false) writes only the aligned mesh, from which compute_master_npz
    --complexity-only computes the area and the SPHARM complexity."""
    m = FateMarkers()
    m.load_mesh_from_file(mesh_path)
    m.align_with_pca()
    if save_coeffs:
        m.precompute_eigens(lmax=15, hks=hks)
        m.save_results(save_path)
    else:
        igl.write_triangle_mesh(save_path + "_transformed_mesh.obj", m.v, m.f)


def _run_one(args):
    """Top-level wrapper for multiprocessing (must be picklable)."""
    mesh_path, save_path, compute_fate, annotation_names, exclusion_rules, hks, save_coeffs = args
    run_fatemarkers(mesh_path, save_path, compute_fate, annotation_names, exclusion_rules, hks,
                    save_coeffs)
    return mesh_path


def _process_batch(tasks, workers, desc=""):
    """Run a list of (mesh_path, save_path, compute_fate, ann, excl, hks, save_coeffs) tuples.
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


def subsample_by_group(files, label_of, n_per_group, group_uid_index, seed=42):
    """Pick up to n_per_group files from each group, the group being the uid token
    at group_uid_index (e.g. 1 -> the well in 'day3p5_C07_12'). Members are
    uid-sorted and each group gets its own RNG ([seed, group position]), so the
    pick depends only on the set of files, and adding files to one group leaves
    the other groups' picks unchanged."""
    groups = {}
    for f in files:
        groups.setdefault(label_of(f).split("_")[group_uid_index], []).append(f)
    picked = []
    for gi, g in enumerate(sorted(groups)):
        members = sorted(groups[g], key=label_of)
        rng = np.random.default_rng([seed, gi])
        idx = rng.choice(len(members), size=min(n_per_group, len(members)), replace=False)
        picked += [members[i] for i in sorted(idx)]
        print(f"  subsample {g}: {min(n_per_group, len(members))}/{len(members)}")
    return sorted(picked, key=label_of)


def apply_subsample(kept, label_of, sub, timepoint):
    """Restrict `kept` to the config subsample. With sub['frozen_list'] (a path,
    '{timepoint}' filled in) the first draw is written there and reused on later
    runs, so later discards (e.g. from review_discards.py) only drop organoids from
    the sample instead of redrawing it. Delete the file to redraw."""
    path = sub.get("frozen_list", "").format(timepoint=timepoint)
    if path and os.path.exists(path):
        with open(path) as f:
            frozen = {l.strip() for l in f if l.strip()}
        picked = [f for f in kept if label_of(f) in frozen]
        print(f"  frozen subsample {path}: {len(picked)}/{len(frozen)} still kept")
        return picked
    picked = subsample_by_group(kept, label_of, sub["n_per_group"],
                                sub["group_uid_index"], sub.get("seed", 42))
    if path:
        with open(path, "w") as f:
            f.write("".join(label_of(p) + "\n" for p in picked))
        print(f"  wrote frozen subsample → {path}")
    return picked


def run_vtp_flat(folder_path, cfg, skip_existing=True, workers=1):
    """Process a 'vtp_flat' dataset: {folder}/{vtp_dir}/{timepoint}/{label_uid}.{mesh_ext}
    (mesh_ext 'vtp' by default; 'stl' for geometry-only meshes). An optional config 'subsample' block
    ({n_per_group, group_uid_index, seed, frozen_list}) restricts good_labels to an
    even per-group subsample of the non-discarded meshes. Config "hks": false is a
    complexity-only run: no true-geometry eigensolve, no eigvals/eigvecs saved
    (build master with compute_master_npz.py --complexity-only). With "hks": false,
    "save_coeffs": false is the lean mode: no _coeffs.npz at all, just the aligned
    .obj; organoids that already have a full _coeffs.npz are kept and reused."""
    discard = discard_set(cfg)
    print(f"discard set: {len(discard)} organoids")
    vtp_dir = cfg.get("vtp_dir", "vtp")
    mesh_ext = cfg.get("mesh_ext", "vtp")

    def label_of(p):
        return os.path.splitext(os.path.basename(p))[0]

    for timepoint in cfg["timepoints"]:
        tp_dir = os.path.join(folder_path, vtp_dir, timepoint)
        if not os.path.isdir(tp_dir):
            print(f"  [skip] no vtp dir {tp_dir}")
            continue
        files = sorted(glob.glob(os.path.join(tp_dir, f"*.{mesh_ext}")))
        kept = [f for f in files if label_of(f) not in discard]
        sub = cfg.get("subsample")
        if sub:
            kept = apply_subsample(kept, label_of, sub, timepoint)

        save_dir = f'{tp_dir}/fm_data/'
        hks = cfg.get("hks", True)
        save_coeffs = cfg.get("save_coeffs", True)
        if not save_coeffs and hks:
            raise SystemExit('config "save_coeffs": false needs "hks": false (HKS needs the eigvecs)')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"good_labels_{timepoint}.npy"),
                np.array([label_of(f) for f in kept]))

        def done(f):   # the mesh, plus the eigendecomposition unless lean
            base = os.path.join(save_dir, label_of(f))
            return (os.path.exists(base + "_transformed_mesh.obj") and
                    (os.path.exists(base + "_coeffs.npz") or not save_coeffs))
        if skip_existing:
            todo = [f for f in kept if not done(f)]
        else:
            todo = kept
        print(f"{timepoint}: {len(todo)}/{len(kept)} to process "
              f"({len(files) - len(kept)} discarded{' or not subsampled' if sub else ''})")

        tasks = [(f, os.path.join(save_dir, label_of(f)), False, None, None, hks, save_coeffs)
                 for f in todo]
        _process_batch(tasks, workers, desc=timepoint)


# ---------------------------------------------------------------------------
# Layout 2: 'fractal_output'  —  the original per-well directory tree
#   {folder}/fractal_output/{tp}/{zarr}/{w[0]}/{w[1:]}/{round}/meshes/{mesh_name}/{label}.vtp
# ---------------------------------------------------------------------------

def run_fractal_output(folder_path, cfg, skip_existing=False, workers=1):
    """Process an old 'fractal_output' dataset."""
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

            tasks = [(f"{mesh_path}{l}.vtp", f"{fm_data}{l}", False, annotation_names, exclusion_rules, True, True)
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
