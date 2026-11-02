"""
Compute all per-organoid features across one or more datasets and save a single
combined master .npz. Every harmonic decomposition is SPHARM: each organoid's true
geometry (PCA-aligned, no per-axis rescaling) is decimated and mapped area-preservingly
onto the unit sphere (src.spharm.SpHarm, N_FACES / PRIMAL_STEPS), and fields are
expanded in real spherical harmonics on that map, so the degree-l blocks are exact.

Arrays saved:

  ids                 (N,)              organoid identifier strings  '{dataset}_{tp}_{well}_{label}'
  datasets            (N,)              source dataset folder name    e.g. 'main_dataset'
  times               (N,)              developmental-day label       e.g. '3p5' (config-driven)
  conditions          (N,)              perturbation/genotype label   'WT' | e.g. 'stem-ChirVpaD1'
  areas               (N,)              true mesh surface area        (m.area)
  spharm_area_error   (N,)              median relative triangle-area error of the spherical
                                        map (how far it is from area-preserving; ~0.003)
  complexity_errors   (N, 9)            SPHARM shape recon error at lmax = 0..8 (col i = degrees 0..i)
  l_cross_values      (N,)              interpolated lmax (max SH degree) where error crosses
                                        L_CROSS_THRESHOLD

  hks_amplitude       (N,)              how strongly the HKS varies over the surface: mean over
                                        the N_VARIABLE_TS variable times of the across-vertex
                                        std of HKS/mean - 1 (hks_unit_area). The vocab encoding
                                        drops this size (build_vocab's Normalizer), so dim_red
                                        appends it as one feature (utils.load_hks_features).
  hks_max_ratio       (N,)              QC: max over vertices and variable times of HKS / its mean over
                                        the vertices. ~1-2.3 for healthy meshes; a narrow pit traps a
                                        localised eigenmode and spikes it (main day2p5_A06_85: ~20),
                                        which skews the whole vocab encoding. Flag > HKS_SPIKE_FLAG
                                        for inspection (manual discard via manage_discards.py).

  For each configured vocabulary (see VOCABS below), one array:
  hks_bof_power__<name>   (N, HKS_LMAX + 1, n_features)   SPHARM power per degree of each
                                                          channel of the vocab-encoded HKS
  bof_vocab_names         (n_vocabs,)                     ordered vocab names

  (--complexity-only omits hks_amplitude, hks_max_ratio and the hks_bof_power__* arrays.)

The spherical map is the expensive step (about 3 s per organoid), so each one is cached in
SPHARM_MAP_DIR/<uid>.npz and reused by later runs (--update, --bof-only, a rebuild):
an interrupted run loses at most the organoids in flight. Delete a map to redo it, and
bump MAP_METHOD when the map method changes (a map from another method is never reused).
The npz holds the decimated + smoothed mesh (PCA-aligned) and its map; the same mesh is
also written as SPHARM_MESH_DIR/<uid>.obj for viewing.

Rows are saved sorted by uid. Organoids that fail are listed in <out>.failed.txt.

Usage:
  python compute_master_npz.py [--folders DIR [DIR ...]] [--out PATH] [--ids FILE]
                               [--workers N] [--update] [--bof-only]
                               [--l-cross-threshold T] [--recompute-l-cross]
                               [--recompute-meta] [--complexity-only]

Defaults:
  --folders  Data/main_dataset Data/sup_dataset Data/pert2
  --out      Data/npz/master.npz
  --workers  1

--ids FILE restricts the run to the uids listed in FILE (one per line); without it
every organoid in the folders' good_labels is used.

--update incrementally syncs an existing --out to the current good_labels (and --ids)
(i.e. after editing the discard lists): rows whose uid is no longer in the target set
are dropped, and only uids that should be present but are missing from the npz are
computed; uids already present are left untouched. The run must use the same vocab
setup as the existing file. Without --update (or if --out does not exist) every
organoid is recomputed from scratch (reusing cached spherical maps).

--recompute-l-cross only re-derives l_cross_values in an existing --out from its
stored complexity_errors at --l-cross-threshold; no meshes are reprocessed. Use
it to pick a complexity threshold (see complexity_analysis.ipynb) without
rebuilding the npz, e.g.
  python compute_master_npz.py --recompute-l-cross --l-cross-threshold 0.02
--l-cross-threshold also applies to full and --update runs.

--bof-only re-encodes HKS with the current output/vocab_*.npz, replacing ONLY the
hks_bof_power__* arrays (and recomputing hks_amplitude and hks_max_ratio from the same HKS) in an
existing --out and keeping every other array. It reads each _coeffs.npz and the cached
spherical map, so it is the cheap way to update master after a vocab change:
  python compute_master_npz.py --bof-only

--recompute-meta re-derives only the metadata arrays (times, conditions) for
every row in an existing --out from the current configs; no meshes reprocessed.

--complexity-only skips everything HKS (hks_amplitude, hks_bof_power__*), keeping
areas / complexity_errors / l_cross_values. Use it for datasets processed with config
"hks": false (run_new_meshes then skips the true-geometry eigensolve and saves only the
aligned mesh), e.g.
  python compute_master_npz.py --folders Data/wnt_pert --out Data/npz/wnt_master.npz --complexity-only
"""

import os
import argparse
import json
import numpy as np
import igl
from tqdm import tqdm
import concurrent.futures

from src.spharm import SpHarm, N_FACES, PRIMAL_STEPS, OUTER_ROUNDS
from src import utils
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_FOLDERS = ["Data/main_dataset", "Data/sup_dataset", "Data/pert2"]
DEFAULT_OUT     = "Data/npz/master.npz"
SPHARM_MAP_DIR  = "Data/spharm_maps_vertex"
SPHARM_MESH_DIR = "Data/spharm_meshes"
MAP_METHOD      = "vertex_cot_lbfgs_radial"    # bump when the map method changes

HKS_LMAX            = 7          # hks_bof_power__* keeps degrees 0..HKS_LMAX
COMPLEXITY_LMAX     = 9
L_CROSS_THRESHOLD   = 0.009       # chosen in complexity_analysis.ipynb
N_VARIABLE_TS       = 20
HKS_SPIKE_FLAG      = 5.0        # hks_max_ratio above this is flagged (healthy max ~2.3)

VOCABS = [
    {"name": "kmeans_variable", "path": "output/vocab_variable_time.npz",
     "encoding": "kmeans", "time": "variable"},
    {"name": "pca_variable",    "path": "output/vocab_pca_variable_time.npz",
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

def spharm_map(uid, v, f):
    """The organoid's spherical map (src.spharm.SpHarm), from SPHARM_MAP_DIR/<uid>.npz
    when a map with the current MAP_METHOD / N_FACES / PRIMAL_STEPS / OUTER_ROUNDS is cached
    there, else computed from the aligned true-geometry mesh (v, f) and cached (written
    atomically), along with its decimated + smoothed mesh as SPHARM_MESH_DIR/<uid>.obj."""
    path = os.path.join(SPHARM_MAP_DIR, f"{uid}.npz")
    if os.path.exists(path):
        sh, meta = SpHarm.load_map(path)
        if (str(meta.get("method")) == MAP_METHOD and int(meta["outer_rounds"]) == OUTER_ROUNDS
                and int(meta["n_faces_target"]) == N_FACES and int(meta["primal_steps"]) == PRIMAL_STEPS):
            return sh
    sh = SpHarm.from_mesh(v, f).map_to_sphere()
    os.makedirs(SPHARM_MAP_DIR, exist_ok=True)
    tmp = path[:-len(".npz")] + ".tmp.npz"
    sh.save_map(tmp, method=MAP_METHOD, n_faces_target=N_FACES, primal_steps=PRIMAL_STEPS,
                outer_rounds=OUTER_ROUNDS, rounds=sh.rounds, start=sh.start)
    os.makedirs(SPHARM_MESH_DIR, exist_ok=True)
    mesh_tmp = os.path.join(SPHARM_MESH_DIR, f"{uid}.tmp.obj")
    igl.write_triangle_mesh(mesh_tmp, sh.v, sh.f)
    os.replace(mesh_tmp, os.path.join(SPHARM_MESH_DIR, f"{uid}.obj"))
    os.replace(tmp, path)
    return sh


def compute_complexity_errors(sh):
    """Column i is the area-normalised RMSE of the SPHARM shape reconstruction with
    degrees 0..i (lmax=i); column 0 is the degree-0 constant (the centre)."""
    return sh.complexity_errors(COMPLEXITY_LMAX - 1)


def hks_unit_area(m, ts):
    eigvecs = m.eigvecs * np.sqrt(m.area)
    hks = np.array([
        np.einsum('i,ji->j', np.exp(-m.eigvals * t), eigvecs ** 2)
        for t in ts
    ]).T
    return hks / np.mean(hks, axis=0) - 1


def hks_amplitude(m, ts=None):
    """Mean over the variable times of the across-vertex std of HKS/mean - 1."""
    return float(hks_unit_area(m, find_times(m.area) if ts is None else ts).std(axis=0).mean())


def hks_max_ratio(m, ts=None):
    """Max over vertices and the variable times of HKS / its vertex mean (QC, see HKS_SPIKE_FLAG)."""
    return float(hks_unit_area(m, find_times(m.area) if ts is None else ts).max() + 1)


def encode_vocab(hks_scaled, v):
    if v["encoding"] == "kmeans":
        dist = np.linalg.norm(
            hks_scaled[:, np.newaxis, :] - v["centres"][np.newaxis, :, :], axis=2
        )
        return np.exp(-dist**2 / (2 * v["sigma"]**2))
    else:
        return (hks_scaled - v["mean"]) @ v["components"].T


def compute_bof_power(m, sh, vocabs):
    """Per vocab: the per-vertex encoding of the HKS on the full mesh (normalised over all
    its vertices), sampled at the map's vertices, SPHARM power per degree (HKS_LMAX + 1, W)."""
    hks_cache = {}
    out = {}
    for v in vocabs:
        ts  = find_times(m.area) if v["time"] == "variable" else np.asarray(v["ts"])
        key = ts.tobytes()
        if key not in hks_cache:
            hks_cache[key] = hks_unit_area(m, ts)
        encoding   = encode_vocab(v["scaler"].transform(hks_cache[key]), v)
        out[v["name"]] = sh.power_spectrum(encoding[sh.birth], HKS_LMAX)
    return out


def _worker_init(n_threads):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    # each worker's JAX (the spherical-map optimiser) gets its share of the cores too
    os.environ["XLA_FLAGS"] = (f"--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={n_threads}")


def load_organoid(save_path):
    """The aligned true-geometry mesh + true area, with the HKS eigendecomposition when
    the organoid has a _coeffs.npz (absent for lean, complexity-only runs)."""
    v, f = igl.read_triangle_mesh(save_path + "_transformed_mesh.obj")
    m = SimpleNamespace(v=v, f=f, eigvals=None, eigvecs=None)
    if os.path.exists(save_path + "_coeffs.npz"):
        d = np.load(save_path + "_coeffs.npz", allow_pickle=True)
        m.area = float(d["area"])
        if "eigvals" in d.files:
            m.eigvals, m.eigvecs = d["eigvals"], d["eigvecs"]
    else:
        m.area = float(igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI).diagonal().sum())
    return m


def shape_features(m, sh):
    """The per-organoid features that need neither HKS nor vocab — everything a
    --complexity-only master holds."""
    return {
        "area":       m.area,
        "area_error": sh.area_error(),
        "complexity": compute_complexity_errors(sh),
    }


# ---------------------------------------------------------------------------
# --bof-only: re-encode HKS with the current vocab, updating ONLY the bof arrays
# (and hks_amplitude) of an existing master. Reuses the cached spherical maps, so
# only the HKS, the encoding and the SH fits are redone.
# ---------------------------------------------------------------------------
_BOF_ONLY_CFGS = {}   # dataset -> its config.json (for vtp_dir / timepoints); all vtp_flat
_BOF_ONLY_VOCABS = None


def _bof_only_init(n_threads, vocabs):
    _worker_init(n_threads)
    global _BOF_ONLY_VOCABS
    _BOF_ONLY_VOCABS = vocabs


def _bof_only_one(task):
    """(uid, dataset) -> (uid, bof dict, (hks_amplitude, hks_max_ratio))."""
    uid, dataset = task
    bare = uid[len(dataset) + 1:]
    if dataset not in _BOF_ONLY_CFGS:
        _BOF_ONLY_CFGS[dataset] = load_config(f"Data/{dataset}")
    coeffs = utils.vtp_flat_coeffs_path(f"Data/{dataset}", _BOF_ONLY_CFGS[dataset], bare)
    m = load_organoid(coeffs[:-len("_coeffs.npz")])
    sh = spharm_map(uid, m.v, m.f)
    return uid, compute_bof_power(m, sh, _BOF_ONLY_VOCABS), (hks_amplitude(m), hks_max_ratio(m))


def run_bof_only(out_path, vocabs, workers):
    """Update ONLY the bof arrays of the existing master at out_path in place."""
    if not vocabs:
        raise SystemExit("--bof-only: no vocabs configured/found (see VOCABS / output/).")
    if not os.path.exists(out_path):
        raise SystemExit(f"--bof-only: {out_path} does not exist (nothing to update).")
    existing = dict(np.load(out_path, allow_pickle=True))
    ids      = existing["ids"].astype(str)
    datasets = existing["datasets"].astype(str)
    print(f"--bof-only: re-encoding {len(ids)} organoids with "
          f"{[v['name'] for v in vocabs]} + hks_amplitude + hks_max_ratio (other arrays kept verbatim)")
    tasks = list(zip(ids, datasets))

    results, amplitude = {}, {}
    if workers == 1:
        global _BOF_ONLY_VOCABS
        _BOF_ONLY_VOCABS = vocabs
        for t in tqdm(tasks, desc="bof-only"):
            uid, bof, amp = _bof_only_one(t)
            results[uid], amplitude[uid] = bof, amp
    else:
        tpw = max(1, 8 // workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, initializer=_bof_only_init, initargs=(tpw, vocabs)) as ex:
            for uid, bof, amp in tqdm(ex.map(_bof_only_one, tasks), total=len(tasks), desc="bof-only"):
                results[uid], amplitude[uid] = bof, amp

    for v in vocabs:
        existing[f"hks_bof_power__{v['name']}"] = np.array([results[u][v["name"]] for u in ids])
    existing["bof_vocab_names"] = np.array([v["name"] for v in vocabs])
    existing["hks_amplitude"] = np.array([amplitude[u][0] for u in ids])
    existing["hks_max_ratio"] = np.array([amplitude[u][1] for u in ids])
    save_npz(out_path, existing)
    print(f"Updated {out_path}: replaced bof arrays + hks_amplitude + hks_max_ratio, kept everything else.")
    report_spikes(ids, existing["hks_max_ratio"])
    for v in vocabs:
        print(f"  hks_bof_power__{v['name']}  {existing['hks_bof_power__'+v['name']].shape}")


# ---------------------------------------------------------------------------
# Per-organoid worker (top-level so it's picklable)
# ---------------------------------------------------------------------------

def _process_one(task):
    """Load one organoid and compute all features. Returns a result dict, or
    {"failed": uid, "error": message} on error."""
    save_path, uid, dataset_name, time, condition, vocabs, hks = task
    try:
        meta = {"uid": uid, "dataset": dataset_name, "time": time, "condition": condition}
        m = load_organoid(save_path)
        if hks and m.eigvecs is None:
            raise ValueError("no HKS eigendecomposition on disk — use --complexity-only")
        sh = spharm_map(uid, m.v, m.f)
        result = {**meta, **shape_features(m, sh)}
        if hks:
            result["hks_amplitude"] = hks_amplitude(m)
            result["hks_max_ratio"] = hks_max_ratio(m)
        if vocabs:
            result["bof"] = compute_bof_power(m, sh, vocabs)
        return result
    except Exception as e:
        print(f"Failed {save_path}: {e}")
        return {"failed": uid, "error": repr(e)}


def _run_tasks(tasks, workers, vocabs):
    """Run tasks in parallel (workers>1) or serial. Returns (result dicts sorted by uid,
    failed {"failed": uid, "error": message} dicts)."""
    results, failed = [], []

    def _collect(r):
        (failed if "failed" in r else results).append(r)

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

    return sorted(results, key=lambda r: r["uid"]), failed


# ---------------------------------------------------------------------------
# Per-dataset task builders
# ---------------------------------------------------------------------------

def load_config(data_path):
    with open(f"{data_path}/config.json") as f:
        return json.load(f)


def derive_time(cfg, timepoint):
    """Developmental-day label for a row. Config-driven so a dataset whose
    subfolders are not days (e.g. pert2: normal/small) can pin a constant day:
      cfg['time']      -> constant for the whole dataset (e.g. pert2: '4p5')
      cfg['time_map']  -> {subfolder: day}
      otherwise        -> strip a leading 'day' (main/sup: 'day4p5' -> '4p5')."""
    if "time" in cfg:
        return cfg["time"]
    if "time_map" in cfg:
        return cfg["time_map"][timepoint]
    return timepoint[3:] if timepoint.startswith("day") else timepoint


def derive_condition(cfg, uid, timepoint, label_uid=None):
    """Perturbation/genotype label for a row. Config-driven:
      cfg['condition']           -> constant (main/sup: 'WT')
      cfg['condition_map']       -> {subfolder: condition}
      cfg['condition_by_token']  -> {"index": i, "map": {token: condition}} on the
                                    label_uid (file stem) tokens, e.g. wnt_pert:
                                    index 1 = the well -> its treatment
      cfg['condition_uid_index'] -> uid.split('_')[i] (pert2: 1 -> the drug token)
      otherwise                  -> 'WT'."""
    if "condition" in cfg:
        return cfg["condition"]
    if "condition_map" in cfg:
        return cfg["condition_map"][timepoint]
    if "condition_by_token" in cfg:
        spec = cfg["condition_by_token"]
        return spec["map"][label_uid.split("_")[spec["index"]]]
    if "condition_uid_index" in cfg:
        return uid.split("_")[cfg["condition_uid_index"]]
    return "WT"


def _build_tasks_vtp_flat(data_path, cfg, vocabs, hks=True):
    dataset_name     = os.path.basename(data_path.rstrip("/"))
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
            if not os.path.exists(save_path + "_transformed_mesh.obj"):
                continue
            uid = f"{dataset_name}_{label_uid}"
            tasks.append((save_path, uid, dataset_name,
                          derive_time(cfg, timepoint),
                          derive_condition(cfg, uid, timepoint, str(label_uid)),
                          vocabs, hks))
    return tasks


def _build_tasks_fractal_output(data_path, cfg, vocabs, hks=True):
    dataset_name     = os.path.basename(data_path.rstrip("/"))
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
                if not os.path.exists(save_path + "_transformed_mesh.obj"):
                    continue
                uid = f"{dataset_name}_{timepoint}_{well_name}_{label}"
                tasks.append((save_path, uid, dataset_name,
                              derive_time(cfg, timepoint),
                              derive_condition(cfg, uid, timepoint,
                                               f"{timepoint}_{well_name}_{label}"),
                              vocabs, hks))
    return tasks


def build_tasks(data_path, cfg, vocabs, hks=True):
    if cfg.get("layout") == "vtp_flat":
        return _build_tasks_vtp_flat(data_path, cfg, vocabs, hks)
    else:
        return _build_tasks_fractal_output(data_path, cfg, vocabs, hks)


# ---------------------------------------------------------------------------
# Assemble / merge saved arrays
# ---------------------------------------------------------------------------

# Arrays that label the columns rather than the rows (one entry per vocab, not per
# organoid). Everything else in the saved npz is per-organoid.
GLOBAL_KEYS = ("bof_vocab_names",)


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


def assemble_save_kwargs(results, vocabs, l_cross_threshold=L_CROSS_THRESHOLD):
    """Turn per-organoid result dicts into the dict of arrays saved to the npz."""
    acc = {k: [r[k] for r in results]
           for k in ("uid", "dataset", "time", "condition", "area", "area_error", "complexity")}

    out = dict(
        ids               = np.array(acc["uid"]),
        datasets          = np.array(acc["dataset"]),
        times             = np.array(acc["time"]),
        conditions        = np.array(acc["condition"]),
        areas             = np.array(acc["area"]),
        spharm_area_error = np.array(acc["area_error"]),
        complexity_errors = np.array(acc["complexity"]),
        l_cross_values    = compute_l_cross(acc["complexity"], l_cross_threshold),
    )
    if "hks_amplitude" in results[0]:       # absent for --complexity-only runs
        out["hks_amplitude"] = np.array([r["hks_amplitude"] for r in results])
        out["hks_max_ratio"] = np.array([r["hks_max_ratio"] for r in results])
    if vocabs:
        out["bof_vocab_names"] = np.array([v["name"] for v in vocabs])
        for v in vocabs:
            out[f"hks_bof_power__{v['name']}"] = np.array([r["bof"][v["name"]] for r in results])
    return out


def report_spikes(ids, ratio):
    """Print the organoids whose hks_max_ratio exceeds HKS_SPIKE_FLAG (for manual inspection)."""
    flagged = [(u, r) for u, r in zip(ids, ratio) if r > HKS_SPIKE_FLAG]
    print(f"hks_max_ratio > {HKS_SPIKE_FLAG:g}: {len(flagged)} organoids flagged for inspection"
          + "".join(f"\n  {u}  {r:.1f}" for u, r in sorted(flagged, key=lambda x: -x[1])))


def save_npz(path, arrays):
    """np.savez, written to a temporary file first so a crash can't leave a broken npz."""
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def report_failures(out_path, failed):
    """List the organoids that failed in <out>.failed.txt and print them."""
    path = out_path[:-len(".npz")] + ".failed.txt"
    with open(path, "w") as fh:
        fh.writelines(f"{r['failed']}\t{r['error']}\n" for r in failed)
    print(f"{len(failed)} organoids failed" + (f", listed in {path}" if failed else ""))


def merge_existing(existing, new_kwargs, keep_mask):
    """Filter existing rows by keep_mask and append the newly computed rows, sorted by uid.

    Global label arrays must agree; per-row column sets must match exactly when
    there are new rows (otherwise the merge would misalign columns)."""
    per_row_existing = sorted(k for k in existing if k not in GLOBAL_KEYS)
    per_row_new      = sorted(k for k in new_kwargs if k not in GLOBAL_KEYS)
    if new_kwargs and per_row_existing != per_row_new:
        raise SystemExit(
            "--update: the existing npz and this run produce different columns "
            f"({per_row_existing} vs {per_row_new}). Match the vocab / --complexity-only "
            "setup, or rebuild without --update"
            + (" (a master from before hks_max_ratio existed: run --bof-only once to add it)."
               if set(per_row_new) - set(per_row_existing) == {"hks_max_ratio"} else "."))

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
    order = np.argsort(merged["ids"].astype(str), kind="stable")
    for k in per_row_existing:
        merged[k] = merged[k][order]
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
    ap.add_argument("--ids", default=None,
                    help="text file of uids (one per line): only these organoids are used")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (default 1). Each gets 8//workers threads.")
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
    ap.add_argument("--bof-only", action="store_true",
                    help="only re-encode HKS with the current vocab, replacing the "
                         "hks_bof_power__* arrays in an existing --out and keeping every "
                         "vocab-independent array (reuses the cached spherical maps); "
                         "use after changing output/vocab_*.npz instead of a full rebuild.")
    ap.add_argument("--complexity-only", action="store_true",
                    help="skip all HKS outputs (hks_amplitude, hks_bof_power__*); for "
                         "datasets run with config \"hks\": false (no eigvecs saved).")
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
        save_npz(args.out, existing)
        new = existing["l_cross_values"]
        print(f"Recomputed l_cross_values for {len(new)} organoids at threshold "
              f"{args.l_cross_threshold} → {args.out}")
        print(f"  mean l_cross: {new.mean():.3f}"
              + (f" (was {np.asarray(old, float).mean():.3f})" if old is not None else ""))
        return

    vocabs  = [] if args.complexity_only else load_vocabs()
    hks     = not args.complexity_only

    # Fast path: re-encode with the current vocab only (after a vocab change),
    # keeping every vocab-independent array.
    if args.bof_only:
        run_bof_only(args.out, vocabs, args.workers)
        return

    configs = [load_config(fp) for fp in args.folders]
    print(f"\nDatasets: {args.folders}\n")

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
            for t in build_tasks(data_path, cfg, vocabs, hks):
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
        save_npz(args.out, existing)
        print(f"Recomputed times+conditions for {len(ids)} rows → {args.out}")
        for arr, label in ((existing["times"], "times"), (existing["conditions"], "conditions")):
            uniq, cnts = np.unique(arr, return_counts=True)
            print(f"  {label}: {dict(zip(uniq.tolist(), cnts.tolist()))}")
        return

    # Build the full task list across all datasets (the target set of uids that
    # should be in the npz, per the current good_labels and --ids).
    wanted = None
    if args.ids:
        with open(args.ids) as fh:
            wanted = {l.strip() for l in fh if l.strip()}
    all_tasks = []
    for data_path, cfg in zip(args.folders, configs):
        tasks = build_tasks(data_path, cfg, vocabs, hks)
        if wanted is not None:
            tasks = [t for t in tasks if t[1] in wanted]
        print(f"{os.path.basename(data_path)}: {len(tasks)} organoids")
        all_tasks.extend(tasks)
    print(f"Total: {len(all_tasks)} organoids"
          + (f" ({len(wanted)} listed in {args.ids})" if wanted is not None else "") + "\n")

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

        results, failed = _run_tasks(missing_tasks, args.workers, vocabs) if missing_tasks else ([], [])
        report_failures(args.out, failed)
        if missing_tasks and not results:
            raise SystemExit("All new organoids failed to process — "
                             "existing npz left unchanged.")
        new_kwargs = (assemble_save_kwargs(results, vocabs, args.l_cross_threshold)
                      if results else {})
        save_kwargs = merge_existing(existing, new_kwargs, keep_mask)
    else:
        results, failed = _run_tasks(all_tasks, args.workers, vocabs)
        report_failures(args.out, failed)
        if not results:
            raise SystemExit("No organoids processed — did you run run_new_meshes.py for each dataset?")
        save_kwargs = assemble_save_kwargs(results, vocabs, args.l_cross_threshold)

    N = len(save_kwargs["ids"])
    print(f"\nSaving {N} organoids")
    if "hks_max_ratio" in save_kwargs:
        report_spikes(save_kwargs["ids"], save_kwargs["hks_max_ratio"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_npz(args.out, save_kwargs)
    print(f"Saved → {args.out}")
    for k, v in save_kwargs.items():
        print(f"  {k:25s}  {np.asarray(v).shape}")


if __name__ == "__main__":
    main()
