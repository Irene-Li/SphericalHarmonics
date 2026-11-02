import concurrent.futures
import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dimensionality-reduction embeddings: save / load / orient / panel style
# ---------------------------------------------------------------------------

def save_embedding(out_dir, name, xy, ids, method, **point_fields):
    """Save a 2D embedding keyed by organoid id.

    Args:
        out_dir: directory to write into (created if missing).
        name:    base filename, e.g. 'percentages_emb' -> {out_dir}/{name}.npz
        xy:      (N, 2) embedding coordinates.
        ids:     (N,) organoid id strings, aligned row-wise with xy.
        method:  str, the reduction method used ('umap'/'tsne'/'pca'/'phate').
        **point_fields: optional (N,) per-point arrays for colouring the
                        scatter (e.g. times=, l_cross=, areas=).

    Returns the written path.
    """
    xy = np.asarray(xy)
    ids = np.asarray(ids)
    assert xy.ndim == 2 and xy.shape[1] >= 2, f"xy must be (N,2+), got {xy.shape}"
    assert xy.shape[0] == ids.shape[0], (
        f"xy/ids length mismatch: {xy.shape[0]} vs {ids.shape[0]}")
    for k, v in point_fields.items():
        v = np.asarray(v)
        assert v.shape[0] == ids.shape[0], (
            f"point field '{k}' length {v.shape[0]} != n_ids {ids.shape[0]}")
        point_fields[k] = v

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(path, xy=xy, ids=ids, method=str(method), name=str(name), **point_fields)
    return path


def load_embedding(path):
    """Load an embedding saved by save_embedding into a plain dict."""
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    # scalars stored as 0-d arrays -> python str
    for k in ("method", "name"):
        if k in out:
            out[k] = str(out[k])
    return out


def orient_embedding(xy, up):
    """Rotate a 2D embedding about its centroid so the per-point score `up` increases
    straight upwards (+y), giving every embedding the same canonical orientation.

    The direction is the least-squares gradient of `up` over the layout, e.g.
    up = -l_cross puts the low-complexity organoids on top. A pure rotation, so all
    distances (and the layout itself) are unchanged; returns xy unchanged when `up`
    has no gradient.
    """
    xy = np.asarray(xy, float)
    c = xy.mean(0)
    g = np.linalg.lstsq(xy - c, np.asarray(up, float) - np.mean(up), rcond=None)[0]
    if not np.linalg.norm(g):
        return xy
    a = np.pi / 2 - np.arctan2(g[1], g[0])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (xy - c) @ rot.T + c


def rotate_to_top(xy, mask):
    """Rotate a 2D embedding about its centroid so the points in `mask` sit straight above
    it (+y), e.g. mask = the lowest-complexity organoids puts the spherical branch on top.
    A pure rotation, so all distances (and the layout itself) are unchanged."""
    xy = np.asarray(xy, float)
    c = xy.mean(0)
    g = xy[mask].mean(0) - c
    a = np.pi / 2 - np.arctan2(g[1], g[0])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (xy - c) @ rot.T + c


def style_embedding_ax(ax, box_aspect=0.8):
    """Same panel shape for every embedding plot: the axes box is height/width =
    box_aspect (0.8 = a bit wider than square) and the data aspect is equal, so the
    layout is undistorted and the data limits absorb the slack."""
    ax.set_box_aspect(box_aspect)
    ax.set_aspect('equal', adjustable='datalim')


def panel_colorbar(fig, mappable, ax, **kwargs):
    """Colorbar exactly as tall as its panel: drawn in an inset just right of the axes box,
    so it follows the panel's fixed box aspect (fig.colorbar(ax=...) instead takes the
    full grid-cell height and ends up taller than a style_embedding_ax panel)."""
    return fig.colorbar(mappable, cax=ax.inset_axes([1.03, 0, 0.04, 1]), **kwargs)


# enrichment condition per organoid, coloured as in the paper's morphospace figure
ENRICHMENT_COLOURS = {'timecourse': '#dddddd', 'ctrl': '#97e258', 'stem': '#fdb549',
                      'abs': '#fb452f', 'TA': '#4d4d4d', 'secretory': '#49c0d7'}


def enrichment_condition(datasets, conditions):
    """Organoid -> enrichment condition: main_dataset = 'timecourse', sup_dataset = 'ctrl'
    (the WT day-4.5 organoids), pert2 by drug prefix: stem-* = 'stem', abs-* = 'abs',
    ta-* = 'TA', sec-* = 'secretory'. Anything else keeps its condition name."""
    by_prefix = {'stem': 'stem', 'abs': 'abs', 'ta': 'TA', 'sec': 'secretory'}
    out = []
    for d, c in zip(np.asarray(datasets).astype(str), np.asarray(conditions).astype(str)):
        if d == 'main_dataset':
            out.append('timecourse')
        elif d == 'sup_dataset':
            out.append('ctrl')
        else:
            out.append(by_prefix.get(c.split('-')[0].lower(), c))
    return np.array(out)


def scatter_enrichment(ax, xy, cats, s=10, seed=0):
    """Scatter coloured by enrichment condition: 'timecourse' in light grey underneath, the
    other conditions on top in a seeded random order (so no condition hides another).
    Returns legend handles in ENRICHMENT_COLOURS order (plus any unknown condition, grey)."""
    cats = np.asarray(cats)
    colour = lambda c: ENRICHMENT_COLOURS.get(c, '#9e9e9e')
    under = cats == 'timecourse'
    ax.scatter(*np.asarray(xy)[under].T, s=s, color=colour('timecourse'), linewidths=0, zorder=1)
    top = np.random.default_rng(seed).permutation(np.where(~under)[0])
    ax.scatter(*np.asarray(xy)[top].T, s=s, c=[colour(c) for c in cats[top]], linewidths=0, zorder=2)
    present = [c for c in ENRICHMENT_COLOURS if c in set(cats)] + \
              sorted(set(cats) - set(ENRICHMENT_COLOURS))
    return [Line2D([], [], ls='none', marker='o', markersize=8, markeredgewidth=0,
                   markerfacecolor=colour(c), label=c) for c in present]


# ---------------------------------------------------------------------------
# Organoid id -> saved mesh and coefficient paths
# ---------------------------------------------------------------------------

def _vtp_flat_fm_path(data_path, cfg, label_uid, suffix):
    """Locate an fm_data file for a vtp_flat organoid.

    Fast path: the first uid token is the timepoint subfolder (main/sup:
    'day4p5_A06_11' -> vtp/day4p5/...). Fallback: the timepoint folder is NOT
    encoded in the uid (pert2: folders are 'normal'/'small' but the uid leads with
    the drug token, e.g. 'sec-DaptHi_day4p5_F02_47'; wnt_pert: folders are the
    conditions, e.g. 'IWP2_60hr_HiWnt_day3p5_C07_1'), so search the
    configured timepoints, else glob the vtp subfolders. Returns the fast-path
    candidate if nothing is found (caller handles the missing file)."""
    vtp_dir = cfg.get("vtp_dir", "vtp")
    fname = f"{label_uid}{suffix}"
    guess = f"{data_path}/{vtp_dir}/{label_uid.split('_')[0]}/fm_data/{fname}"
    if os.path.exists(guess):
        return guess
    tps = cfg.get("timepoints")
    subdirs = ([f"{data_path}/{vtp_dir}/{t}" for t in tps] if tps
               else glob.glob(f"{data_path}/{vtp_dir}/*"))
    for d in subdirs:
        cand = f"{d}/fm_data/{fname}"
        if os.path.exists(cand):
            return cand
    return guess


def vtp_flat_obj_path(data_path, cfg, label_uid):
    """PCA-transformed mesh (.obj) for a vtp_flat organoid."""
    return _vtp_flat_fm_path(data_path, cfg, label_uid, "_transformed_mesh.obj")


def vtp_flat_coeffs_path(data_path, cfg, label_uid):
    """Saved coefficients (.npz) for a vtp_flat organoid."""
    return _vtp_flat_fm_path(data_path, cfg, label_uid, "_coeffs.npz")



# ---------------------------------------------------------------------------
# Spherical coordinates
# ---------------------------------------------------------------------------

def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x) % (2*np.pi)
    return r, theta, phi

def sph2latlon(theta, phi):
    lat = (np.pi/2 - theta)*180/np.pi
    lon = phi*180/np.pi
    return lat, lon


# ---------------------------------------------------------------------------
# HKS shape features  (used by dim_red.ipynb)
# ---------------------------------------------------------------------------

def load_hks_features(master_path, mode_cut=8, amplitude_weight=1.2, keep_ids=None):
    """Shape features: the HKS bag-of-features SPHARM spectrum plus one HKS-amplitude column.

    Returns (ids, features) with features (N, mode_cut * n_vocab [+ 1]). The stored power per
    degree of each vocab channel (compute_master_npz.compute_bof_power) becomes sqrt(power), the
    RMS amplitude of that degree, and each (degree, channel) is standardised, so every degree has
    an equal say in the distance (raw power leaves ~90% of it to degrees 0 and 2). With
    amplitude_weight > 0 one column is appended: log(hks_amplitude) z-scored, scaled so its mean
    pairwise |difference| equals the mean pairwise distance of the spectrum block, times
    amplitude_weight. The amplitude restores how strongly the HKS varies over the surface, which
    the vocab encoding normalises away. `keep_ids` restricts the file to those organoids, in that
    order, before anything is computed, so the scaling comes from them alone.
    """
    m = np.load(master_path, allow_pickle=True)
    ids = m["ids"].astype(str)
    power = np.array(m["hks_bof_power__kmeans_variable"], dtype=float)  # (N, n_degrees, n_vocab)
    if mode_cut > power.shape[1]:
        raise ValueError(f"{master_path} has only {power.shape[1]} degrees "
                         f"(mode_cut={mode_cut}; see compute_master_npz.HKS_LMAX)")
    row = {u: i for i, u in enumerate(ids)}
    rows = np.arange(len(ids)) if keep_ids is None else np.array([row[u] for u in keep_ids])
    ids = ids[rows]
    amps = np.sqrt(power[rows, :mode_cut])
    H = ((amps - amps.mean(axis=0, keepdims=True)) / amps.std(axis=0, keepdims=True)).reshape(len(ids), -1)
    if not amplitude_weight:
        return ids, H
    z = np.log(m["hks_amplitude"])[rows]
    z = (z - z.mean()) / z.std()
    scale = pdist(H).mean() / pdist(z[:, None]).mean()
    return ids, np.hstack([H, amplitude_weight * scale * z[:, None]])


def _fate_from_csv(data_root, datasets, csv_name):
    """Slow path: build per-organoid fate fractions from the per-cell CSVs.

    Iterates every row in the CSV and only stores organoids actually found
    (no zero-fill for missing ids). Returns (found_ids, P, col_names).
    """
    perc_dfs = {}
    for ds in sorted(set(datasets)):
        path = os.path.join(data_root, ds, csv_name)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            header = fh.readline().strip().split(",")
        excl = [c for c in header if c.endswith("_exclusive")]
        df = pd.read_csv(path, usecols=["label_uid", "projected_to_mesh"] + excl,
                         low_memory=False)
        df = df[df["projected_to_mesh"].astype(str) == "True"]
        pos = (df[excl] > 0).astype(float)
        pos["label_uid"] = df["label_uid"].values
        perc_dfs[ds] = pos.groupby("label_uid")[excl].mean()
    if not perc_dfs:
        return np.zeros(0, dtype=str), np.zeros((0, 0)), []
    ct_cols = sorted(set.intersection(*[set(df.columns) for df in perc_dfs.values()]))
    col_names = [c.split(".")[0] for c in ct_cols]
    found_ids, rows_P = [], []
    for ds in sorted(set(datasets)):
        if ds not in perc_dfs:
            continue
        for bare in perc_dfs[ds].index:
            found_ids.append(f"{ds}_{bare}")
            rows_P.append(perc_dfs[ds].loc[bare, ct_cols].values)
    found_ids = np.array(found_ids, dtype=str)
    P = np.array(rows_P, dtype=float) if rows_P else np.zeros((0, len(col_names)))
    return found_ids, P, col_names


def load_fate_percentages(data_root, ids, datasets,
                          csv_name="cell_features_class_with_projection_exclusive.csv",
                          cache_path="Data/npz/fate_percentages.npz", update=False):
    """Per-organoid fate-marker fractions for organoids found in the CSV.

    Returns (ids_found, percentages, col_names) where ids_found is the subset of
    `ids` that appear in the fate CSV, in the same relative order as `ids`.
    Organoids absent from the CSV are silently excluded (never zero-filled), so
    the caller can filter its own arrays by the returned ids.

    The slow CSV parse (millions of rows) is cached to `cache_path`. The cache
    stores only found organoids. Pass update=True to force a rebuild.
    """
    ids = np.asarray(ids).astype(str)
    csv_paths = [os.path.join(data_root, ds, csv_name) for ds in sorted(set(datasets))]
    csv_paths = [p for p in csv_paths if os.path.exists(p)]

    cache_ids, cache_P, col_names = None, None, None
    if cache_path and not update and os.path.exists(cache_path):
        fresh = not csv_paths or (
            os.path.getmtime(cache_path) >= max(os.path.getmtime(p) for p in csv_paths))
        if fresh:
            z = np.load(cache_path, allow_pickle=True)
            cache_ids = z["ids"].astype(str)
            cache_P   = z["percentages"]
            col_names = list(z["col_names"])

    if cache_ids is None:                       # cold start or stale cache
        cache_ids, cache_P, col_names = _fate_from_csv(data_root, datasets, csv_name)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, ids=cache_ids, percentages=cache_P,
                     col_names=np.array(col_names))

    cache_pos = {u: i for i, u in enumerate(cache_ids)}
    found_mask = np.array([u in cache_pos for u in ids])
    ids_found  = ids[found_mask]
    rows       = [cache_pos[u] for u in ids_found]
    return ids_found, cache_P[rows], col_names


# ---------------------------------------------------------------------------
# Per-organoid HKS cache (used by build_vocab.py)
#
# Loading a mesh's saved eigendecomposition and computing its decimated per-vertex
# HKS is the slow step of vocab training. These helpers cache that result (the
# decimated per-vertex HKS at N_TIMES times + frac/complexity/area) keyed to the
# current good_labels + N_TIMES/DECIMATE_FACE, so re-running vocab experiments
# (different CURV_Z, KMeans k, ...) reads one ~60 MB npz instead of re-reading
# every mesh. build_vocab supplies the per-mesh loader `run_one` so this module
# stays free of the mesh/FateMarkers dependency (avoids a circular import).
# ---------------------------------------------------------------------------

def _hks_cache_worker_init(n_threads):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)


def load_hks_cache(cache_path, save_paths, n_times, decimate_face):
    """Return (hkss, fracs, complexities) from the HKS cache if it matches the given
    good_labels (`save_paths`) and params, else None. Degenerate (empty-HKS)
    organoids are dropped. `hkss` is a list of (n_decim_verts, n_times) arrays."""
    if not os.path.exists(cache_path):
        return None
    z = np.load(cache_path, allow_pickle=True)
    if (int(z["meta_n_times"]) != n_times or int(z["meta_decimate_face"]) != decimate_face
            or not np.array_equal(z["input_save_paths"].astype(str),
                                  np.asarray(save_paths, dtype=str))):
        return None
    hkss = np.split(z["hks"], np.cumsum(z["lengths"])[:-1])
    fracs, comps = z["fracs"], z["complexities"]
    keep = np.array([h.shape[0] > 0 for h in hkss])
    if not keep.all():
        hkss = [h for h, k in zip(hkss, keep) if k]
        fracs, comps = fracs[keep], comps[keep]
    return hkss, fracs, comps


def build_hks_cache(cache_path, save_paths, run_one, n_times, decimate_face, datasets, workers=6):
    """Create/update the per-organoid HKS cache at `cache_path`.

    Runs `run_one(save_path)` — which returns (save_path, hks, frac, complexity,
    area) or None for a failed/degenerate mesh — over `save_paths` in parallel and
    saves the decimated per-vertex HKS + scalars. `run_one` is passed in by the
    caller (build_vocab._run_one) so this stays independent of the mesh loader."""
    res = [None] * len(save_paths)
    if workers == 1:
        for i, p in enumerate(tqdm(save_paths, desc="hks cache")):
            res[i] = run_one(p)
    else:
        tpw = max(1, 8 // workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, initializer=_hks_cache_worker_init, initargs=(tpw,)) as ex:
            for i, r in enumerate(tqdm(ex.map(run_one, save_paths),
                                       total=len(save_paths), desc="hks cache")):
                res[i] = r

    results = [r for r in res if r is not None]
    kept_paths = [r[0] for r in results]
    hkss  = [r[1] for r in results]
    fracs = np.array([r[2] for r in results])
    comps = np.array([r[3] for r in results])
    areas = np.array([r[4] for r in results])
    lengths = np.array([h.shape[0] for h in hkss])
    hks = np.concatenate(hkss).astype(np.float64) if hkss else np.zeros((0, n_times))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path,
             input_save_paths=np.array(save_paths), kept_save_paths=np.array(kept_paths),
             hks=hks, lengths=lengths, fracs=fracs, complexities=comps, areas=areas,
             meta_n_times=n_times, meta_decimate_face=decimate_face,
             meta_datasets=np.array(datasets))
    print(f"  HKS cache: {len(kept_paths)}/{len(save_paths)} organoids -> {cache_path} "
          f"({hks.nbytes / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# Mesh topology  (manage_discards.py's hole rule)
#
# A clean organoid surface is one closed, genus-0 sheet. Per mesh the table
# {dataset}/feature_tables/mesh_topology.csv records:
#   n_components         connected pieces of the surface
#   n_boundary_edges     edges with only one adjacent face (open surface)
#   n_boundary_loops     open holes = connected loops of boundary edges
#   n_nonmanifold_edges  edges shared by > 2 faces (self-intersecting / fin-like surface)
#   n_pinch_vertices     extra fans at vertices where the surface touches itself at a
#                        point (split before computing the genus; not a hole)
#   euler                V - E + F (referenced vertices only)
#   genus                handles / tunnels through the surface: (2C - euler' - B) / 2 with
#                        euler' after splitting pinch vertices (NaN when an edge is
#                        non-manifold, where the genus is ill-defined)
#   has_hole             1 if n_boundary_loops > 0 or genus > 0 (or genus undefined)
# A deep dead-end pit is NOT a topological hole (the surface stays closed, genus 0);
# compute_master_npz's hks_max_ratio flags the narrow pits that matter.
# ---------------------------------------------------------------------------

TOPOLOGY_FIELDS = ["label_uid", "n_vertices", "n_faces", "n_components", "n_boundary_edges",
                   "n_boundary_loops", "n_nonmanifold_edges", "n_pinch_vertices", "euler",
                   "genus", "has_hole"]


def mesh_topology(f):
    """Topology summary of a triangle mesh given its (F, 3) face array.

    Pinch vertices (the surface touching itself at a single point) are split
    before the genus is computed — they are not holes. Each face corner is a
    node; corners of the same vertex are joined across every edge shared by
    exactly two faces, so the connected groups are the vertex's fans."""
    used = np.unique(f)
    nv = len(used)
    remap = np.full(f.max() + 1, -1)
    remap[used] = np.arange(nv)
    f = remap[f]
    nf = len(f)

    # half-edges (face, corner a, corner b) keyed by their undirected edge
    he_face = np.repeat(np.arange(nf), 3)
    ka, kb = np.tile([0, 1, 2], nf), np.tile([1, 2, 0], nf)
    va, vb = f[he_face, ka], f[he_face, kb]
    key = np.sort(np.stack([va, vb], 1), 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key_s = key[order]
    edges, start, count = np.unique(key_s, axis=0, return_index=True, return_counts=True)
    n_edges = len(edges)
    bnd = edges[count == 1]
    n_nonmanifold = int((count > 2).sum())

    # manifold edges: join the two faces, and the matching corners at both endpoints
    two = start[count == 2]
    h1, h2 = order[two], order[two + 1]
    def corner(h, v):
        return 3 * he_face[h] + np.where(va[h] == v, ka[h], kb[h])
    lo, hi = key_s[two, 0], key_s[two, 1]
    rows = np.concatenate([corner(h1, lo), corner(h1, hi)])
    cols = np.concatenate([corner(h2, lo), corner(h2, hi)])
    n_fans = connected_components(sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                                                shape=(3 * nf, 3 * nf)), directed=False)[0]
    n_comp = connected_components(sp.coo_matrix((np.ones(len(two)), (he_face[h1], he_face[h2])),
                                                shape=(nf, nf)), directed=False)[0]

    # boundary loops = connected components of the boundary-edge graph
    if len(bnd):
        bv, inv = np.unique(bnd, return_inverse=True)
        inv = inv.reshape(-1, 2)
        badj = sp.coo_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])), shape=(len(bv), len(bv)))
        n_loops = connected_components(badj, directed=False)[0]
    else:
        n_loops = 0

    euler = nv - n_edges + nf
    euler_split = n_fans - n_edges + nf          # after splitting pinch vertices
    genus = np.nan if n_nonmanifold else (2 * n_comp - euler_split - n_loops) / 2
    has_hole = int(n_loops > 0 or not np.isfinite(genus) or genus > 0)
    return dict(n_vertices=nv, n_faces=nf, n_components=int(n_comp),
                n_boundary_edges=len(bnd), n_boundary_loops=int(n_loops),
                n_nonmanifold_edges=n_nonmanifold, n_pinch_vertices=int(n_fans - nv),
                euler=int(euler), genus=genus, has_hole=has_hole)


def _mesh_topology_row(path):
    """One row of the topology table, the mesh read exactly as the pipeline reads it
    (FateMarkers.load_mesh_from_file: .vtp / .stl / .obj with the same point merging)."""
    from src.fatemarkers import FateMarkers         # here: vtk is heavy, and only this needs it
    uid = os.path.splitext(os.path.basename(path))[0]
    try:
        m = FateMarkers()
        m.load_mesh_from_file(path)
        return dict(label_uid=uid, **mesh_topology(np.asarray(m.f, dtype=np.int64)))
    except Exception as e:
        print(f"\n  [skip] {uid}: {e}")
        return None


def update_mesh_topology(data_path, workers=1, refresh=False, verbose=True):
    """Bring {data_path}/feature_tables/mesh_topology.csv up to date: compute every mesh of the
    (vtp_flat) dataset that is not in it yet, or all of them with refresh. Returns its path.
    manage_discards.py calls this before reading the table."""
    data_path = data_path.rstrip("/")
    with open(f"{data_path}/config.json") as fh:
        cfg = json.load(fh)
    if cfg.get("layout") != "vtp_flat":
        raise SystemExit("mesh topology supports the vtp_flat layout only")
    out = f"{data_path}/feature_tables/mesh_topology.csv"

    rows = {}
    if os.path.exists(out) and not refresh:
        with open(out) as fh:
            rows = {r["label_uid"]: r for r in csv.DictReader(fh)}
    ext, vtp_dir = cfg.get("mesh_ext", "vtp"), cfg.get("vtp_dir", "vtp")
    paths = sorted(p for tp in cfg["timepoints"]
                   for p in glob.glob(os.path.join(data_path, vtp_dir, tp, f"*.{ext}")))
    todo = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in rows]
    if verbose or todo:
        print(f"{len(rows)} meshes already in {out} | {len(todo)} to compute")
    if not todo and rows:
        return out

    if workers == 1:
        results = [_mesh_topology_row(p) for p in tqdm(todo)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(_mesh_topology_row, todo, chunksize=8), total=len(todo)))
    rows.update({r["label_uid"]: r for r in results if r is not None})

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TOPOLOGY_FIELDS)
        w.writeheader()
        w.writerows(rows[u] for u in sorted(rows))

    holes = [r for r in rows.values() if int(r["has_hole"])]
    def n(key): return sum(1 for r in rows.values() if float(r[key]) > 0)
    print(f"wrote {out}: {len(rows)} meshes | {len(holes)} with holes "
          f"(open: {n('n_boundary_loops')}, non-manifold: {n('n_nonmanifold_edges')}, "
          f"handles: {sum(1 for r in rows.values() if str(r['genus']) not in ('nan', '') and float(r['genus']) > 0)}) "
          f"| multi-piece: {sum(1 for r in rows.values() if int(r['n_components']) > 1)}")
    return out


# ---------------------------------------------------------------------------
# Cluster representatives  (dim_red.ipynb's highlight figure)
# ---------------------------------------------------------------------------

def update_highlight_json(path, ids, features, clusters, runners_up=2):
    """Pick one representative organoid per aggregated shape cluster: the cluster MEDOID, the
    member with the smallest mean feature distance to the other members (the organoid most
    similar to the rest of its cluster). The low-complexity group 0 is skipped (the
    figure's 'spherical' row already illustrates it).

    ids, features, clusters are row-aligned (dim_red's m_ids, shape_features and
    aggregated_shape_clusters). Prints, per cluster, the medoid, the runners-up and where the
    json's current pick ranks, then replaces the json's 'highlight' list with the medoids, carrying
    over each cluster's 'name'; 'spherical' and any other keys are kept. Returns the picks."""
    ids = np.asarray(ids).astype(str)
    F = np.asarray(features, float)
    agg = np.asarray(clusters).astype(int)
    doc = json.load(open(path)) if os.path.exists(path) else {}
    pos = {u: i for i, u in enumerate(ids)}
    current = {}                                   # cluster -> the json's current entry for it
    for h in doc.get("highlight", []):
        if h["uid"] in pos:
            current.setdefault(int(agg[pos[h["uid"]]]), h)

    picks = []
    for c in sorted(set(agg.tolist())):
        if c == 0:
            continue
        idx = np.where(agg == c)[0]
        n = len(idx)
        if n < 2:
            print(f"cluster {c}: only {n} member — skipped")
            continue
        D = cdist(F[idx], F[idx])
        mean_d = D.sum(1) / (n - 1)                # mean distance to the OTHER members
        order = np.argsort(mean_d, kind="stable")
        print(f"cluster {c} (n={n}, mean distance between members {D.sum() / (n * (n - 1)):.3f})")
        for r, k in enumerate(order[:1 + runners_up]):
            print(f"  {'medoid' if r == 0 else f'#{r + 1}':>6s}  {ids[idx[k]]:45s} mean distance {mean_d[k]:.3f}")
        old = current.get(c)
        if old is not None:
            k_old = int(np.where(idx[order] == pos[old["uid"]])[0][0])
            print(f"  {'was':>6s}  {old['uid']:45s} mean distance {mean_d[order[k_old]]:.3f}"
                  f"  (rank {k_old + 1}/{n})")
        picks.append({"uid": str(ids[idx[order[0]]]),
                      "note": f"agg_cluster {c} medoid: smallest mean feature distance to the other "
                              f"{n - 1} members ({mean_d[order[0]]:.3f}; cluster average "
                              f"{D.sum() / (n * (n - 1)):.3f})",
                      "name": (old or {}).get("name", "")})
    doc["highlight"] = picks
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(picks)} representatives to {path}")
    return picks
