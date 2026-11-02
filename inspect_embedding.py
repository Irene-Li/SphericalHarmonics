#!/usr/bin/env python3
"""
Interactive viewer for a saved 2D embedding of organoids.

One window with two side-by-side panels (drag the divider to resize them):
  * Embedding (left) - the embedding as a true 2D scatter. Click a point to
                       load that organoid; hover to see its id in the status
                       bar. "Colour points by" picks the scatter colouring and
                       the toolbar underneath zooms / pans / saves the plot.
  * Organoid (right) - ONLY the selected organoid's 3D mesh, which you can
                       rotate (drag), zoom (scroll) and pan (shift-drag)
                       independently of the scatter. "Colour organoid by" picks
                       the per-vertex overlay (fate marker, or HKS with
                       --source pipeline); the choice is kept across organoids.
                       Below it, a bar chart of the organoid's fate fractions.
  The 3D camera is kept when you click another organoid, so meshes (all
  PCA-aligned and scaled to a unit sphere) are compared from the same
  viewpoint; "Reset view" goes back to the default front view.

The embedding is written by `src.utils.save_embedding` (e.g. from
dim_red.ipynb).

The scatter can be colored by l_cross / area / time / condition or by any fate
marker's percentage (the l=0 harmonic coefficient saved on the embedding as 'perc_<name>'
point-fields by dim_red.ipynb; log scale, shared range), or by
cell-type diversity (Hill numbers saved by dim_red.ipynb as 'hill_q<val>' point
fields, one per order q in [0, 1]; magma, each q with its own colour range).

Mesh source is selectable with --source:
  vtp       (default) load straight from the original .vtp — needs only the
            embedding .npz and the .vtp meshes, no pipeline outputs.
  pipeline  load the precomputed per-organoid '{stem}_coeffs.npz' (PCA-aligned
            eigendecomposition) + '{stem}_transformed_mesh.obj' written by
            run_new_meshes.py. This skips re-aligning/re-eigendecomposing the
            mesh, and unlocks two HKS-based overlay families computed straight
            from the saved eigenvalues/eigenvectors:
              - raw HKS at fixed times ('hks_t1'/'hks_t4'/'hks_t25'/'hks_t100',
                same times as elsewhere in the pipeline; hks_t25 shown by default)
              - bag-of-features ('bof_<vocab>_word<k>'): per-vertex soft
                assignment to each vocabulary in compute_master_npz.VOCABS
                (kmeans_variable, pca_variable — same vocabs/encoding as the
                'hks_bof_power__*' columns in master.npz), loaded from output/.
            Fate fields are read from a matching .vtp if one is found (the
            pipeline does not decompose fate fields).

In both modes, --data_root is walked once and meshes/pipeline files are found
by filename regardless of nesting, so any layout works.

Organoid ids are dataset-prefixed ('{dataset}_{timepoint}_{well}_{label}'); the
trailing part is the filename stem used to find the mesh / pipeline files. If a
dataset's config.json is present next to the meshes it is used (and
auto-discovered) to give the fate fields friendly names; otherwise the raw vtp
field names are shown.

Run from the repository root, in the `scmpx` conda env:

    python inspect_embedding.py \
        --embedding output/embeddings/hks_emb.npz --data_root Data

    python inspect_embedding.py \
        --embedding output/embeddings/hks_emb.npz --data_root Data --source pipeline

Click a point in the scatter to load that organoid. Close the window to quit.
"""

import argparse
import json
import os
import sys

import numpy as np
# conda-forge's vtk links its optional C++ Qt module against Qt6; `import vtk`
# (here and in src/) would load it next to PyQt5's Qt5 and the two clash on
# macOS. The viewer doesn't use it, and vtkmodules.all skips it on ImportError.
sys.modules.setdefault("vtkmodules.vtkRenderingQt", None)
# Import the Qt binding first so matplotlib's Qt backend and VTK's Qt widget
# both pick up PyQt5.
from PyQt5 import QtCore, QtWidgets
import matplotlib
matplotlib.use("QtAgg")   # any stray pyplot figure lands in Qt too
import matplotlib.ticker as mticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure
import vtk
import vtkmodules.qt
vtkmodules.qt.PyQtImpl = "PyQt5"   # VTK would otherwise prefer PySide6/PyQt6 if installed
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import ID_TYPE_CODE, numpy_to_vtk, numpy_to_vtkIdTypeArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import utils
from src.fatemarkers import FateMarkers
from compute_master_npz import find_times, hks_unit_area, encode_vocab, load_vocabs

DEFAULT_MARKER = "lgr"
COMBO_MARKERS = ("STEM", "EC", "PANETH")   # must all be present in perc_* fields
HKS_TIMES = [1, 4, 25, 100]   # matches compute_master_npz.SPARSE_TS / output/encodings_*.vtp
HKS_DEFAULT_TIME = 25         # shown by default in --source pipeline (mid diffusion scale)
COEFFS_SUFFIX = "_coeffs.npz"
SURFACE_ONLY = "(surface only)"   # 'Colour organoid by' entry: plain shaded mesh
MESH_COLOR = (0.55, 0.65, 0.85)
PICK_RADIUS_PT = 12           # a click selects the nearest point within this radius


def _fit_mesh(v, radius=1.0):
    """Center a mesh at the origin and scale it into a sphere of `radius`."""
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean(axis=0)
    v = v / (np.linalg.norm(v, axis=1).max() or 1.0) * radius
    return v.astype(np.float32)


def _vtk_mesh(v, f):
    """vtkPolyData from a (V, 3) vertex array and an (F, k) face array."""
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(v, dtype=np.float32), deep=True))
    f = np.asarray(f, dtype=ID_TYPE_CODE)
    offsets = np.arange(0, f.size + 1, f.shape[1], dtype=ID_TYPE_CODE)
    polys = vtk.vtkCellArray()
    polys.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(f.ravel()), deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(pts)
    mesh.SetPolys(polys)
    return mesh


def _vtk_lut(cmap_name="viridis", n=256):
    """A matplotlib colormap as a vtkLookupTable (NaN -> light grey)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.Build()
    for i, rgba in enumerate(matplotlib.colormaps[cmap_name](np.linspace(0, 1, n))):
        lut.SetTableValue(i, *rgba)
    lut.SetNanColor(0.8, 0.8, 0.8, 1.0)
    return lut


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True,
                    help="path to an embedding .npz (from utils.save_embedding)")
    ap.add_argument("--data_root", default="Data",
                    help="folder to search (recursively) for the original .vtp / .stl "
                         "meshes. Meshes are found by filename, so any layout works.")
    ap.add_argument("--source", choices=["vtp", "pipeline"], default="vtp",
                    help="'vtp' (default) loads straight from the .vtp; 'pipeline' "
                         "loads the precomputed '{stem}_coeffs.npz' + "
                         "'{stem}_transformed_mesh.obj' instead, which also enables "
                         "HKS scalar overlays (see module docstring).")
    ap.add_argument("--groups", default=None,
                    help="hand-picked shape groups json (a list of lists of organoid ids), "
                         "shown as the 'uid_group' colouring. Default: "
                         "<data_root>/uid_groups.json when it exists.")
    args = ap.parse_args()
    print(f"Mesh source: {args.source}"
          + ("  (HKS overlays enabled)" if args.source == "pipeline" else ""))

    # ---- load embedding -------------------------------------------------
    emb = utils.load_embedding(args.embedding)
    ids = emb["ids"].astype(str)
    xy = np.asarray(emb["xy"], dtype=float)
    method = emb.get("method", "?")
    times = emb["times"].astype(str) if "times" in emb else np.array(["?"] * len(ids))
    l_cross = emb["l_cross"].astype(float) if "l_cross" in emb else np.full(len(ids), np.nan)
    areas = emb["areas"].astype(float) if "areas" in emb else np.full(len(ids), np.nan)
    # optional per-point shape-cluster labels (saved on the shape embedding)
    shape_clusters = emb["shape_cluster"].astype(float) if "shape_cluster" in emb else None
    # optional aggregated clusters (0 = low-complexity merged group, 1..n = high-complexity)
    agg_clusters = emb["agg_cluster"].astype(float) if "agg_cluster" in emb else None
    # optional perturbation/genotype condition ('WT' or a pert2 drug token)
    conditions = emb["condition"].astype(str) if "condition" in emb else None
    print(f"Loaded {len(ids)} points from {args.embedding} (method={method})")

    # ---- locate the original .vtp meshes and/or the pipeline outputs ---
    # Build {filename-stem: [paths]} indexes by walking data_root once. The
    # collaborator only needs to drop the meshes (and/or pipeline fm_data/
    # output) somewhere under data_root; the exact nesting doesn't matter. An
    # embedding id is the dataset name followed by the mesh's filename stem
    # ('main_dataset' + '_' + 'day2p5_A04_67'), and run_new_meshes.py writes
    # pipeline outputs under the same stem ('{stem}_coeffs.npz' /
    # '{stem}_transformed_mesh.obj'), so both are found by matching the
    # longest trailing part of the id against an indexed stem.
    vtp_index = {}
    pipeline_index = {}   # stem -> [base path, i.e. path without '_coeffs.npz']
    for root, _dirs, files in os.walk(args.data_root, followlinks=True):   # datasets may be symlinks
        for fn in files:
            if fn.endswith((".vtp", ".stl")):     # .stl: geometry only (e.g. wnt_pert)
                vtp_index.setdefault(fn[:-4], []).append(os.path.join(root, fn))
            elif fn.endswith(COEFFS_SUFFIX):
                stem = fn[:-len(COEFFS_SUFFIX)]
                pipeline_index.setdefault(stem, []).append(os.path.join(root, stem))
    n_vtp = sum(len(v) for v in vtp_index.values())
    n_pipeline = sum(len(v) for v in pipeline_index.values())
    print(f"Indexed {n_vtp} .vtp/.stl meshes and {n_pipeline} pipeline coeffs under {args.data_root}")
    if n_vtp == 0 and args.source == "vtp":
        print(f"  [warn] no .vtp meshes found under {args.data_root!r} — "
              f"meshes won't load. Pass --data_root <folder with the .vtp files>.")
    if n_pipeline == 0 and args.source == "pipeline":
        print(f"  [warn] no '*{COEFFS_SUFFIX}' files found under {args.data_root!r} — "
              f"meshes won't load. Run run_new_meshes.py first, or use --source vtp.")

    def _resolve_by_stem(full_id, index):
        """Map an embedding id to a path via `index` (or None if not found).

        Tries progressively shorter trailing parts of the id ('a_b_c_d' ->
        'b_c_d' -> 'c_d' ...) and returns the first that is an indexed stem;
        if several files share that stem, prefers one whose path contains the
        leading dataset prefix.
        """
        toks = full_id.split("_")
        for i in range(len(toks)):
            stem = "_".join(toks[i:])
            cands = index.get(stem)
            if not cands:
                continue
            if len(cands) == 1:
                return cands[0]
            prefix = "_".join(toks[:i])
            for p in cands:
                if prefix and prefix in p.replace(os.sep, "_"):
                    return p
            return cands[0]
        return None

    def resolve_vtp(full_id):
        return _resolve_by_stem(full_id, vtp_index)

    def resolve_pipeline_base(full_id):
        return _resolve_by_stem(full_id, pipeline_index)

    # ---- bag-of-features vocabularies (--source pipeline only) ---------
    # Same vocabs/encoding compute_master_npz uses for the 'hks_bof_power__*'
    # columns in master.npz (VOCABS: kmeans_variable, pca_variable), loaded once
    # here and reused for every organoid as per-vertex 'bof_<vocab>_word<k>'
    # overlays (soft-assignment encoding, not the spherical-harmonic coeffs).
    bof_vocabs = load_vocabs() if args.source == "pipeline" else []

    # ---- optional config.json for friendly fate names ------------------
    # config.json is not required.  If one sits next to (or above) a mesh, it
    # is used to combine/exclude markers and rename the raw vtp fields to
    # friendly names; otherwise the raw field names are shown.
    _cfg_cache = {}
    _root_abs = os.path.abspath(args.data_root)

    def find_cfg(mesh_path):
        """(cfg_or_None, field_to_friendly) by searching up from a mesh/pipeline path."""
        d = os.path.dirname(os.path.abspath(mesh_path))
        if d in _cfg_cache:
            return _cfg_cache[d]
        cur, result = d, (None, {})
        while True:
            cfg_path = os.path.join(cur, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
                f2f = {}
                for name, flds in cfg.get("annotation_names", {}).items():
                    for fl in ([flds] if isinstance(flds, str) else flds):
                        f2f[fl] = name
                result = (cfg, f2f)
                break
            if os.path.normpath(cur) == _root_abs or cur == os.path.dirname(cur):
                break
            cur = os.path.dirname(cur)
        _cfg_cache[d] = result
        return result

    # ---- fate percentages from the embedding ---------------------------
    # dim_red saves each fate as a 'perc_<name>' point-field: the
    # lowest-order (l=0) harmonic coefficient / sqrt(area), i.e. the fraction of
    # surface covered by that marker in [0, 1]. No cell-type CSV needed.
    fate_labels = [k[len("perc_"):] for k in emb if k.startswith("perc_")]
    perc_matrix = (np.column_stack([emb[f"perc_{lab}"].astype(float) for lab in fate_labels])
                   if fate_labels else np.zeros((len(ids), 0)))
    log_perc = np.log(perc_matrix + 1e-3)
    log_lo, log_hi = ((float(log_perc.min()), float(log_perc.max()))
                      if fate_labels else (0.0, 1.0))

    # timepoints -> numeric code for coloring
    uniq_t = sorted(set(times))
    tcode = np.array([uniq_t.index(t) for t in times], dtype=float)

    # coloring options for the scatter: (values, cmap, clim) -- cmaps match the notebook
    color_fields = {"l_cross": (l_cross, "Greens", (0, 7)),
                    "area":    (areas,   "Blues",  None),
                    "time":    (tcode,   "viridis", None)}
    color_options = ["l_cross", "area", "time"]
    cat_levels = {}   # categorical field -> (code, label) pairs for the colorbar
    cat_cmaps = {}    # categorical field -> explicit colour list (else tab10/tab20)

    # 3-marker presence-absence combo (8-way categorical), from the perc fields
    if all(m in fate_labels for m in COMBO_MARKERS):
        a, b, c = COMBO_MARKERS
        pj = {m: fate_labels.index(m) for m in COMBO_MARKERS}
        combo = np.array([f"{a}{'+' if perc_matrix[i, pj[a]] > 0 else '-'} "
                          f"{b}{'+' if perc_matrix[i, pj[b]] > 0 else '-'} "
                          f"{c}{'+' if perc_matrix[i, pj[c]] > 0 else '-'}"
                          for i in range(len(ids))])
        combo_levels = sorted(set(combo))
        combo_codes = np.array([combo_levels.index(c) for c in combo], dtype=float)
        color_fields["fate_combo"] = (combo_codes, "tab10", (0, max(1, len(combo_levels) - 1)))
        cat_levels["fate_combo"] = list(enumerate(combo_levels))
        color_options.append("fate_combo")

    for j, lab in enumerate(fate_labels):
        color_fields[lab] = (log_perc[:, j], "Reds", (log_lo, log_hi))
    color_options += list(fate_labels)
    fate_set = set(fate_labels)

    # Hill-number diversity (saved by dim_red as 'hill_q<val>'): effective number
    # of cell types per organoid at order q. Continuous, magma like the notebook;
    # each q gets its own colour range (clim=None -> per-field min/max).
    hill_labels = sorted((k for k in emb if k.startswith("hill_q")),
                         key=lambda k: float(k[len("hill_q"):]))
    hill_div = emb[hill_labels[-1]].astype(float) if hill_labels else None
    if hill_labels:
        for k in hill_labels:
            color_fields[k] = (emb[k].astype(float), "magma", None)
        color_options += hill_labels
        print(f"hill diversity: {len(hill_labels)} orders q "
              f"({hill_labels[0]}..{hill_labels[-1]})")

    # shape-cluster colouring (only present on the shape embedding)
    if shape_clusters is not None:
        color_fields["shape_cluster"] = (shape_clusters, "tab10",
                                         (shape_clusters.min(), shape_clusters.max()))
        color_options.insert(3, "shape_cluster")
        uniq_cl = sorted(set(shape_clusters.astype(int)))
        cat_levels["shape_cluster"] = [(c, f"cluster {c}") for c in uniq_cl]

    # aggregated-cluster colouring: label 0 is the low-complexity merged group
    # (shown grey), 1..n are the high-complexity clusters (distinct colours).
    if agg_clusters is not None:
        uniq_ag = sorted(set(agg_clusters.astype(int)))
        color_fields["agg_cluster"] = (agg_clusters, "tab20", (min(uniq_ag), max(uniq_ag)))
        cat_levels["agg_cluster"] = [(c, "low-cplx" if c == 0 else f"cluster {c}")
                                     for c in uniq_ag]
        n_high = sum(c != 0 for c in uniq_ag)
        base = matplotlib.colormaps["tab10" if n_high <= 10 else "tab20"]
        hi = [base(i % base.N) for i in range(n_high)]
        cat_cmaps["agg_cluster"] = (["lightgrey"] + hi) if 0 in uniq_ag else hi
        color_options.insert(4, "agg_cluster")

    # distance to the organoid's aggregated-cluster centre in weighted-HKS space
    # (saved by dim_red as 'centre_dist'; 0 = at the centre / most representative).
    # Continuous; reversed viridis so near-centre reads bright, with a robust
    # upper clip so a few far outliers don't wash out the scale.
    if "centre_dist" in emb:
        cdist = emb["centre_dist"].astype(float)
        hi_clip = float(np.percentile(cdist, 98)) if np.isfinite(cdist).any() else None
        color_fields["centre_dist"] = (cdist, "viridis_r", (0.0, hi_clip) if hi_clip else None)
        color_options.insert(5, "centre_dist")

    # perturbation/genotype condition (categorical): point-field 'condition'
    # saved by dim_red ('WT' for the wild-type datasets, the drug token for pert2).
    if conditions is not None:
        uniq_cond = sorted(set(conditions))
        if len(uniq_cond) > 1:
            cond_code = np.array([uniq_cond.index(c) for c in conditions], dtype=float)
            color_fields["condition"] = (cond_code, "tab10", (0, len(uniq_cond) - 1))
            cat_levels["condition"] = list(enumerate(uniq_cond))
            color_options.append("condition")
            print(f"condition: {len(uniq_cond)} levels {uniq_cond}")

    # hand-picked shape groups (categorical), read from the groups json rather than the
    # embedding, so any embedding can be coloured by them: 0 = in no group, g + 1 = in
    # group g (numbered from 1 as in the json's order).
    groups_path = args.groups or os.path.join(args.data_root, "uid_groups.json")
    if os.path.isfile(groups_path):
        with open(groups_path) as fh:
            groups = json.load(fh)
        group_of = {}
        for g, members in enumerate(groups):
            for u in members:
                group_of.setdefault(u, g + 1)
        grp_code = np.array([group_of.get(u, 0) for u in ids], dtype=float)
        if (grp_code > 0).any():
            ng = len(groups)
            color_fields["uid_group"] = (grp_code, "tab10", (0, ng))
            cat_levels["uid_group"] = ([(0, "none")]
                                       + [(g + 1, f"group {g + 1}") for g in range(ng)])
            base = matplotlib.colormaps["tab10" if ng <= 10 else "tab20"]
            cat_cmaps["uid_group"] = ["lightgrey"] + [base(i % base.N) for i in range(ng)]
            color_options.append("uid_group")
            print(f"uid_group: {int((grp_code > 0).sum())} of {len(ids)} points in "
                  f"{ng} hand-picked groups ({groups_path})")
    elif args.groups:
        print(f"  [warn] groups file {args.groups} not found — no 'uid_group' colouring")

    # ====================================================================
    # mesh loaders: (vertices, faces, overlay categories, overlay values)
    # ====================================================================
    def _load_organoid_vtp(full_id):
        """Load mesh + fate fields straight from the .vtp (current default)."""
        vtp = resolve_vtp(full_id)
        if vtp is None:
            print(f"  [warn] no .vtp found for {full_id} under {args.data_root}")
            return None
        try:
            m = FateMarkers()
            m.load_mesh_from_file(vtp)
            cfg, field_to_friendly = find_cfg(vtp)
            if cfg is not None and cfg.get("annotation_names"):
                m._refine_markers(cfg["annotation_names"], cfg.get("exclusion_rules", {}))
            m.align_with_pca()
            v, f, fields, field_names = m.v, m.f, m.fields, m.field_names
        except Exception as e:
            print(f"  [warn] failed to load {full_id} from {vtp}: {e}")
            return None

        # overlays grouped into Fate vs HKS/shape; load_organoid shows the
        # persisted choice.
        cats = {"Fate": [], "HKS / shape": []}
        vals = {}
        if fields is not None and len(field_names):
            for i, fld in enumerate(field_names):
                friendly = field_to_friendly.get(fld, fld)
                if friendly in vals:             # skip duplicate field names
                    continue
                vals[friendly] = fields[:, i]
                cats["Fate"].append(friendly)
        return _fit_mesh(v), f, cats, vals

    def _load_organoid_pipeline(full_id):
        """Load the precomputed coeffs.npz + transformed_mesh.obj (--source pipeline).

        Geometry and eigenvalues/eigenvectors come from the npz/obj — no
        re-alignment or re-eigendecomposition. The saved eigendecomposition also
        unlocks HKS scalar overlays. Fate fields are the raw fields of a matching
        .vtp; if none is found the mesh still loads with HKS overlays only.
        """
        base = resolve_pipeline_base(full_id)
        if base is None:
            print(f"  [warn] no pipeline '*{COEFFS_SUFFIX}' found for {full_id} "
                  f"under {args.data_root}")
            return None
        try:
            m = FateMarkers()
            m.load_results(base)
        except Exception as e:
            print(f"  [warn] failed to load pipeline files for {full_id} from {base}: {e}")
            return None

        cfg, field_to_friendly = find_cfg(base)

        # fate fields: a matching .vtp's raw fields (the pipeline does not decompose
        # fate fields, see run_new_meshes).
        fields, field_names = None, None
        vtp = resolve_vtp(full_id)
        if vtp is not None and not vtp.endswith(".stl"):   # .stl has no fate fields
            try:
                mv = FateMarkers()
                mv.load_mesh_from_file(vtp)
                if cfg is not None and cfg.get("annotation_names"):
                    mv._refine_markers(cfg["annotation_names"], cfg.get("exclusion_rules", {}))
                if mv.fields.shape[0] == m.v.shape[0]:
                    fields, field_names = mv.fields, mv.field_names
                else:
                    print(f"  [warn] vertex-count mismatch between pipeline mesh "
                          f"and .vtp for {full_id} — skipping fate fields")
            except Exception as e:
                print(f"  [warn] failed to read fate fields from .vtp for {full_id}: {e}")

        # All overlays are grouped into Fate vs HKS/shape; load_organoid shows
        # the persisted choice (default: an HKS overlay).
        cats = {"Fate": [], "HKS / shape": []}
        vals = {}
        if fields is not None and len(field_names):
            for i, fld in enumerate(field_names):
                friendly = field_to_friendly.get(fld, fld)
                if friendly in vals:             # skip duplicate field names
                    continue
                vals[friendly] = fields[:, i]
                cats["Fate"].append(friendly)

        # per-vertex HKS at a few diffusion times, straight from the saved
        # eigendecomposition (no recomputation needed). hks_t25 is shown by
        # default; the others are listed alongside it.
        try:
            hks = m.compute_hks_for_new_times(HKS_TIMES)
            for k, t in enumerate(HKS_TIMES):
                nm = f"hks_t{t}"
                vals[nm] = hks[:, k]
                cats["HKS / shape"].append(nm)
        except Exception as e:
            print(f"  [warn] failed to compute HKS for {full_id}: {e}")

        # bag-of-features: soft-assign per-vertex HKS (sampled at each vocab's own
        # diffusion times) to that vocabulary's words. Same encoding as
        # compute_master_npz.compute_bof_power, but evaluated per-vertex instead
        # of expanded in spherical harmonics.
        for v in bof_vocabs:
            try:
                ts = find_times(m.area) if v["time"] == "variable" else np.asarray(v["ts"])
                hks_scaled = v["scaler"].transform(hks_unit_area(m, ts))
                encoding = encode_vocab(hks_scaled, v)        # (n_verts, n_words)
                for k in range(encoding.shape[1]):
                    nm = f"bof_{v['name']}_word{k}"
                    vals[nm] = encoding[:, k]
                    cats["HKS / shape"].append(nm)
            except Exception as e:
                print(f"  [warn] failed to compute '{v['name']}' bag-of-features "
                      f"for {full_id}: {e}")

        return _fit_mesh(m.v), m.f, cats, vals

    def _default_quantity(cats):
        """Fallback colouring when the remembered overlay is absent here."""
        if DEFAULT_MARKER in cats.get("Fate", []):
            return DEFAULT_MARKER
        hks_def = f"hks_t{HKS_DEFAULT_TIME}"
        if hks_def in cats.get("HKS / shape", []):
            return hks_def
        for names in cats.values():
            if names:
                return names[0]
        return SURFACE_ONLY

    # ====================================================================
    # one Qt window: embedding scatter (left) | organoid panel (right)
    # ====================================================================
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Embedding — {os.path.basename(args.embedding)} "
                       f"({method}, {len(ids)} organoids)")
    split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    win.setCentralWidget(split)
    status = win.statusBar()

    def _labelled_row(label, widget):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        row.addWidget(widget)
        row.addStretch(1)
        return row

    # ---- left: 2D scatter + colour selector -----------------------------
    left = QtWidgets.QWidget()
    lv = QtWidgets.QVBoxLayout(left)
    lv.setContentsMargins(6, 6, 6, 0)
    color_combo = QtWidgets.QComboBox()
    color_combo.addItems(color_options)
    color_combo.setMaxVisibleItems(30)
    fig = Figure(layout="constrained")
    canvas = FigureCanvasQTAgg(fig)
    toolbar = NavigationToolbar2QT(canvas, left)
    lv.addLayout(_labelled_row("Colour points by:", color_combo))
    lv.addWidget(canvas, 1)
    lv.addWidget(toolbar)
    split.addWidget(left)

    ax = fig.add_subplot()
    init_c, init_cmap, _ = color_fields["l_cross"]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=init_c, cmap=init_cmap, s=6, alpha=0.85,
                    linewidths=0)
    hl = ax.scatter([], [], s=160, facecolors="none", edgecolors="red", linewidths=2,
                    zorder=6)
    cbar = fig.colorbar(sc, ax=ax)
    ax.set_title(f"{method} embedding ({len(ids)} organoids) — click a point")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    def recolor(field):
        c, cmap, clim = color_fields[field]
        c = np.asarray(c, float)
        sc.set_array(c)
        if field in cat_levels:
            # discrete colormap: one solid colour band per class (codes 0..n-1)
            codes, labels = zip(*cat_levels[field])
            n = len(codes)
            if field in cat_cmaps:
                disc = ListedColormap(cat_cmaps[field])
            else:
                base = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
                disc = ListedColormap([base(i % base.N) for i in range(n)])
            sc.set_cmap(disc)
            sc.set_norm(BoundaryNorm(np.arange(n + 1) - 0.5, n))
            cbar.update_normal(sc)
            cbar.set_ticks(list(codes))
            cbar.set_ticklabels(list(labels))
        else:
            sc.set_cmap(cmap)
            if clim is not None:
                lo, hi = clim
            else:
                finite = np.isfinite(c)
                lo, hi = (np.min(c[finite]), np.max(c[finite])) if finite.any() else (0, 1)
            sc.set_norm(Normalize(lo, hi))
            cbar.update_normal(sc)
            cbar.locator = mticker.AutoLocator()
            cbar.update_ticks()
        cbar.set_label(f"log({field}+1e-3)" if field in fate_set else field)
        canvas.draw_idle()
    color_combo.currentTextChanged.connect(recolor)

    # ---- right: organoid 3D view over the fate-fraction bars ------------
    right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    org_panel = QtWidgets.QWidget()
    rv = QtWidgets.QVBoxLayout(org_panel)
    rv.setContentsMargins(6, 6, 6, 0)
    info = QtWidgets.QLabel("Click a point in the embedding to load an organoid.")
    info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)   # copyable id
    info.setWordWrap(True)
    quant_combo = QtWidgets.QComboBox()
    quant_combo.setMaxVisibleItems(30)
    quant_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
    reset_btn = QtWidgets.QPushButton("Reset view")
    quant_row = _labelled_row("Colour organoid by:", quant_combo)
    quant_row.addWidget(reset_btn)
    vtk_widget = QVTKRenderWindowInteractor(org_panel)
    rv.addWidget(info)
    rv.addLayout(quant_row)
    rv.addWidget(vtk_widget, 1)
    right.addWidget(org_panel)

    bar_fig = Figure(layout="constrained")
    bar_canvas = FigureCanvasQTAgg(bar_fig)
    bar_canvas.setMinimumHeight(120)
    right.addWidget(bar_canvas)
    right.setStretchFactor(0, 3)
    right.setStretchFactor(1, 1)
    split.addWidget(right)

    # fate % horizontal bar
    ax_bar = bar_fig.add_subplot()
    bars = ax_bar.barh(np.arange(len(fate_labels)), np.zeros(len(fate_labels)),
                       color="tab:blue")
    ax_bar.set_yticks(np.arange(len(fate_labels)))
    ax_bar.set_yticklabels(fate_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("cell fraction")
    if not fate_labels:
        bar_canvas.hide()

    # VTK scene: one mesh actor whose input is swapped per organoid, coloured
    # by a selectable point-data array through a shared viridis lookup table.
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.SetBackground2(0.82, 0.86, 0.92)
    ren.GradientBackgroundOn()
    render_window = vtk_widget.GetRenderWindow()
    render_window.AddRenderer(ren)
    iren = render_window.GetInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    lut = _vtk_lut("viridis")
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetLookupTable(lut)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.InterpolateScalarsBeforeMappingOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*MESH_COLOR)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(20)
    actor.VisibilityOff()
    ren.AddActor(actor)

    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetLabelFormat("%.3g")
    scalar_bar.SetPosition(0.88, 0.15)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.7)
    scalar_bar.SetMaximumWidthInPixels(60)
    scalar_bar.UnconstrainedFontSizeOn()
    labels_prop = scalar_bar.GetLabelTextProperty()
    labels_prop.SetFontSize(14)
    labels_prop.SetColor(0.15, 0.15, 0.15)
    labels_prop.ShadowOff()
    labels_prop.ItalicOff()
    labels_prop.BoldOff()
    scalar_bar.VisibilityOff()
    ren.AddActor2D(scalar_bar)

    # active_quantity persists the chosen colouring across organoid clicks;
    # quant_values holds the current organoid's per-vertex overlay arrays.
    state = {"active_quantity": DEFAULT_MARKER, "quant_values": {}, "camera_set": False}

    def reset_view():
        """Default front view: y up, looking down -z, mesh filling the panel."""
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetPosition(0.0, 0.0, 1.0)
        cam.SetViewUp(0.0, 1.0, 0.0)
        ren.ResetCamera()
        render_window.Render()
    reset_btn.clicked.connect(reset_view)

    def show_quantity(name):
        """Colour the mesh by overlay `name` (anything else -> plain surface)."""
        q = state["quant_values"].get(name)
        if q is None:
            mapper.ScalarVisibilityOff()
            scalar_bar.VisibilityOff()
        else:
            finite = np.isfinite(q)
            lo, hi = (float(q[finite].min()), float(q[finite].max())) if finite.any() else (0.0, 1.0)
            hi = hi if hi > lo else lo + 1.0   # constant field (e.g. absent marker)
            mapper.SelectColorArray(name)
            mapper.SetScalarRange(lo, hi)
            mapper.ScalarVisibilityOn()
            scalar_bar.VisibilityOn()
        render_window.Render()

    def on_quantity_chosen(_index):
        name = quant_combo.currentData()
        if name is None:          # section header / separator
            return
        state["active_quantity"] = name
        show_quantity(name)
    quant_combo.currentIndexChanged.connect(on_quantity_chosen)

    def fill_quantity_combo(cats, active):
        """Rebuild 'Colour organoid by' (section headers are unselectable)."""
        quant_combo.blockSignals(True)
        quant_combo.clear()
        quant_combo.addItem(SURFACE_ONLY, SURFACE_ONLY)
        model = quant_combo.model()
        for cat, names in cats.items():
            if not names:
                continue
            quant_combo.insertSeparator(quant_combo.count())
            quant_combo.addItem(cat)
            header = model.item(quant_combo.count() - 1)
            header.setEnabled(False)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            for nm in names:
                quant_combo.addItem(f"   {nm}", nm)
        quant_combo.setCurrentIndex(max(0, quant_combo.findData(active)))
        quant_combo.blockSignals(False)

    def load_organoid(idx):
        full_id = ids[idx]
        loader = _load_organoid_pipeline if args.source == "pipeline" else _load_organoid_vtp
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            result = loader(full_id)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if result is None:
            state["quant_values"] = {}
            actor.VisibilityOff()
            scalar_bar.VisibilityOff()
            quant_combo.clear()
            render_window.Render()
            status.showMessage(f"Could not load {full_id} — see the terminal for details.")
            return
        v, f, cats, vals = result

        mesh = _vtk_mesh(v, f)
        for nm, q in vals.items():
            arr = numpy_to_vtk(np.ascontiguousarray(q, dtype=np.float64), deep=True)
            arr.SetName(nm)
            mesh.GetPointData().AddArray(arr)
        normals = vtk.vtkPolyDataNormals()   # smooth shading
        normals.SetInputData(mesh)
        normals.SplittingOff()
        normals.ComputePointNormalsOn()
        normals.Update()
        mapper.SetInputData(normals.GetOutput())
        actor.VisibilityOn()
        state["quant_values"] = vals

        # re-apply the remembered colouring so it persists across organoids;
        # fall back to a sensible default if this organoid lacks that overlay.
        remembered = state["active_quantity"]
        active = remembered if remembered in vals or remembered == SURFACE_ONLY \
            else _default_quantity(cats)
        fill_quantity_combo(cats, active)
        if not state["camera_set"]:
            reset_view()
            state["camera_set"] = True
        else:
            ren.ResetCameraClippingRange()
        show_quantity(active)

    def select(idx):
        uid = ids[idx]
        hl.set_offsets([xy[idx]])
        canvas.draw_idle()
        details = [f"time {times[idx]}", f"l_cross {l_cross[idx]:.1f}", f"area {areas[idx]:.0f}"]
        if hill_div is not None:
            details.append(f"diversity {hill_div[idx]:.2f}")
        if agg_clusters is not None:
            details.append(f"agg_cluster {int(agg_clusters[idx])}")
        if conditions is not None:
            details.append(f"condition {conditions[idx]}")
        info.setText(f"<b>{uid}</b><br>{' · '.join(details)}")
        if fate_labels:
            vals = perc_matrix[idx]
            for b, v in zip(bars, vals):
                b.set_width(v)
            ax_bar.set_xlim(0, max(0.05, 1.05 * float(np.max(vals))))
            bar_canvas.draw_idle()
        print(f"selected {uid}")
        load_organoid(idx)

    def nearest_point(event):
        """Index of the scatter point nearest a mouse event (None if too far)."""
        if event.inaxes is not ax or event.x is None:
            return None
        pix = ax.transData.transform(xy)
        d = np.hypot(pix[:, 0] - event.x, pix[:, 1] - event.y)
        i = int(np.nanargmin(d))
        return i if d[i] <= PICK_RADIUS_PT * fig.dpi / 72 else None

    def on_click(event):
        if event.button != 1 or toolbar.mode:   # ignore clicks while zooming/panning
            return
        i = nearest_point(event)
        if i is not None:
            select(i)
    canvas.mpl_connect("button_press_event", on_click)

    def on_hover(event):
        i = nearest_point(event)
        if i is None:
            status.clearMessage()
        else:
            status.showMessage(f"{ids[i]}   (time {times[i]}, l_cross {l_cross[i]:.1f})")
    canvas.mpl_connect("motion_notify_event", on_hover)

    # ---- run -------------------------------------------------------------
    app.aboutToQuit.connect(vtk_widget.Finalize)
    win.resize(1500, 850)
    split.setSizes([780, 720])
    win.show()
    iren.Initialize()
    recolor("l_cross")   # apply the notebook clim/label to the initial view
    select(0)            # show something on startup
    app.exec_()
    print("closed.")


if __name__ == "__main__":
    main()
