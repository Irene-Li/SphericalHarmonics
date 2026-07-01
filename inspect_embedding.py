#!/usr/bin/env python3
"""
Interactive viewer for a saved 2D embedding of organoids.

Two linked windows:
  * a matplotlib window  - the embedding as a true 2D scatter (click a point)
                           plus a bar chart of the selected organoid's fate %.
  * a polyscope window   - ONLY the selected organoid's 3D mesh, which you can
                           rotate/zoom freely and independently of the scatter.
                           Switch which fate marker colors the mesh with
                           polyscope's mesh UI.

The embedding is written by `src.utils.save_embedding` (e.g. from
double_clustering.ipynb).

The scatter can be colored by l_cross / area / time / condition or by any fate
marker's percentage (the l=0 harmonic coefficient saved on the embedding as 'perc_<name>'
point-fields by double_clustering.ipynb; log scale, shared range), or by
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
                'hks_bof_coeffs__*' columns in master.npz), loaded from sim/.
            Fate fields come from the saved fate coefficients when the npz has
            them (fractal_output runs with compute_fate=True); otherwise
            they're read from a matching .vtp if one is found
            (main_dataset/sup_dataset/pert, which only save shape coefficients).

In both modes, --data_root is walked once and meshes/pipeline files are found
by filename regardless of nesting, so any layout works.

Organoid ids are dataset-prefixed ('{dataset}_{timepoint}_{well}_{label}'); the
trailing part is the filename stem used to find the mesh / pipeline files. If a
dataset's config.json is present next to the meshes it is used (and
auto-discovered) to give the fate fields friendly names; otherwise the raw vtp
field names are shown.

Opens two separate windows (scatter + organoid); drag them apart so they
don't overlap.

Run from the repository root, in the `scmpx` conda env:

    python inspect_embedding.py \
        --embedding Data/embeddings/hks_emb.npz --data_root Data

    python inspect_embedding.py \
        --embedding Data/embeddings/hks_emb.npz --data_root Data --source pipeline

Click a point in the scatter to load that organoid. Close either window to quit.
"""

import argparse
import json
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.widgets import RadioButtons

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import utils
from src.fatemarkers import FateMarkers
from compute_master_npz import find_times, hks_unit_area, encode_vocab, load_vocabs

ORG_NAME = "organoid"
DEFAULT_MARKER = "lgr"
COMBO_MARKERS = ("STEM", "EC", "PANETH")   # must all be present in perc_* fields
HKS_TIMES = [1, 4, 25, 100]   # matches compute_master_npz.SPARSE_TS / sim/encodings_*.vtp
HKS_DEFAULT_TIME = 25         # shown by default in --source pipeline (mid diffusion scale)
COEFFS_SUFFIX = "_coeffs.npz"


def _fit_mesh(v, radius=1.0):
    """Center a mesh at the origin and scale it into a sphere of `radius`."""
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean(axis=0)
    v = v / (np.linalg.norm(v, axis=1).max() or 1.0) * radius
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True,
                    help="path to an embedding .npz (from utils.save_embedding)")
    ap.add_argument("--data_root", default="Data",
                    help="folder to search (recursively) for the original .vtp "
                         "meshes. Meshes are found by filename, so any layout works.")
    ap.add_argument("--source", choices=["vtp", "pipeline"], default="vtp",
                    help="'vtp' (default) loads straight from the .vtp; 'pipeline' "
                         "loads the precomputed '{stem}_coeffs.npz' + "
                         "'{stem}_transformed_mesh.obj' instead, which also enables "
                         "HKS scalar overlays (see module docstring).")
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
    # optional perturbation/genotype condition ('WT' or a pert drug token)
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
    for root, _dirs, files in os.walk(args.data_root):
        for fn in files:
            if fn.endswith(".vtp"):
                vtp_index.setdefault(fn[:-4], []).append(os.path.join(root, fn))
            elif fn.endswith(COEFFS_SUFFIX):
                stem = fn[:-len(COEFFS_SUFFIX)]
                pipeline_index.setdefault(stem, []).append(os.path.join(root, stem))
    n_vtp = sum(len(v) for v in vtp_index.values())
    n_pipeline = sum(len(v) for v in pipeline_index.values())
    print(f"Indexed {n_vtp} .vtp files and {n_pipeline} pipeline coeffs under {args.data_root}")
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
    # Same vocabs/encoding compute_master_npz uses for the 'hks_bof_coeffs__*'
    # columns in master.npz (VOCABS: kmeans_variable, pca_variable), loaded once
    # here and reused for every organoid as per-vertex 'bof_<vocab>_word<k>'
    # overlays (soft-assignment encoding, not the spherical-harmonic coeffs).
    bof_vocabs = load_vocabs() if args.source == "pipeline" else []

    def shorten_id(full_id):
        """'main_dataset_day1p5_A01_42' -> 'md_1p5_a01_42' (cosmetic title)."""
        toks = full_id.split("_")
        for i, t in enumerate(toks):
            if t.startswith("day"):
                ds = "".join(w[0] for w in toks[:i]) or "?"
                rest = "_".join(toks[i + 1:]).lower()
                return f"{ds}_{t[len('day'):]}_{rest}"
        return full_id

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
    # double_clustering saves each fate as a 'perc_<name>' point-field: the
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
        base = plt.get_cmap("tab10" if n_high <= 10 else "tab20")
        hi = [base(i % base.N) for i in range(n_high)]
        cat_cmaps["agg_cluster"] = (["lightgrey"] + hi) if 0 in uniq_ag else hi
        color_options.insert(4, "agg_cluster")

    # uid_groups membership (categorical): point-field saved by double_clustering
    # as 'uid_group' (0 = none, g+1 = membership in hand-picked group g).
    if "uid_group" in emb:
        grp_code = emb["uid_group"].astype(float)
        if (grp_code > 0).any():
            ng = int(grp_code.max())
            color_fields["uid_group"] = (grp_code, "tab10", (0, ng))
            cat_levels["uid_group"] = ([(0, "none")]
                                       + [(g + 1, f"group {g + 1}") for g in range(ng)])
            base = plt.get_cmap("tab10" if ng <= 9 else "tab20")
            cat_cmaps["uid_group"] = ["lightgrey"] + [base(i % base.N) for i in range(ng)]
            color_options.append("uid_group")
            print(f"uid_group: {int((grp_code > 0).sum())} of {len(ids)} points labelled "
                  f"({ng} groups)")

    # trusted organoids (categorical): point-field 'trusted' (1 if the organoid
    # is in any confident small-chamfer pair, i.e. it supervises the chamfer term).
    if "trusted" in emb:
        is_trusted = emb["trusted"].astype(float)
        if is_trusted.any():
            color_fields["trusted"] = (is_trusted, "tab10", (0, 1))
            cat_levels["trusted"] = [(0, "untrusted"), (1, "trusted")]
            cat_cmaps["trusted"] = ["lightgrey", "crimson"]
            color_options.append("trusted")
            print(f"trusted: {int(is_trusted.sum())} of {len(ids)} points")

    # perturbation/genotype condition (categorical): point-field 'condition'
    # saved by dim_red ('WT' for the wild-type datasets, the drug token for pert).
    if conditions is not None:
        uniq_cond = sorted(set(conditions))
        if len(uniq_cond) > 1:
            cond_code = np.array([uniq_cond.index(c) for c in conditions], dtype=float)
            color_fields["condition"] = (cond_code, "tab10", (0, len(uniq_cond) - 1))
            cat_levels["condition"] = list(enumerate(uniq_cond))
            color_options.append("condition")
            print(f"condition: {len(uniq_cond)} levels {uniq_cond}")

    # active_quantity persists the chosen colouring across organoid clicks;
    # categories groups overlay names into Fate / HKS-shape; quant_values holds
    # each overlay's per-vertex array so we can re-enable it by re-adding.
    state = {"org": None, "active_quantity": DEFAULT_MARKER,
             "categories": {}, "quant_values": {}}

    def _set_active(name):
        """Show the overlay `name` (and hide the previously shown one), then
        remember it so the choice persists across organoids.

        polyscope's add_scalar_quantity returns None (no handle to toggle), but
        re-adding a quantity with an existing name replaces it in place — so we
        re-add the target enabled and the previous one disabled. Only these two
        are touched, regardless of how many overlays the organoid has."""
        org, vals = state["org"], state["quant_values"]
        if org is None or name not in vals:
            return
        prev = state["active_quantity"]
        if prev and prev != name and prev in vals:
            org.add_scalar_quantity(prev, vals[prev], cmap="viridis", enabled=False)
        org.add_scalar_quantity(name, vals[name], cmap="viridis", enabled=True)
        state["active_quantity"] = name

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
        return None

    ACCENT = (0.35, 0.85, 1.0, 1.0)   # cyan section headers, to stand out

    def quantity_ui_callback():
        """Standalone floating panel (separate from polyscope's structure UI):
        overlays split into Fate vs HKS/shape categories, radio-selected (one
        active at a time), persisted across organoids via _set_active."""
        if not state["quant_values"]:
            return
        # own window, offset to the right of polyscope's top-left panel so it
        # reads as a distinct control; draggable/resizable after first show.
        psim.SetNextWindowPos((360.0, 20.0), psim.ImGuiCond_FirstUseEver)
        psim.SetNextWindowSize((250.0, 430.0), psim.ImGuiCond_FirstUseEver)
        psim.Begin("Colour organoid by")
        chosen = None
        for cat, names in state["categories"].items():
            if not names:
                continue
            psim.PushStyleColor(psim.ImGuiCol_Text, ACCENT)
            psim.SeparatorText(cat)
            psim.PopStyleColor()
            for nm in names:
                # '##<cat>' keeps the visible label but gives each radio a unique
                # ImGui ID (avoids conflicting-ID errors on duplicate names).
                if psim.RadioButton(f"{nm}##{cat}", state["active_quantity"] == nm):
                    chosen = nm
        psim.End()
        if chosen is not None:
            _set_active(chosen)

    # ====================================================================
    # polyscope window: only the organoid mesh (independent rotation)
    # ====================================================================
    ps.init()
    ps.set_program_name("Organoid")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("turntable")
    # show only our custom 'Colour organoid by' window: drop polyscope's default
    # panels (its Structures / Quantities selectors) and the auto wrapper window
    # it would otherwise open around the user callback (we open our own Begin).
    ps.set_build_default_gui_panels(False)
    ps.set_open_imgui_window_for_user_callback(False)

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

        org = ps.register_surface_mesh(ORG_NAME, _fit_mesh(v), f, smooth_shade=True,
                                       color=(0.55, 0.65, 0.85))
        # overlays added disabled; load_organoid enables the persisted choice.
        cats = {"Fate": [], "HKS / shape": []}
        vals = {}
        if fields is not None and len(field_names):
            for i, fld in enumerate(field_names):
                friendly = field_to_friendly.get(fld, fld)
                if friendly in vals:             # skip duplicate field names
                    continue
                org.add_scalar_quantity(friendly, fields[:, i], cmap="viridis", enabled=False)
                vals[friendly] = fields[:, i]
                cats["Fate"].append(friendly)
        return org, cats, vals

    def _load_organoid_pipeline(full_id):
        """Load the precomputed coeffs.npz + transformed_mesh.obj (--source pipeline).

        Geometry, eigenvalues/eigenvectors, and (when saved) the fate-field
        coefficients all come from the npz/obj — no re-alignment or
        re-eigendecomposition. The saved eigendecomposition also unlocks HKS
        scalar overlays. Datasets whose pipeline run only computed shape
        coefficients (no coeffs_fm) fall back to a matching .vtp for raw fate
        fields; if none is found the mesh still loads with HKS overlays only.
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

        # fate fields: prefer the saved fate coefficients (inverse spherical-harmonic
        # reconstruction back to per-vertex values); fall back to a matching .vtp's
        # raw fields when the npz only has shape coefficients (vtp_flat runs never
        # compute fate, see run_new_meshes.run_vtp_flat).
        fields, field_names = None, None
        if getattr(m, "coeffs_fm", None) is not None:
            try:
                fields = m.reconstruct_from_coeffs(m.coeffs_fm, lmax=m.lmax)
                field_names = m.field_names
            except Exception as e:
                print(f"  [warn] failed to reconstruct fate fields for {full_id}: {e}")
        if fields is None:
            vtp = resolve_vtp(full_id)
            if vtp is not None:
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

        org = ps.register_surface_mesh(ORG_NAME, _fit_mesh(m.v), m.f, smooth_shade=True,
                                       color=(0.55, 0.65, 0.85))
        # All overlays are added disabled and grouped into Fate vs HKS/shape;
        # load_organoid enables the persisted choice (default: an HKS overlay).
        cats = {"Fate": [], "HKS / shape": []}
        vals = {}
        if fields is not None and len(field_names):
            for i, fld in enumerate(field_names):
                friendly = field_to_friendly.get(fld, fld)
                if friendly in vals:             # skip duplicate field names
                    continue
                org.add_scalar_quantity(friendly, fields[:, i], cmap="viridis", enabled=False)
                vals[friendly] = fields[:, i]
                cats["Fate"].append(friendly)

        # per-vertex HKS at a few diffusion times, straight from the saved
        # eigendecomposition (no recomputation needed). hks_t25 is shown by
        # default; the others are added alongside it for the quantities list.
        try:
            hks = m.compute_hks_for_new_times(HKS_TIMES, coeffs=False)
            for k, t in enumerate(HKS_TIMES):
                nm = f"hks_t{t}"
                org.add_scalar_quantity(nm, hks[:, k], cmap="viridis", enabled=False)
                vals[nm] = hks[:, k]
                cats["HKS / shape"].append(nm)
            print(f"  HKS overlays added: {[f'hks_t{t}' for t in HKS_TIMES]} "
                  f"(pick one in the 'Colour organoid by' panel — HKS / shape section)")
        except Exception as e:
            print(f"  [warn] failed to compute HKS for {full_id}: {e}")

        # bag-of-features: soft-assign per-vertex HKS (sampled at each vocab's own
        # diffusion times) to that vocabulary's words. Same encoding as
        # compute_master_npz.compute_bof_coeffs, but evaluated per-vertex instead
        # of projected to spherical-harmonic coefficients.
        for v in bof_vocabs:
            try:
                ts = find_times(m.area) if v["time"] == "variable" else np.asarray(v["ts"])
                hks_scaled = v["scaler"].transform(hks_unit_area(m, ts))
                encoding = encode_vocab(hks_scaled, v)        # (n_verts, n_words)
                for k in range(encoding.shape[1]):
                    nm = f"bof_{v['name']}_word{k}"
                    org.add_scalar_quantity(nm, encoding[:, k], cmap="viridis", enabled=False)
                    vals[nm] = encoding[:, k]
                    cats["HKS / shape"].append(nm)
                print(f"  bag-of-features added: bof_{v['name']}_word0..{encoding.shape[1] - 1}")
            except Exception as e:
                print(f"  [warn] failed to compute '{v['name']}' bag-of-features "
                      f"for {full_id}: {e}")

        return org, cats, vals

    def load_organoid(idx):
        full_id = ids[idx]
        if state["org"] is not None:
            state["org"].remove()
            state["org"] = None
        state["categories"], state["quant_values"] = {}, {}

        loader = _load_organoid_pipeline if args.source == "pipeline" else _load_organoid_vtp
        result = loader(full_id)
        if result is None:
            return
        org, cats, vals = result

        org.reset_transform()
        state["org"] = org
        state["categories"] = cats
        state["quant_values"] = vals
        # re-apply the remembered colouring so it persists across organoids;
        # fall back to a sensible default if this organoid lacks that overlay.
        allnames = [n for names in cats.values() for n in names]
        active = state["active_quantity"] if state["active_quantity"] in allnames \
            else _default_quantity(cats)
        if active is not None:
            _set_active(active)
        ps.reset_camera_to_home_view()

    # ====================================================================
    # matplotlib window: 2D scatter + fate% bar + color selector
    # ====================================================================
    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title("Embedding")
    ax_radio = fig.add_axes([0.005, 0.08, 0.085, 0.86], frameon=True)  # color selector
    ax = fig.add_axes([0.16, 0.10, 0.46, 0.82])      # scatter
    ax_bar = fig.add_axes([0.82, 0.10, 0.16, 0.82])  # fate % bar

    markersize = 5

    init_c, init_cmap, _ = color_fields["l_cross"]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=init_c, cmap=init_cmap, s=markersize, alpha=0.85)
    hl = ax.scatter([], [], s=160, facecolors="none", edgecolors="red", linewidths=2,
                    zorder=6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("l_cross")
    ax.set_title(f"{method} embedding  ({len(ids)} organoids) — click a point")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    ax_radio.set_title("color by", fontsize=9)
    radio = RadioButtons(ax_radio, color_options)
    for t in radio.labels:
        t.set_fontsize(8)

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
                base = plt.get_cmap("tab10" if n <= 10 else "tab20")
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
        fig.canvas.draw_idle()
    radio.on_clicked(recolor)
    recolor("l_cross")   # apply the notebook clim/label to the initial view

    # fate % horizontal bar
    bars = ax_bar.barh(np.arange(len(fate_labels)), np.zeros(len(fate_labels)),
                       color="tab:blue")
    ax_bar.set_yticks(np.arange(len(fate_labels)))
    ax_bar.set_yticklabels(fate_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("cell fraction")
    ax_bar.set_title("fate")

    def select(idx):
        uid = ids[idx]
        hl.set_offsets([xy[idx]])
        vals = perc_matrix[idx] if len(fate_labels) else np.zeros(0)
        for b, v in zip(bars, vals):
            b.set_width(v)
        ax_bar.set_xlim(0, max(0.05, float(np.max(vals)) if len(vals) else 0.05))
        div_str = f" D={hill_div[idx]:.2f}" if hill_div is not None else ""
        cond_str = f" c={conditions[idx]}" if conditions is not None else ""
        ax_bar.set_title(f"{shorten_id(uid)}\n"
                         f"t={times[idx]} L={l_cross[idx]:.1f} a={areas[idx]:.0f}{div_str}{cond_str}")
        fig.canvas.draw_idle()
        load_organoid(idx)

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        d = (xy[:, 0] - event.xdata) ** 2 + (xy[:, 1] - event.ydata) ** 2
        select(int(np.argmin(d)))
    fig.canvas.mpl_connect("button_press_event", on_click)

    # ---- run both event loops together ---------------------------------
    ps.set_user_callback(quantity_ui_callback)
    plt.ion()
    plt.show(block=False)
    select(0)  # show something on startup

    while plt.fignum_exists(fig.number) and not ps.window_requests_close():
        ps.frame_tick()
        plt.pause(0.02)

    print("closed.")


if __name__ == "__main__":
    main()
