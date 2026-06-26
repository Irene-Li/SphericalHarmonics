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

The scatter can be colored by l_cross / area / time or by any fate marker's
percentage (the l=0 harmonic coefficient saved on the embedding as 'perc_<name>'
point-fields by double_clustering.ipynb; log scale, shared range).

Self-contained: this viewer needs only the embedding .npz and the original .vtp
meshes — no pipeline outputs (no per-organoid coeffs/eigendecomposition, no
transformed .obj). Point --data_root at any folder containing the .vtp files;
they are found by filename regardless of how they're nested, so you can just
drop the original meshes there. Each .vtp already carries the per-vertex fate
fields, which are shown on the mesh.

Organoid ids are dataset-prefixed ('{dataset}_{timepoint}_{well}_{label}'); the
trailing '{timepoint}_{well}_{label}' is the .vtp filename stem used to find the
mesh. If a dataset's config.json is present next to the meshes it is used (and
auto-discovered) to give the fate fields friendly names; otherwise the raw vtp
field names are shown.

Opens two separate windows (scatter + organoid); drag them apart so they
don't overlap.

Run from the repository root, in the `scmpx` conda env:

    python inspect_embedding.py \
        --embedding Data/embeddings/hks_emb.npz --data_root Data

Click a point in the scatter to load that organoid. Close either window to quit.
"""

import argparse
import json
import os
import sys

import numpy as np
import polyscope as ps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.widgets import RadioButtons

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import utils
from src.fatemarkers import FateMarkers

ORG_NAME = "organoid"
DEFAULT_MARKER = "lgr"
COMBO_MARKERS = ("STEM", "EC", "PANETH")   # must all be present in perc_* fields


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
    args = ap.parse_args()

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
    print(f"Loaded {len(ids)} points from {args.embedding} (method={method})")

    # ---- locate the original .vtp meshes -------------------------------
    # Build a {filename-stem: [paths]} index by walking data_root once.  The
    # collaborator only needs to drop the original .vtp files somewhere under
    # data_root; the exact nesting doesn't matter.  An embedding id is the
    # dataset name followed by the mesh's filename stem
    # ('main_dataset' + '_' + 'day2p5_A04_67'), so a mesh is found by matching
    # the longest trailing part of the id against an indexed stem.
    vtp_index = {}
    for root, _dirs, files in os.walk(args.data_root):
        for fn in files:
            if fn.endswith(".vtp"):
                vtp_index.setdefault(fn[:-4], []).append(os.path.join(root, fn))
    n_vtp = sum(len(v) for v in vtp_index.values())
    print(f"Indexed {n_vtp} .vtp files under {args.data_root}")
    if n_vtp == 0:
        print(f"  [warn] no .vtp meshes found under {args.data_root!r} — "
              f"meshes won't load. Pass --data_root <folder with the .vtp files>.")

    def resolve_vtp(full_id):
        """Map an embedding id to a .vtp path (or None if not found).

        Tries progressively shorter trailing parts of the id ('a_b_c_d' ->
        'b_c_d' -> 'c_d' ...) and returns the first that is an indexed stem;
        if several files share that stem, prefers one whose path contains the
        leading dataset prefix.
        """
        toks = full_id.split("_")
        for i in range(len(toks)):
            stem = "_".join(toks[i:])
            cands = vtp_index.get(stem)
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

    def find_cfg(vtp_path):
        """(cfg_or_None, field_to_friendly) by searching up from a mesh path."""
        d = os.path.dirname(os.path.abspath(vtp_path))
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
    color_fields = {"l_cross": (l_cross, "Greens", (1, 8)),
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

    # shape-cluster colouring (only present on the shape embedding)
    if shape_clusters is not None:
        color_fields["shape_cluster"] = (shape_clusters, "tab10",
                                         (shape_clusters.min(), shape_clusters.max()))
        color_options.insert(3, "shape_cluster")
        uniq_cl = sorted(set(shape_clusters.astype(int)))
        cat_levels["shape_cluster"] = [(c, f"cluster {c}") for c in uniq_cl]

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

    state = {"org": None}

    # ====================================================================
    # polyscope window: only the organoid mesh (independent rotation)
    # ====================================================================
    ps.init()
    ps.set_program_name("Organoid")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("turntable")

    def load_organoid(idx):
        full_id = ids[idx]
        if state["org"] is not None:
            state["org"].remove()
            state["org"] = None

        vtp = resolve_vtp(full_id)
        if vtp is None:
            print(f"  [warn] no .vtp found for {full_id} under {args.data_root}")
            return
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
            return

        org = ps.register_surface_mesh(ORG_NAME, _fit_mesh(v), f, smooth_shade=True,
                                       color=(0.55, 0.65, 0.85))
        if fields is not None and len(field_names):
            for i, fld in enumerate(field_names):
                friendly = field_to_friendly.get(fld, fld)
                org.add_scalar_quantity(friendly, fields[:, i], cmap="viridis",
                                        enabled=(friendly == DEFAULT_MARKER))

        org.reset_transform()
        state["org"] = org
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
        ax_bar.set_title(f"{shorten_id(uid)}\n"
                         f"t={times[idx]} L={l_cross[idx]:.1f} a={areas[idx]:.0f}")
        fig.canvas.draw_idle()
        load_organoid(idx)

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        d = (xy[:, 0] - event.xdata) ** 2 + (xy[:, 1] - event.ydata) ** 2
        select(int(np.argmin(d)))
    fig.canvas.mpl_connect("button_press_event", on_click)

    # ---- run both event loops together ---------------------------------
    plt.ion()
    plt.show(block=False)
    select(0)  # show something on startup

    while plt.fignum_exists(fig.number) and not ps.window_requests_close():
        ps.frame_tick()
        plt.pause(0.02)

    print("closed.")


if __name__ == "__main__":
    main()
