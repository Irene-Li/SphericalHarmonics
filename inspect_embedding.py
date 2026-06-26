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

Organoid ids are dataset-prefixed ('{dataset}_{timepoint}_{well}_{label}'); each
mesh is loaded from '{data_root}/{dataset}' using that dataset's own config.json.

Opens two separate windows (scatter + organoid); drag them apart so they
don't overlap.

Run from the repository root, in the `scmpx` conda env:

    python inspect_embedding.py \
        --embedding Data/embeddings/hks_full_emb.npz --data_root Data

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
                    help="parent dir of the dataset folders. Organoid ids of the form "
                         "'{dataset}_{timepoint}_{well}_{label}' are resolved to "
                         "'{data_root}/{dataset}', each with its own config.json.")
    ap.add_argument("--vocab", default="sim/vocab_new.npz",
                    help="HKS bag-of-features vocab; adds per-vertex 'vocab-k' "
                         "overlays on the mesh (set to '' to disable)")
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

    # ---- per-dataset config resolution ---------------------------------
    # Ids are dataset-prefixed ('{dataset}_{bare_uid}').  Dataset names may
    # contain underscores (e.g. 'main_dataset'), so we prefix-match against
    # the actual folders present under data_root rather than splitting blindly.
    known_datasets = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isfile(os.path.join(args.data_root, d, "config.json")))
    print(f"Known datasets: {known_datasets}")

    # short prefix per dataset: first letter of each '_'-separated word, e.g.
    # 'main_dataset' -> 'md', 'sup_dataset' -> 'sd'
    _ds_short = {"_".join(w): "".join(w[0] for w in ds.split("_"))
                 for ds in known_datasets
                 for w in [ds.split("_")]}

    def shorten_id(full_id):
        """'main_dataset_day1p5_A01_42' -> 'md_1p5_a01_42'"""
        for ds in known_datasets:
            if full_id.startswith(ds + "_"):
                bare = full_id[len(ds) + 1:]          # 'day1p5_A01_42'
                parts = bare.split("_")
                tp = parts[0].lstrip("day")            # '1p5'
                rest = "_".join(parts[1:]).lower()     # 'a01_42'
                return f"{_ds_short[ds]}_{tp}_{rest}"
        return full_id

    def split_uid(full):
        """(data_path, bare_uid) for a dataset-prefixed id."""
        for ds in known_datasets:
            if full.startswith(ds + "_"):
                return os.path.join(args.data_root, ds), full[len(ds) + 1:]
        raise ValueError(f"Cannot determine dataset for id: {full!r}")

    _cfg_cache = {}

    def load_cfg(data_path):
        """(cfg, field_to_friendly) for a dataset, cached."""
        if data_path not in _cfg_cache:
            with open(f"{data_path}/config.json") as fh:
                cfg = json.load(fh)
            f2f = {}
            for name, flds in cfg["annotation_names"].items():
                for fl in ([flds] if isinstance(flds, str) else flds):
                    f2f[fl] = name
            _cfg_cache[data_path] = (cfg, f2f)
        return _cfg_cache[data_path]

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

    # ---- optional HKS bag-of-features vocabulary -----------------------
    # Adds per-vertex 'vocab-k' overlays on the mesh, computed exactly like
    # double_clustering's load_bof (HKS at the vocab time-scales -> soft-assign
    # to each vocab word).
    vocab = None
    if args.vocab and os.path.exists(args.vocab):
        vr = np.load(args.vocab, allow_pickle=True)
        vocab = {"words": vr["vocab"],
                 "sigma": float(np.ravel(vr["sigma"])[0]),
                 "scaler": vr["scaler"].item(),
                 "ts": np.ravel(vr["ts"])}
        print(f"Loaded HKS vocab from {args.vocab} ({vocab['words'].shape[0]} words)")
    elif args.vocab:
        print(f"No vocab at {args.vocab} -- skipping vocab overlays")

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
        data_path, uid = split_uid(ids[idx])
        cfg, field_to_friendly = load_cfg(data_path)
        if state["org"] is not None:
            state["org"].remove()
            state["org"] = None

        flat = cfg.get("layout") == "vtp_flat"
        if flat:
            vtp    = utils.vtp_flat_path(data_path, cfg, uid)
            obj    = utils.vtp_flat_obj_path(data_path, cfg, uid)
            c_path = utils.vtp_flat_coeffs_path(data_path, cfg, uid)
        else:
            vtp    = utils.organoid_vtp_path(data_path, cfg, uid)
            obj    = utils.organoid_obj_path(data_path, cfg, uid)
            c_path = utils.organoid_coeffs_path(data_path, cfg, uid)

        fields, field_names = None, None
        try:
            m = FateMarkers()
            if os.path.exists(vtp):
                m.load_mesh_from_file(vtp)
                m._refine_markers(cfg["annotation_names"], cfg["exclusion_rules"])
                m.align_with_pca()
                v, f, fields, field_names = m.v, m.f, m.fields, m.field_names
            elif os.path.exists(obj):
                import igl
                v, f = igl.read_triangle_mesh(obj)
            else:
                print(f"  [warn] no mesh found for {uid}")
                return
        except Exception as e:
            print(f"  [warn] failed to load {uid}: {e}")
            return

        org = ps.register_surface_mesh(ORG_NAME, _fit_mesh(v), f, smooth_shade=True,
                                       color=(0.55, 0.65, 0.85))
        if fields is not None:
            for i, fld in enumerate(field_names):
                friendly = field_to_friendly.get(fld, fld)
                org.add_scalar_quantity(friendly, fields[:, i], cmap="viridis",
                                        enabled=(friendly == DEFAULT_MARKER))

        # per-vertex HKS vocab encoding (uses the saved eigendecomposition)
        if vocab is not None:
            try:
                cf = np.load(c_path)
                eigvecs = cf["eigvecs"]
                if eigvecs.shape[0] != v.shape[0]:
                    raise ValueError(f"eigvec/vertex mismatch "
                                     f"({eigvecs.shape[0]} vs {v.shape[0]})")
                m.eigvals, m.eigvecs = cf["eigvals"], eigvecs
                hks = m.compute_hks_for_new_times(vocab["ts"], coeffs=False)
                hks_s = vocab["scaler"].transform(hks / np.mean(hks, axis=0) - 1)
                dist = np.linalg.norm(hks_s[:, None, :] - vocab["words"][None, :, :], axis=2)
                enc = np.exp(-dist ** 2 / (2 * vocab["sigma"] ** 2))   # (nverts, n_words)
                for k in range(enc.shape[1]):
                    org.add_scalar_quantity(f"vocab-{k}", enc[:, k], cmap="viridis")
            except Exception as e:
                print(f"  [warn] vocab overlay failed for {uid}: {e}")

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
