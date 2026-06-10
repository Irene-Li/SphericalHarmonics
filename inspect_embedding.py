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
percentage (log scale, shared range, matching double_clustering.ipynb).

Opens two separate windows (scatter + organoid); drag them apart so they
don't overlap.

Run from the repository root, in the `scmpx` conda env:

    python inspect_embedding.py \
        --embedding Data/20260224/embeddings/hks_emb.npz \
        --data_path Data/20260224 --csv Data/cell_types_norm_neg.csv

Click a point in the scatter to load that organoid. Close either window to quit.
"""

import argparse
import json
import os
import sys

import numpy as np
import polyscope as ps
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import utils
from src.fatemarkers import FateMarkers

ORG_NAME = "organoid"
DEFAULT_MARKER = "lgr"


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
    ap.add_argument("--data_path", default="Data/20260224",
                    help="dataset root containing config.json + fractal_output/")
    ap.add_argument("--csv", default="Data/cell_types_norm_neg.csv",
                    help="cell-fate percentages CSV (indexed by label_uid)")
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
    print(f"Loaded {len(ids)} points from {args.embedding} (method={method})")

    # ---- config + fate-name map ----------------------------------------
    with open(f"{args.data_path}/config.json") as f:
        cfg = json.load(f)
    field_to_friendly = {fld: name for name, fld in cfg["annotation_names"].items()}

    # ---- fate percentages CSV ------------------------------------------
    import pandas as pd
    df = pd.read_csv(args.csv).set_index("label_uid")
    fate_cols = [c for c in df.columns if c.endswith(".cnt_exclusive")]
    fate_labels = [c.split(".")[0] for c in fate_cols]

    def percentages_for(uid):
        if uid in df.index:
            return np.nan_to_num(df.loc[uid, fate_cols].to_numpy(dtype=float))
        return np.zeros(len(fate_cols))

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

    # per-organoid fate percentages aligned to the embedding ids.
    # Colour on a log scale with a shared range across fates, as in the notebook.
    perc_matrix = np.array([percentages_for(u) for u in ids])      # (N, n_fates)
    log_perc = np.log(perc_matrix + 1e-3)
    log_lo, log_hi = float(log_perc.min()), float(log_perc.max())

    # timepoints -> numeric code for coloring
    uniq_t = sorted(set(times))
    tcode = np.array([uniq_t.index(t) for t in times], dtype=float)

    # coloring options for the scatter: (values, cmap, clim) -- cmaps match the notebook
    color_fields = {"l_cross": (l_cross, "Greens", (1, 8)),
                    "area":    (areas,   "Blues",  None),
                    "time":    (tcode,   "viridis", None)}
    for j, lab in enumerate(fate_labels):
        color_fields[lab] = (log_perc[:, j], "Reds", (log_lo, log_hi))
    color_options = ["l_cross", "area", "time"] + list(fate_labels)
    fate_set = set(fate_labels)

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
        uid = ids[idx]
        if state["org"] is not None:
            state["org"].remove()
            state["org"] = None

        vtp = utils.organoid_vtp_path(args.data_path, cfg, uid)
        obj = utils.organoid_obj_path(args.data_path, cfg, uid)
        fields, field_names = None, None
        try:
            m = FateMarkers()
            if os.path.exists(vtp):
                m.load_mesh_from_file(vtp)
                m._refine_lgr5_marker()  # match the rerun_fm / coeffs pipeline
                m.align_with_pca()       # canonical PCA pose, matching the coeffs frame
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
                cf = np.load(utils.organoid_coeffs_path(args.data_path, cfg, uid))
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
    ax_bar = fig.add_axes([0.71, 0.10, 0.27, 0.82])  # fate % bar

    init_c, init_cmap, _ = color_fields["l_cross"]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=init_c, cmap=init_cmap, s=14, alpha=0.85)
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
        sc.set_cmap(cmap)
        if clim is not None:
            sc.set_clim(*clim)
        else:
            finite = np.isfinite(c)
            if finite.any():
                sc.set_clim(np.min(c[finite]), np.max(c[finite]))
        cbar.set_label(f"log({field}+1e-3)" if field in fate_set else field)
        cbar.update_normal(sc)
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
    ax_bar.set_xlabel("fraction of cells")
    ax_bar.set_title("fate %")

    def select(idx):
        uid = ids[idx]
        hl.set_offsets([xy[idx]])
        vals = percentages_for(uid)
        for b, v in zip(bars, vals):
            b.set_width(v)
        ax_bar.set_xlim(0, max(0.05, float(np.max(vals))))
        ax_bar.set_title(f"fate %  —  {uid}\n time {times[idx]} | L {l_cross[idx]:.2f}"
                         f" | area {areas[idx]:.0f}")
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
