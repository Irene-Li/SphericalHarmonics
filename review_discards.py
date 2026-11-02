#!/usr/bin/env python
"""
Step through a dataset's organoids from most to least complex and mark each one
keep / discard. Discards go straight into the dataset's manual discard list
(feature_tables/manual_discards.txt, via manage_discards.py) and
labels_to_discard.csv is rebuilt; every decision is also logged to
feature_tables/manual_review.csv so a later session resumes where you stopped.

Order: l_cross descending, ties (e.g. at the cap) broken by the residual
reconstruction error at the highest degree. Meshes are the PCA-aligned
'_transformed_mesh.obj' from run_new_meshes.py (what l_cross was computed on).

Keys (polyscope window focused) — or use the buttons in the Review panel:
    K  keep        D  discard        B  back (undoes the previous decision)
    Q  quit (decisions are saved as you go)

Run from the repo root in the scmpx env, e.g.:
    KMP_DUPLICATE_LIB_OK=TRUE python review_discards.py \
        --dataset Data/wnt_pert --master Data/npz/wnt_master.npz

Afterwards apply the discards (nothing is recomputed; a frozen subsample just
loses the discarded organoids):
    python run_new_meshes.py Data/wnt_pert
    python compute_master_npz.py --folders Data/wnt_pert \
        --out Data/npz/wnt_master.npz --complexity-only --update
"""
import os
import io
import csv
import json
import argparse
import contextlib
from datetime import datetime

import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim

import manage_discards as md
from src import utils

ORG_NAME = "organoid"
COLOUR = (0x1b / 255, 0xaf / 255, 0x7a / 255)
ACCENT = (0.35, 0.85, 1.0, 1.0)
DISCARD_RED = (1.0, 0.45, 0.4, 1.0)


def load_order(master_path, dataset_name):
    """(label_uids, l_cross, err_at_max_degree, area) sorted most -> least complex."""
    m = np.load(master_path, allow_pickle=True)
    ids = m["ids"].astype(str)
    keep = m["datasets"].astype(str) == dataset_name
    uids = np.array([u[len(dataset_name) + 1:] for u in ids[keep]])
    l_cross = m["l_cross_values"][keep]
    err = m["complexity_errors"][keep][:, -1]
    area = m["areas"][keep]
    order = np.lexsort((-err, -l_cross))          # primary: l_cross desc
    return uids[order], l_cross[order], err[order], area[order]


def freeze_subsample(data_path, cfg):
    """If the config subsample has a frozen_list that doesn't exist yet, write the
    CURRENT sample (good_labels) to it before any discard is added — otherwise the
    next run_new_meshes.py would redraw the sample from the changed pool."""
    sub = cfg.get("subsample")
    if not sub or not sub.get("frozen_list"):
        return
    for tp in cfg["timepoints"]:
        path = sub["frozen_list"].format(timepoint=tp)
        gl = os.path.join(data_path, cfg.get("vtp_dir", "vtp"), tp, "fm_data", f"good_labels_{tp}.npy")
        if not os.path.exists(path) and os.path.exists(gl):
            labels = np.load(gl, allow_pickle=True)
            with open(path, "w") as f:
                f.write("".join(f"{l}\n" for l in labels))
            print(f"froze current subsample ({len(labels)}) → {path}")


class ReviewLog:
    """feature_tables/manual_review.csv: label_uid, decision, l_cross, reviewed_at."""
    FIELDS = ["label_uid", "decision", "l_cross", "reviewed_at"]

    def __init__(self, path):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path) as f:
                self.rows = {r["label_uid"]: r for r in csv.DictReader(f)}

    def _write(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows.values())

    def set(self, uid, decision, l_cross):
        self.rows[uid] = {"label_uid": uid, "decision": decision, "l_cross": f"{l_cross:.4f}",
                          "reviewed_at": datetime.now().isoformat(timespec="seconds")}
        self._write()

    def drop(self, uid):
        self.rows.pop(uid, None)
        self._write()


def set_manual(uid, discard, sphericity_tol):
    """Add/remove one id in manual_discards.txt and rebuild labels_to_discard.csv
    (manage_discards' own rebuild; its chatter is suppressed)."""
    with contextlib.redirect_stdout(io.StringIO()):
        manual = md.load_manual()
        md.save_manual(manual | {uid} if discard else manual - {uid})
        md.rebuild(sphericity_tol)


def _key(name):
    """True on the frame the key goes down (no auto-repeat); False if this
    polyscope/imgui build doesn't expose named keys (buttons still work)."""
    k = getattr(psim, f"ImGuiKey_{name}", None)
    if k is None:
        return False
    try:
        return psim.IsKeyPressed(k, False)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset root, e.g. Data/wnt_pert")
    ap.add_argument("--master", required=True, help="npz with that dataset's l_cross, e.g. Data/npz/wnt_master.npz")
    ap.add_argument("--redo", action="store_true", help="show already-reviewed organoids again")
    ap.add_argument("--sphericity-tol", type=float, default=1.3,
                    help="passed to manage_discards' rebuild (keep it equal to its default)")
    args = ap.parse_args()

    data_path = args.dataset.rstrip("/")
    dataset_name = os.path.basename(data_path)
    with open(f"{data_path}/config.json") as f:
        cfg = json.load(f)
    md.configure_paths(data_path)
    freeze_subsample(data_path, cfg)

    uids, l_cross, err, area = load_order(args.master, dataset_name)
    rank = {u: i for i, u in enumerate(uids)}
    log = ReviewLog(f"{data_path}/feature_tables/manual_review.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        manual = md.load_manual()
    queue = [u for u in uids if args.redo or (u not in log.rows and u not in manual)]
    print(f"{len(uids)} organoids in {args.master} | {len(log.rows)} already reviewed | "
          f"{len(manual)} manual discards | {len(queue)} to review")
    if not queue:
        print("nothing left to review (use --redo to go through them again).")
        return

    state = {"i": 0, "history": [], "quit": False, "n_discard": 0, "n_keep": 0}

    def show(i):
        ps.remove_all_structures()
        if i >= len(queue):
            return
        uid = queue[i]
        v, f = igl.read_triangle_mesh(utils.vtp_flat_obj_path(data_path, cfg, uid))
        v = v - v.mean(0)
        v = v / (np.linalg.norm(v, axis=1).max() or 1.0)
        ps.register_surface_mesh(ORG_NAME, v, f, color=COLOUR, smooth_shade=True)
        ps.reset_camera_to_home_view()

    def decide(discard):
        uid = queue[state["i"]]
        j = rank[uid]
        set_manual(uid, discard, args.sphericity_tol)
        log.set(uid, "discard" if discard else "keep", l_cross[j])
        state["n_discard" if discard else "n_keep"] += 1
        state["history"].append((state["i"], discard))
        print(f"  {'DISCARD' if discard else 'keep   '}  {uid}  l_cross {l_cross[j]:.2f}")
        state["i"] += 1
        show(state["i"])

    def back():
        if not state["history"]:
            return
        i, was_discard = state["history"].pop()
        uid = queue[i]
        if was_discard:
            set_manual(uid, False, args.sphericity_tol)
        log.drop(uid)
        state["n_discard" if was_discard else "n_keep"] -= 1
        print(f"  undo     {uid}")
        state["i"] = i
        show(i)

    def callback():
        psim.SetNextWindowPos((20.0, 20.0), psim.ImGuiCond_FirstUseEver)
        psim.SetNextWindowSize((330.0, 250.0), psim.ImGuiCond_FirstUseEver)
        psim.Begin("Review")
        i = state["i"]
        psim.TextUnformatted(f"{i} / {len(queue)} reviewed this session  "
                             f"(kept {state['n_keep']}, discarded {state['n_discard']})")
        if i < len(queue):
            uid = queue[i]
            j = rank[uid]
            psim.PushStyleColor(psim.ImGuiCol_Text, ACCENT)
            psim.SeparatorText(uid)
            psim.PopStyleColor()
            cap = "  (at cap)" if l_cross[j] >= 7.999 else ""
            psim.TextUnformatted(f"complexity rank {j + 1} / {len(uids)}")
            psim.TextUnformatted(f"l_cross {l_cross[j]:.2f}{cap}   err@deg8 {err[j]:.4f}")
            psim.TextUnformatted(f"area {area[j] * 100:.0f} um^2")
            if uid in log.rows:
                psim.TextUnformatted(f"previously: {log.rows[uid]['decision']}")
            psim.Separator()
            keep_hit = psim.Button("Keep  [K]")
            psim.SameLine()
            psim.PushStyleColor(psim.ImGuiCol_Text, DISCARD_RED)
            disc_hit = psim.Button("Discard  [D]")
            psim.PopStyleColor()
            if keep_hit or _key("K"):
                decide(False)
            elif disc_hit or _key("D"):
                decide(True)
        else:
            psim.TextUnformatted("All done — close the window or press Q.")
        psim.Separator()
        if psim.Button("Back  [B]") or _key("B"):
            back()
        psim.SameLine()
        if psim.Button("Quit  [Q]") or _key("Q"):
            state["quit"] = True
        psim.End()

    ps.init()
    ps.set_program_name(f"Review {dataset_name}")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("turntable")
    ps.set_build_default_gui_panels(False)
    ps.set_open_imgui_window_for_user_callback(False)
    ps.set_user_callback(callback)
    show(0)
    while not state["quit"] and not ps.window_requests_close():
        ps.frame_tick()

    with contextlib.redirect_stdout(io.StringIO()):
        n_manual = len(md.load_manual())
    print(f"\nsession: kept {state['n_keep']}, discarded {state['n_discard']} | "
          f"{n_manual} manual discards in {md.MANUAL_FILE}")
    print("apply them (nothing is recomputed):\n"
          f"  python run_new_meshes.py {data_path}\n"
          f"  python compute_master_npz.py --folders {data_path} --out {args.master} "
          f"--complexity-only --update")


if __name__ == "__main__":
    main()
