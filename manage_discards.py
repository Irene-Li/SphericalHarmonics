#!/usr/bin/env python3
"""
Builds and maintains combined_labels_to_discard.npy for a dataset.

Organoid ids in that file are excluded from processing by run_new_meshes.py
and rerun_fm.py.

============================================================
CONFIGURATION  (edit these two lines at the top of the file)
============================================================

  FOLDER          dataset root — where manual_discards.txt and
                  combined_labels_to_discard.npy are written.

  FEATURES_PATH   folder containing the feature CSVs
                  (mesh_features.csv  and  organoids_to_discard.csv)

============================================================
TWO SOURCES
============================================================

  1. Auto-derived — read from FEATURES_PATH on every run:
       • sphericity > --sphericity-tol (default 1.3)
         organoids with an irregular / fused shape
       • C01 (DAPI) min-intensity == 0
         segmentation mask extends outside the imaged region

  2. Hand-coded — stored in {folder}/manual_discards.txt:
       organoids you want to exclude for any other reason
       (e.g. identified visually via the polyscope inspector)

  combined_labels_to_discard.npy  =  auto ∪ manual
  It is rebuilt from scratch every time any command runs.

============================================================
COMMANDS
============================================================

  update  -- rebuild the .npy from feature tables + manual list
  list    -- print every discarded id with its source
  add     -- add one or more ids to the manual list
  remove  -- remove one or more ids from the manual list
  import  -- bulk-add from a text file (one id or *_mesh.obj per line)

============================================================
EXAMPLES
============================================================

  # initial build
  python manage_discards.py update

  # stricter sphericity cutoff
  python manage_discards.py update --sphericity-tol 1.2

  # found a bad organoid in the inspector
  python manage_discards.py add day3p5_A01_42

  # bulk-import previously deleted meshes
  python manage_discards.py import deleted.txt

  # remove a mistakenly added id
  python manage_discards.py remove day3p5_A01_42

  # see everything with source labels (auto / manual / auto+manual)
  python manage_discards.py list
"""

import sys
import os
import argparse
import numpy as np

FOLDER         = "Data/20260224/"
FEATURES_PATH  = "Data/20251001/features"

MANUAL_FILE    = "discard.txt"
COMBINED_FILE  = "combined_labels_to_discard.npy"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_manual():
    path = os.path.join(FOLDER, MANUAL_FILE)
    if not os.path.exists(path):
        print(f"No manual discard file found at {path}. Starting with empty manual list.")
        return set()
    with open(path) as f:
        return {l.strip() for l in f if l.strip()}


def save_manual(ids):
    with open(os.path.join(FOLDER, MANUAL_FILE), "w") as f:
        for uid in sorted(ids):
            f.write(uid + "\n")


def load_auto(sphericity_tol):
    """Derive discards from feature CSVs.

    Returns a dict with keys 'sphericity' and 'intensity', each a set of ids.
    The combined auto set is the union of both.
    """
    import pandas as pd
    by_sphericity = set()
    by_intensity  = set()

    mesh_path = os.path.join(FEATURES_PATH, "mesh_features.csv")
    if os.path.exists(mesh_path):
        mesh = pd.read_csv(mesh_path)
        bad = mesh.loc[mesh["sphericity"] > sphericity_tol, "label_uid"]
        by_sphericity.update(bad.astype(str))
        print(f"  sphericity > {sphericity_tol}: {len(bad)} organoids")
    else:
        print(f"  [skip] {mesh_path} not found")

    disc_path = os.path.join(FEATURES_PATH, "organoids_to_discard.csv")
    if os.path.exists(disc_path):
        disc = pd.read_csv(disc_path)
        bad = disc.loc[disc["C01.min_intensity"] < 1, "label_uid"]
        by_intensity.update(bad.astype(str))
        print(f"  DAPI min-intensity == 0: {len(bad)} organoids")
    else:
        print(f"  [skip] {disc_path} not found")

    return {"sphericity": by_sphericity, "intensity": by_intensity}


def rebuild(sphericity_tol):
    """Combine auto + manual → save combined_labels_to_discard.npy."""
    print("Auto-derived:")
    auto_sets = load_auto(sphericity_tol)
    auto   = auto_sets["sphericity"] | auto_sets["intensity"]
    manual = load_manual()
    combined = sorted(auto | manual)
    out = os.path.join(FOLDER, COMBINED_FILE) 
    np.save(out, np.array(combined, dtype=str))
    print(f"  manual: {len(manual)} organoids")
    print(f"Combined ({len(combined)} total) → {out}")
    return combined


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_update(sphericity_tol):
    rebuild(sphericity_tol)


def cmd_list(sphericity_tol):
    auto_sets  = load_auto(sphericity_tol)
    by_sph     = auto_sets["sphericity"]
    by_int     = auto_sets["intensity"]
    auto       = by_sph | by_int
    manual     = load_manual()
    all_ids    = sorted(auto | manual)

    print(f"\n{'ID':<35} {'SOURCE':<20} REASON")
    print("-" * 70)
    for uid in all_ids:
        reasons = []
        if uid in by_sph: reasons.append("sphericity")
        if uid in by_int: reasons.append("intensity")
        if uid in manual: reasons.append("manual")
        src = "auto + manual" if (uid in auto and uid in manual) else \
              "auto"          if uid in auto else "manual"
        print(f"{uid:<35} {src:<20} {', '.join(reasons)}")
    print(f"\nauto: {len(auto)}  (sphericity: {len(by_sph)}, "
          f"intensity: {len(by_int)}, both: {len(by_sph & by_int)})  "
          f"manual: {len(manual)}  total unique: {len(all_ids)}")


def cmd_add(sphericity_tol, new_ids):
    manual = load_manual()
    added = set(new_ids) - manual
    manual |= set(new_ids)
    save_manual(manual)
    print(f"Added {len(added)} to manual list.")
    rebuild(sphericity_tol)


def cmd_remove(sphericity_tol, targets):
    manual = load_manual()
    missing = set(targets) - manual
    manual -= set(targets)
    save_manual(manual)
    if missing:
        print(f"Not in manual list (ignored): {sorted(missing)}")
    print(f"Removed {len(targets) - len(missing)} from manual list.")
    rebuild(sphericity_tol)


def cmd_import(sphericity_tol, txt_path):
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    new_ids = [l.replace("_mesh.obj", "") for l in lines]
    print(f"Importing {len(new_ids)} ids from {txt_path}")
    cmd_add(sphericity_tol, new_ids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd",
                        choices=["update", "list", "add", "remove", "import"])
    parser.add_argument("rest", nargs="*",
                        help="ids (add/remove), or path (import)")
    parser.add_argument("--sphericity-tol", type=float, default=1.3,
                        help="Sphericity threshold for auto filter (default: 1.3)")
    args = parser.parse_args()

    tol = args.sphericity_tol

    if args.cmd == "update":
        cmd_update(tol)
    elif args.cmd == "list":
        cmd_list(tol)
    elif args.cmd == "add":
        if not args.rest:
            print("Usage: manage_discards.py add <id> [<id> ...]")
            sys.exit(1)
        cmd_add(tol, args.rest)
    elif args.cmd == "remove":
        if not args.rest:
            print("Usage: manage_discards.py remove <id> [<id> ...]")
            sys.exit(1)
        cmd_remove(tol, args.rest)
    elif args.cmd == "import":
        if not args.rest:
            print("Usage: manage_discards.py import <file.txt>")
            sys.exit(1)
        cmd_import(tol, args.rest[0])
