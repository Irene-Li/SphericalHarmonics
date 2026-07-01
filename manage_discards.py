#!/usr/bin/env python3
"""
Builds and maintains labels_to_discard.csv for a dataset.

Organoid ids in that file are excluded from processing by run_new_meshes.py
(its config 'discard.labels_to_discard_csv' points here).

============================================================
CONFIGURATION
============================================================

  --dataset              dataset root (holds export_status.csv). Defaults to
                         Data/main_dataset; pass Data/sup_dataset to manage the
                         supplementary dataset's discard list instead.
  FEATURES_DIR           {dataset}/feature_tables: holds mesh_features.csv +
                         manual_discards.txt; labels_to_discard.csv is written
                         here too. A dataset without mesh_features.csv (e.g.
                         sup_dataset) simply has no auto-derived discards — its
                         labels_to_discard.csv is built from manual_discards.txt.
  CORRECTED_TIMEPOINTS   timepoints that were hand-corrected — their organoids
                         are never discarded (excluded from the output).

The discard list is keyed to the dataset's OWN label_uids — it is rebuilt from
`mesh_features.csv` (which shares the export's labels) intersected with the set
of actually-exported organoids (`export_status.csv`), so the list only ever
references meshes that exist.

============================================================
THREE INGREDIENTS
============================================================

  1. Auto-derived — read from mesh_features.csv on every run:
       • sphericity > --sphericity-tol (default 1.3)
         organoids with an irregular / fused shape

  2. Hand-coded — stored in {FEATURES_DIR}/manual_discards.txt:
       organoids you want to exclude for any other reason
       (e.g. identified visually via the polyscope inspector). One id per line;
       a trailing '_mesh.obj' is stripped.

  3. Corrected timepoints — CORRECTED_TIMEPOINTS:
       hand-corrected timepoints whose organoids are kept regardless.

  labels_to_discard.csv = ((auto − corrected-timepoints) ∪ manual) ∩ exported
  Manual discards survive the corrected-timepoints filter; auto discards do not.
  It is rebuilt from scratch every time any command runs.

============================================================
COMMANDS
============================================================

  update  -- rebuild the .csv from mesh_features + manual list
  list    -- print every discarded id with its source
  add     -- add one or more ids to the manual list
  remove  -- remove one or more ids from the manual list
  import  -- bulk-add from a text file (one id or *_mesh.obj per line)

============================================================
EXAMPLES
============================================================

  python manage_discards.py update
  python manage_discards.py update --sphericity-tol 1.2
  python manage_discards.py update --dataset Data/sup_dataset
  python manage_discards.py add day3p5_A01_42
  python manage_discards.py add day4p5_B07_78 --dataset Data/sup_dataset
  python manage_discards.py import deleted.txt
  python manage_discards.py remove day3p5_A01_42
  python manage_discards.py list
"""

import sys
import os
import csv
import argparse

DEFAULT_DATASET = "Data/sup_dataset"

CORRECTED_TIMEPOINTS = ["day4p5", "day4p5-more"]

# Path globals; populated by configure_paths() from the --dataset argument.
DATASET = FEATURES_DIR = MANUAL_FILE = OUTPUT_FILE = MESH_FEATURES = EXPORT_STATUS = None


def configure_paths(dataset):
    """Point all the file globals at the given dataset root."""
    global DATASET, FEATURES_DIR, MANUAL_FILE, OUTPUT_FILE, MESH_FEATURES, EXPORT_STATUS
    DATASET       = dataset.rstrip("/")
    FEATURES_DIR  = f"{DATASET}/feature_tables"
    MANUAL_FILE   = f"{FEATURES_DIR}/manual_discards.txt"
    OUTPUT_FILE   = f"{FEATURES_DIR}/labels_to_discard.csv"
    MESH_FEATURES = f"{FEATURES_DIR}/mesh_features.csv"
    EXPORT_STATUS = f"{DATASET}/export_status.csv"


configure_paths(DEFAULT_DATASET)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _strip(uid):
    uid = uid.strip()
    return uid[:-len("_mesh.obj")] if uid.endswith("_mesh.obj") else uid


def universe():
    """label_uids that were actually exported (the only meshes that can be
    processed). From export_status.csv; falls back to mesh_features.csv."""
    if os.path.exists(EXPORT_STATUS):
        with open(EXPORT_STATUS) as f:
            return {r["label_uid"].strip() for r in csv.DictReader(f) if r.get("label_uid")}
    print(f"  [warn] {EXPORT_STATUS} not found — using all mesh_features ids as the universe")
    with open(MESH_FEATURES) as f:
        return {r["label_uid"].strip() for r in csv.DictReader(f) if r.get("label_uid")}


def load_manual():
    if not os.path.exists(MANUAL_FILE):
        print(f"No manual discard file at {MANUAL_FILE}. Starting empty.")
        return set()
    with open(MANUAL_FILE) as f:
        return {_strip(l) for l in f if l.strip()}


def save_manual(ids):
    with open(MANUAL_FILE, "w") as f:
        for uid in sorted(ids):
            f.write(uid + "\n")


def load_auto(sphericity_tol):
    """Derive discards from mesh_features.csv. Returns {'sphericity': set}."""
    by_sphericity = set()
    if os.path.exists(MESH_FEATURES):
        with open(MESH_FEATURES) as f:
            for r in csv.DictReader(f):
                uid = (r.get("label_uid") or "").strip()
                sph = r.get("sphericity")
                if uid and sph not in (None, "") and float(sph) > sphericity_tol:
                    by_sphericity.add(uid)
        print(f"  sphericity > {sphericity_tol}: {len(by_sphericity)} organoids")
    else:
        print(f"  [skip] {MESH_FEATURES} not found")
    return {"sphericity": by_sphericity}


def write_combined(ids):
    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_uid"])
        for uid in sorted(ids):
            w.writerow([uid])


def rebuild(sphericity_tol):
    """((auto − corrected-timepoints) ∪ manual) ∩ exported  →  labels_to_discard.csv."""
    uni = universe()
    corrected = set(CORRECTED_TIMEPOINTS)
    print("Auto-derived:")
    auto = load_auto(sphericity_tol)["sphericity"]
    manual = load_manual()

    # Auto discards are suppressed for corrected timepoints; manual discards are not.
    auto_filtered = {i for i in auto if i.split("_")[0] not in corrected}
    present = (auto_filtered | manual) & uni
    combined = present
    dropped = (auto | manual) - uni
    auto_suppressed = auto - auto_filtered
    write_combined(combined)

    print(f"  manual: {len(manual)} organoids")
    if auto_suppressed:
        print(f"  [note] {len(auto_suppressed)} auto-discard ids are in corrected timepoints "
              f"{CORRECTED_TIMEPOINTS} and were kept (manual discards in those timepoints still apply)")
    if dropped:
        print(f"  [note] {len(dropped)} discard ids are not in the export and were "
              f"dropped (e.g. {sorted(dropped)[:3]})")
    print(f"Combined ({len(combined)} of {len(uni)} exported) → {OUTPUT_FILE}")
    return combined


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_update(sphericity_tol):
    rebuild(sphericity_tol)


def cmd_list(sphericity_tol):
    uni = universe()
    corrected = set(CORRECTED_TIMEPOINTS)
    by_sph = load_auto(sphericity_tol)["sphericity"]
    manual = load_manual()
    auto_filtered = {i for i in by_sph if i.split("_")[0] not in corrected}
    all_ids = sorted((auto_filtered | manual) & uni)

    print(f"\n{'ID':<35} {'SOURCE':<16} REASON")
    print("-" * 64)
    for uid in all_ids:
        reasons = []
        if uid in by_sph: reasons.append("sphericity")
        if uid in manual: reasons.append("manual")
        src = "auto + manual" if (uid in by_sph and uid in manual) else \
              "auto"          if uid in by_sph else "manual"
        print(f"{uid:<35} {src:<16} {', '.join(reasons)}")
    print(f"\nauto(sphericity): {len(by_sph & uni)}  manual: {len(manual & uni)}  "
          f"total unique (exported): {len(all_ids)}")


def cmd_add(sphericity_tol, new_ids):
    new_ids = {_strip(i) for i in new_ids}
    manual = load_manual()
    added = new_ids - manual
    save_manual(manual | new_ids)
    print(f"Added {len(added)} to manual list.")
    rebuild(sphericity_tol)


def cmd_remove(sphericity_tol, targets):
    targets = {_strip(i) for i in targets}
    manual = load_manual()
    missing = targets - manual
    save_manual(manual - targets)
    if missing:
        print(f"Not in manual list (ignored): {sorted(missing)}")
    print(f"Removed {len(targets) - len(missing)} from manual list.")
    rebuild(sphericity_tol)


def cmd_import(sphericity_tol, txt_path):
    with open(txt_path) as f:
        new_ids = [_strip(l) for l in f if l.strip()]
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
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"Dataset root holding feature_tables/ and export_status.csv "
                             f"(default: {DEFAULT_DATASET}). E.g. Data/sup_dataset")
    args = parser.parse_args()

    configure_paths(args.dataset)
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
