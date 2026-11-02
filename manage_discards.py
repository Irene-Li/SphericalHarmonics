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

  1. Auto-derived — read on every run:
       • sphericity > --sphericity-tol (default 1.3), from mesh_features.csv:
         organoids with an irregular / fused shape. A dataset whose config.json has
         "discard": {"sphericity": false} skips this rule (pert2: its elongated
         stem-ChirVpaD1 organoids are real shapes, not fused ones)
       • topological holes, from {FEATURES_DIR}/mesh_topology.csv: open holes or
         tunnels through the surface (has_hole == 1). The table is brought up to
         date first (utils.update_mesh_topology computes any mesh not in it yet;
         --topology-workers N in parallel, --refresh-topology recomputes all).
         On for every dataset; a dataset's config.json can switch it off with
         "discard": {"topology_holes": false}.

  2. Hand-coded — stored in {FEATURES_DIR}/manual_discards.txt:
       organoids you want to exclude for any other reason
       (e.g. identified visually with review_discards.py). One id per line;
       a trailing '_mesh.obj' is stripped.

  3. Corrected timepoints — CORRECTED_TIMEPOINTS:
       hand-corrected timepoints whose organoids are kept by the sphericity rule.

  labels_to_discard.csv = ((sphericity − corrected-timepoints) ∪ holes ∪ manual) ∩ exported
  Holes are mesh defects (no spherical map, no clean HKS), so they are discarded in the
  corrected timepoints too; so are manual discards.
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
  python manage_discards.py update --dataset Data/wnt_pert --refresh-topology --topology-workers 6
  python manage_discards.py add day3p5_A01_42
  python manage_discards.py add day4p5_B07_78 --dataset Data/sup_dataset
  python manage_discards.py import deleted.txt
  python manage_discards.py remove day3p5_A01_42
  python manage_discards.py list
"""

import sys
import os
import csv
import json
import argparse

DEFAULT_DATASET = "Data/sup_dataset"

CORRECTED_TIMEPOINTS = ["day4p5", "day4p5-more"]

# Path globals; populated by configure_paths() from the --dataset argument.
DATASET = FEATURES_DIR = MANUAL_FILE = OUTPUT_FILE = MESH_FEATURES = EXPORT_STATUS = None
TOPOLOGY_FILE = None
TOPOLOGY_WORKERS, TOPOLOGY_REFRESH = 1, False     # set from the command line


def configure_paths(dataset):
    """Point all the file globals at the given dataset root."""
    global DATASET, FEATURES_DIR, MANUAL_FILE, OUTPUT_FILE, MESH_FEATURES, EXPORT_STATUS
    global TOPOLOGY_FILE
    DATASET       = dataset.rstrip("/")
    FEATURES_DIR  = f"{DATASET}/feature_tables"
    MANUAL_FILE   = f"{FEATURES_DIR}/manual_discards.txt"
    OUTPUT_FILE   = f"{FEATURES_DIR}/labels_to_discard.csv"
    MESH_FEATURES = f"{FEATURES_DIR}/mesh_features.csv"
    EXPORT_STATUS = f"{DATASET}/export_status.csv"
    TOPOLOGY_FILE = f"{FEATURES_DIR}/mesh_topology.csv"


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


def topology_rule_enabled():
    """The hole rule is on unless config.json says "discard": {"topology_holes": false}."""
    path = f"{DATASET}/config.json"
    if not os.path.exists(path):
        return True
    with open(path) as f:
        return bool(json.load(f).get("discard", {}).get("topology_holes", True))


def load_topology():
    """Organoids whose mesh has a topological hole (mesh_topology.csv, has_hole == 1)."""
    if not topology_rule_enabled():
        return set()
    from src import utils                            # heavy (vtk), so only when the rule is on
    utils.update_mesh_topology(DATASET, workers=TOPOLOGY_WORKERS, refresh=TOPOLOGY_REFRESH,
                               verbose=False)
    with open(TOPOLOGY_FILE) as f:
        holes = {r["label_uid"].strip() for r in csv.DictReader(f) if int(r["has_hole"])}
    print(f"  topological holes: {len(holes)} organoids")
    return holes


def sphericity_rule_enabled():
    """The sphericity rule is on unless config.json says "discard": {"sphericity": false}."""
    path = f"{DATASET}/config.json"
    if not os.path.exists(path):
        return True
    with open(path) as f:
        return bool(json.load(f).get("discard", {}).get("sphericity", True))


def load_auto(sphericity_tol):
    """Derive auto discards. Returns {'sphericity': set, 'topology': set}."""
    by_sphericity = set()
    if not sphericity_rule_enabled():
        print("  sphericity rule: off for this dataset (config.json)")
    elif os.path.exists(MESH_FEATURES):
        with open(MESH_FEATURES) as f:
            for r in csv.DictReader(f):
                uid = (r.get("label_uid") or "").strip()
                sph = r.get("sphericity")
                if uid and sph not in (None, "") and float(sph) > sphericity_tol:
                    by_sphericity.add(uid)
        print(f"  sphericity > {sphericity_tol}: {len(by_sphericity)} organoids")
    else:
        print(f"  [skip] {MESH_FEATURES} not found")
    return {"sphericity": by_sphericity, "topology": load_topology()}


def write_combined(ids):
    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_uid"])
        for uid in sorted(ids):
            w.writerow([uid])


def rebuild(sphericity_tol):
    """((sphericity − corrected-timepoints) ∪ holes ∪ manual) ∩ exported  →  labels_to_discard.csv."""
    uni = universe()
    corrected = set(CORRECTED_TIMEPOINTS)
    print("Auto-derived:")
    manual = load_manual()

    # Sphericity discards are suppressed for corrected timepoints; holes and manual are not.
    auto_d = load_auto(sphericity_tol)
    auto = set().union(*auto_d.values())
    auto_filtered = {i for i in auto_d["sphericity"] if i.split("_")[0] not in corrected} | auto_d["topology"]
    present = (auto_filtered | manual) & uni
    combined = present
    dropped = (auto | manual) - uni
    auto_suppressed = auto - auto_filtered
    write_combined(combined)

    print(f"  manual: {len(manual)} organoids")
    if auto_suppressed:
        print(f"  [note] {len(auto_suppressed)} sphericity discards are in corrected timepoints "
              f"{CORRECTED_TIMEPOINTS} and were kept (holes and manual discards there still apply)")
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
    auto_d = load_auto(sphericity_tol)
    by_sph, by_topo = auto_d["sphericity"], auto_d["topology"]
    manual = load_manual()
    auto_filtered = {i for i in by_sph if i.split("_")[0] not in corrected} | by_topo
    all_ids = sorted((auto_filtered | manual) & uni)

    print(f"\n{'ID':<35} {'SOURCE':<16} REASON")
    print("-" * 64)
    for uid in all_ids:
        reasons = []
        if uid in by_sph: reasons.append("sphericity")
        if uid in by_topo: reasons.append("topology")
        if uid in manual: reasons.append("manual")
        is_auto = uid in by_sph or uid in by_topo
        src = "auto + manual" if (is_auto and uid in manual) else \
              "auto"          if is_auto else "manual"
        print(f"{uid:<35} {src:<16} {', '.join(reasons)}")
    print(f"\nauto(sphericity): {len(by_sph & uni)}  auto(topology): {len(by_topo & uni)}  "
          f"manual: {len(manual & uni)}  "
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
    parser.add_argument("--topology-workers", type=int, default=1,
                        help="parallel workers for meshes missing from mesh_topology.csv")
    parser.add_argument("--refresh-topology", action="store_true",
                        help="recompute mesh_topology.csv for every mesh of the dataset")
    args = parser.parse_args()

    configure_paths(args.dataset)
    TOPOLOGY_WORKERS, TOPOLOGY_REFRESH = args.topology_workers, args.refresh_topology
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
