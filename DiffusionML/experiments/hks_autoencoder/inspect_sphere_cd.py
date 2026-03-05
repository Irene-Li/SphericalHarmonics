#!/usr/bin/env python3
"""
Interactive polyscope viewer for inspecting meshes by sphere-CD score.

Loads the sphere_cd_cache.json produced by data_sweep.py, sorts meshes from
most to least non-spherical, and lets you page through them with a reference
sphere overlay.  Mark meshes for deletion; after closing the window the
script confirms before permanently removing the paired _mesh.obj + _data.npz
files and pruning the cache.

Run from the repository root:

    python DiffusionML/experiments/hks_autoencoder/inspect_sphere_cd.py \
        --data_path Data/small_meshes [--start_at_top] [--min_cd 0.0] [--max_cd 999]

Controls (imgui sidebar):
    [< Prev] / [Next >]  — navigate
    [Mark / Unmark]      — toggle deletion flag for the current mesh
    Close window         — proceed to deletion confirmation
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import potpourri3d as pp3d

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dataset import _mesh_surface_area


# ---------------------------------------------------------------------------
# Icosphere (verts + faces) – needed for the reference sphere overlay
# ---------------------------------------------------------------------------

def _icosphere(subdivisions: int = 3):
    """Return (verts, faces) for a unit icosphere."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    for _ in range(subdivisions):
        new_f, mid = [], {}

        def _mid(a, b):
            nonlocal v
            key = (min(a, b), max(a, b))
            if key not in mid:
                m = (v[a] + v[b]) / 2.0
                mid[key] = len(v)
                v = np.vstack([v, m / np.linalg.norm(m)])
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = np.array(new_f, dtype=np.int64)

    return v.astype(np.float32), f


_SPHERE_VERTS, _SPHERE_FACES = None, None


def _get_sphere():
    global _SPHERE_VERTS, _SPHERE_FACES
    if _SPHERE_VERTS is None:
        _SPHERE_VERTS, _SPHERE_FACES = _icosphere(subdivisions=3)
    return _SPHERE_VERTS, _SPHERE_FACES


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def load_and_show(mesh_path: str):
    """Register the organoid mesh and a scaled reference sphere in polyscope."""
    ps.remove_all_structures()

    verts, faces = pp3d.read_mesh(mesh_path)
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)

    # Centre at origin (same as the CD computation)
    verts_c = verts - verts.mean(axis=0)

    area = _mesh_surface_area(verts_c, faces)
    r = math.sqrt(max(area, 1e-12) / (4.0 * math.pi))

    sphere_v, sphere_f = _get_sphere()
    sphere_v_scaled = sphere_v * r

    org = ps.register_surface_mesh(
        "organoid", verts_c, faces,
        color=(0.25, 0.55, 0.90),
        smooth_shade=True,
    )
    org.set_transparency(0.95)

    ref = ps.register_surface_mesh(
        "reference sphere", sphere_v_scaled, sphere_f,
        color=(0.95, 0.55, 0.20),
        smooth_shade=True,
    )
    ref.set_transparency(0.35)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactively inspect meshes by sphere-CD score")
    parser.add_argument('--data_path', default='Data/small_meshes')
    parser.add_argument('--min_cd', type=float, default=0.0,
                        help='Only show meshes with sphere_cd >= this value')
    parser.add_argument('--max_cd', type=float, default=float('inf'),
                        help='Only show meshes with sphere_cd <= this value')
    parser.add_argument('--start_at_top', action='store_true',
                        help='Start at the most non-spherical end (highest CD). '
                             'Default: start at lowest CD (most spherical).')
    args = parser.parse_args()

    cache_path = os.path.join(args.data_path, 'sphere_cd_cache.json')
    if not os.path.exists(cache_path):
        print(f"ERROR: cache not found at {cache_path}")
        print("Run data_sweep.py first to compute sphere-CD values.")
        sys.exit(1)

    with open(cache_path) as f:
        cache: dict = json.load(f)

    # Build list of (cd, abs_mesh_path), filter, sort
    entries = []
    for key, cd in cache.items():
        mesh_path = os.path.join(args.data_path, key)
        npz_path = mesh_path[:-len('_mesh.obj')] + '_data.npz'
        if not os.path.exists(mesh_path) or not os.path.exists(npz_path):
            continue
        if args.min_cd <= cd <= args.max_cd:
            entries.append((cd, mesh_path))

    entries.sort(key=lambda x: x[0], reverse=args.start_at_top)

    if not entries:
        print("No meshes match the given CD range. Exiting.")
        sys.exit(0)

    print(f"Loaded {len(entries)} meshes "
          f"(CD range [{entries[0][0]:.4f}, {entries[-1][0]:.4f}])")
    print("Close the polyscope window when done inspecting.")

    # ------------------------------------------------------------------
    # Mutable viewer state
    # ------------------------------------------------------------------
    state = {
        'idx': 0,
        'marked': set(),       # set of mesh_path strings
        'needs_refresh': True,
    }

    def callback():
        cd, path = entries[state['idx']]
        basename = os.path.basename(path)
        is_marked = path in state['marked']

        # ── Info ──────────────────────────────────────────────────────
        psim.SeparatorText("Mesh info")
        psim.Text(f"  {state['idx'] + 1} / {len(entries)}")
        psim.Text(f"  sphere_cd = {cd:.4f}")
        psim.TextWrapped(f"  {basename}")

        if is_marked:
            psim.TextColored((1.0, 0.3, 0.3, 1.0), "  *** MARKED FOR DELETION ***")

        # ── Navigation ────────────────────────────────────────────────
        psim.Separator()
        prev_clicked = psim.Button("< Prev")
        psim.SameLine()
        next_clicked = psim.Button("Next >")

        changed = False
        if prev_clicked and state['idx'] > 0:
            state['idx'] -= 1
            changed = True
        if next_clicked and state['idx'] < len(entries) - 1:
            state['idx'] += 1
            changed = True

        # ── Mark / Unmark ─────────────────────────────────────────────
        psim.Separator()
        if is_marked:
            if psim.Button("Unmark"):
                state['marked'].discard(path)
        else:
            if psim.Button("Mark for deletion"):
                state['marked'].add(path)

        psim.Text(f"  Marked: {len(state['marked'])} file(s)")

        # ── Legend ────────────────────────────────────────────────────
        psim.Separator()
        psim.TextColored((0.25, 0.55, 0.90, 1.0), "  Blue  = organoid mesh")
        psim.TextColored((0.95, 0.55, 0.20, 1.0), "  Orange = reference sphere")
        psim.TextDisabled("  (same surface area)")
        psim.Separator()
        psim.TextDisabled("  Close window to finish.")

        # ── Refresh 3D view when needed ───────────────────────────────
        if changed or state['needs_refresh']:
            state['needs_refresh'] = False
            load_and_show(entries[state['idx']][1])

    # ------------------------------------------------------------------
    # Launch polyscope
    # ------------------------------------------------------------------
    ps.init()
    ps.set_program_name("Sphere-CD Inspector")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_transparency_mode("pretty")

    load_and_show(entries[0][1])
    ps.set_user_callback(callback)
    ps.show()

    # ------------------------------------------------------------------
    # Post-session: confirm and delete
    # ------------------------------------------------------------------
    if not state['marked']:
        print("No files marked for deletion. Done.")
        return

    print(f"\n{'─'*60}")
    print(f"Files marked for deletion ({len(state['marked'])}):")
    for path in sorted(state['marked']):
        print(f"  {os.path.basename(path)}")

    ans = input("\nPermanently delete these files? [y/N]: ").strip().lower()
    if ans != 'y':
        print("Aborted — no files deleted.")
        return

    deleted, errors = [], []
    for mesh_path in state['marked']:
        stem = mesh_path[:-len('_mesh.obj')]
        npz_path = stem + '_data.npz'
        for fpath in (mesh_path, npz_path):
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    deleted.append(fpath)
            except OSError as e:
                errors.append(f"{fpath}: {e}")

    # Prune deleted keys from cache
    pruned = 0
    for mesh_path in state['marked']:
        key = os.path.basename(mesh_path)
        if key in cache:
            del cache[key]
            pruned += 1
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"\nDeleted {len(deleted)} file(s), pruned {pruned} cache entries.")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")


if __name__ == '__main__':
    main()
