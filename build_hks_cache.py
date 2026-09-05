"""Build a per-organoid HKS cache so vocab training / curvature analysis can skip
the slow mesh reload.

For every good-labels organoid across main+sup+pert2 it stores the decimated
per-vertex HKS (at N_TIMES variable times) plus frac/complexity/area — i.e. the
output of build_vocab.load_one — so a run reads one ~60 MB npz instead of
re-reading ~230 GB of saved eigendecompositions.

Rebuild after run_new_meshes.py changes good_labels, or if N_TIMES /
DECIMATE_FACE change (build_vocab validates those before using the cache).

Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/anaconda3/envs/scmpx/bin/python build_hks_cache.py --workers 6
"""
import os
import sys
import argparse
import concurrent.futures

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
import build_vocab as BV

CACHE = "sim/hks_cache_variable_time.npz"


def build(workers=6):
    save_paths = BV.gather_save_paths()
    print(f"loading {len(save_paths)} organoids (this is the slow part, once)...")
    res = [None] * len(save_paths)
    tpw = max(1, 8 // workers)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=BV._init_worker, initargs=(tpw,)) as ex:
        for i, r in enumerate(tqdm(ex.map(BV._run_one, save_paths),
                                   total=len(save_paths), desc="load")):
            res[i] = r

    results = [r for r in res if r is not None]   # (save_path, hks, frac, complexity, area)
    BV.save_cache(save_paths, results)             # canonical writer lives in build_vocab
    print(f"  ({len(results)}/{len(save_paths)} organoids kept; "
          f"{len(save_paths) - len(results)} skipped as failed/degenerate)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    build(ap.parse_args().workers)
