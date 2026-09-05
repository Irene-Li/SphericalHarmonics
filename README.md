# Organoid shape analysis

This repo quantifies the **3D shape** of organoid surface meshes and relates it to
**cell-fate composition**. Each organoid mesh is turned into rotation-invariant
shape descriptors — spherical-harmonic (SH) reconstruction complexity and a
heat-kernel-signature (HKS) bag-of-features "power spectrum" — which are then
weighted, clustered, and embedded in 2D, and compared against per-organoid fate
marker fractions.

Three datasets feed the pipeline:

- **`main_dataset`** — wild-type (WT) developmental time course (day 1.5 → 4.5).
- **`sup_dataset`** — supplementary WT organoids (day 4.5).
- **`pert2`** — drug-perturbed organoids at day 4.5 (five conditions).

> Just want to explore an already-computed result? Skip to
> [Viewing embeddings without running the pipeline](#viewing-embeddings-without-running-the-pipeline).

---

## Environments

Two conda environments are used:

- **Pipeline (heavy)** — anything with `igl`, `vtk`, `numpy`, `scipy`,
  `scikit-learn`, `umap-learn`, and `tqdm`. In development this is the `scmpx`
  env. All pipeline commands below are prefixed with `KMP_DUPLICATE_LIB_OK=TRUE`
  to avoid an OpenMP clash between `igl` and the BLAS libraries, e.g.
  `KMP_DUPLICATE_LIB_OK=TRUE python compute_master_npz.py`.
- **Viewer (light)** — created from `environment.yml`
  (`conda env create -f environment.yml && conda activate organoid-viewer`);
  only what `inspect_embedding.py` needs.

---

## Data layout

Each dataset folder holds its meshes plus the tables the pipeline reads:

```
Data/
├── main_dataset/
│   ├── config.json            # layout, timepoints, fate marker names, discard rules
│   ├── vtp/<timepoint>/*.vtp  # one mesh per organoid (+ fm_data/ after run_new_meshes)
│   ├── feature_tables/
│   │   ├── mesh_features.csv         # classical shape features (volume, sphericity, ...)
│   │   └── labels_to_discard.csv     # built by manage_discards.py
│   └── cell_features_class_with_projection_exclusive.csv   # per-cell fate table
├── sup_dataset/               # same structure (single timepoint day4p5)
└── pert2/                     # perturbation dataset (day4p5)
    ├── config.json
    └── vtp/{normal,small}/*.vtp
```

`main_dataset` / `sup_dataset` use `{dataset}/vtp/{timepoint}/{timepoint}_{well}_{label}.vtp`.
`pert2` differs (no rearranging needed): its `vtp/` subfolders are a **shape-size
split** (`normal` / `small`), not timepoints (every organoid is day `4p5`), and
its filenames lead with the **drug condition** —
`{condition}_day4p5_{well}_{label}.vtp`. The five conditions are `stem-ChirVpaD1`
and `ta-Yapa` (in `normal`), `abs-Iwp2`, `sec-DaptHi`, `sec-DaptIwp2iMekD1` (in
`small`). Each dataset's `config.json` (`condition_uid_index`, `time`, etc.) tells
the pipeline how to read time/condition from the filename.

Organoid IDs are `{dataset}_{filename-stem}`, e.g.
`pert2_stem-ChirVpaD1_day4p5_C06_101`.

> `Data/` is git-ignored (large binaries). The code in this repo reproduces every
> artifact under `Data/` and `sim/` from the meshes.

---

## Pipeline

Data flows through these stages; each is a committed script/notebook so the run is
reproducible end-to-end. Commands assume the pipeline env (`KMP_DUPLICATE_LIB_OK=TRUE`).

### 1. Per-mesh descriptors — `run_new_meshes.py`
PCA-aligns each mesh, computes its Laplace–Beltrami eigendecomposition and SH
coefficients, and writes `vtp/<tp>/fm_data/{stem}_coeffs.npz` +
`{stem}_transformed_mesh.obj` and `good_labels_<tp>.npy`.
```bash
python run_new_meshes.py Data/main_dataset --workers 6
python run_new_meshes.py Data/sup_dataset  --workers 6
python run_new_meshes.py Data/pert2        --workers 6
```

### 2. Discards — `manage_discards.py` (optional)
Builds `feature_tables/labels_to_discard.csv` (auto: `sphericity` outliers; manual:
`manual_discards.txt`). A dataset's `config.json` `discard` block points at it, and
`run_new_meshes.py` excludes those organoids from `good_labels`.
```bash
python manage_discards.py add <label_uid> --dataset Data/pert2   # blacklist one organoid
python manage_discards.py update --dataset Data/main_dataset      # rebuild from sphericity + manual
```

### 3. HKS vocabulary — `build_vocab.py`
Fits the bag-of-features codebooks (KMeans + PCA) on per-vertex HKS pooled across
main+sup+pert2 (complexity-balanced subsample). The slow per-mesh load is cached
to `sim/hks_cache_variable_time.npz` via `utils.build_hks_cache` / `load_hks_cache`
(built automatically on first run; reused when it matches the current
`good_labels`). Writes `sim/vocab_variable_time.npz` (KMeans) and
`sim/vocab_pca_variable_time.npz` (PCA), each with `meta_*` provenance keys.
```bash
python build_vocab.py --workers 6
python build_vocab.py --refresh-cache --workers 6   # after good_labels change
```

### 4. Master feature table — `compute_master_npz.py`
Assembles `Data/npz/master.npz`: per-organoid `areas`, `complexity_errors`,
`l_cross_values` (SH-degree complexity), `hks_coeffs_sparse`, and the
vocab-encoded `hks_bof_coeffs__{kmeans,pca}_variable`. Default folders are
main+sup+pert2.
```bash
python compute_master_npz.py --workers 6                 # full build
python compute_master_npz.py --update --workers 6        # incremental (after discards)
python compute_master_npz.py --bof-only --workers 6      # re-project after a vocab change (cheap)
```
Only the `hks_bof_coeffs__*` arrays depend on the vocab — `--bof-only` rebuilds
just those and keeps everything else, so after re-fitting the vocab you do **not**
need a full rebuild. (Chamfer, by contrast, does **not** depend on the vocab.)

### 5. Chamfer supervision — `compute_chamfer.py`
Pairwise (decimated) chamfer distance between high-complexity organoids
(`l_cross > L_CROSS_MIN`) → `Data/npz/chamfer_highcomplexity.npz`. Used only as
weak, trustworthy-when-small supervision for the weights (never as ground truth).
Non-manifold meshes `igl.decimate` can't collapse are skipped.
```bash
python compute_chamfer.py --workers 8
```

### 6. HKS weights — `optimize_hks_weights.py`
Learns per-feature weights on the HKS power spectrum from two signals: chamfer
correlation (confident pairs) and hand-picked-group compactness
(`Data/uid_groups.json`, including a "spheres" group). Writes
`Data/npz/hks_weights_full.npz` (weights + all hyperparameters).
```bash
python optimize_hks_weights.py --chamfer Data/npz/chamfer_highcomplexity.npz \
    --mode_cut 8 --beta_group 0.05 --cv_threshold 0.1
```

### 7. Embeddings — `dim_red.ipynb`
Loads master + weights, filters to organoids present in the fate table
(`utils.load_fate_percentages`, cached to `Data/npz/fate_percentages.npz`),
applies the weights, subsamples (uid-canonical, seeded → reproducible), clusters,
and UMAP-embeds. Saves `Data/embeddings/{hks_emb,mesh_feat_emb,all_feat_emb,shape_fate_emb}.npz`
for the viewer. A `VERSION` toggle at the top selects the artifact set (the final
run is `"new"`).

### Supporting: `complexity_analysis.ipynb`
Inspects per-degree reconstruction error to choose the `l_cross` threshold and
writes `Data/complexity_recon_errors.csv` (per-degree errors + `l_cross` at a range
of thresholds).

### Dependency chain
```
meshes ─run_new_meshes→ fm_data ─build_vocab→ vocab ─compute_master_npz→ master.npz
                                                               │
                                              compute_chamfer ─┤ (from l_cross + meshes, vocab-independent)
                                                               ▼
                                            optimize_hks_weights → weights ─dim_red→ embeddings
```
Change the **vocab** → rebuild master (`--bof-only`) → re-fit weights → re-run dim_red.
Change the **chamfer** → re-fit weights → re-run dim_red (master untouched).

### Reproducibility notes
- Seeds are fixed (`SEED=42`, `random_state=42`); the dim_red subsample is
  uid-canonical so it survives a master rebuild/reorder.
- `vocab_*.npz` and `chamfer_highcomplexity.npz` embed `meta_*` provenance
  (datasets, thresholds, decimation, seed, …).
- Superseded artifacts from the pert→pert2 migration are kept under
  `archive/pert2_run_backups/` (see its README).
- UMAP is deterministic only within one `umap-learn`/`numba`/`pynndescent`
  install; pin those for cross-machine reproduction.

---

# Viewing embeddings without running the pipeline

`inspect_embedding.py` is an interactive viewer for a computed embedding. It shows
the organoids as a 2D scatter; clicking a point opens that organoid's 3D mesh in a
separate window to rotate/zoom. It is **self-contained**: it needs only the
embedding `.npz` and the original `.vtp` meshes — no other pipeline outputs — so
the embedding + meshes are enough to explore shapes without re-running anything.

## Inputs

1. **The code** — `inspect_embedding.py` and the `src/` folder beside it.
2. **An embedding file** — a single `.npz` (e.g. `Data/embeddings/hks_emb.npz`)
   with the 2D coordinates and organoid IDs. Read as-is.
3. **The original meshes** — the `.vtp` files (see [Data layout](#data-layout)).

By default no coefficient files or `.obj` meshes are needed — the `.vtp` files are
read directly. (`--source pipeline` uses the precomputed outputs for extra
overlays; see below.)

## Environment + running

```bash
conda env create -f environment.yml   # one time
conda activate organoid-viewer
python inspect_embedding.py --embedding Data/embeddings/hks_emb.npz --data_root Data
```

- `--embedding` — path to the `.npz` embedding file.
- `--data_root` — root data folder; searched recursively for the `.vtp` files, so
  it only needs to sit above the dataset folders.

Two windows open:

- **Embedding window** — 2D scatter, one dot per organoid. "Color by" buttons
  recolor by timepoint, cluster, complexity, cell-type diversity, per-marker fate
  fraction, or **condition** (WT vs the `pert2` drug treatments). Click a dot to
  load its 3D mesh.
- **Organoid window** — the 3D mesh; drag to rotate, scroll to zoom. A draggable
  **"Colour organoid by"** panel lists the per-vertex fate markers.

Drag the windows apart so they don't overlap. Closing either window quits.

### Optional: `--source pipeline`

By default the viewer reads meshes straight from the `.vtp` files (`--source vtp`).
`--source pipeline` instead loads the precomputed `{stem}_coeffs.npz` +
`{stem}_transformed_mesh.obj` (from `run_new_meshes.py`) and unlocks an
**HKS / shape** overlay section (raw heat-kernel signatures at several diffusion
times, plus bag-of-features words). It needs those files present under
`--data_root`; if you only have the `.vtp` meshes, stay on `--source vtp`.

## Troubleshooting

- **`Indexed 0 .vtp files under ...`** — `--data_root` points at the wrong folder;
  point it above the `.vtp` files.
- **`no .vtp found for <id>`** after clicking a dot — that organoid's `.vtp` isn't
  under `--data_root`, or its filename doesn't match the expected stem. Other
  organoids are unaffected.
- **`ModuleNotFoundError`** — the env isn't active; run `conda activate organoid-viewer`.
- **Fate-marker names show as raw columns** (e.g. `LGR5_intensity` not `lgr`) —
  harmless; friendly names come from each dataset's `config.json`.
