# Organoid shape analysis

This repo quantifies the **3D shape** of organoid surface meshes and relates it to
**cell-fate composition**. Each organoid mesh is turned into rotation-invariant
shape descriptors — spherical-harmonic (SH) reconstruction complexity and a
heat-kernel-signature (HKS) bag-of-features "power spectrum", plus one HKS-amplitude
feature — which are then clustered and embedded in 2D, and compared against
per-organoid fate marker fractions.

All harmonic decompositions are **SPHARM**: each organoid's true geometry (PCA-aligned, no
per-axis rescaling) is decimated to 5000 faces and mapped area-preservingly onto the unit
sphere (`src/spharm.py`, `SpHarm.from_mesh(...).map_to_sphere()`; median triangle-area error
~1%), and fields are expanded in real spherical harmonics on that map by area-weighted least
squares, so the degree-l blocks of the power spectrum are exact. (The earlier pipeline projected
onto manifold harmonics, the Laplace-Beltrami eigenvectors of a per-axis rescaled mesh, grouped
into degree blocks by counting; on complex organoids those blocks are not degenerate clusters.)
Fate fields are not decomposed.

Three datasets feed the pipeline:

- **`main_dataset`** — wild-type (WT) developmental time course (day 1.5 → 4.5).
- **`sup_dataset`** — supplementary WT organoids (day 4.5).
- **`pert2`** — drug-perturbed organoids at day 4.5 (five conditions).

A fourth, **`wnt_pert`** (WNT perturbation, day 3.5, shape only — no fate data), is
analysed for complexity only and kept in its own `Data/npz/wnt_master.npz`; see
[WNT perturbation (complexity only)](#wnt-perturbation-complexity-only).

> [!IMPORTANT]
> **This repo does not work without the organoid meshes.** They are not included here: `Data/`
> is git-ignored, so a fresh clone has only the code. Every step of the pipeline starts from
> the meshes, and the viewer needs them to show an organoid. You have to obtain them separately
> and place them as described in [Data layout](#data-layout).

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
├── npz/                       # master_full.npz, master.npz, wnt_master.npz, fate_percentages.npz (dim_red cache)
├── spharm_maps_vertex/<uid>.npz  # cached spherical map + decimated, smoothed mesh per organoid (compute_master_npz)
├── spharm_meshes/<uid>.obj    # the same mesh for viewing, written whenever a map is computed; not read by the pipeline
├── highlight_organoids.json   # organoids drawn on dim_red's highlight figure
├── uid_groups.json            # hand-picked shape groups (inspect_embedding.py 'uid_group' colouring)
├── main_dataset/
│   ├── config.json            # layout, timepoints, fate marker names, discard rules
│   ├── vtp/<timepoint>/*.vtp  # one mesh per organoid (+ fm_data/ after run_new_meshes)
│   ├── feature_tables/
│   │   ├── mesh_features.csv         # classical shape features (volume, sphericity, ...)
│   │   └── labels_to_discard.csv     # built by manage_discards.py
│   └── cell_features_class_with_projection_exclusive.csv   # per-cell fate table
├── sup_dataset/               # same structure (single timepoint day4p5)
├── pert2/                     # perturbation dataset (day4p5)
│   ├── config.json
│   └── vtp/{normal,small}/*.vtp
└── wnt_pert/                  # Wnt plate, shape only (day3p5)
    ├── config.json            # mesh_ext "stl", hks false, save_coeffs false (lean)
    ├── feature_tables/        # mesh_features, mesh_topology, discards, review log
    └── stl/{control,IWP2_24hr,IWP2_60hr,IWP2_60hr_HiWnt}/*.stl
```

`main_dataset` / `sup_dataset` use `{dataset}/vtp/{timepoint}/{timepoint}_{well}_{label}.vtp`.
`pert2` differs (no rearranging needed): its `vtp/` subfolders are a **shape-size
split** (`normal` / `small`), not timepoints (every organoid is day `4p5`), and
its filenames lead with the **drug condition** —
`{condition}_day4p5_{well}_{label}.vtp`. The five conditions are `stem-ChirVpaD1`
and `ta-Yapa` (in `normal`), `abs-Iwp2`, `sec-DaptHi`, `sec-DaptIwp2iMekD1` (in
`small`). `wnt_pert` follows the same scheme with one subfolder per condition:
`stl/{condition}/{condition}_day3p5_{well}_{label}.stl`. Its condition names contain
underscores (`IWP2_60hr_HiWnt`), so ids are parsed from the end (well = second-to-last
token) and the condition comes from a per-well map (`condition_by_token`, index -2).
Each dataset's `config.json` (`condition_uid_index`, `condition_by_token`, `time`, etc.)
tells the pipeline how to read time/condition from the filename.

Organoid IDs are `{dataset}_{filename-stem}`, e.g.
`pert2_stem-ChirVpaD1_day4p5_C06_101`.

> `Data/` is git-ignored (large binaries). The code in this repo reproduces every
> artifact under `Data/` and `output/` from the meshes.

---

## Pipeline

Data flows through these stages; each is a committed script/notebook so the run is
reproducible end-to-end. Commands assume the pipeline env (`KMP_DUPLICATE_LIB_OK=TRUE`).

### 1. Per-mesh preprocessing — `run_new_meshes.py`
PCA-aligns each mesh and computes its true-geometry Laplace–Beltrami eigendecomposition
(what the HKS is built from), and writes `vtp/<tp>/fm_data/{stem}_coeffs.npz` +
`{stem}_transformed_mesh.obj` and `good_labels_<tp>.npy`. (Older `_coeffs.npz` files also
hold manifold-harmonic modes; nothing reads them any more.)
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
A second auto rule drops meshes with **topological holes** (open holes or tunnels through the
surface; a deep dead-end pit is not one), in every timepoint including the hand-corrected ones.
`manage_discards.py update` first brings `feature_tables/mesh_topology.csv` up to date
(`utils.update_mesh_topology`: components, boundary loops, non-manifold edges, pinch vertices,
Euler characteristic, genus, `has_hole`, computed once per mesh). The rule is on for every dataset
unless its `config.json` has `"discard": {"topology_holes": false}`; `"sphericity": false`
switches the sphericity rule off (pert2: its elongated organoids are real shapes).
```bash
python manage_discards.py update --dataset Data/wnt_pert --topology-workers 6
python manage_discards.py update --dataset Data/wnt_pert --refresh-topology   # recompute every mesh
```
To curate by eye, `review_discards.py` steps through a dataset's organoids from most
to least complex (polyscope window; K keep / D discard / B back / Q quit), writing
discards to `manual_discards.txt` and logging every decision to
`feature_tables/manual_review.csv` so a later session resumes where it stopped:
```bash
python review_discards.py --dataset Data/wnt_pert --master Data/npz/wnt_master.npz
```

### 3. HKS vocabulary — `build_vocab.py`
Fits the bag-of-features codebooks (KMeans + PCA) on per-vertex HKS pooled across
main+sup+pert2 (complexity-balanced subsample on the SPHARM `l_cross_values` of
`Data/npz/master.npz`; organoids not in it are left out). The slow per-mesh load is cached
to `output/hks_cache_variable_time.npz` via `utils.build_hks_cache` / `load_hks_cache`
(built automatically on first run; reused when it matches the current
`good_labels`). Writes `output/vocab_variable_time.npz` (KMeans) and
`output/vocab_pca_variable_time.npz` (PCA), each with `meta_*` provenance keys.
```bash
python build_vocab.py --workers 6
python build_vocab.py --refresh-cache --workers 6   # after good_labels change
```

### 4. Master feature table — `compute_master_npz.py`
Assembles `Data/npz/master.npz`: per-organoid `areas`, `spharm_area_error` (how far the
spherical map is from area-preserving), `complexity_errors` (SPHARM shape reconstruction error
at lmax 0..8), `l_cross_values` (SH-degree complexity), the SPHARM power spectra
`hks_bof_power__{kmeans,pca}_variable` (N, 8 degrees, words) of the vocab-encoded HKS, and
`hks_amplitude`. Default folders are main+sup+pert2; `--ids FILE` restricts it to a list of uids.
`dim_red.ipynb` and `complexity_analysis.ipynb` read the table of every organoid, built with
`--out Data/npz/master_full.npz`.

The spherical map is the slow step (~4 s per organoid single-process, ~1 s with
`--workers 6`); each one is cached to `Data/spharm_maps_vertex/<uid>.npz` and reused by every later
run (`--update`, `--bof-only`, a rebuild), so an interrupted run loses only the organoids in
flight.

`hks_amplitude` is how strongly the heat signature varies over the surface: the mean
over the 20 variable diffusion times of the across-vertex std of `HKS/mean − 1`. The
vocab encoding throws this size away (build_vocab's `Normalizer` scales every vertex
curve to length 1), so `dim_red` adds it back as one feature.
```bash
python compute_master_npz.py --workers 6                 # full build
python compute_master_npz.py --update --workers 6        # incremental (after discards)
python compute_master_npz.py --bof-only --workers 6      # re-encode after a vocab change (cheap)
```
Only the `hks_bof_power__*` arrays depend on the vocab — `--bof-only` rebuilds
just those (and recomputes `hks_amplitude` from the same HKS) on the cached maps and keeps
everything else, so after re-fitting the vocab you do **not** need a full rebuild.

### 5. Embeddings — `dim_red.ipynb`
**Features.** The HKS SPHARM spectrum as sqrt(power) per degree and vocab word (degrees
0..`MODE_CUT` − 1 = 7), each feature standardised to unit variance so every degree has an equal
say, plus the HKS amplitude with weight `AMPLITUDE_WEIGHT` = 1.2, built by
`utils.load_hks_features` (log amplitude, z-scored, scaled so weight 1 = the typical spread of the
whole HKS block). Chosen on the hand-picked groups and neighbour reviews: raw or max-normalised
power leaves ~90% of the distance to degrees 0 and 2; learned per-feature weights did not improve
on the standardised features.

**Selection** — one cell chooses the rows from `Data/npz/master_full.npz`: organoids in the fate
table (`utils.load_fate_percentages`, cached to `Data/npz/fate_percentages.npz`), `l_cross >=
L_CROSS_MIN`, minus `EXCLUDE_CONDITIONS`, then an optional per-timepoint subsample
(`SUBSAMPLE_BY_TIME`; seeded with `SUBSAMPLE_SEED`, from each timepoint's organoids in uid order, so
reproducible while the discards stay fixed). The features are scaled over the selected organoids.

**Clusters.** Spectral clustering of the shape features (`n_shape_clusters` = 20, kNN graph 5). Clusters whose
100th-percentile (highest) `l_cross` is below `LOW_COMPLEXITY_THRESHOLD` (3) merge into the low-complexity
group 0 — empty when none qualifies — and the rest are the shape clusters 1..n (`agg_cluster`).

**Embeddings** (UMAP), each saved to `output/embeddings/` with the same per-organoid overlays
(time, condition, `l_cross`, area, clusters, fate fractions, Hill numbers) for the viewer:

| file | embeds |
|---|---|
| `hks_emb.npz` | the shape features (the main embedding) |
| `mesh_feat_emb.npz` | classical `mesh_features.csv` descriptors (baseline) |
| `all_feat_emb.npz` | shape + classical features, each block balanced |
| `shape_fate_emb.npz` | shape + fate composition (ILR), each block scaled to unit spread |

**Plot conventions.** Every embedding is rotated right after it is computed so the
low-complexity organoids sit on top (`utils.orient_embedding`: a pure rotation along the
`l_cross` gradient, so the saved npz match the plots). Every embedding panel has the same shape —
height/width `EMB_BOX_ASPECT` = 0.8 with equal data aspect (`utils.style_embedding_ax`) and
colourbars as tall as the panel (`utils.panel_colorbar`).

**Fate composition** — `Figures/fate_on_embedding.{png,pdf}`: the main embedding once per cell
type (stem, absorptive, secretory progenitor, Paneth, EE progenitor, EC, TA, none), coloured by the
% of each organoid's cells of that type — turbo on a cube-root power scale so small percentages stay
readable; organoids with none of that type in grey.

**Enrichment condition** — `Figures/condition_on_embedding.{png,pdf}`: the main embedding
coloured by `utils.enrichment_condition` — `main_dataset` = timecourse, `sup_dataset` = ctrl (WT
day 4.5), pert2 drugs = stem / abs / TA / secretory — in the paper figure's colours
(`utils.ENRICHMENT_COLOURS`), timecourse in grey underneath. The "by condition" panels of the HKS
embedding and the shape + fate co-embedding use the same colours (`utils.scatter_enrichment`).

**Other outputs:** `Figures/hill_diversity.{png,pdf}`, `Figures/shape_vs_fate_distance.pdf` and
`output/shape_fate_distance.npz`.

#### Highlight figure — `Figures/agg_clusters_highlight.{png,pdf}`
The main embedding coloured by aggregated cluster, with example organoids drawn around it:
- **dots** — diameter ∝ √area, so dot area ∝ organoid area; the size key (equivalent-sphere
  diameter) sits bottom right;
- **meshes** — the organoids in `Data/highlight_organoids.json` `highlight`, at one physical
  scale (100 µm bar; the fm_data meshes are the µm `.vtp` / 10), joined to their points by
  leader lines placed so they don't cross; each takes its current cluster's colour and its
  `name` as label (empty → `cluster N`);
- **sphere row** — the json's `spherical` organoids, smallest to largest, standing in for the
  low-complexity group;
- **colours** — a fixed palette, assigned by increasing median `l_cross` (most spherical =
  yellow … most complex = dark teal); grey = low complexity.

The cell before the figure fills `highlight` with one organoid per shape cluster
(`utils.update_highlight_json`): the **medoid**, the member with the smallest mean shape-feature
distance to the rest of its cluster (the low-complexity group is skipped), also saved as a table in `output/cluster_representatives.csv`. It prints each medoid and
where the previous pick ranked, and keeps each cluster's `name` and the `spherical` row. Set
`UPDATE_HIGHLIGHTS = False` in that cell to keep a hand-edited highlight list.

### WNT perturbation (complexity only)
`Data/wnt_pert` is one day-3.5 plate (20260919_IWP2_LGR5) — since 2026-10-08 the collaborator's **cleaned selection**: 890 meshes classified as healthy organoids from 2D DAPI MIPs (`Data/wnt_pert/README.txt`), flattened into the tree below; results for meshes byte-identical to the earlier full export were reused. Four conditions in duplicate wells — control (D02, D03),
IWP2_24hr (E02, E03), IWP2_60hr (D07, D08), IWP2_60hr_HiWnt (C07, C08) — set per well by
`config.json` `condition_by_token`. It holds geometry-only `.stl` meshes (`"mesh_ext": "stl"`) in
the pert2-style tree `stl/{condition}/{condition}_day3p5_{well}_{label}.stl` (one subfolder, with
its own `fm_data/`, per condition; `"time": "3p5"`). Only SH complexity is
computed for it: `"hks": false` makes `run_new_meshes.py` skip the true-geometry eigensolve (only
HKS needs it), and `"save_coeffs": false` is the **lean mode**: instead of the ~35 MB
`_coeffs.npz` eigendecomposition it keeps only the aligned `.obj`, from which
`compute_master_npz.py --complexity-only` computes the area and the SPHARM complexity errors
(degrees 0–8). Organoids that already have a full `_coeffs.npz` are reused as is. All non-discarded organoids are processed. (An earlier
`"subsample"` config block — `n_per_group`, `group_uid_index` -2, `seed`, `frozen_list` — can
restrict `good_labels` to an even per-well sample; the first wnt analysis used 63/well, recorded
in `feature_tables/subsample_{condition}.txt`.)
```bash
python manage_discards.py update --dataset Data/wnt_pert          # sphericity > 1.3
python run_new_meshes.py Data/wnt_pert --workers 6
python compute_master_npz.py --folders Data/wnt_pert --out Data/npz/wnt_master.npz --complexity-only
```
`wnt_complexity.ipynb` then writes the complexity of every organoid in `wnt_master.npz` to `output/wnt_complexity.csv` and compares the control wells with each IWP2 condition in `Figures/wnt_complexity_by_condition.png`.

### Supporting: `complexity_analysis.ipynb`
Inspects per-degree reconstruction error to choose the `l_cross` threshold and
writes `output/complexity_recon_errors.csv` (per-degree errors + `l_cross` at a range
of thresholds).

### Dependency chain
```
meshes ─run_new_meshes→ fm_data ─build_vocab→ vocab ─compute_master_npz→ master_full.npz ─dim_red→ embeddings
dim_red: clusters ─(medoids)→ highlight_organoids.json ─(highlight cell)→ highlight figure
```
Change the **vocab** → rebuild master (`--bof-only`) → re-run dim_red. The highlight medoids follow
the clusters within the same dim_red run.

### Reproducibility notes
- Seeds are fixed (`SEED=42`, `random_state=42`, `SUBSAMPLE_SEED=42`); the dim_red subsample picks
  from uid-sorted organoids so it survives a master rebuild/reorder, but a change to the discards
  changes the pick. Embeddings are rotated to a canonical
  orientation (low complexity on top), and the highlight medoids are deterministic.
- `vocab_*.npz` embed `meta_*` provenance (datasets, thresholds, decimation, seed, …).
- The retired HKS weight-training / chamfer / hand-labelling tools and their data are
  kept under `archive/weight_training_20261008/` (see its README).
- Superseded artifacts from the pert→pert2 migration are kept under
  `archive/pert2_run_backups/` (see its README).
- UMAP is deterministic only within one `umap-learn`/`numba`/`pynndescent`
  install; pin those for cross-machine reproduction.

---

# Viewing embeddings without running the pipeline

`inspect_embedding.py` is an interactive viewer for a computed embedding. It shows
the organoids as a 2D scatter; clicking a point shows that organoid's 3D mesh in a
panel beside it to rotate/zoom. It is **self-contained**: it needs only the
embedding `.npz` and the original `.vtp` meshes — no other pipeline outputs — so
the embedding + meshes are enough to explore shapes without re-running anything.

## Inputs

1. **The code** — `inspect_embedding.py` and the `src/` folder beside it.
2. **An embedding file** — a single `.npz` (e.g. `output/embeddings/hks_emb.npz`)
   with the 2D coordinates and organoid IDs. Read as-is.
3. **The original meshes** — the `.vtp` files (see [Data layout](#data-layout)).

By default no coefficient files or `.obj` meshes are needed — the `.vtp` files are
read directly. (`--source pipeline` uses the precomputed outputs for extra
overlays; see below.)

## Environment + running

```bash
conda env create -f environment.yml   # one time
conda activate organoid-viewer
python inspect_embedding.py --embedding output/embeddings/hks_emb.npz --data_root Data
```

- `--embedding` — path to the `.npz` embedding file.
- `--data_root` — root data folder; searched recursively for the `.vtp` files, so
  it only needs to sit above the dataset folders.

One window opens with two panels side by side (drag the divider to resize):

- **Embedding (left)** — 2D scatter, one dot per organoid. **"Colour points by"**
  recolors by timepoint, cluster, complexity, cell-type diversity, per-marker fate
  fraction, or **condition** (WT vs the `pert2` drug treatments). Click a dot to
  load its 3D mesh; hover to see its id in the status bar. The toolbar below
  zooms, pans and saves the plot.
- **Organoid (right)** — the selected organoid's id and stats, its 3D mesh (drag to
  rotate, scroll to zoom, shift-drag to pan) and a bar chart of its fate
  fractions. **"Colour organoid by"** picks the per-vertex fate marker; the choice
  and the camera are kept when you click the next organoid, and **Reset view**
  returns to the front view.

Closing the window quits.

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
