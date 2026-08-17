# Organoid shape viewer

`inspect_embedding.py` is an interactive viewer for the organoid shapes produced
by this pipeline. Once the embedding has been computed, it shows the organoids as
a 2D scatter plot; clicking a point opens that organoid's 3D mesh in a separate
window for free rotation and zoom.

It is self-contained: it needs only the embedding `.npz` and the original `.vtp`
meshes, with no other pipeline outputs. This makes it convenient for sharing —
the embedding and meshes are enough to explore the shapes without re-running
anything.

The notes below cover where the data lives and how to run the viewer, and are
written to be followed step by step (including by an AI assistant — pasting this
whole file gives it every path and command it needs).

---

## Inputs

The viewer requires three things:

1. **The code** — `inspect_embedding.py` and the `src/` folder beside it.
2. **An embedding file** — a single `.npz` (for example `hks_emb.npz`) holding
   the 2D scatter coordinates and the list of organoid IDs. It is read as-is;
   nothing in it needs to be opened or edited.
3. **The original meshes** — the `.vtp` files, one per organoid, as generated at
   the start of the pipeline.

By default no coefficient files, eigendecompositions, or transformed `.obj`
meshes are needed — the `.vtp` files are read directly. (An optional
`--source pipeline` mode does use those precomputed outputs for extra shape
overlays; see [Optional: `--source pipeline`](#optional---source-pipeline) below.)

---

## Data layout

The meshes come from three datasets, `main_dataset`, `sup_dataset`, and `pert2`,
kept in their original structure. The IDs in the embedding `.npz` refer directly
to that structure:

```
organoid-viewer/
├── inspect_embedding.py        # the viewer script
├── src/                        # helper code it imports — keep this whole folder
│   ├── utils.py
│   ├── fatemarkers.py
│   ├── meshharm.py
│   └── ...
├── environment.yml             # the conda environment (see below)
├── hks_emb.npz                 # the embedding file
└── Data/                       # passed as --data_root
    ├── main_dataset/
    │   ├── config.json         # optional; gives the fate markers friendly names
    │   └── vtp/
    │       ├── day1p5/         #   day1p5_A01_1.vtp, day1p5_A01_102.vtp, ...
    │       ├── day2/
    │       ├── day2p5/         #   day2p5_A04_67.vtp, ...
    │       ├── day3/
    │       ├── day3p5/
    │       ├── day4/
    │       ├── day4p5/
    │       └── day4p5-more/
    ├── sup_dataset/
    │   ├── config.json
    │   └── vtp/
    │       └── day4p5/         #   day4p5_B02_100.vtp, ...
    └── pert2/                  # perturbation dataset — drug-treated organoids
        ├── config.json
        └── vtp/
            ├── normal/         #   stem-ChirVpaD1_day4p5_C06_101.vtp, ta-Yapa_day4p5_B03_3.vtp, ...
            └── small/          #   abs-Iwp2_day4p5_G06_10.vtp, sec-DaptHi_day4p5_F02_103.vtp, sec-DaptIwp2iMekD1_day4p5_C04_1.vtp, ...
```

`main_dataset` and `sup_dataset` use the standard
`{dataset}/vtp/{timepoint}/{filename}.vtp` layout. `pert2` differs in two ways,
but needs no rearranging either:

- Its `vtp/` subfolders are a **shape-size split** (`normal` / `small`), not
  timepoints — every `pert2` organoid is day `4p5`.
- Its filenames are prefixed with the **drug condition**
  (`{condition}_day4p5_{well}_{label}.vtp`, e.g. `stem-ChirVpaD1_day4p5_C06_101.vtp`),
  whereas `main_dataset` / `sup_dataset` filenames are plain
  `{timepoint}_{well}_{label}.vtp`. The five conditions are `stem-ChirVpaD1` and
  `ta-Yapa` (in `normal`), and `abs-Iwp2`, `sec-DaptHi`, and `sec-DaptIwp2iMekD1`
  (in `small`).

`--data_root` simply points at the folder that contains all three dataset folders
(here, `Data`).

> Internally each mesh is located by filename, searching `--data_root`
> recursively, so any directory arrangement works as long as the `.vtp` files
> sit somewhere beneath it. The structure above is the expected one.

### How IDs map to files

Each organoid ID is the **dataset name** followed by the mesh's **filename stem**,
and resolves to a `.vtp` like this:

```
main_dataset_day2p5_A04_67           →  Data/main_dataset/vtp/day2p5/day2p5_A04_67.vtp
sup_dataset_day4p5_B02_100           →  Data/sup_dataset/vtp/day4p5/day4p5_B02_100.vtp
pert2_stem-ChirVpaD1_day4p5_C06_101  →  Data/pert2/vtp/normal/stem-ChirVpaD1_day4p5_C06_101.vtp
```

For `pert2`, the stem carries the drug condition (`stem-ChirVpaD1`, `abs-Iwp2`,
`sec-DaptHi`, …) as its first token. Note the `normal` / `small` subfolder is
**not** part of the ID:
meshes are located by filename, searched recursively under `--data_root`, so the
viewer finds the file whichever subfolder it sits in.

So the embedding only references meshes that already exist in the three dataset
folders; nothing needs to be renamed or copied.

---

## Environment setup (one time)

With [conda](https://docs.conda.io/en/latest/miniconda.html) (or mamba)
installed, run from the project folder:

```bash
conda env create -f environment.yml
conda activate organoid-viewer
```

This creates an environment named `organoid-viewer` containing only the packages
the viewer needs.

---

## Running the viewer

With the environment activated, from the project folder:

```bash
python inspect_embedding.py --embedding hks_emb.npz --data_root Data
```

- `--embedding` — path to the `.npz` embedding file.
- `--data_root` — the root data folder, `Data`. It is searched recursively for
  the `.vtp` files, so it only needs to sit above the dataset folders.

Two windows open:

- **Embedding window** — a 2D scatter plot, one dot per organoid. The "color by"
  buttons on the left recolor the dots — by timepoint, cluster, complexity,
  cell-type diversity, per-marker fate fraction, or **condition** (WT vs the
  `pert2` drug treatments). Clicking a dot loads that organoid's 3D mesh.
- **Organoid window** — the 3D mesh. Drag to rotate, scroll to zoom. A draggable
  **"Colour organoid by"** panel lists the per-vertex fate markers; click one to
  colour the surface by it, and the choice persists as you click through
  organoids.

Dragging the two windows apart keeps them from overlapping. Closing either window
quits.

### Optional: `--source pipeline`

By **default** the viewer reads meshes straight from the `.vtp` files
(`--source vtp`) and needs nothing else — this is the self-contained mode the
rest of this README assumes.

Passing `--source pipeline` instead loads precomputed per-organoid pipeline
outputs (`{stem}_coeffs.npz` + `{stem}_transformed_mesh.obj`, written by
`run_new_meshes.py`) and unlocks extra shape overlays: the **"Colour organoid
by"** panel gains an **HKS / shape** section (raw heat-kernel signatures at
several diffusion times, plus bag-of-features words) alongside the fate markers.

This requires having run that processing first, so those files exist under
`--data_root`; without them the meshes won't load. If you only have the `.vtp`
meshes, stay on the default `--source vtp`.

---

## Troubleshooting

- **`Indexed 0 .vtp files under ...`** — `--data_root` points at the wrong folder.
  It should point at the folder containing the `.vtp` files.
- **`no .vtp found for <id>`** after clicking a dot — that organoid's `.vtp` is
  not under `--data_root`, or its filename does not match the expected stem (e.g.
  the embedding expects `day2p5_A04_67.vtp`). The scatter and other organoids are
  unaffected.
- **`ModuleNotFoundError`** — the environment is not active; run
  `conda activate organoid-viewer` first.
- **Fate-marker names appear as raw column names** (e.g. `LGR5_intensity` rather
  than `lgr`) — harmless. Friendly names come from the `config.json` inside each
  dataset folder (`Data/main_dataset/config.json`, etc.); without it the raw
  names are shown.
