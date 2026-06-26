import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
import vtk


# ---------------------------------------------------------------------------
# Dimensionality-reduction embeddings: save / load
# ---------------------------------------------------------------------------

def save_embedding(out_dir, name, xy, ids, method, **point_fields):
    """Save a 2D embedding keyed by organoid id.

    Args:
        out_dir: directory to write into (created if missing).
        name:    base filename, e.g. 'percentages_emb' -> {out_dir}/{name}.npz
        xy:      (N, 2) embedding coordinates.
        ids:     (N,) organoid id strings, aligned row-wise with xy.
        method:  str, the reduction method used ('umap'/'tsne'/'pca'/'phate').
        **point_fields: optional (N,) per-point arrays for colouring the
                        scatter (e.g. times=, l_cross=, areas=).

    Returns the written path.
    """
    xy = np.asarray(xy)
    ids = np.asarray(ids)
    assert xy.ndim == 2 and xy.shape[1] >= 2, f"xy must be (N,2+), got {xy.shape}"
    assert xy.shape[0] == ids.shape[0], (
        f"xy/ids length mismatch: {xy.shape[0]} vs {ids.shape[0]}")
    for k, v in point_fields.items():
        v = np.asarray(v)
        assert v.shape[0] == ids.shape[0], (
            f"point field '{k}' length {v.shape[0]} != n_ids {ids.shape[0]}")
        point_fields[k] = v

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(path, xy=xy, ids=ids, method=str(method), name=str(name), **point_fields)
    return path


def load_embedding(path):
    """Load an embedding saved by save_embedding into a plain dict."""
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    # scalars stored as 0-d arrays -> python str
    for k in ("method", "name"):
        if k in out:
            out[k] = str(out[k])
    return out


# ---------------------------------------------------------------------------
# Organoid id -> on-disk mesh paths
# ---------------------------------------------------------------------------

def _parse_uid(uid):
    """'day3p5_A01_42' -> (timepoint, well, label)."""
    timepoint, well, label = uid.split("_")
    return timepoint, well, label


def _well_dir(uid, cfg):
    """Common '{tp}/{zarr}/{well[0]}/{well[1:]}/{round}' fragment for a uid."""
    timepoint, well, label = _parse_uid(uid)
    zarr_name = cfg["zarr_names"][timepoint]
    round_name = cfg["rounds"][timepoint]
    return f"{timepoint}/{zarr_name}/{well[0]}/{well[1:]}/{round_name}", label


def organoid_vtp_path(data_path, cfg, uid):
    """Path to the original .vtp mesh (carries per-vertex fate fields)."""
    frag, label = _well_dir(uid, cfg)
    mesh_name = cfg["mesh_name"]
    return f"{data_path}/fractal_output/{frag}/meshes/{mesh_name}/{label}.vtp"


def vtp_flat_path(data_path, cfg, label_uid):
    """Mesh path for the 'vtp_flat' layout: {data}/{vtp_dir}/{timepoint}/{label_uid}.vtp."""
    timepoint = label_uid.split("_")[0]
    return f"{data_path}/{cfg.get('vtp_dir', 'vtp')}/{timepoint}/{label_uid}.vtp"


def vtp_flat_obj_path(data_path, cfg, label_uid):
    """PCA-transformed mesh (.obj) for a vtp_flat organoid."""
    timepoint = label_uid.split("_")[0]
    return f"{data_path}/{cfg.get('vtp_dir', 'vtp')}/{timepoint}/fm_data/{label_uid}_transformed_mesh.obj"


def vtp_flat_coeffs_path(data_path, cfg, label_uid):
    """Saved coefficients (.npz) for a vtp_flat organoid."""
    timepoint = label_uid.split("_")[0]
    return f"{data_path}/{cfg.get('vtp_dir', 'vtp')}/{timepoint}/fm_data/{label_uid}_coeffs.npz"


def organoid_obj_path(data_path, cfg, uid):
    """Path to the saved PCA-transformed shape (.obj, geometry only)."""
    frag, label = _well_dir(uid, cfg)
    return f"{data_path}/fractal_output/{frag}/fm_data/{label}_transformed_mesh.obj"


def organoid_coeffs_path(data_path, cfg, uid):
    """Path to the saved per-organoid coefficients/eigendecomposition npz."""
    frag, label = _well_dir(uid, cfg)
    return f"{data_path}/fractal_output/{frag}/fm_data/{label}_coeffs.npz"

def read_stl(filename): 
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    v = np.array(polydata.GetPoints().GetData())

    # Extract faces (cells)
    cells = polydata.GetPolys()
    cells.InitTraversal()

    # Create an array to store the faces
    n_faces = polydata.GetNumberOfPolys()
    faces = np.zeros((n_faces, 3), dtype=np.int64)

    # Extract all faces
    for i in range(n_faces):
        cell = vtk.vtkIdList()
        cells.GetNextCell(cell)
        for j in range(3):  # Assuming triangular faces
            faces[i, j] = cell.GetId(j)
    f = faces 
    return v, f 


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x) % (2*np.pi)
    return r, theta, phi

def sph2latlon(theta, phi):
    lat = (np.pi/2 - theta)*180/np.pi
    lon = phi*180/np.pi
    return lat, lon

def latlon2sph(lat, lon):
    theta = np.pi/2 - lat*np.pi/180
    phi = lon*np.pi/180
    return theta, phi


def draw_2d_surface(r, theta_grid, phi_grid):
    plt.tricontourf(phi_grid.flatten(), theta_grid.flatten(), r.flatten(), 100)
    plt.gca().invert_yaxis()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    plt.colorbar()
    plt.show()  


def draw_2d_scatter(r, theta_grid, phi_grid, title='r'):
    plt.scatter(phi_grid.flatten(), theta_grid.flatten(), c=r.flatten(),
                s=2, cmap='viridis', alpha=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    plt.title(title)
    plt.colorbar()
    plt.show()  




def draw_3d_surface(x, y, z, r):
    fig = go.Figure(data=[
            go.Surface(
                x=x, y=y, z=z,
                surfacecolor=r,
                colorscale='Viridis',
                opacity=1,
                showscale=True,
            )
        ])

    fig.update_layout(
            scene={
                'xaxis_title': 'X',
                'yaxis_title': 'Y',
                'zaxis_title': 'Z',
                'aspectmode': 'data',
            },
            width=800,
            height=800,
            margin=dict(l=0, r=0, b=0, t=40)
        )
    return fig 


def plot_3d_scatter(x, y, z, r):
    fig = go.Figure(data=[go.Scatter3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color=r.flatten(),  # color points by z-value
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # Update layout
    fig.update_layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # maintain aspect ratio
        ),
        width=800,
        height=800
    )

    return fig


# ---------------------------------------------------------------------------
# HKS shape features  (shared by optimize_hks_weights.py and dim_red.ipynb)
# ---------------------------------------------------------------------------

UID_GROUPS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Data", "uid_groups.json")
)


def load_uid_groups(path=UID_GROUPS_FILE):
    """Load hand-picked organoid groups from JSON as list[list[str]]."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"uid groups json not found: {path}. Expected a list of UID groups."
        )
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(g, list) for g in data):
        raise ValueError(
            f"Invalid UID groups format in {path}. Expected list[list[str]]."
        )
    for gi, grp in enumerate(data):
        if not all(isinstance(uid, str) for uid in grp):
            raise ValueError(
                f"Invalid UID type in group {gi} in {path}. Expected strings only."
            )
    return data


# Hand-picked organoid groups = distinct shape categories. Used as supervision
# when learning HKS weights (within-group close, between-group far) and as an
# overlay when visualising embeddings. Source of truth: Data/uid_groups.json
UID_GROUPS = load_uid_groups()


def load_power_spectrum(master_path, mode_cut=8,
                        vocab_key="hks_bof_coeffs__kmeans_variable"):
    """HKS bag-of-features coeffs -> normalized power spectrum (N, mode_cut, n_vocab).

    Per degree l, sums the squared coefficients in the l-shell, then divides each
    vocab channel by its max across samples and modes. `mode_cut` keeps only the
    first that-many modes (degrees); higher modes capture fine surface detail and
    are dominated by small imperfections, so dropping them denoises the shape.
    Pass mode_cut=None to keep all modes. Returns (ids, hps).
    """
    m = np.load(master_path, allow_pickle=True)
    ids = m["ids"].astype(str)
    bof = m[vocab_key]                                  # (N, n_modes^2, n_vocab)
    nd = int(np.sqrt(bof.shape[1]))
    hps = np.zeros((bof.shape[0], nd, bof.shape[2]))
    for n in range(nd):
        hps[:, n, :] = np.sum(bof[:, n ** 2:(n + 1) ** 2, :] ** 2, axis=1)
    if mode_cut is not None:
        hps = hps[:, :mode_cut, :]
    hps /= np.max(hps, axis=(0, 1), keepdims=True)
    return ids, hps


def apply_hks_weights(hps, weights):
    """Weight an (N, M, V) power spectrum by a (M, V) matrix and flatten to (N, M*V).

    Euclidean distance in the returned feature space is the weighted HKS shape
    distance used for clustering / dimensionality reduction.
    """
    w = np.asarray(weights)
    assert w.shape == hps.shape[1:], \
        f"weights {w.shape} != power-spectrum modes×vocab {hps.shape[1:]}"
    return (hps * w[np.newaxis, :, :]).reshape(hps.shape[0], -1)


def group_pair_indices(uid_groups, id_to_row):
    """Resolve hand-picked groups to row indices -> within/between pair arrays.

    Returns (within_i, within_j, between_i, between_j). Ids missing from
    `id_to_row` are skipped; ids repeated inside a group are de-duped so no
    zero-distance self-pair is created.
    """
    from itertools import combinations
    grp_rows = [list(dict.fromkeys(id_to_row[u] for u in g if u in id_to_row))
                for g in uid_groups]
    wi, wj = [], []
    for rr in grp_rows:
        for a, b in combinations(rr, 2):
            wi.append(a); wj.append(b)
    oi, oj = [], []
    for p in range(len(grp_rows)):
        for q in range(p + 1, len(grp_rows)):
            for a in grp_rows[p]:
                for b in grp_rows[q]:
                    oi.append(a); oj.append(b)
    return np.array(wi), np.array(wj), np.array(oi), np.array(oj)


def _fate_from_csv(data_root, ids, datasets, csv_name):
    """Slow path: compute per-organoid fate fractions from the per-cell CSVs."""
    import pandas as pd
    perc_dfs = {}
    for ds in sorted(set(datasets)):
        path = os.path.join(data_root, ds, csv_name)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            header = fh.readline().strip().split(",")
        excl = [c for c in header if c.endswith("_exclusive")]
        df = pd.read_csv(path, usecols=["label_uid", "projected_to_mesh"] + excl,
                         low_memory=False)
        df = df[df["projected_to_mesh"].astype(str) == "True"]
        pos = (df[excl] > 0).astype(float)
        pos["label_uid"] = df["label_uid"].values
        perc_dfs[ds] = pos.groupby("label_uid")[excl].mean()
    if not perc_dfs:
        return np.zeros((len(ids), 0)), []
    ct_cols = sorted(set.intersection(*[set(df.columns) for df in perc_dfs.values()]))
    col_names = [c.split(".")[0] for c in ct_cols]
    P = np.zeros((len(ids), len(col_names)))
    for i, (full_id, ds) in enumerate(zip(ids, datasets)):
        bare = full_id[len(ds) + 1:]
        if ds in perc_dfs and bare in perc_dfs[ds].index:
            P[i] = perc_dfs[ds].loc[bare, ct_cols].values
    return P, col_names


def load_fate_percentages(data_root, ids, datasets,
                          csv_name="cell_features_class_with_projection_exclusive.csv",
                          cache_path="Data/fate_percentages.npz", rebuild=False):
    """Per-organoid fate-marker fractions aligned to `ids` -> (percentages (N,K), col_names).

    The source CSVs are per-CELL (millions of rows), so parsing them is slow. The
    computed per-organoid fractions are cached to `cache_path` (npz). A cached
    result is reused when it is newer than every source CSV and already covers all
    requested ids; otherwise it is recomputed and the cache refreshed. Pass
    rebuild=True (or cache_path=None) to force the slow CSV path.
    """
    ids = np.asarray(ids).astype(str)
    csv_paths = [os.path.join(data_root, ds, csv_name) for ds in sorted(set(datasets))]
    csv_paths = [p for p in csv_paths if os.path.exists(p)]

    if cache_path and not rebuild and os.path.exists(cache_path):
        fresh = not csv_paths or (
            os.path.getmtime(cache_path) >= max(os.path.getmtime(p) for p in csv_paths))
        if fresh:
            z = np.load(cache_path, allow_pickle=True)
            pos = {u: i for i, u in enumerate(z["ids"].astype(str))}
            if all(u in pos for u in ids):                  # cache covers every requested id
                rows = [pos[u] for u in ids]
                return z["percentages"][rows], list(z["col_names"])

    P, col_names = _fate_from_csv(data_root, ids, datasets, csv_name)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, ids=ids, percentages=P, col_names=np.array(col_names))
    return P, col_names