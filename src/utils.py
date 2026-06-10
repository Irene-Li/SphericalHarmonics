import os
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
    assert xy.ndim == 2 and xy.shape[1] == 2, f"xy must be (N,2), got {xy.shape}"
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