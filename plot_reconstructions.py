"""SPHARM reconstructions of a few organoids at increasing order, with their complexity curves.

For each organoid (rows), its decimated mesh is rebuilt on its cached spherical map
(compute_master_npz.SPHARM_MAP_DIR/<uid>.npz) from x, y, z expanded in spherical harmonics up to
each ℓ_max in --lmax (columns), the same reconstruction compute_master_npz.compute_complexity_errors
uses (src.spharm.SpHarm.complexity_errors). Next to the meshes are the complexity score
(master.npz l_cross_values) and the area-normalised RMSE vs ℓ_max curves (master.npz
complexity_errors) with the l_cross threshold and the crossing points.

Meshes are rendered with VTK off screen: smooth Phong shading under a studio light kit, in the
PCA frame viewed down the shortest axis, all at one physical scale. They are tinted with each
organoid's curve colour. Outputs go to --out_dir (default Figures/reconstructions):
  - reconstructions.png / .pdf: the full figure,
  - renders/org-{i}_lmax{l}.png: each mesh render alone on a transparent background, for slides,
  - stl/org-{i}_lmax{l}.stl: each reconstructed mesh (binary STL), centred, in the PCA frame, in
    the fm_data mesh units (10 µm).

    python plot_reconstructions.py
    python plot_reconstructions.py --uids UID1 UID2 UID3 --lmax 1 2 4 6 14
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import vtk
from matplotlib.colors import to_rgb
from PIL import Image
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

import igl

from compute_master_npz import L_CROSS_THRESHOLD, SPHARM_MAP_DIR
from src.spharm import SpHarm

DEFAULT_UIDS = ['sup_dataset_day4p5_B02_8',            # three-armed (group 2), high complexity
                'main_dataset_day4_A02_42',            # medium
                'main_dataset_day4p5-more_C04_127']    # near-spherical
COLOURS = ['#e8894a', '#5fa85a', '#5b8fd6']            # org-1, org-2, org-3 (as the reference)


def reconstructions(uid, lmaxs):
    """{lmax: vertices} of the SPHARM reconstruction on the cached spherical map, plus faces,
    in the map's PCA frame, and the reconstruction error at each lmax."""
    path = os.path.join(SPHARM_MAP_DIR, f"{uid}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{uid}: no spherical map at {path} (run compute_master_npz.py on it)")
    sh, _ = SpHarm.load_map(path)
    v, f = sh.v, sh.f
    lat, lon, _ = sh._latlon_weights()
    rec = {l: np.stack([sh.expand(v[:, i], l).expand(lat=lat, lon=lon) for i in range(3)], axis=1)
           for l in lmaxs}
    # sanity: the same reconstruction error as the pipeline's complexity_errors
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI).diagonal()
    err = {l: np.sqrt(np.sum(M * ((v - rec[l]) ** 2).sum(1)) / M.sum()) / np.sqrt(M.sum())
           for l in lmaxs}
    centre = v.mean(0)
    return {l: r - centre for l, r in rec.items()}, f, err


def polydata(v, f):
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(v, float), deep=True))
    pd.SetPoints(pts)
    cells = np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(f), numpy_to_vtkIdTypeArray(cells, deep=True))
    pd.SetPolys(ca)
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(pd); nrm.SplittingOff(); nrm.ConsistencyOn(); nrm.AutoOrientNormalsOn()
    nrm.Update()
    return nrm.GetOutput()


def write_stl(path, v, f):
    """Binary STL of a mesh, with its triangles wound consistently."""
    w = vtk.vtkSTLWriter()
    w.SetFileName(path); w.SetFileTypeToBinary(); w.SetInputData(polydata(v, f)); w.Write()


def render(v, f, colour, R, px=500, supersample=3, ambient=0.32, diffuse=0.68, specular=0.2,
           specular_power=28, key_to_fill=1.5, tint=0.18):
    """RGBA render (transparent background) of a mesh viewed down +z, covering ±R mesh units.
    Soft studio look: generous ambient, low key-to-fill ratio (gentle shadows), a soft gloss,
    and the base colour mixed `tint` of the way towards white."""
    big = px * supersample
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1); ren.SetBackgroundAlpha(0.0)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(polydata(v, f))
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    p = actor.GetProperty()
    p.SetColor(*(np.array(to_rgb(colour)) * (1 - tint) + tint)); p.SetInterpolationToPhong()
    p.SetAmbient(ambient); p.SetDiffuse(diffuse); p.SetSpecular(specular); p.SetSpecularPower(specular_power)
    p.SetSpecularColor(1, 1, 1)
    ren.AddActor(actor)
    kit = vtk.vtkLightKit()
    kit.SetKeyLightIntensity(0.8); kit.SetKeyToFillRatio(key_to_fill); kit.SetKeyToBackRatio(2.5)
    kit.AddLightsToRenderer(ren)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn(); cam.SetPosition(0, 0, 10 * R); cam.SetFocalPoint(0, 0, 0)
    cam.SetViewUp(0, 1, 0); cam.SetParallelScale(R); ren.ResetCameraClippingRange()
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1); win.SetAlphaBitPlanes(1); win.SetMultiSamples(0)
    win.AddRenderer(ren); win.SetSize(big, big); win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win); grab.SetInputBufferTypeToRGBA(); grab.ReadFrontBufferOff(); grab.Update()
    img = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(big, big, 4)[::-1]
    win.Finalize()
    # supersampled -> smooth silhouettes; resize premultiplied so edges don't pick up the background
    out = Image.fromarray(np.ascontiguousarray(img)).convert('RGBa').resize((px, px), Image.LANCZOS)
    return np.asarray(out.convert('RGBA'))


def on_white(img):
    """Flatten an RGBA render onto white (matplotlib resampling of transparent edges leaves a halo)."""
    a = img[..., 3:4] / 255.0
    return (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uids", nargs='+', default=DEFAULT_UIDS)
    ap.add_argument("--lmax", nargs='+', type=int, default=[1, 2, 4, 6, 14])
    ap.add_argument("--master", default="Data/npz/master.npz")
    ap.add_argument("--out_dir", default="Figures/reconstructions")
    args = ap.parse_args()
    colours = (COLOURS * len(args.uids))[:len(args.uids)]

    m = np.load(args.master, allow_pickle=True)
    row = {u: i for i, u in enumerate(m['ids'].astype(str))}
    E, L = m['complexity_errors'], m['l_cross_values']

    recs, faces = [], []
    for u in args.uids:
        r, f, err = reconstructions(u, args.lmax)
        stored = {l: E[row[u], l] for l in args.lmax if l < E.shape[1]}
        assert all(np.isclose(err[l], s, rtol=1e-4) for l, s in stored.items()), (u, err, stored)
        recs.append(r); faces.append(f)
        print(f"{u:42s} l_cross {L[row[u]]:.2f} | error at ℓ_max {args.lmax}: "
              + ' '.join(f"{err[l]:.4f}" for l in args.lmax))

    # --- renders: one physical scale for every panel ----------------------------------------
    R = max(np.abs(v[:, :2]).max() for r in recs for v in r.values()) * 1.04
    os.makedirs(f"{args.out_dir}/renders", exist_ok=True)
    os.makedirs(f"{args.out_dir}/stl", exist_ok=True)
    imgs = []
    for i, (r, f, col) in enumerate(zip(recs, faces, colours)):
        imgs.append([])
        for l in args.lmax:
            img = render(r[l], f, col, R)
            Image.fromarray(img).save(f"{args.out_dir}/renders/org-{i + 1}_lmax{l}.png")
            imgs[-1].append(img)
            write_stl(f"{args.out_dir}/stl/org-{i + 1}_lmax{l}.stl", r[l], f)

    # --- figure -------------------------------------------------------------------------------
    plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'custom',
                         'mathtext.it': 'Arial:italic', 'mathtext.rm': 'Arial'})
    n, k = len(args.uids), len(args.lmax)
    cell, label_w, score_w, gap = 1.15, 0.85, 1.1, 0.25        # inches
    plot_w, top, bottom = 4.4, 0.75, 0.15
    W = label_w + k * cell + score_w + gap + plot_w + 0.35
    H = top + n * cell + bottom
    fig = plt.figure(figsize=(W, H), facecolor='white')
    fx, fy = (lambda x: x / W), (lambda y: y / H)

    for i in range(n):
        y0 = H - top - (i + 1) * cell
        fig.text(fx(label_w - 0.12), fy(y0 + cell / 2), f"org-{i + 1}", ha='right', va='center',
                 fontsize=15, fontweight='bold', color=colours[i])
        for j in range(k):
            ax = fig.add_axes([fx(label_w + j * cell), fy(y0), fx(cell), fy(cell)])
            ax.imshow(on_white(imgs[i][j]), interpolation='antialiased'); ax.axis('off')
        # complexity score box
        sx = label_w + k * cell + score_w / 2
        fig.text(fx(sx), fy(y0 + cell / 2), f"{L[row[args.uids[i]]]:.2f}", ha='center', va='center',
                 fontsize=15, color=colours[i],
                 bbox=dict(boxstyle='square,pad=0.45', fc='white', ec='#222222', lw=1.0))
    for j, l in enumerate(args.lmax):
        fig.text(fx(label_w + (j + 0.5) * cell), fy(H - top + 0.08), rf"$\ell_{{\max}} = {l}$",
                 ha='center', va='bottom', fontsize=13)
    fig.text(fx(label_w + 0.05), fy(H - 0.12), 'reconstruction to harmonic order:', ha='left', va='top',
             fontsize=14)
    fig.text(fx(label_w + k * cell + score_w / 2), fy(H - 0.12), 'complexity\nscore', ha='center',
             va='top', fontsize=13, linespacing=1.0)

    # RMSE vs lmax with the threshold and each organoid's crossing
    px0 = label_w + k * cell + score_w + gap + 0.55
    ax = fig.add_axes([fx(px0), fy(bottom + 0.42), fx(plot_w - 0.55), fy(n * cell - 0.6)])
    ls = np.arange(E.shape[1])
    ax.axhline(L_CROSS_THRESHOLD, color='#e05a4f', lw=1.4, label=f'threshold {L_CROSS_THRESHOLD:g}', zorder=1)
    for i, u in enumerate(args.uids):
        ax.plot(ls, E[row[u]], 'o-', color=colours[i], lw=1.6, ms=4.5, label=f'org-{i + 1}', zorder=3)
        lc = L[row[u]]
        ax.axvline(lc, color=colours[i], lw=0.9, ls=':', zorder=2)
        ax.annotate(f"{lc:.2f}", xy=(lc, 1), xycoords=('data', 'axes fraction'), xytext=(0, 14),
                    textcoords='offset points', ha='center', va='bottom', fontsize=11, color=colours[i],
                    arrowprops=dict(arrowstyle='->', color=colours[i], lw=1.0, shrinkA=1, shrinkB=0))
    ax.annotate(r'$\ell_{\mathrm{cross}}$ =', xy=(0, 1), xycoords='axes fraction', xytext=(-4, 14),
                textcoords='offset points', ha='right', va='bottom', fontsize=12)
    ax.set_yscale('log'); ax.set_xticks(ls); ax.set_xlim(-0.3, ls[-1] + 0.3)
    ax.set_xlabel(r'max harmonic order $\ell_{\max}$', fontsize=11)
    ax.set_ylabel('area-normalized RMSE', fontsize=11)
    ax.grid(axis='y', which='major', color='#dddddd', lw=0.6, ls='--')
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=True, framealpha=0.95, edgecolor='#cccccc', loc='upper right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    for ext in ('png', 'pdf'):
        fig.savefig(f"{args.out_dir}/reconstructions.{ext}", dpi=300, facecolor='white')
    print(f"saved {args.out_dir}/reconstructions.png/.pdf and {n * k} renders and STLs in {args.out_dir}/renders/ and stl/")


if __name__ == "__main__":
    main()
