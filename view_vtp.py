#!/usr/bin/env python
"""View a .vtp mesh coloured by its per-vertex (point-data) arrays.

Interactive window (default) — rotate with the mouse, cycle fields with n / p:
    python view_vtp.py sim/encodings_main_dataset_day4p5_A06_11.vtp
    python view_vtp.py <file.vtp> --field hks_t25

Static montage to a PNG (offscreen, good for a quick look / sharing):
    python view_vtp.py <file.vtp> --screenshot out.png
    python view_vtp.py <file.vtp> --screenshot out.png --fields fate_aldob hks_t25 bof_word0

Run with the scmpx env (has vtk + matplotlib):
    /opt/homebrew/anaconda3/envs/scmpx/bin/python view_vtp.py <file.vtp>
"""
import argparse
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read(path):
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def array_names(pd):
    a = pd.GetPointData()
    return [a.GetArrayName(i) for i in range(a.GetNumberOfArrays())]


def _lut():
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.667, 0.0)   # blue (low) -> red (high)
    lut.Build()
    return lut


def make_actor(pd, field, lut):
    rng = pd.GetPointData().GetArray(field).GetRange()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(field)
    mapper.SetScalarRange(rng if rng[1] > rng[0] else (rng[0], rng[0] + 1e-9))
    mapper.SetLookupTable(lut)
    mapper.InterpolateScalarsBeforeMappingOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor, rng


def render_rgb(pd, field, size=(520, 520)):
    lut = _lut()
    actor, rng = make_actor(pd, field, lut)
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(1, 1, 1)
    ren.ResetCamera()
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(*size)
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1]
    return arr, rng


def montage(pd, fields, out):
    import matplotlib.pyplot as plt
    ncol = min(len(fields), 2)
    nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow), squeeze=False)
    flat = axes.ravel()
    for ax, f in zip(flat, fields):
        arr, rng = render_rgb(pd, f)
        ax.imshow(arr)
        ax.set_title(f"{f}   [{rng[0]:.3g}, {rng[1]:.3g}]")
        ax.axis('off')
    for ax in flat[len(fields):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print("wrote", out)


def interactive(pd, field):
    names = array_names(pd)
    state = {'i': names.index(field) if field in names else 0}
    lut = _lut()
    actor, _ = make_actor(pd, names[state['i']], lut)
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0.1, 0.1, 0.15)
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle(names[state['i']])
    ren.AddActor2D(bar)
    rw = vtk.vtkRenderWindow()
    rw.AddRenderer(ren)
    rw.SetSize(900, 700)
    rw.SetWindowName(names[state['i']])
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    def set_field(name):
        r = pd.GetPointData().GetArray(name).GetRange()
        m = actor.GetMapper()
        m.SelectColorArray(name)
        m.SetScalarRange(r if r[1] > r[0] else (r[0], r[0] + 1e-9))
        bar.SetTitle(name)
        rw.SetWindowName(name)
        rw.Render()

    def keypress(obj, _ev):
        k = obj.GetKeySym()
        if k in ('n', 'p'):
            state['i'] = (state['i'] + (1 if k == 'n' else -1)) % len(names)
            set_field(names[state['i']])

    iren.AddObserver('KeyPressEvent', keypress)
    ren.ResetCamera()
    rw.Render()
    print("fields:", names)
    print("keys: n / p = next / previous field,  q = quit")
    iren.Start()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('vtp')
    ap.add_argument('--field', default=None, help='point-data array to colour by')
    ap.add_argument('--screenshot', default=None, help='write a montage PNG instead of a window')
    ap.add_argument('--fields', nargs='+', default=None, help='arrays for the montage')
    a = ap.parse_args()

    pd = read(a.vtp)
    names = array_names(pd)
    if not names:
        raise SystemExit("no point-data arrays found in " + a.vtp)
    print(f"{a.vtp}: {pd.GetNumberOfPoints()} points, {len(names)} arrays")

    if a.screenshot:
        want = a.fields or ['fate_aldob', 'fate_ta', 'hks_t25', 'bof_word0']
        fields = [f for f in want if f in names] or names[:4]
        montage(pd, fields, a.screenshot)
    else:
        interactive(pd, a.field or names[0])
