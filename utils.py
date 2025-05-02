import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt 
import vtk 

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