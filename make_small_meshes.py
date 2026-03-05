from ast import parse

from tqdm import tqdm
from src.fatemarkers import FateMarkers
import numpy as np 
import os 
import anndata as ad
import pandas as pd
from collections import defaultdict
import json 
import argparse
import igl 

def map_scalar_fields_to_new_mesh(V_old, F_old, V_new, S_old):
    V_old = V_old.astype(np.float64)
    if S_old.ndim == 1:
        S_old = S_old[:, np.newaxis]
    
    # 1. Find closest points
    _, I_face, C = igl.point_mesh_squared_distance(V_new, V_old, F_old)
    
    # 2. Get the vertices of the closest triangles
    v0 = V_old[F_old[I_face, 0]]
    v1 = V_old[F_old[I_face, 1]]
    v2 = V_old[F_old[I_face, 2]]
    
    # 3. Calculate Barycentric Coordinates
    B = igl.barycentric_coordinates_tri(C, v0, v1, v2)
    
    # 4. Interpolate (Vectorized version is much faster than a loop)
    # We use F_old[I_face, k] to get the VERTEX indices for the k-th corner of the face
    S_new = (S_old[F_old[I_face, 0]] * B[:, [0]] + 
             S_old[F_old[I_face, 1]] * B[:, [1]] + 
             S_old[F_old[I_face, 2]] * B[:, [2]]) 
             
    return S_new.squeeze()

def create_small_mesh(mesh_path, save_path, ts=[1, 4, 25, 10], rescale=True, lmax=8, target_size=2562, sec_cell_names=None, annotation_names=None): 
    m = FateMarkers()
    m.load_mesh_from_file(mesh_path) 
    m._refine_lgr5_marker(sec_cell_names=sec_cell_names) 
    m.align_with_pca() 
    mass_matrix = igl.massmatrix(m.v, m.f, igl.MASSMATRIX_TYPE_VORONOI)
    area = mass_matrix.diagonal().sum()
    if rescale: 
        m.v = m.v / np.sqrt(area) 
        # rescale fields by 75% percentile of nonzero components 
        for i in range(m.fields.shape[1]):
            nonzero_vals = m.fields[:, i][m.fields[:, i] > 0]
            if len(nonzero_vals) > 0:
                p75 = np.percentile(nonzero_vals, 75)
                if p75 > 0:
                    m.fields[:, i] = m.fields[:, i] / p75

    m.precompute_eigens(lmax=lmax)
    hks = m.compute_hks_for_new_times(new_ts=ts, coeffs=False)
    _, v_dec, f_dec, _, _ = igl.decimate(m.v, m.f, target_size) # Decimate to make meshes smaller (for testing purposes)

    correct_order = list(annotation_names.keys())
    indices = [m.field_names.index(annotation_names[name]) for name in correct_order]
    scalar_fields = np.hstack([hks, m.fields[:, indices]]) # (V_old, K) where K = num_hks + num_cell_fates
    scalar_fields_dec = map_scalar_fields_to_new_mesh(m.v, m.f, v_dec, scalar_fields) # (V_new, K)

    # Save geometry
    igl.write_triangle_mesh(f"{save_path}_mesh.obj", v_dec, f_dec)
    
    # Save data separately for easier loading later
    np.savez(f"{save_path}_data.npz", 
             scalars=scalar_fields_dec, 
             names=([f"hks_{t}" for t in ts] + correct_order))

def nested_dict():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def convert_to_dict(strings): 
    tree = nested_dict()
    for s in strings:
        parts = s.split('_')
        day = parts[0]
        well_letter = parts[1][0]
        well_number = parts[1][1:]
        number = int(parts[2])
        
        tree[day][well_letter][well_number].append(number)
    return tree 

    

if __name__ == "__main__":

    # parse arguments given the run script 
    parser = argparse.ArgumentParser(description="Make meshes smaller and do hks precomputation")
    parser.add_argument("folder_path", type=str, help="Path to the data folder (e.g. Data/20260211/)")
    parser.add_argument("save_path", type=str, help="Path to save the processed meshes (e.g. Data/small_meshes/)")
    parser.add_argument("--discard_labels", type=str, default=None,
                        help="Path to a .npy file of label names to discard (e.g. Data/20260211/discard_labels.npy)")
    args = parser.parse_args()

    folder_path = args.folder_path
    save_path = args.save_path
    if not folder_path.endswith('/'):
        folder_path += '/'

    with open(f"{folder_path}config.json", 'r') as f:
        cfg = json.load(f)

    timepoints = cfg['timepoints']
    zarr_names = cfg['zarr_names']
    wells = cfg['wells']
    mesh_name = cfg['mesh_name']    
    rounds = cfg['rounds']
    sec_cell_names = cfg['sec_cell_names']
    annotation_names = cfg['annotation_names']

    ts = np.exp(np.linspace(np.log(0.01), np.log(1), 16))

    # load the discard labels if provided
    discard_tree = None
    if args.discard_labels is not None:
        discard_labels = np.load(args.discard_labels)
        discard_tree = convert_to_dict(discard_labels)

    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint]
            path = f"{folder_path}/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            mesh_path = f"{path}/meshes/{mesh_name}/"
            all_labels = [int(l.split('.')[0]) for l in os.listdir(mesh_path)]

            if discard_tree is not None:
                to_discard = discard_tree[timepoint][well_name[0]][well_name[1:]]
                good_labels = [l for l in all_labels if l not in to_discard]
            else:
                good_labels = all_labels

            if not os.path.exists(f"{path}/fm_data/"):
                os.mkdir(f"{path}/fm_data/")
    
            np.save(f"{path}/fm_data/good_labels.npy", np.array(good_labels))
            print(f"Processing {len(good_labels)} meshes for {timepoint} {well_name}...")
            for label in tqdm(good_labels):
                mesh_file = f"{mesh_path}/{label}.vtp"
                if not os.path.exists(mesh_file):
                    print(f"Mesh file does not exist: {mesh_file}")
                    continue
                save_loc = f"{save_path}/{timepoint}_{well_name}_{label}"
                try: 
                    create_small_mesh(mesh_file, save_loc,ts=ts, sec_cell_names=sec_cell_names, annotation_names=annotation_names)
                except Exception as e:
                    print(f"Error processing mesh {mesh_file}: {e}")
                    continue 


