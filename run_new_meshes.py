from tqdm import tqdm
from src.fatemarkers import FateMarkers
import numpy as np 
import os 
import anndata as ad
import pandas as pd
from collections import defaultdict
import json 
import argparse

def run_fatemarkers(mesh_path, save_path, sec_cell_names=None):
    m = FateMarkers()
    m.load_mesh_from_file(mesh_path)
    m._refine_lgr5_marker(sec_cell_names=sec_cell_names)
    m.align_with_pca()
    m.precompute_eigens(lmax=15)
    m.compute_coefficients()
    m.save_results(save_path)

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
    parser = argparse.ArgumentParser(description="Run FateMarkers on mesh data.")
    parser.add_argument("folder_path", type=str, help="Path to the data folder (e.g. Data/20260224/)")
    parser.add_argument("--discard_labels", type=str, default=None,
                        help="Path to a .npy file of label names to discard (e.g. Data/20260224/combined_labels_to_discard.npy)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a mesh if its _coeffs.npz and _transformed_mesh.obj already exist")
    args = parser.parse_args()

    folder_path = args.folder_path
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

    # load the discard labels if provided
    discard_tree = None
    if args.discard_labels is not None:
        discard_labels = np.load(args.discard_labels)
        discard_tree = convert_to_dict(discard_labels)

    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint]
            path = f"{folder_path}fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            mesh_path = f"{path}meshes/{mesh_name}/"
            all_labels = [int(l.split('.')[0]) for l in os.listdir(mesh_path)]

            if discard_tree is not None:
                to_discard = discard_tree[timepoint][well_name[0]][well_name[1:]]
                good_labels = [l for l in all_labels if l not in to_discard]
            else:
                good_labels = all_labels

            fm_data = f"{path}fm_data/"
            if not os.path.exists(fm_data):
                os.mkdir(fm_data)

            np.save(f"{fm_data}good_labels.npy", np.array(good_labels))

            if args.skip_existing:
                todo = [l for l in good_labels
                        if not (os.path.exists(f"{fm_data}{l}_coeffs.npz") and
                                os.path.exists(f"{fm_data}{l}_transformed_mesh.obj"))]
            else:
                todo = good_labels
            print(f"{timepoint} {well_name}: {len(todo)}/{len(good_labels)} meshes to process")
            if not todo:
                continue

            for label in tqdm(todo):
                mesh_file = f"{mesh_path}{label}.vtp"
                if not os.path.exists(mesh_file):
                    print(f"Mesh file does not exist: {mesh_file}")
                    continue
                save_loc = f"{fm_data}{label}"
                try:
                    # rerun_fatemarkers(mesh_file, save_loc, sec_cell_names=sec_cell_names)
                    run_fatemarkers(mesh_file, save_loc, sec_cell_names=sec_cell_names)
                except Exception as e:
                    print(f"Error processing mesh {mesh_file}: {e}")
                    continue 


