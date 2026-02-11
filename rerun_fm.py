from tqdm import tqdm
from src.fatemarkers import FateMarkers
import numpy as np 
import os 
import anndata as ad
import pandas as pd
from collections import defaultdict

def rerun_fatemarkers(mesh_path, save_path): 
    m = FateMarkers()
    m.load_results(save_path) 
    # load mesh to get the scalar fields and recompute the coefficients 
    m.load_mesh_from_file(mesh_path)  
    m._refine_lgr5_marker()
    m.align_with_pca()
    m.compute_coefficients(lmax=15)
    m.save_results(save_path)

def run_fatemarkers(mesh_path, save_path): 
    m = FateMarkers()
    m.load_mesh_from_file(mesh_path) 
    m._refine_lgr5_marker() 
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

    timepoints = ['day1p5', 'day2', 'day2p5', 'day3', 'day3p5', 'day4', 'day4p5', 'day4p5-more']

    zarr_names = {
        'day1p5': 'r0.zarr',
        'day2':'r0.zarr',
        'day2p5':'r0.zarr',
        'day3':'r0.zarr',
        'day3p5':'r0.zarr',
        'day4':'r0.zarr',
        'day4p5':'r0.zarr',
        'day4p5-more':'r0.zarr',
    }


    wells = {
        'day1p5': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
        'day2': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
        'day2p5': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
        'day3': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'B02', 'B03'],
        'day3p5': ['A01', 'A02', 'A03', 'A04', 'B03'],
        'day4': ['A02', 'A03', 'A04', 'A05', 'A06', 'B01', 'B02'],
        'day4p5': ['A06', 'B06'],
        'day4p5-more': ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'],
    }

    rounds = {
    'day1p5': ['0_fused_zillum_registered'],
    'day2': ['0_fused_zillum_registered'],
    'day2p5': ['0_fused_zillum_registered'],
    'day3': ['0_fused_zillum_registered'],
    'day3p5': ['0_fused_zillum_registered'],
    'day4': ['0_fused_zillum_registered'],
    'day4p5': ['0_fused_zillum_registered'],
    'day4p5-more': ['0_fused_zillum_registered'],
    }

    mesh_name = 'nnorg_linked_multi_annotated_class' 

    tables = {
        'day1p5':['cell_features'],
        'day2':['cell_features'],
        'day2p5':['cell_features'],
        'day3':['cell_features'],
        'day3p5':['cell_features'],
        'day4':['cell_features'],
        'day4p5':['cell_features'],
        'day4p5-more':['cell_features'],
    }

    # this is where data are saved 
    folder_path = 'Data/20251001/'
    discard_label_path = 'Data/combined_labels_to_discard.npy'
    discard_labels = np.load(discard_label_path)
    discard_tree = convert_to_dict(discard_labels)



    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint][0]
            path = f"{folder_path}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            to_discard = discard_tree[timepoint][well_name[0]][well_name[1:]]
            mesh_path = f"{folder_path}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/meshes/{mesh_name}/"
            all_labels = [int(l.split('.')[0]) for l in os.listdir(mesh_path)]
            good_labels = [l for l in all_labels if l not in to_discard]
            np.save(f"{path}/fm_data/good_labels.npy", np.array(good_labels))
            for label in tqdm(good_labels):
                mesh_file = f"{mesh_path}/{label}.vtp"
                if not os.path.exists(mesh_file):
                    print(f"Mesh file does not exist: {mesh_file}")
                    continue
                save_path = f"{folder_path}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/fm_data"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_loc = f"{save_path}/{label}"
                try: 
                    run_fatemarkers(mesh_file, save_loc)
                except Exception as e:
                    print(f"Error processing mesh {mesh_file}: {e}")
                    continue 


