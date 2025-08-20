from tqdm import tqdm
from src.fatemarkers import FateMarkers
import numpy as np 
import os 
import anndata as ad
import pandas as pd

def run_fatemarkers(mesh_path, save_path): 
    m = FateMarkers()
    m.load_mesh_from_file(mesh_path)  
    m.align_with_pca() 
    m.compute_coefficients(lmax=15)
    m.save_results(save_path)

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

    meshes = {
        'day1p5': ['nnorg_linked_multi_annotated'],
        'day2': ['nnorg_linked_multi_annotated'],
        'day2p5': ['nnorg_linked_multi_annotated'],
        'day3': ['nnorg_linked_multi_annotated'],
        'day3p5': ['nnorg_linked_multi_annotated'],
        'day4': ['nnorg_linked_multi_annotated'],
        'day4p5': ['nnorg_linked_multi_annotated'],
        'day4p5-more': ['nnorg_linked_multi_annotated'],
    }

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

    folder_path = 'Data/20250818/fractal_output/'
    # blacklist = []

    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint][0]
            path = f"{folder_path}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            labels = np.load(path + 'good_labels.npy').astype('int')
            mesh_name = meshes[timepoint][0]
            print(path) 
            for label in tqdm(labels):
                mesh_path = f"{folder_path}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/meshes/{mesh_name}/{label}.vtp"
                if not os.path.exists(mesh_path):
                    print(f"Mesh file does not exist: {mesh_path}")
                    continue
                save_path = f"{folder_path}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/fm_data"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = f"{save_path}/{label}"
                if not os.path.exists(save_path + '_coeffs.npz'):
                    try: 
                        run_fatemarkers(mesh_path, save_path)
                    except Exception as e:
                        print(f"Error processing mesh {mesh_path}: {e}")
                        continue 

