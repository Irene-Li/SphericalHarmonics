from tqdm import tqdm
from spharm import SpHarm
import numpy as np 
import os 

def run_spharm(mesh_path, save_path): 
    m = SpHarm()
    m.load_mesh_from_file(mesh_path)  
    m.align_with_pca() 
    m.compute_initial_parameterization()
    m.optimize(max_outer_iterations=100, primal_steps=100, verbose=False)
    clms = m.compute_sh_coefficients(lmax=15)
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
        'day1p5': ['A01', 'A02', 'A03', 'A04', 'A06'],
        'day2': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
        'day2p5': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
        'day3': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'B02', 'B03'],
        'day3p5': ['A01', 'A02', 'A03', 'A04', 'B03'],
        'day4': ['A02', 'A03', 'A04', 'A05', 'A06', 'B01', 'B02'],
        'day4p5': ['A06', 'B06'],
        'day4p5-more': ['C01', 'C02', 'C03', 'C04', 'C06'],
    }

    rounds = {
        'day1p5': ['0_fused'],
        'day2': ['0_fused'],
        'day2p5': ['0_fused'],
        'day3': ['0_fused'],
        'day3p5': ['0_fused'],
        'day4': ['0_fused'],
        'day4p5': ['0_fused'],
        'day4p5-more': ['0_fused'],
    }

    meshes = {
        'day1p5': ['nnorg_linked'],
        'day2': ['nnorg_linked'],
        'day2p5': ['nnorg_linked'],
        'day3': ['nnorg_linked'],
        'day3p5': ['nnorg_linked'],
        'day4': ['nnorg_linked'],
        'day4p5': ['nnorgb3'],
        'day4p5-more': ['nnorgb3'],
    }

    tables = {
        'day1p5':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day2':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day2p5':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day3':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day3p5':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day4':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day4p5':['mesh_features', 'nnorgb3_expanded_features'],
        'day4p5-more':['mesh_features', 'nnorgb3_expanded_features'],
    }



    
    blacklist = [
    'Data/fractal_output/day4p5-more/r0.zarr/C/01/0_fused/meshes/nnorgb3/123', 
    'Data/fractal_output/day4p5/r0.zarr/B/06/0_fused/meshes/nnorgb3/133'] 

    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint][0]
            path = f"Data/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            labels = np.load(path + 'good_labels.npy').astype('int')
            mesh_name = meshes[timepoint][0]
            print(path) 
            for label in labels:
                mesh_path = f"Data/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/meshes/{mesh_name}/{label}.stl"
                if not os.path.exists(mesh_path):
                    print(f"Mesh file does not exist: {mesh_path}")
                    continue
                save_path = f"Data/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/meshes/{mesh_name}/{label}"
                if not os.path.exists(save_path + '_clms.npy') and save_path not in blacklist:
                    print(f"Processing mesh: {mesh_path}")
                    run_spharm(mesh_path, save_path)

