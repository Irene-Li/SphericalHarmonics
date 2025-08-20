import numpy as np 
import os 
import anndata as ad
import pandas as pd


def filter_out_organoids(path, feature_table_name, mesh_table_name, save_path, tol=1):
    adata = ad.read_zarr(path + feature_table_name)
    df = adata.to_df()
    m1 = (df["C01.min_intensity"]<tol) 

    adata = ad.read_zarr(path + mesh_table_name)
    df = adata.to_df()
    m2 = (df["sphericity"] > 1.3)
    df_labels = adata.obs
    good_labels = df_labels["label"][~(m1|m2)].to_numpy().astype(int)

    np.save(save_path + "/good_labels.npy", good_labels)


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
        'day4p5': ['nnorg_linked'],
        'day4p5-more': ['nnorg_linked'],
    }

    tables = {
        'day1p5':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day2':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day2p5':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day3':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day3p5':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day4':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day4p5':['mesh_features', 'nnorg_linked_expanded1_features'],
        'day4p5-more':['mesh_features', 'nnorg_linked_expanded1_features'],
    }

    save_folder_path = 'Data/20250818/fractal_output/'
    save_extra = '_zillum_registered'

    # filter out organoids based on mesh segmentation quality 
    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint][0]
            path = f"Data/mesh_dataset/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            feature_table_name = f"tables/{tables[timepoint][1]}"
            mesh_table_name = f"tables/{tables[timepoint][0]}"
            save_path = f"{save_folder_path}/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}{save_extra}/"
            filter_out_organoids(path, feature_table_name, mesh_table_name, save_path, tol=1)