from tqdm import tqdm
import os 

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

    # this is where data are saved 
    folder_path2 = 'Data/20250818/fractal_output/'
    folder_path = 'Data/20251001/'

    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint][0]
            copy_path = f"{folder_path}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/fm_data"
            new_path = f"{folder_path2}{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/fm_data"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            # move entire folder  
            print('moving..')
            os.system(f"cp {copy_path}/* {new_path}")


