from tqdm import tqdm
from spharm import SpHarm
import numpy as np 
import os 
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

def run_spharm(mesh_path, save_path): 
    """Original spharm processing function"""
    try:
        print(f"Processing mesh: {mesh_path}")
        m = SpHarm()
        m.load_mesh_from_file(mesh_path)  
        m.align_with_pca() 
        m.compute_initial_parameterization()
        m.optimize(max_outer_iterations=100, primal_steps=100, verbose=False)
        clms = m.compute_sh_coefficients(lmax=15)
        m.save_results(save_path)
        print(f"Completed mesh: {mesh_path}")
        return True, f"Successfully processed {mesh_path}"
    except Exception as e:
        print(f"Failed mesh: {mesh_path} - Error: {str(e)}")
        return False, f"Error processing {mesh_path}: {str(e)}\n{traceback.format_exc()}"

def process_mesh_wrapper(args):
    """Wrapper function for multiprocessing"""
    mesh_path, save_path = args
    return run_spharm(mesh_path, save_path)

def collect_mesh_tasks(timepoints, zarr_names, wells, rounds, meshes, blacklist):
    """Collect all mesh processing tasks"""
    tasks = []
    
    for timepoint in timepoints:
        zarr_name = zarr_names[timepoint]
        for well_name in wells[timepoint]:
            round_name = rounds[timepoint][0]
            path = f"Data/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/"
            
            # Check if good_labels.npy exists
            labels_file = path + 'good_labels.npy'
            if not os.path.exists(labels_file):
                print(f"Labels file does not exist: {labels_file}")
                continue
                
            labels = np.load(labels_file).astype('int')
            mesh_name = meshes[timepoint][0]
            
            for label in labels:
                mesh_path = f"Data/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/meshes/{mesh_name}/{label}.stl"
                save_path = f"Data/fractal_output/{timepoint}/{zarr_name}/{well_name[0]}/{well_name[1:]}/{round_name}/meshes/{mesh_name}/{label}"
                
                # Check if mesh file exists and hasn't been processed yet
                if (os.path.exists(mesh_path) and 
                    not os.path.exists(save_path + '_clms.npy') and 
                    save_path not in blacklist):
                    tasks.append((mesh_path, save_path))
    
    return tasks

def run_parallel_spharm(n_processes=None, chunk_size=1):
    """
    Run SpHarm processing in parallel
    
    Args:
        n_processes: Number of processes to use (default: cpu_count())
        chunk_size: Number of tasks per chunk for multiprocessing
    """
    
    # Configuration data
    # timepoints = ['day1p5', 'day2', 'day2p5', 'day3', 'day3p5', 'day4', 'day4p5', 'day4p5-more']
    timepoints = ['day2']

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

    blacklist = [
        'Data/fractal_output/day4p5-more/r0.zarr/C/01/0_fused/meshes/nnorgb3/123', 
        'Data/fractal_output/day4p5/r0.zarr/B/06/0_fused/meshes/nnorgb3/133'
    ] 

    # Collect all tasks
    print("Collecting mesh processing tasks...")
    tasks = collect_mesh_tasks(timepoints, zarr_names, wells, rounds, meshes, blacklist)
    
    if not tasks:
        print("No tasks to process!")
        return
    
    print(f"Found {len(tasks)} meshes to process")
    
    # Set number of processes
    if n_processes is None:
        n_processes = min(cpu_count(), len(tasks))
    
    print(f"Using {n_processes} processes")
    
    # Process tasks in parallel
    successful = 0
    failed = 0
    
    with Pool(processes=n_processes) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_mesh_wrapper, tasks, chunksize=chunk_size),
            total=len(tasks),
            desc="Processing meshes"
        ))
    
    # Count results
    for success, message in results:
        if success:
            successful += 1
        else:
            failed += 1
            print(f"FAILED: {message}")
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tasks)}")

if __name__ == "__main__":
    # Run with default settings (uses all CPU cores)
    # run_parallel_spharm()
    
    # Alternative: specify number of processes and chunk size
    run_parallel_spharm(n_processes=6, chunk_size=2)