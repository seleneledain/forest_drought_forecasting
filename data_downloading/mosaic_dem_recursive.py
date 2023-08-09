import os
import rasterio
from rasterio.merge import merge
import glob


def mosaic_dem_recursive(folder_path, n_sub, output_folder):
    """Mosaic/merge the DEM tiles together in a recursive manner. First, n_sub files will be merged to form intermediate tiles. 
    Then these tiles will be merged to produce the final DEM.

    Args:
        folder_path (str): path to where tiles are stored.
        n_sub (int): Number of tiles to merge recursively.
        output_folder (str): path to folder store the intermediate tiles and resulting DEM.
    """

    # Get a list of all raster files in the folder
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tif")]

    for i in range(len(files)//n_sub+1):
        # Open all raster files and store them in a list
        datasets = []
        print(i*n_sub, i*n_sub+n_sub)
        for file in files[i*n_sub:i*n_sub+n_sub]:
            
            dataset = rasterio.open(file)
            datasets.append(dataset)
        
        # Merge the datasets
        mosaic, transform = merge(datasets)
        
        # Set the output file path for the mosaic
        out_path = os.path.join(output_folder, f"DEM_{i}_rec.tif")
        
        metadata = dataset.meta.copy()
        metadata.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})
        
        with rasterio.open(out_path, "w", **metadata) as dest:
            dest.write(mosaic)
        
        for dataset in datasets:
            dataset.close()
            

    ## Mosaic the intermediate files

    # Get a list of all raster files in the folder
    files = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith("_rec.tif")]

    datasets = []
    for file in files:
        dataset = rasterio.open(file)
        datasets.append(dataset)
        
    # Merge the datasets
    mosaic, transform = merge(datasets)

    metadata = dataset.meta.copy()
    metadata.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})

    with rasterio.open(os.path.join(output_folder, "DEM.tif"), "w", **metadata) as dest:
        dest.write(mosaic)

    for dataset in datasets:
        dataset.close()
        
    # Delete intermediate files
    file_pattern = "DEM_*_rec.tif"
    matching_files = glob.glob(os.path.join(output_folder, file_pattern))
    # Delete each matching file
    for file_path in matching_files:
        os.remove(file_path)
        
    print('Merging DEM finished!')



if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="/data/scratch/selene/dem_reproj_resamp/")
    parser.add_argument('--n_sub', type=int, default=1000) 
    parser.add_argument('--output_folder', type=str, default="/data/scratch/selene/static_data/")

    args = parser.parse_args()
    
    mosaic_dem_recursive(args.folder, args.n_sub, args.output_folder)
