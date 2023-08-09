import os
import rasterio
from rasterio.merge import merge


def mosaic_dem(folder_path, output_path):
    """Mosaic files in a folder (tiles) into one DEM

    Args:
        folder_path (str): path to DEM tiles
        output_path (str): path to store the resulting DEM
    """
    
    # Get a list of all raster files in the folder
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tiff")]

    # Open all raster files and store them in a list
    datasets = []
    for file in files:
        
        dataset = rasterio.open(file)
        datasets.append(dataset)
        
    # Merge the datasets
    mosaic, transform = merge(datasets)

    metadata = dataset.meta.copy()
    metadata.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})

    with rasterio.open(output_path, "w", **metadata) as dest:
        dest.write(mosaic)

    for dataset in datasets:
        dataset.close()
        
    print("Mosaic created successfully!")


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default="/data/scratch/selene/dem_reproj_resamp/")
    parser.add_argument('--output_path', type=str, default="/data/scratch/selene/static_data/DEM.tif")

    args = parser.parse_args()
    
    mosaic_dem(args.folder, args.output_path)

