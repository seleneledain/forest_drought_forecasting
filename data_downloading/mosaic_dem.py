import os
import rasterio
from rasterio.merge import merge

# Specify the folder path containing the raster tiles
folder_path = "/data/scratch/selene/dem_reproj_resamp/" # TO UPDATE with path to reprojected tiles

# Get a list of all raster files in the folder
files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".tif")]

# Open all raster files and store them in a list
datasets = []
for file in files:
    
    dataset = rasterio.open(file)
    datasets.append(dataset)
    
# Merge the datasets
mosaic, transform = merge(datasets)

# Set the output file path for the mosaic
output_path = f"/data/scratch/selene/static_data/DEM.tif" # TO UPDATE with output path

metadata = dataset.meta.copy()
metadata.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})

with rasterio.open(output_path, "w", **metadata) as dest:
    dest.write(mosaic)

for dataset in datasets:
    dataset.close()
    
print("Mosaic created successfully!")
