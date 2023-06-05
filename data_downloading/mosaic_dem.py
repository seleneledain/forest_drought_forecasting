import os
import rasterio
from rasterio.merge import merge

# Specify the folder path containing the raster tiles
folder_path = "swiss_dem/downloads/" # TO UPDATE with path to reprojected tiles

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
output_path = "swiss_dem/downloads/DEM.tif" # TO UPDATE with output path

# Update the metadata for the mosaic
metadata = dataset.meta.copy()
metadata.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})

# Write the mosaic to the output file
with rasterio.open(output_path, "w", **metadata) as dest:
    dest.write(mosaic)

# Close all datasets
for dataset in datasets:
    dataset.close()

print("Mosaic created successfully!")
