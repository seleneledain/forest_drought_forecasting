"""
Remove cubes generated from wrong coordinates
"""

import os
import glob
import xarray as xr

cube_path = '/data/scratch/selene/neg/cubes'
data_path = '/data/scratch/selene/neg/'
coords_keep_path = '/data/scratch/selene/sampling_rasters/coords_negative.txt'

# Open coords_keep_path: its lines are coordinates lon, lat
with open(coords_keep_path, 'r') as coords_keep_file:
    coords_to_keep = set(coords_keep_file.read().splitlines())

# Loop over the cubes, their format is f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon_lat[0]}_{lon_lat[1]}_{width}_{height}*.nc'
cube_files = glob.glob(os.path.join(cube_path, '*.nc'))
for cube_file in cube_files:
    cube_filename = os.path.basename(cube_file)
    cube_parts = cube_filename.split('_')
    lon_lat = (cube_parts[-6], cube_parts[-5])  # Extract lon_lat from the filename

    # If {lon_lat[0]}_{lon_lat[1]} is not in coords_keep_path, delete the file
    if '_'.join(lon_lat) not in coords_to_keep:
        # Remove .npy files in data_path that has {lon_lat[0]}_{lon_lat[1]} in its name
        npy_files_to_remove = glob.glob(os.path.join(data_path, f'*{lon_lat[0]}_{lon_lat[1]}*.npy'))
        for npy_file in npy_files_to_remove:
            os.remove(npy_file)
        
        # Open the cube with xarray and iterate through its lon and lat. 
        cube = xr.open_dataset(os.path.join(cube_path, cube_file))
        for i_lat, lat in enumerate(cube.lat):
            for i_lon, lon in enumerate(cube.lon): 
                # Remove any .npz with {lon.values}_{lat.values} in its name
                lon_lat_str = f'{lon.values}_{lat.values}'
                npz_files_to_remove = glob.glob(os.path.join(data_path, f'*{lon_lat_str}*.npz'))
                for npz_file in npz_files_to_remove:
                    os.remove(npz_file)
                    print(f'Deleted sample: {npz_file}')
        
    
        os.remove(cube_file)
        print(f"Deleted: {cube_filename}\n -------------")
        
