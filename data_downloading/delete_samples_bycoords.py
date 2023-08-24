"""
Remove cubes generated from wrong coordinates
"""

import os
import glob

cube_path = '/data/scratch/selene/neg/cubes'
data_path = '/data/scratch/selene/neg/'
# Conditions to delete
lat_bound = 46.25
lat_operation = 'less' #'more'
lon_bound = 9.387
lon_operation = 'more' #'more'

# Loop over the files, their format is f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon_lat[0]}_{lon_lat[1]}_{width}_{height}*.nc'
# f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon_lat[0]}_{lon_lat[1]}_{width}_{height}_{shift}*.npz' or _min.npy or _max.npy
# Delete file if {lon_lat[0]} < lon_bound if lon_operation is 'less', else > lon_bound
# AND if {lon_lat[1]} < lat_bound if lat_operation is 'less', else > lat_bound

files_to_delete = []

for file_path in glob.glob(os.path.join(data_path, '*.np*')):
    file_name = os.path.basename(file_path)
    _, _, _, _, _, _, lon, lat, _, _, _ = file_name[:-4].split('_')
    lon = float(lon)
    lat = float(lat)
    lon_condition = (lon < lon_bound) if lon_operation == 'less' else (lon > lon_bound)
    lat_condition = (lat < lat_bound) if lat_operation == 'less' else (lat > lat_bound)
    
    if lon_condition and lat_condition:
        files_to_delete.append(file_path)

print('nbr npz/npy files to del', len(files_to_delete))
for file_path in files_to_delete:
    os.remove(file_path)
    #print(f"Deleted: {file_path}")
    #pass
    


# Same for files in cube_path
    
files_to_delete = []

for file_path in glob.glob(os.path.join(cube_path, '*_raw.nc')):  # For files ending in _raw.nc
    file_name = os.path.basename(file_path)
    _, _, _, _, _, _, lon, lat, _, _ = file_name[:-7].split('_')
    lon = float(lon)
    lat = float(lat)
    
    lon_condition = (lon < lon_bound) if lon_operation == 'less' else (lon > lon_bound)
    lat_condition = (lat < lat_bound) if lat_operation == 'less' else (lat > lat_bound)
    
    if lon_condition and lat_condition:
        files_to_delete.append(file_path)

for file_path in glob.glob(os.path.join(cube_path, '*.nc')):  # For files ending in .nc
    file_name = os.path.basename(file_path)
    if file_name.endswith('_raw.nc'):  # Skip the _raw.nc files since we already processed them
        continue
    
    _, _, _, _, _, _, lon, lat, _, _ = file_name[:-3].split('_')
    lon = float(lon)
    lat = float(lat)
    
    lon_condition = (lon < lon_bound) if lon_operation == 'less' else (lon > lon_bound)
    lat_condition = (lat < lat_bound) if lat_operation == 'less' else (lat > lat_bound)
    
    if lon_condition and lat_condition:
        files_to_delete.append(file_path)

print('nbr cubes to del', len(files_to_delete))
for file_path in files_to_delete:
    os.remove(file_path)
    print(f"Deleted: {file_path}")