"""
Generate minicube dataset

Author: Selene Ledain
Date: June 12th, 2023
"""

import os
import torch
import warnings
warnings.filterwarnings('ignore')
import sys
# Add the path to the repository containing the file
sys.path.insert(0, '/Users/led/Documents/earthnet-minicuber/') # To modify
# Import the module
from earthnet_minicuber.minicuber import *
from add_bands import *
from cloud_cleaning import *


def save_cube(cube, cube_name, split, cont_targ, root_dir):
    """
    Save cube to split directory. Will create the directories if not existing yet
    
    :param cube: data cube to save
    :param cube_name: name of .npz file for saving cube. Format is {start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}.npz
    :param split: str, train/test/val or other name of split
    :param cont_targ: any between ['context', 'target']
    :param root_dir: str, root directory where data will be saved and folders created
    """
    
    path_to_save = os.path.join(root_dir, split, cont_targ)
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        print(f"Created new directory: {path_to_save}.")
    
    numpy_array = cube.to_array().values
    np.savez(os.path.join(path_to_save,cube_name), data=numpy_array)

    
def save_context_target(cube_context, cube_target, file_name, split, root_dir):
    """
    Save cube to split directory. Will create the directories if not existing yet
    
    :param cube_context: context data cube to save
    :param cube_target: target data cube to save
    :param file_name: name of .npz file for saving cube. Format is {start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}.npz
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    """
    
    path_to_save = os.path.join(root_dir, split)
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        print(f"Created new directory: {path_to_save}.")
    
    context_array = cube_context.to_array().values
    target_array = cube_target.to_array().values
    np.savez(os.path.join(path_to_save,file_name), context=context_array, target=target_array)
    
    
    
def obtain_context_target(cube, context, target, split, root_dir, specs, shift=0):
    """
    Split into context target pairs given whole time interval of the cube
    
    :param cube: data cube containing data across whole time interval
    :param context: int, number of context frames
    :param target: int, number of target frames
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    :param specs: minicuber specifications
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    """
    n_frames = len(cube.time)
    lon = specs["lon_lat"][0]
    lat = specs["lon_lat"][1]
    width = specs["xy_shape"][0]
    height = specs["xy_shape"][1]
    
    if context+target > n_frames-shift:
        raise Exception("Time interval of data cube is not big enough for request context+target frames! Please download more data, or change context/target length.")
    
    n_pairs = n_frames-shift-context-target+1 # number of context-target pairs possible
    start_t = 0
    end_t = context+target
    
    for pair in range(n_pairs):
        if shift>0:
            era_variable_names = [name for name in list(cube.variables) if name.startswith('era5_')]
            era_data = cube[filtered_variable_names]
            sub_context_era = era_data.isel(time=slice(start_t+pair+shift, start_t+pair+context+shift))
            other_variable_names = [name for name in list(cube.variables) if not name.startswith('era5_')]
            other_data =  cube[other_variable_names]
            sub_context_other = other_data.isel(time=slice(start_t+pair, start_t+pair+context))
            sub_context_era['time'] = sub_context_other.time
            sub_context = xr.merge([sub_context_other, sub_context_era])
        else:
            sub_context = cube.isel(time=slice(start_t+pair, start_t+pair+context))
        start_yr = sub_context.time.to_index().date[0].year
        start_month = sub_context.time.to_index().date[0].month
        start_day = sub_context.time.to_index().date[0].day
        end_yr = sub_context.time.to_index().date[-1].year
        end_month = sub_context.time.to_index().date[-1].month
        end_day = sub_context.time.to_index().date[-1].day
        cube_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}_{shift}.npz'
        # Save sub-cube here with a function
        save_cube(sub_context, cube_name, split, 'context', root_dir)
        
        if shift>0:
            era_variable_names = [name for name in list(cube.variables) if name.startswith('era5_')]
            era_data = cube[filtered_variable_names]
            sub_target_era = era_data.isel(time=slice(start_t+pair+context+shift, end_t+pair+shift))
            other_variable_names = [name for name in list(cube.variables) if not name.startswith('era5_')]
            other_data =  cube[other_variable_names]
            sub_target_other = other_data.isel(time=slice(start_t+pair+context, end_t+pair))
            sub_target_era['time'] = sub_target_other.time
            sub_target = xr.merge([sub_target_other, sub_target_era])
        else:
            sub_target = cube.isel(time=slice(start_t+pair+context, end_t+pair))
        start_yr = sub_target.time.to_index().date[0].year
        start_month = sub_target.time.to_index().date[0].month
        start_day = sub_target.time.to_index().date[0].day
        end_yr = sub_target.time.to_index().date[-1].year
        end_month = sub_target.time.to_index().date[-1].month
        end_day = sub_target.time.to_index().date[-1].day
        cube_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}_{shift}.npz'
        # Save sub-cube here with a function
        save_cube(sub_target, cube_name, split, 'target', root_dir)

    return 

def save_min_max(cube, split, root_dir, specs):
    
    min_vals = cube.min().to_array().values
    max_vals = cube.max().to_array().values
    
    # Need to name the cube intelligently and save min/max to numpy array
    lon = specs["lon_lat"][0]
    lat = specs["lon_lat"][1]
    width = specs["xy_shape"][0]
    height = specs["xy_shape"][1]
    start_yr = cube.time.to_index().date[0].year
    start_month = cube.time.to_index().date[0].month
    start_day = cube.time.to_index().date[0].day
    end_yr = cube.time.to_index().date[-1].year
    end_month = cube.time.to_index().date[-1].year
    end_day = cube.time.to_index().date[-1].year
    cube_stat_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}'
    # Save values
    path_to_save = os.path.join(root_dir, split, cube_stat_name)
    
    np.save(path_to_save+'_min.npy', min_vals)
    np.save(path_to_save+'_max.npy', max_vals)

def generate_samples(specs, specs_add_bands, context, target, split, root_dir, cloud_cleaning=0, normalisation=False, shift=0):
    """
    Generate datacubes for a split, with a given context and target length and minicuber specifications.
    Save the cubes locally.
    
    :param specs: specifications for minicuber
    :param specs_add_band: specifications for additional/static bands in minicubes
    :param context: context length
    :param target: target legnth
    :param split: str, split name
    :param root_dir: str, path to save data
    :param normalisation: boolean. Compute min/max in space and time for each variables and save values
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    """
    
    # Generate cube given specs and specs_add_band
    emc = Minicuber(specs)
    cube = emc.load_minicube(specs, compute = True)
    print('Downloaded data')
    cube = get_additional_bands(specs_add_bands, cube)
    print('Added local data')
    
    # Cloud cleaning
    if cloud_cleaning:
        cube = smooth_s2_timeseries(cube, cloud_cleaning)
        print('Performed cloud cleaning')
        
    # Split to context/target pairs and save
    obtain_context_target_pixels(cube, context, target, split, root_dir, specs, shift)
    print(f"Created {split} samples from cube!")
    
    # Compute normalisation stats (min, max) if normalisation=True (usually for train split)

    if normalisation and split=='train':
        save_min_max(cube, split, root_dir, specs)
        print('Computed normalisation statistics')
        
    return 



def obtain_context_target_pixels(cube, context, target, split, root_dir, specs, shift=0):
    """
    Split into context target pairs given whole time interval of the cube
    
    :param cube: data cube containing data across whole time interval
    :param context: int, number of context frames
    :param target: int, number of target frames
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    :param specs: minicuber specifications
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    """
    n_frames = len(cube.time)
    lon = specs["lon_lat"][0]
    lat = specs["lon_lat"][1]
    width = specs["xy_shape"][0]
    height = specs["xy_shape"][1]
    
    if context+target > n_frames-shift:
        raise Exception("Time interval of data cube is not big enough for request context+target frames! Please download more data, or change context/target length.")
    
    n_pairs = n_frames-shift-context-target+1 # number of context-target pairs possible
    start_t = 0
    end_t = context+target
    
    for pair in range(n_pairs):
        if shift>0:
            era_variable_names = [name for name in list(cube.variables) if name.startswith('era5_')]
            era_data = cube[era_variable_names]
            sub_context_era = era_data.isel(time=slice(start_t+pair+shift, start_t+pair+context+shift))
            other_variable_names = [name for name in list(cube.variables) if not name.startswith('era5_')]
            other_data =  cube[other_variable_names]
            sub_context_other = other_data.isel(time=slice(start_t+pair, start_t+pair+context))
            sub_context_era['time'] = sub_context_other.time
            sub_context = xr.merge([sub_context_other, sub_context_era])
        else:
            sub_context = cube.isel(time=slice(start_t+pair, start_t+pair+context))
        
        if shift>0:
            era_variable_names = [name for name in list(cube.variables) if name.startswith('era5_')]
            era_data = cube[era_variable_names]
            sub_target_era = era_data.isel(time=slice(start_t+pair+context+shift, end_t+pair+shift))
            other_variable_names = [name for name in list(cube.variables) if not name.startswith('era5_')]
            other_data =  cube[other_variable_names]
            sub_target_other = other_data.isel(time=slice(start_t+pair+context, end_t+pair))
            sub_target_era['time'] = sub_target_other.time
            sub_target = xr.merge([sub_target_other, sub_target_era])
        else:
            sub_target = cube.isel(time=slice(start_t+pair+context, end_t+pair))
        
        # Extract pixels at lat lons here and save context+target together
        extract_pixel_timeseries(sub_context, sub_target, shift, split, root_dir)

    return 


def extract_pixel_timeseries(cube_context, cube_target, shift, split, root_dir):
    """
    From a datacube, will extract individual pixel timeseries based on forest mask and to_sample variable.
    Then save them as .npz files.
    
    :param cube_context: xarray from which to extract data for context frames
    :param cube_target: xarray from which to extract data for target frames
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    """
    # TO DO: set to_sample = 1 everywhere if cloud cleaning is not performed
    start_yr = cube_context.time.to_index().date[0].year
    start_month = cube_context.time.to_index().date[0].month
    start_day = cube_context.time.to_index().date[0].day
    end_yr = cube_target.time.to_index().date[-1].year
    end_month = cube_target.time.to_index().date[-1].month
    end_day = cube_target.time.to_index().date[-1].day
    width = len(cube_context.lon) 
    height = len(cube_context.lat)
    
    # Get all the pixels that satisfy condition (forest mask + valid after cloud cleaning)
    for i_lat, lat in enumerate(cube_context.lat):
        for i_lon, lon in enumerate(cube_context.lon):
            if cube_context.sel(lat=lat, lon=lon).to_sample.values and cube_context.sel(lat=lat, lon=lon).FOREST_MASK.values:
                pixel_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon.values}_{lat.values}_{width}_{height}_{shift}.npz'
                save_context_target(cube_context.sel(lat=lat, lon=lon), cube_target.sel(lat=lat, lon=lon), pixel_name, split, root_dir)

    
