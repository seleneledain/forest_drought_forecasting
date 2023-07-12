"""
Generate minicube dataset

Author: Selene Ledain
Date: June 12th, 2023
"""

import os
import torch
import glob
import xarray as xr
import warnings
warnings.filterwarnings('ignore')
import sys
# Add the path to the repository containing the file
sys.path.insert(0, '/Users/led/Documents/earthnet-minicuber/') # To modify
# Import the module
from earthnet_minicuber.minicuber import *

sys.path.append('..')  # Add parent directory to the sys.path
from data_downloading.cloud_cleaning import *
from feature_engineering.add_bands import *


def save_cube(cube, split, root_dir, lon_lat):
    """
    Save cube to split directory. Will create the directories if not existing yet
    
    :param cube: data cube to save
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    :param lon_lat: tuple, center coordinates of the cube
    """
    
    path_to_save = os.path.join(root_dir, split, 'cubes')
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        print(f"Created new directory: {path_to_save}.")
    
    start_yr = cube.time.to_index().date[0].year
    start_month = cube.time.to_index().date[0].month
    start_day = cube.time.to_index().date[0].day
    end_yr = cube.time.to_index().date[-1].year
    end_month = cube.time.to_index().date[-1].month
    end_day = cube.time.to_index().date[-1].day
    width = len(cube.lon) 
    height = len(cube.lat)
    cube_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon_lat[0]}_{lon_lat[1]}_{width}_{height}.npz'

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

def generate_samples(config):
    """
    Generate datacubes for a split, with a given context and target length and minicuber specifications.
    Save the cubes locally.
    
    Config containing
    :param specs: specifications for minicuber
    :param specs_add_band: specifications for additional/static bands in minicubes
    :param context: context length
    :param target: target legnth
    :param split: str, split name
    :param root_dir: str, path to save data
    :param normalisation: boolean. Compute min/max in space and time for each variables and save values
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    """

    
    start_yr = config.specs["time_interval"].split("-")[0]
    start_month = config.specs["time_interval"].split("-")[1]
    end_yr = config.specs["time_interval"].split("-")[2].split("/")[1]
    end_month = config.specs["time_interval"].split("-")[3]
    lon = config.specs["lon_lat"][0]
    lat = config.specs["lon_lat"][1]
    width = config.specs["xy_shape"][0]
    height = config.specs["xy_shape"][1]
    cube_name = f'{start_yr}_{start_month}*{end_yr}_{end_month}*{lon}_{lat}_{width}_{height}.npz'
    search_cube = glob.glob(os.path.join(config.root_dir, config.split, 'cubes', cube_name))
    
    # Check if cube already exists
    if search_cube:
        # Load cube 
        cube = xr.open_dataset(search_cube[0])
    else:
        # Generate cube given specs and specs_add_band
        emc = Minicuber(specs)
        cube = emc.load_minicube(config.specs, compute = True)
        print('Downloaded data')
        cube = get_additional_bands(config.specs_add_bands, cube)
        print('Added local data')
    
    # Cloud cleaning
    if cloud_cleaning:
        cube = smooth_s2_timeseries(cube, config.cloud_cleaning)
        print('Performed cloud cleaning')
        
    # Deal with NaNs (or -9999)
    # For era5 linear interpolation
    cube_tmp = cube[[name for name in list(cube.variables) if name.startswith('era5')]]
    if np.isnan(cube_tmp.to_array()).any():
        cube[[name for name in list(cube.variables) if name.startswith('era5')]] = cube[[name for name in list(cube.variables) if name.startswith('era5')]].interpolate_na(dim='time', method='linear')
    # For static layers to average in space
    variables_to_fill = [name for name in list(cube.variables) if not name.startswith('era5') and not name.startswith('s2') and name not in ['time', 'lat', 'lon']]
    cube_tmp = cube[variables_to_fill]
    cube_tmp = cube_tmp.where(cube_tmp != -9999, np.nan)
    if np.isnan(cube_tmp.to_array()).any():
        mean = cube_tmp[variables_to_fill].mean()
        cube[variables_to_fill] = cube[variables_to_fill].fillna(mean)
    print('Dealed with misisng values')
    
        
    # Save cube 
    save_cube(cube, cube_name, config.split, config.root_dir, config.specs["lon_lat"])
        
    # Split to context/target pairs and save
    obtain_context_target_pixels(cube, config.context, config.target, config.split, config.root_dir, config.specs, config.bands_to_drop, config.shift)
    print(f"Created {split} samples from cube!")
    
    # Compute normalisation stats (min, max) if normalisation=True (usually for train split)
    if config.normalisation and config.split=='train':
        save_min_max(cube, config.split, config.root_dir, config.specs)
        print('Computed normalisation statistics')
        
    return 



def obtain_context_target_pixels(cube, context, target, split, root_dir, specs, bands_to_drop, shift=0):
    """
    Split into context target pairs given whole time interval of the cube
    
    :param cube: data cube containing data across whole time interval
    :param context: int, number of context frames
    :param target: int, number of target frames
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    :param specs: minicuber specifications
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    :param bands_to_drop: variables to remove when saving pixel timeseries
    
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
        extract_pixel_timeseries(sub_context, sub_target, shift, split, root_dir, bands_to_drop)

    return 


def extract_pixel_timeseries(cube_context, cube_target, shift, split, root_dir, bands_to_drop):
    """
    From a datacube, will extract individual pixel timeseries based on forest mask and to_sample variable.
    Then save them as .npz files.
    
    :param cube_context: xarray from which to extract data for context frames
    :param cube_target: xarray from which to extract data for target frames
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    :param bands_to_drop: variables to remove when saving pixel timeseries
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
            if cube_context.sel(lat=lat, lon=lon).to_sample.values: # and cube_context.sel(lat=lat, lon=lon).FOREST_MASK.values:
                pixel_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon.values}_{lat.values}_{width}_{height}_{shift}.npz'
                save_context_target(cube_context.sel(lat=lat, lon=lon).drop_vars(bands_to_drop), cube_target.sel(lat=lat, lon=lon).drop_vars(bands_to_drop), pixel_name, split, root_dir)

    
