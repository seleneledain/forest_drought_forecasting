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
from datetime import datetime
import numpy as np
import random

import sys
# Add the path to the repository containing the file
sys.path.insert(0, '/Users/led/Documents/earthnet-minicuber/') # To modify
# Import the module
from earthnet_minicuber.minicuber import *

sys.path.append('..')  # Add parent directory to the sys.path
from data_downloading.cloud_cleaning import *
from feature_engineering.add_bands import *


def save_cube(cube, split, root_dir, lon_lat, raw=False):
    """
    Save cube to split directory. Will create the directories if not existing yet
    
    :param cube: data cube to save
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    :param lon_lat: tuple, center coordinates of the cube
    :para, raw: Boolean. Whether the data is raw or processed
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
    if raw:
        cube_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon_lat[0]}_{lon_lat[1]}_{width}_{height}_raw.nc'
    else:
        cube_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon_lat[0]}_{lon_lat[1]}_{width}_{height}.nc'

    #numpy_array = cube.to_array().values
    cube.to_netcdf(os.path.join(path_to_save,cube_name))

    
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
    
    
def obtain_context_target_cubes(cube, context, target, split, root_dir, specs, shift=0):
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
    """
    Min and max values per band, saved to a numpy array. The order of the bands should be the same as in the data samples after dropping bands: Sentinel2, custom data in the same order as provided in config, ERA5
    """

    # Modifying the dataset variable order to match final output
    new_order = list([name for name in list(cube.variables) if name.startswith('s2')])  +list([name for name in list(cube.variables) if not name.startswith('era5') and not name.startswith('s2') and name not in ['time', 'lat', 'lon']]) + list([name for name in list(cube.variables) if name.startswith('era5')]) 
    cube = cube[new_order]
    
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
    :param pixs_per_scene: number of pixels to sample per scene. None if no limit
    :param cloud_cleaning: max number of consecutive nan value in timeseries allowed, after which cloud cleaning will be performed
    :param target_in_summer: If target start date should be included in June-Sep
    :param drought_labels: If to use drought mask for sampling pixels
    :param forest_thresh: threshold of forest to consider to sample pixel
    :param drought_thresh: threshold of drought to consider to sample pixel
    """

    
    start_yr = config.specs["time_interval"].split("-")[0]
    start_month = config.specs["time_interval"].split("-")[1]
    start_month = start_month[1:] if start_month.startswith('0') else start_month
    end_yr = config.specs["time_interval"].split("-")[2].split("/")[1]
    end_month = config.specs["time_interval"].split("-")[3]
    end_month = end_month[1:] if end_month.startswith('0') else end_month
    lon = config.specs["lon_lat"][0]
    lat = config.specs["lon_lat"][1]
    width = config.specs["xy_shape"][0]
    height = config.specs["xy_shape"][1]
    cube_name = f'{start_yr}_{start_month}*{end_yr}_{end_month}*{lon}_{lat}_{width}_{height}.nc'
    search_cube = glob.glob(os.path.join(config.root_dir, config.split, 'cubes', cube_name))

    # Check if cube already exists
    if search_cube:
        # Load cube 
        cube = xr.open_dataset(os.path.join(config.root_dir,search_cube[0]), engine='netcdf4')        
    else:
        # Generate cube given specs and specs_add_band
        emc = Minicuber(config.specs)
        cube = emc.load_minicube(config.specs, compute = True)
        print('Downloaded data')
        cube = get_additional_bands(config.specs_add_bands, cube)
        print('Added local data')
        
        # Save raw cube
        save_cube(cube, config.split, config.root_dir, config.specs["lon_lat"], raw=True)
        
        # Cloud cleaning
        if config.cloud_cleaning:
            cube = smooth_s2_timeseries(cube, config.cloud_cleaning, config.remove_pct, config.loess_frac)
            print('Performed cloud cleaning')
        else:
            # Set to_sample = 1 everywhere if cloud cleaning is not performed
            ones_dataarray = xr.DataArray(np.ones((len(cube.lat), len(cube.lon))), dims=('lat', 'lon'))
            cube["to_sample"] = ones_dataarray
            

        # Deal with NaNs (or -9999)
        # For era5 linear interpolation
        cube_tmp = cube[[name for name in list(cube.variables) if name.startswith('era5')]]
        if np.isnan(cube_tmp.to_array()).any():
            cube[[name for name in list(cube.variables) if name.startswith('era5')]] = cube[[name for name in list(cube.variables) if name.startswith('era5')]].interpolate_na(dim='time', method='linear')
        # For static layers to average in space
        variables_to_fill = [name for name in list(cube.variables) if not name.startswith('era5') and not name.startswith('s2') and name not in ['time', 'lat', 'lon', 'to_sample']]
        cube_tmp = cube[variables_to_fill]
        cube_tmp = cube_tmp.where(cube_tmp != -9999, np.nan)
        if np.isnan(cube_tmp.to_array()).any():
            mean = cube_tmp[variables_to_fill].mean().to_array().values
            for i, v in enumerate(variables_to_fill):
                cube[v].fillna(mean[i])
        print('Dealed with missing values')
        
        # Save cube 
        save_cube(cube, config.split, config.root_dir, config.specs["lon_lat"], raw=False)
        
        # Compute normalisation stats (min, max) if normalisation=True (usually for train split)
        if config.normalisation and 'train' in config.split:
            save_min_max(cube.drop_vars(config.bands_to_drop), config.split, config.root_dir, config.specs)
            print('Computed normalisation statistics')
            
        
    # Split to context/target pairs and save
    if config.target_in_summer:
        obtain_context_target_pixels_summer(cube, config.context, config.target, config.split, config.root_dir, config.specs, config.bands_to_drop, config.pixs_per_scene, config.shift, config.drought_labels, config.forest_thresh, config.drought_thresh)
    else:
        obtain_context_target_pixels(cube, config.context, config.target, config.split, config.root_dir, config.specs, config.bands_to_drop, config.pixs_per_scene, config.shift, config.drought_labels, config.forest_thresh, config.drought_thresh)
   
    print(f"Created {config.split} samples from cube!")
        
    return


def find_date_closest_summer_end(time_yr):
    """
    Returns the last date before Sep. 1st. of that year
    
    :param time_yr: List of datetime objects, all in same year
    """
    from datetime import datetime
    
    target_date = datetime(datetime.now().year, 9, 1)

    closest_index = None
    closest_value = None
    closest_timedelta = None

    for i, dt in enumerate(time_yr):
        if dt.month < target_date.month or (dt.month == target_date.month and dt.day < target_date.day):
            timedelta = target_date - dt

            if closest_timedelta is None or timedelta < closest_timedelta:
                closest_index = i
                closest_value = dt
                closest_timedelta = timedelta

    if closest_index is not None:
        return closest_index, closest_value
    else:
        #print("No value found before September 1st.")
        return None, None


def obtain_context_target_pixels_summer(cube, context, target, split, root_dir, specs, bands_to_drop, pixs_per_scene, shift=0, drought_labels=False, forest_thresh=0.5, drought_thresh=0):
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
    :param drought_labels: Select pixels that are labeled with a drought event using drought mask in data
    :param forest_thresh: threshold to pass in sampling for forest ratio in pixel 
    :param drought_thresh: threshold to pass in sampling for drought ratio in pixel 
    :param pixs_per_scene: number of pixels to sample per scene. None if no limit
    """
    from datetime import datetime
    
    n_frames = len(cube.time)
    lon = specs["lon_lat"][0]
    lat = specs["lon_lat"][1]
    width = specs["xy_shape"][0]
    height = specs["xy_shape"][1]
    
    if context+target > n_frames-shift:
        raise Exception("Time interval of data cube is not big enough for request context+target frames! Please download more data, or change context/target length.")
    
    unique_years = set()
    all_cube_times = [datetime.strptime(np.datetime_as_string(date, unit='s'), '%Y-%m-%dT%H:%M:%S') for date in cube.time.values]
    for dt in all_cube_times:
        unique_years.add(dt.year)
        
    # Find the last possible target date for the dataset (needs to allow for data shifting)
    last_target_start_date = all_cube_times[-target-shift] # last_date - target len - shift

    # Operate over each summer
    for yr in sorted(unique_years):
        #print(yr)
        
        # Get the dates that fall in that year
        yr_dt = datetime(yr, 1, 1)
        cube_yr = cube.where(cube.time.dt.year == yr_dt.year, drop=True)
        time_yr = cube_yr.time.values
        # Convert NumPy datetime64 objects to datetime objects
        time_yr = [datetime.strptime(np.datetime_as_string(date, unit='s'), '%Y-%m-%dT%H:%M:%S') for date in time_yr]
        
        
        # Check if all dates would be usable or not (limited by target length and shift). Goal is to find last_target_start_idx in time_yr and its value
        
        # Find the last target start date before Sep 1st. It must be in summer.
        last_idx, last_target_start_date = find_date_closest_summer_end(time_yr)
        # Need to adjust last_target_start_date_idx because for now it is an idx for time_yr, not all_cube_times
        last_target_start_date_idx = [idx for idx, t in enumerate(all_cube_times) if t==time_yr[last_idx]][0]

        if last_target_start_date is None:
            # Pass this year
            continue
        # Check that it is in summer
        if last_target_start_date.month >= 6 and last_target_start_date.month < 9 and last_target_start_date.day >= 1:
            #print('in summer')
            if (last_target_start_date_idx + target + shift) <= len(all_cube_times): # Need to fix idx because not adapt to comparing
                #print('enough dates')
                # If this exists then we can just truncate time_yr
                target_start = time_yr[:last_idx+1]
            else:
                #print('not enough dates')
                # Find the last possible target start date, and check if it is in summer
                last_target_start_date = all_cube_times[-target-shift+1]
                # Check if last_target_start_date + target + shift exists
                if last_target_start_date.month >= 6 and last_target_start_date.month < 9 and last_target_start_date.day >= 1:
                    #print('change dates')
                    # It is in summer
                    target_start = time_yr[:-target-shift]
                    last_idx = len(time_yr) - target - shift -1
                else:
                    #print('cant change dates')
                    # Pass this year as there are not enough summer values
                    continue
        else:
            #print('not in summer')
            # Pass this year as there are no dates in summer
            continue
    

        # Filter the datetime list for dates between June 1st and September 1st 
        target_start = [dt for dt in target_start if dt.month >= 6 and dt.month < 9 and dt.day >= 1]
        context_start = [all_cube_times[i-context] if i-context >= 0 else None for td in target_start for i, t in enumerate(all_cube_times) if td == all_cube_times[i]]
        
        # Find the indices where context_start is None
        none_indices = [i for i, cs in enumerate(context_start) if cs is None]
        
        # Drop elements at the none_indices from both lists
        context_start = [cs for i, cs in enumerate(context_start) if i not in none_indices]
        target_start = [ts for i, ts in enumerate(target_start) if i not in none_indices]
        
        if not len(context_start) or not len(target_start):
            print('Available time frame not enough for context generation. Skipping year as target.')
            continue
        else:
            last_context_start_date_idx = [idx for idx, t in enumerate(all_cube_times) if t==context_start[-1]][0]
            # Need to adjust last_target_start_date_idx because for now it is an idx for time_yr, not all_cube_times
            last_target_start_date_idx = [idx for idx, t in enumerate(all_cube_times) if t==time_yr[last_idx]][0]

            # Now add all the other dates (not just start context/target) and convert to NumPy datetime64 objects 
            dt_context = [np.datetime64(dt) for dt in context_start] + [np.datetime64(dt) for dt in all_cube_times[last_context_start_date_idx+1:last_context_start_date_idx+context+shift]]
            dt_target = [np.datetime64(dt) for dt in target_start] + [np.datetime64(dt) for dt in all_cube_times[last_target_start_date_idx+1:last_target_start_date_idx+target+shift]]
            #print('Should be cst between yrs', len(dt_context)-len(context_start), len(dt_target)-len(target_start))


            # Generate all sub-timeseries
            for i,d in enumerate(dt_context[:-context-shift]):
                #print('i', i)
                if shift>0:
                    # Need to use cube and not cube_yr since the dates might extend beyond that year
                    era_variable_names = [name for name in list(cube.variables) if name.startswith('era5_')]
                    era_data = cube[era_variable_names]
                    other_variable_names = [name for name in list(cube.variables) if not name.startswith('era5_')]
                    other_data =  cube[other_variable_names]
                    # Context data
                    sub_context_era = era_data.sel(time=slice(dt_context[i+shift],dt_context[i+context+shift-1]))
                    sub_context_other = other_data.sel(time=slice(d,dt_context[i+context-1]))
                    sub_context_era['time'] = sub_context_other.time
                    sub_context = xr.merge([sub_context_other, sub_context_era])
                    # Target data
                    sub_target_era = era_data.sel(time=slice(dt_target[i+shift],dt_target[i+target+shift-1]))
                    sub_target_other = other_data.sel(time=slice(dt_target[i],dt_target[i+target-1]))
                    sub_target_era['time'] = sub_target_other.time
                    sub_target = xr.merge([sub_target_other, sub_target_era])
                else:
                    sub_context = cube_yr.sel(time=slice(d,time[i+context]))
                    sub_target = cube_yr.sel(time=slice(dt_target[i],dt_target[i+target]))

                # Extract pixels at lat lons here and save context+target together
                extract_pixel_timeseries(sub_context, sub_target, shift, split, root_dir, bands_to_drop, drought_labels, forest_thresh, drought_thresh, pixs_per_scene)


def extract_pixel_timeseries(cube_context, cube_target, shift, split, root_dir, bands_to_drop, drought_labels, forest_thresh, drought_thresh, pixs_per_scene):
    """
    From a datacube, will extract individual pixel timeseries based on forest mask and to_sample variable.
    Then save them as .npz files.
    
    :param cube_context: xarray from which to extract data for context frames
    :param cube_target: xarray from which to extract data for target frames
    :param shift: int, number of itmeframes of shift between ERA5 and S2. The final dates in the xarray will be the non-shifted ones
    :param split: str, train/test/val or other name of split
    :param root_dir: str, root directory where data will be saved and folders created
    :param bands_to_drop: variables to remove when saving pixel timeseries
    :param forest_thresh: threshold to pass in sampling for forest ratio in pixel 
    :param drought_thresh: threshold to pass in sampling for drought ratio in pixel 
    :param pixs_per_scene: number of pixels to sample per scene. None if no limit
    """
    
    start_yr = cube_context.time.to_index().date[0].year
    start_month = cube_context.time.to_index().date[0].month
    start_day = cube_context.time.to_index().date[0].day
    end_yr = cube_target.time.to_index().date[-1].year
    end_month = cube_target.time.to_index().date[-1].month
    end_day = cube_target.time.to_index().date[-1].day
    width = len(cube_context.lon) 
    height = len(cube_context.lat)
    
    # Get all the pixels that satisfy condition (forest mask + valid after cloud cleaning)
    
    if pixs_per_scene is None:
        
        for i_lat, lat in enumerate(cube_context.lat):
            for i_lon, lon in enumerate(cube_context.lon): 
                
                check_to_sample = cube_context.sel(lat=lat, lon=lon).to_sample.values 
                check_forest_mask = cube_context.sel(lat=lat, lon=lon).FOREST_MASK.values
                if drought_labels:
                    check_drought_mask = cube_context.sel(lat=lat, lon=lon).DROUGHT_MASK.values
                else:
                    check_drought_mask = 1
                
                if check_to_sample and check_forest_mask > forest_thresh and check_drought_mask > drought_thresh:
                    pixel_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon.values}_{lat.values}_{width}_{height}_{shift}.npz'
                    save_context_target(cube_context.sel(lat=lat, lon=lon).drop_vars(bands_to_drop), cube_target.sel(lat=lat, lon=lon).drop_vars(bands_to_drop), pixel_name, split, root_dir)
                    
    else:
        # Initialize a counter for successful samples and tested coords
        successful_samples = 0
        trials = 0

        # Loop until the desired number of samples is achieved
        while successful_samples < pixs_per_scene and trials < len(cube_context.lat)*len(cube_context.lon):
            # Generate random indices for latitudes and longitudes
            random_lat_index = random.randint(0, len(cube_context.lat) - 1)
            random_lon_index = random.randint(0, len(cube_context.lon) - 1)
            lat = cube_context.lat[random_lat_index]
            lon = cube_context.lon[random_lon_index]
            trials += 1

            check_to_sample = cube_context.sel(lat=lat, lon=lon).to_sample.values
            check_forest_mask = cube_context.sel(lat=lat, lon=lon).FOREST_MASK.values
            if drought_labels:
                check_drought_mask = cube_context.sel(lat=lat, lon=lon).DROUGHT_MASK.values
            else:
                check_drought_mask = 1

            if check_to_sample and check_forest_mask > forest_thresh and check_drought_mask > drought_thresh:
                pixel_name = f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon.values}_{lat.values}_{width}_{height}_{shift}.npz'
                save_context_target(cube_context.sel(lat=lat, lon=lon).drop_vars(bands_to_drop), cube_target.sel(lat=lat, lon=lon).drop_vars(bands_to_drop), pixel_name, split, root_dir)
                successful_samples += 1

                            
            
def obtain_context_target_pixels(cube, context, target, split, root_dir, specs, bands_to_drop, pixs_per_scene, shift=0, forest_thresh=0.5, drought_thresh=0):
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
    :param forest_thresh: threshold to pass in sampling for forest ratio in pixel 
    :param drought_thresh: threshold to pass in sampling for drought ratio in pixel 
    :param pixs_per_scene: number of pixels to sample per scene. None if no limit
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
        extract_pixel_timeseries(sub_context, sub_target, shift, split, root_dir, bands_to_drop, forest_thresh, drought_thresh)

    return 

