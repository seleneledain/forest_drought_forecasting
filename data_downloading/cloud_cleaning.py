"""
Replace cloud covered values in the dataset

Author: Selene Ledain
Date: June 23rd, 2023
"""

import numpy as np
import xarray as xr
import pandas as pd

def check_if_interp(data, max_nan):
    """
    For each pixel timeseries of NDVI, will check if there are sufficient values to allow timeseries smoothing (not too many NaN).
    
    :param data: xarray dataset with data that needs to be smoothed
    :param max_nan: maximum number of consecutive NaNs allowed. If more, the pixel will not be used/sampled
    """
    
    #for variable_name in filtered_ds_10.data_vars:
    #if variable_name.startswith('s2'):

    # Select the variable (now just NDVI)
    variable = data['s2_ndvi']

    # Initialise a new variable to store the data check
    new_variable = xr.DataArray(np.zeros((len(variable.lat), len(variable.lon))),
                                dims=('lat', 'lon'),
                                coords={'lat': variable.lat, 'lon': variable.lon})

    # Add the new variable to the dataset
    data['to_sample'] = new_variable

    # Loop along the pixel dimensions (lat, lon)
    for i_lat, lat in enumerate(variable.lat):
        for i_lon, lon in enumerate(variable.lon):

            # Access the data for the current pixel
            subset = variable.sel(lat=lat, lon=lon)

            # If all nan to skip and add indictator
            if subset.isnull().all():
                data['to_sample'].loc[dict(lat=lat, lon=lon)] = False

            # Compute max number of consecutive
            max_count, max_index = find_max_consec_nan(subset.isnull().values)
            #print(f'{max_count} consecutive missing timestamps - i.e. {max_count*5} missing days\nMissing from {str(filtered_ds_10.time.isel(time=max_index).values)} to {str(filtered_ds_10.time.isel(time=max_index+max_count-1).values)}')

            if max_count > max_nan:
                data['to_sample'].loc[dict(lat=lat, lon=lon)] = False
            else:
                data['to_sample'].loc[dict(lat=lat, lon=lon)] = True
                
    return data

def find_max_consec_nan(timeseries):
    """
    Returns maximum consecutive NaNs in a mask. Will also return the start index of these NaNs.
    """
    max_index = 0
    max_count = 0  # Initialize max_count to 0 for counting NaNs
    count = 0  # Initialize count to 0 for counting NaNs
    for i, x in enumerate(timeseries):
        if x:  # Check if the value is True (indicating NaN)
            count += 1
        else:
            if count > max_count:
                max_count = count
                max_index = i - count  # Update max_index to the start index of consecutive NaNs
            count = 0  # Reset count when a non-NaN value is encountered
    if count > max_count:  # Check if the maximum count occurs at the end of the series
        max_count = count
        max_index = len(timeseries) - count
    return max_count, max_index



import statsmodels.api as sm

def interpolate_loess(y, data_array, loess_frac):
    """
    Perform LOESS regression with NaN handling
    """
    
    # Get the time values as numeric representation
    time_numeric = np.arange(len(data_array['time']))
    mask = y.isnull().values
    smoothed_data = np.zeros_like(mask)

    # Perform LOESS regression with NaN handling
    if np.any(mask):  # Interpolate missing values
        x = time_numeric[~mask]
        y_interp = np.interp(time_numeric, x, y[~mask]) #linear interpolation
        lowess = sm.nonparametric.lowess(y_interp, time_numeric, frac=loess_frac)
    else:  # No missing values, perform LOESS directly
        lowess = sm.nonparametric.lowess(y, time_numeric, frac=loess_frac)

    # Extract the smoothed values
    smoothed_values = lowess[:, 1]

    return smoothed_values



def remove_lower_pct_per_week(cube, pct):
    """
    For each pixel timeseries and each Sentinel-2 variable, find pct percentile per week of year and remove values below.
    """
    
    s2_vars = [name for name in list(cube.variables) if name.startswith('s2')]
    filtered_cloud = cube[s2_vars].where(cube.s2_mask==0)

    # Find lower 10% per week of year for each pixel
    filtered_cloud['time'] = pd.to_datetime(filtered_cloud.time.values)
    ds_weekly = filtered_cloud.groupby('time.week')
    ds_percentile = ds_weekly.reduce(np.nanquantile, q=pct, dim='time')

    # Remove lower 10% per week of year for each pixel (for S2)
    
    filtered_ds = xr.zeros_like(filtered_cloud)
    for i, t in enumerate(filtered_cloud['time']):
        # Filter the data for that week
        data =  filtered_cloud.sel(time=t)
        quant = ds_percentile.sel(week=t.dt.week)
        above_quant = data.where(data>=quant)
        filtered_ds.loc[dict(time=t)] = above_quant
        
    cube[s2_vars] = filtered_ds
    
    return cube


def smooth_s2_timeseries(cube, max_nan, remove_pct, loess_frac):
    """
    Apply interpolation and smoothing to S2 timseries
    
    :param cube: xarray with data that needs to be smoothed
    :param max_nan: maximum number of consecutive NaNs allowed. If more, the pixel will not be used/sampled
    :param pct: percentile (scaled between 0 and 1) under which to drop values per week of year
    :param frac: the fraction of the data used when estimating each y-value in the LOESS smoothing
    """
    # Filter out clouds and lower 10 percentile
    cube = remove_lower_pct_per_week(cube, remove_pct)
    print('Cloud cleaning - removed lower 10% per week of yr')
    
    # First check if to interpolate or not
    cube = check_if_interp(cube, max_nan)
    print('Cloud cleaning - checked if interp')
    
    # Loop through variables and interpolate/smooth pixel timeseries for Sentinel-2
    for variable_name in cube.data_vars:
        if variable_name.startswith('s2_B') or variable_name.startswith('s2_ndvi'): # only smooth continuous variables

            # Select the variable (now just NDVI)
            variable = cube[variable_name]

            # Loop along the pixel dimensions (lat, lon)
            for i_lat, lat in enumerate(variable.lat):
                for i_lon, lon in enumerate(variable.lon):

                    # Check if should smooth the timeseries for that pixel
                    if cube.sel(lat=lat, lon=lon).to_sample.values:
                        subset = variable.isel(lat=i_lat, lon=i_lon)
                        smoothed_values = interpolate_loess(subset, cube, loess_frac)
                        # Replace with smoothed values
                        cube[variable_name].loc[dict(lat=lat, lon=lon)] = smoothed_values
                        
    return cube