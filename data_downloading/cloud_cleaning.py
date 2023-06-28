"""
Replace cloud covered values in the dataset

Author: Selene Ledain
Date: June 23rd, 2023
"""

import numpy as np
import xarray as xr

def check_if_interp(data_array, max_nan):
    """
    For each pixel timeseries of NDVI, will check if there are sufficient values to allow timeseries smoothing (not too many NaN).
    
    :param data_array: xarray with data that needs to be smoothed
    :param max_nan: maximum number of consecutive NaNs allowed. If more, the pixel will not be used/sampled
    """
    
    #for variable_name in filtered_ds_10.data_vars:
    #if variable_name.startswith('s2'):

    # Select the variable (now just NDVI)
    variable = data_array['s2_ndvi']

    # Initialise a new variable to store the data check
    new_variable = xr.DataArray(np.zeros((len(variable.lat), len(variable.lon))),
                                dims=('lat', 'lon'),
                                coords={'lat': variable.lat, 'lon': variable.lon})

    # Add the new variable to the dataset
    data_array['to_sample'] = new_variable

    # Loop along the pixel dimensions (lat, lon)
    for i_lat, lat in enumerate(variable.lat):
        for i_lon, lon in enumerate(variable.lon):

            # Access the data for the current pixel
            subset = variable.sel(lat=lat, lon=lon)

            # If all nan to skip and add indictator
            if subset.isnull().all():
                data_array['to_sample'].loc[dict(lat=lat, lon=lon)] = False

            # Compute max number of consecutive
            max_count, max_index = find_max_consec_nan(subset.isnull().values)
            #print(f'{max_count} consecutive missing timestamps - i.e. {max_count*5} missing days\nMissing from {str(filtered_ds_10.time.isel(time=max_index).values)} to {str(filtered_ds_10.time.isel(time=max_index+max_count-1).values)}')

            if max_count > max_nan:
                data_array['to_sample'].loc[dict(lat=lat, lon=lon)] = False
            else:
                data_array['to_sample'].loc[dict(lat=lat, lon=lon)] = True
                
    return data_array

def find_max_consec_nan(timeseries):
    """
    Returns maximum consecutive nans in a mask. Will also return the start index of these nans.
    """
    max_index = 0
    max_count = 1
    max_num = timeseries[0]
    count = 1
    for i, x in enumerate(timeseries):
        if i < len(timeseries)-1:
            next_t = timeseries[i+1]
            if x != next_t:
                if count > max_count:
                    max_count = count
                    max_num = x
                count = 1
                max_index = i + 1
            else:
                count += 1
    return count, max_index



import statsmodels.api as sm

def interpolate_loess(y, data_array):
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
        lowess = sm.nonparametric.lowess(y_interp, time_numeric, frac=0.1)
    else:  # No missing values, perform LOESS directly
        lowess = sm.nonparametric.lowess(y, time_numeric, frac=0.1)

    # Extract the smoothed values
    smoothed_values = lowess[:, 1]

    return smoothed_values



def remove_lower_10_per_week(cube):
    """
    For each pixel timeseries and each variable, find 10th percentile per week of year and remove values below.
    """

    filtered_cloud = cube.where(cube.s2_mask==0)

    # Find lower 10% per week of year for each pixel
    filtered_cloud['time'] = pd.to_datetime(filtered_cloud.time.values)
    ds_weekly = filtered_cloud.groupby('time.week')
    ds_10th_percentile = ds_weekly.reduce(np.nanquantile, q=0.1, dim='time')

    # Remove lower 10% per week of year for each pixel
    filtered_ds_10 = xr.zeros_like(filtered_cloud)
    for i, t in enumerate(filtered_cloud['time']):
        # Filter the data for that week
        data =  filtered_cloud.sel(time=t)
        quant = ds_10th_percentile.sel(week=t.dt.week)
        above_quant = data.where(data>=quant)
        filtered_ds_10.loc[dict(time=t)] = above_quant
        
    return filtered_ds_10


def smooth_s2_timeseries(cube, max_nan):
    """
    Apply interpolation and smoothing to S2 timseries
    
    :param cube: xarray with data that needs to be smoothed
    :param max_nan: maximum number of consecutive NaNs allowed. If more, the pixel will not be used/sampled
    """
    # Filter out clouds and lower 10 percentile
    #cube = remove_lower_10_per_week(cube)
    
    # First check if to interpolate or not
    cube = check_if_interp(cube, max_nan)
    
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
                        smoothed_values = interpolate_loess(subset, cube)
                        # Replace with smoothed values
                        cube[variable_name].loc[dict(lat=lat, lon=lon)] = smoothed_values
                        
    return cube