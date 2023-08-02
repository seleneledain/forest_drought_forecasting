# Configuration file for minicube generation

coord_list_paths = ['/Users/led/Desktop/sampling_rasters/coords_sample_negative.txt'] 

specs = {
    "lon_lat": None,
    "xy_shape": (35, 35), # width, height of cutout around center pixel
    "resolution": 20, # in meters.. will use this on a local UTM grid..
    "time_interval": "2015-08-01/2021-12-31",
    "providers": [
        {
            "name": "s2",
            "kwargs": {"bands": ["B02", "B03", "B04", "B08", "B8A"], "best_orbit_filter": True, "five_daily_filter": True, "brdf_correction": True, "cloud_mask": True, "aws_bucket": "planetary_computer", "cloud_mask_rescale_factor":2}
        },
        {
          "name": "era5",
           "kwargs": {"bands": ['sp', 'tp', 'sr', 't', 'maxt', 'mint', 'ap', 'dp'], "aws_bucket": "s3", "n_daily_filter": None, "agg_list": ['mean', 'sum', 'sum', 'mean', 'max', 'min', 'mean', 'mean'], "match_s2": True} 
        }        
        ]
}


specs_add_bands = {
    "bands": ['SAND0_5', 'FED100_200', 'NDVI', "FOREST_MASK", "DEM", "DROUGHT_MASK"],
    "static_dir": '/Users/led/Desktop/' # Path_to_local_data
}

bands_to_drop = ["s2_mask", "s2_avail", "s2_SCL", "to_sample", "FOREST_MASK", "DROUGHT_MASK"]

# Where to save data
root_dir = '/Users/selene/Desktop/Unibe/New code/'
split = 'train_neg'

# Sample format
context = 54 # 9 months 5-daily 
target =  18 # 3 months 5-daily
shift = 18

# Cleaning
cloud_cleaning = 36 # max_count for cloud_cleaning
normalisation = True
remove_pct = 0.05 # Drop lower percentile of data per week of year
loess_frac = 0.07 # Percent of data to use for LOESS smoothing

# Samples criteria
target_in_summer = True
drought_labels = False 
forest_thresh = 0.8 
drought_thresh = 0
pixs_per_scene = 50

