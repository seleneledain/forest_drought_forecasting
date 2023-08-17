# Configuration file for minicube generation

coord_list_paths = ['/data/scratch/selene/sampling_rasters/coords_sample_negative.txt'] 

specs = {
    "lon_lat": None,
    "xy_shape": (128, 128), # width, height of cutout around center pixel
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
    "bands": ['NDVI', "FOREST_MASK", "DEM", "DROUGHT_MASK", 
              "CLAY0_5", "CLAY100_200", "CLAY15_30", "CLAY30_60", "CLAY5_15", "CLAY60_100", 
              "FED0_5", "FED100_200", "FED15_30", "FED30_60", "FED5_15", "FED60_100", 
              "FC", "FH",
              "GRAV0_5", "GRAV100_200", "GRAV15_30", "GRAV30_60", "GRAV5_15", "GRAV60_100",
              "SAND0_5", "SAND100_200", "SAND15_30", "SAND30_60", "SAND5_15", "SAND60_100",
              "SDEP",
              "CARB0_5", "CARB100_200", "CARB15_30", "CARB30_60", "CARB5_15", "CARB60_100",
              "PH0_5", "PH100_200", "PH15_30", "PH30_60", "PH5_15", "PH60_100",
              "DEM_100", "DEM_500", "slope", "easting", "northing", "rugg", "curv", "twi",
              "slope_100", "easting_100", "northing_100", "rugg_100", "curv_100", "twi_100",
              "slope_500", "easting_500", "northing_500", "rugg_500", "curv_500", "twi_500"],
    "static_dir":  '/data/scratch/selene/static_data/' # Path_to_local_data
}

bands_to_drop = ["s2_mask", "s2_avail", "s2_SCL", "to_sample", "FOREST_MASK", "DROUGHT_MASK"]

# Where to save data
root_dir = '/data/scratch/selene/' # Path_to_local_data/selene/Desktop/Unibe/New code/'
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
pixs_per_scene = 100
gen_samples = False

