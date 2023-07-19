# Configuration file for minicube generation

coord_list_paths = ['/Users/led/Desktop/sampling_rasters/coords_sample_negative.txt'] #, '/Users/led/Desktop/sampling_rasters/coords_drought.txt'] 

specs = {
    "lon_lat": None, #(6.73570, 46.93912), # center pixel Creux du Van
    "xy_shape": (35, 35), # width, height of cutout around center pixel
    "resolution": 20, # in meters.. will use this on a local UTM grid..
    "time_interval": "2015-08-01/2022-12-31",
    "providers": [
        {
            "name": "s2",
            "kwargs": {"bands": ["B02", "B03", "B04", "B08", "B8A"], "best_orbit_filter": True, "five_daily_filter": False, "brdf_correction": True, "cloud_mask": True, "aws_bucket": "planetary_computer", "cloud_mask_rescale_factor":2}
        },
        {
          "name": "era5",
           "kwargs": {"bands": ['sp', 'tp', 'sr', 't', 'maxt', 'mint', 'ap', 'dp'], "aws_bucket": "s3", "n_daily_filter": None, "agg_list": ['mean', 'sum', 'sum', 'mean', 'max', 'min', 'mean', 'mean'], "match_s2": True} 
        }        
        ]
}


specs_add_bands = {
    "bands": ['SAND0_5', 'FED100_200', 'NDVI', "FOREST_MASK", "DEM", "DROUGHT_MASK"],
    "static_dir": '/Users/led/Desktop/Master_Thesis/Data/original_data/soil_maps/' # Path_to_local_data
}

bands_to_drop = ["s2_mask", "s2_avail", "s2_SCL", "to_sample", "FOREST_MASK", "DROUGHT_MASK"]

root_dir = '/Users/led/Desktop/New code/'
split = 'train_full'
context = 54 # 9 months 5-daily 
target =  18 # 3 months 5-daily
shift = 18
cloud_cleaning = 18 # max_count for cloud_cleaning
normalisation = True
target_in_summer = True
drought_labels = False # Change if using drought coord
