# Configuration file for minicube generation

specs = {
    "lon_lat": (6.73570, 46.93912), # center pixel Creux du Van
    "xy_shape": (30, 30), # width, height of cutout around center pixel
    "resolution": 20, # in meters.. will use this on a local UTM grid..
    "time_interval": "2018-08-01/2018-09-30",
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
    "bands": ['SAND0_5', 'FED100_200', 'NDVI'],
    "static_dir": '/Users/led/Desktop/Master_Thesis/Data/original_data/soil_maps/' # Path_to_local_data
}

root_dir = ''
split = 'train'
context = 1
target = 1
shift=2
cloud_cleaning= 15 #max_count for cloud_cleaning
normalisation=False
