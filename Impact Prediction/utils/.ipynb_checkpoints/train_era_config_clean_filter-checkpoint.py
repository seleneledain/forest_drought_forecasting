"""
Experiment and training configuration file for drought impact forecasting models
Authors:    Selene Ledain (selene.ledain@ibm.com)
Created:    22 Jan 2023
"""



sim_params = {
    "learning_rate": 0.0001,
    "num_epochs": 80,
    "optimizer": "Adam",
    "momentum":0.9,
    "loss_function": "MSE",
    "n_batches": 559,
    "method":'dir',
    "norm_method": 'minmax',
    "num_steps":1,
    "exp": "filtered_clean",
    "exp_val": "filtered_clean",
    "exp_test": "filtered_clean",
    "sample_type": "pixel_data",
    "batch_size_val": 40,
    "n_batches_val": 58,
    "batch_size_te": 40,
    "n_batches_te": 42,
    "cp_idx": None #(0,1)
}



train_sampler_params = {
    # On CCC
    "size": [1,1],
    "batch_size": 40, 
    "replacement": False,
    "mask_dir": "/dccstor/cimf/drought_impact/downloads/forest_mask", #default None
    "set_seed": True, #default True
    "roi":  [7.142, 7.632, 46.912, 47.33], #default None
    "mask_threshold":0.8,
    "static_dir":"/dccstor/cimf/drought_impact/downloads/ENV_DATA"
}


model_params = {
    "hidden_dim": 10,
    "num_layers": 2,
    "output_dim": 1 #n_feats_out
}





############################
# NOT IMPORTANT DURING TRAIING AS SAMPLES ARE ALREADY CREATED
# Only model_params and multiple_labels could be important



train_dataset_params = {
    # On CCC
    "s2_path": "/dccstor/cimf/drought_impact/downloads/SENTINEL 2/", #default None
    "era_path": "/dccstor/cimf/drought_impact/downloads/ERA5/", #default None
    "dem_path": "/dccstor/cimf/drought_impact/downloads/DEM_ch/21", #default None
    "env_path": "/dccstor/cimf/drought_impact/downloads/ENV_DATA", #default None
    "ts_delta": 10, #default None
    "ts_len":9, #default None
    "len_preds":1, #default 1
    "focus_time": ['2018-01-01', '2019-12-31'], #default [None, None]
    "focus_list": [[46.907, 7.137, 47.407, 7.637]], #default []
    "ratio": 0.5, #default 0.5
    "data_file_extension": "tiff",
    "feature_set": ["MSI", "NDMI", "NDVIre", "NDWI", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "CP", "NDVI", "SCL",
               "VPDX", "VPDN", "AWVC", "MAXT", "MINT", "SR" ,"SP", "T", "TCC", "TP", 
               "DEM", 
               "CURV17", "CURV19", "CURV21", "EAST17", "EAST19", "EAST21", "NORT17", "NORT19", "NORT21", "RUGG17", "RUGG19", "RUGG21", "SLOP17", "SLOP19", "SLOP21", "TWI17", "TWI19", "TWI21",
               "CLAY0_5", "CLAY100_200", "CLAY15_30", "CLAY30_60", "CLAY5_15", "CLAY60_100",
               "FED0_5", "FED100_200", "FED15_30",  "FED30_60", "FED5_15", "FED60_100",
               "FC", "FH",
               "GRAV0_5", "GRAV100_200", "GRAV15_30", "GRAV30_60", "GRAV5_15", "GRAV60_100",
               "SAND0_5", "SAND100_200", "SAND15_30", "SAND30_60","SAND5_15", "SAND60_100",
               "SDEP",
               "CARB0_5", "CARB100_200", "CARB15_30", "CARB30_60", "CARB5_15", "CARB60_100",
               "PH0_5", "PH100_200", "PH15_30", "PH30_60","PH5_15", "PH60_100"],
    "remove_bands": ["CP", "SCL", "VPDX", "VPDN", "AWVC", "MAXT", "MINT", "SR" ,"SP", "T", "TCC", "TP", "SAND60_100"],
    "agg_funct_dict": {"MSI":"mean", "NDMI":"mean", "NDVIre":"mean", "NDWI":"mean", "B11":"mean", "B12":"mean", "B02":"mean", "B03":"mean", "B04":"mean", "B05":"mean", "B06":"mean", "B07":"mean", "B08":"mean", "B8a":"mean", "CP":"mean", "NDVI":"mean", "SCL":"mean", "TCC":"mean", "vpd_min":"min", "vpd_max":"max", "AWVC":"mean", "MaxT":"max", "MinT":"min", "SR":"sum","SP":"mean", "Temp":"mean", "Total Prec":"sum"}, # naming format should be same as in filename
    "multiple_labels": True,
    "correct_ndvi":None
    #"agg_time": True,
    #"keep_out_list":[],    
}



val_dataset_params = { #Same as train, only change parts that differ
    # On CCC
    "focus_time": ['2019-05-01', '2019-09-30'], #default [None, None]
    "focus_list": [[46.907, 7.137, 47.407, 7.637]], #default []    
}



val_sampler_params = {
    # On CCC
    "roi": [7.142, 7.632, 46.912, 47.33], #default None
    "length": 4000
    #"mask_threshold":0 
}


test_dataset_params = { #Same as train, only change parts that differ
    # On CCC
    "focus_time": ['2020-05-01', '2020-09-30'], #default [None, None]
    "focus_list": [[46.907, 7.137, 47.407, 7.637]], #default []    
}



test_sampler_params = {
    # On CCC
    "roi": [7.142, 7.632, 46.912, 47.33], #default None
    "length": 4000
    #"mask_threshold":0 
}

