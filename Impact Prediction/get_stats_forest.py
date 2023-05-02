""" 
Script to create data samples in a dataset.

Authors: 
Emine Didem Durukan (emine.didem.durukan@ibm.com)
Selene Ledain (selene.ledain@ibm.com)
"""

# Import packages
from pathlib import Path
import torch as th
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import time
from tqdm.auto import tqdm # Instantly make your loops show a smart progress meter
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import uuid
import pandas as pd
from torch.autograd import Variable 
import time
import argparse
import importlib


#mlflow
import mlflow as mlflow

#other scripts
from drought_impact_dataset import DroughtImpactDataset
from drought_impact_sampler import DroughtImpactSampler
from utils.utils_pixel import *
import glob
from collections import OrderedDict


def get_stats(split, sample_type, exp, start_idx, config_id, samples_per_job):
    """
    :param split: 'train', 'val', 'test'
    :param sample_type: 'pixel_data' or 'scene_data'
    :param exp: experiment name
    :param start_idx: number at which to start numbering the samples
    :param config_id: identifier for the congif file to use
    :param samples_per_job: number of smaples in a job, for slicing the dataloader
    """
 
    module = importlib.import_module(f'utils.{config_id}')
    train_dataset_params = module.train_dataset_params
    train_sampler_params = module.train_sampler_params
    val_dataset_params = module.val_dataset_params
    val_sampler_params = module.val_sampler_params
    test1_dataset_params = module.test1_dataset_params
    test2_dataset_params = module.test2_dataset_params
    test_sampler_params = module.test_sampler_params
    sim_params = module.sim_params
    model_params = module.model_params
    
   
    
    # ------------- Load info from config file -----------------------
    print("=================== Welcome to data cube creator!====================")
    print("==> Reading parameters...")
    
    # Common to all datasets
    s2_path =  Path(train_dataset_params['s2_path'])
    era_path =  Path(train_dataset_params['era_path'])
    dem_path =   Path(train_dataset_params['dem_path'])
    env_path =  Path(train_dataset_params['env_path'])
    ts_delta =  train_dataset_params['ts_delta']
    ts_len =  train_dataset_params['ts_len']
    len_preds =  train_dataset_params['len_preds']
    ratio = train_dataset_params['ratio']
    data_file_extension = train_dataset_params['data_file_extension'] 
    feature_set = train_dataset_params['feature_set']
    remove_bands = train_dataset_params['remove_bands']
    agg_funct_dict = train_dataset_params['agg_funct_dict']
    multiple_labels = train_dataset_params['multiple_labels']
    correct_ndvi = train_dataset_params['correct_ndvi']
    
    focus_time_train = train_dataset_params['focus_time']
    focus_list_train = train_dataset_params['focus_list']
    sampler_roi_train = train_sampler_params['roi']
    batch_size_tr =  train_sampler_params['batch_size'] 
    n_batch = sim_params['n_batches']
    
    # Common to all samplers
    sampler_size = train_sampler_params["size"]
    sampler_replacement =  train_sampler_params['replacement'] 
    mask_dir =  train_sampler_params['mask_dir'] 
    sampler_set_seed =  train_sampler_params['set_seed'] 
    static_dir = train_sampler_params['static_dir'] 
    mask_threshold = 0.8 #train_sampler_params["mask_threshold"]

    print("==> Read parameters!")
    
    
    
    # ------------- Initialise datasets, samplers, loaders -----------------------
    print("==> Creating dataset, sampler, loader class....")
             
   
    # Create dataset
    train_ds = DroughtImpactDataset(s2_path=s2_path, era_path=era_path, env_path=env_path, dem_path=dem_path, focus_list=focus_list_train,
                          focus_time=[focus_time_train], ts_delta=ts_delta, ts_len=ts_len, ratio=ratio, len_preds=len_preds, feature_set=feature_set, agg_funct_dict=agg_funct_dict, multiple_labels=multiple_labels, correct_ndvi=correct_ndvi)
    # Create sampler
    train_sampler = DroughtImpactSampler(train_ds, size=sampler_size, length=batch_size_tr*n_batch, replacement=sampler_replacement, 
                                     mask_dir=mask_dir, roi=sampler_roi_train, set_seed=sampler_set_seed, static_dir=static_dir, 
                                         mask_threshold=mask_threshold)
    # Create dataloader
    train_dl = DataLoader(train_ds, sampler=train_sampler, num_workers=4)


    print("==> Created dataset objects!")
    
    
    
    # -------------------------------- Create folders for storing data-------------------------
    print("==> Create storage folders...")

    # Check if the folders exist
    folders = [f'{sample_type}/{exp}/{split}/x/',f'{sample_type}/{exp}/{split}/y/']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"==> Created {folder}")
        else:
            print(f"==> {folder} folder already exists no need to create")
    
    
    
    #---------------------- Do statistics on the training set to normalize the entire dataset--------------------------
    print("==> Calculating dataset statistics...")
    # Temporal bands
    bands = []
    n_temp = train_ds.bands_s2 + train_ds.bands_era
    for i in list(train_ds.feature_set.keys())[:n_temp]:
        bands.append(train_ds.feature_set[i])
    tmp_bands_vals, tmp_band_means_or_mins, tmp_band_stds_or_maxs = dataset_stats(train_dl, bands=bands, temporal=True, norm_method=sim_params["norm_method"], 
                                                                                  start_idx=start_idx, samples_per_job=samples_per_job)

    # Static bands
    bands = []
    for i in list(train_ds.feature_set.keys())[n_temp:]:
        bands.append(train_ds.feature_set[i])
    stat_bands_vals, stat_band_means_or_mins, stat_band_stds_or_maxs = dataset_stats(train_dl, bands=bands, temporal=False, norm_method=sim_params["norm_method"],
                                                                                     start_idx=start_idx, samples_per_job=samples_per_job)

    # First temporal, then static
    all_band_vals = list(tmp_bands_vals) + list(stat_bands_vals)
    all_band_means_or_mins = list(tmp_band_means_or_mins) + list(stat_band_means_or_mins)
    all_band_stds_or_maxs = list(tmp_band_stds_or_maxs) + list(stat_band_stds_or_maxs)

    # If already exist, need to recompute
    #save_dataset_stats([all_band_means_or_mins,all_band_stds_or_maxs], sample_type, exp)
    save_dataset_stats([all_band_means_or_mins,all_band_stds_or_maxs], sample_type, exp, config_id +f'_{start_idx}')

    
    print("==> Finished dataset stats!")            

    return









########################################################################
# CALL FUNCTION

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str)
parser.add_argument('--sample_type', type=str)
parser.add_argument('--exp', type=str)
parser.add_argument('--start_idx', type=int)
parser.add_argument('--config_id', type=str)
parser.add_argument('--samples_per_job', type=int)
args = parser.parse_args()

get_stats(args.split, args.sample_type, args.exp, args.start_idx, args.config_id, args.samples_per_job)
