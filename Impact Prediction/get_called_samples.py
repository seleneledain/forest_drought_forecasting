""" 
Script to create data samples in a dataset.

Authors: 
Emine Didem Durukan (emine.didem.durukan@ibm.com)
Selene Ledain (selene.ledain@ibm.com)
"""

# Import packages
from pathlib import Path
import torch as th
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


def get_called_samples(split, sample_type, exp, start_idx, config_id, samples_per_job, name_start_idx):
    """
    :param split: 'train', 'val', 'test'
    :param sample_type: 'pixel_data' or 'scene_data'
    :param exp: experiment name
    :param start_idx: number at which to start numbering the samples
    :param config_id: identifier for the congif file to use
    :param samples_per_job: number of smaples in a job, for slicing the dataloader
    :param name_start_idx: idx where the naming starts in the dataset
    """
 
    module = importlib.import_module(f'utils.{config_id}')
    train_dataset_params = module.train_dataset_params
    train_sampler_params = module.train_sampler_params
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
    
    if split=='train':
        focus_time_train = train_dataset_params['focus_time']
        focus_list_train = train_dataset_params['focus_list']
        sampler_roi_train = train_sampler_params['roi']
        batch_size_tr = train_sampler_params['batch_size'] 
        n_batch = sim_params['n_batches']
    if split=='val':
        val_dataset_params = module.val_dataset_params
        val_sampler_params = module.val_sampler_params
        focus_time_val = val_dataset_params['focus_time']
        focus_list_val = val_dataset_params['focus_list']
        sampler_roi_val = val_sampler_params['roi']
        sampler_length_val = val_sampler_params['length'] 
    if 'test' in split:
        test_dataset_params = module.test_dataset_params
        test_sampler_params = module.test_sampler_params
        focus_list_test = test_dataset_params['focus_list']
        focus_time_test = test_dataset_params['focus_time']
        sampler_roi_test = test_sampler_params['roi']
        sampler_length_te = test_sampler_params['length'] 
    
    # Common to all samplers
    sampler_size = train_sampler_params["size"]
    sampler_replacement =  train_sampler_params['replacement'] 
    mask_dir =  train_sampler_params['mask_dir'] 
    mask_threshold = train_sampler_params["mask_threshold"]
    sampler_set_seed =  train_sampler_params['set_seed'] 
    static_dir = train_sampler_params['static_dir'] 

    print("==> Read parameters!")
    
    
    
    
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
    
    


    # -------------------------------- Normalise data and set-up dataloaders-------------------------
    print("==> Creating dataset...")
           

    if split=='train':
        # Dataset
        ds = DroughtImpactDataset(s2_path=s2_path, era_path=era_path, dem_path=dem_path, env_path=env_path, focus_list=focus_list_train,
                              focus_time=[focus_time_train], ts_delta=ts_delta, ts_len=ts_len, ratio=ratio, len_preds=len_preds, feature_set=feature_set, agg_funct_dict=agg_funct_dict, multiple_labels=multiple_labels, correct_ndvi=correct_ndvi)
        # Sampler
        train_sampler = DroughtImpactSampler(ds, size=sampler_size, length=batch_size_tr*n_batch, replacement=sampler_replacement, 
                                         mask_dir=mask_dir, set_seed=sampler_set_seed, roi=sampler_roi_train, mask_threshold=mask_threshold, static_dir=static_dir)
        # Loader
        DL = DataLoader(ds, sampler=train_sampler) 
        
        
    if split=='val':
        
        # Dataset
        ds = DroughtImpactDataset(s2_path=s2_path, era_path=era_path, dem_path=dem_path, env_path=env_path, focus_list=focus_list_val,
                              focus_time=[focus_time_val], ts_delta=ts_delta, ts_len=ts_len, ratio=ratio, len_preds=len_preds, feature_set=feature_set, agg_funct_dict=agg_funct_dict, multiple_labels = multiple_labels, correct_ndvi=correct_ndvi)
        # Sampler
        val_sampler = DroughtImpactSampler(ds, size=sampler_size, length=sampler_length_val, replacement=sampler_replacement, 
                                         mask_dir=mask_dir, set_seed=sampler_set_seed, roi=sampler_roi_val, mask_threshold=mask_threshold, static_dir=static_dir)
        # Loader
        DL = DataLoader(ds, sampler=val_sampler) 
        

    
    if 'test' in split: # could also provide a split called test2

        # Dataset
        ds = DroughtImpactDataset(s2_path=s2_path, era_path=era_path, dem_path=dem_path, env_path=env_path, focus_list=focus_list_test,
                              focus_time=[focus_time_test], ts_delta=ts_delta, ts_len=ts_len, ratio=ratio, len_preds=len_preds, feature_set=feature_set, agg_funct_dict=agg_funct_dict, multiple_labels = multiple_labels, correct_ndvi=correct_ndvi)
        # Sampler
        test_sampler = DroughtImpactSampler(ds, size=sampler_size, length=sampler_length_te, replacement=sampler_replacement, 
                                         mask_dir=mask_dir, set_seed=sampler_set_seed, roi=sampler_roi_test, mask_threshold=mask_threshold, static_dir=static_dir)
        # Dataloader
        DL = DataLoader(ds, sampler=test_sampler)
        
    
    
    
    print("==> Created!")
    
    
    
    #---------------------- Save called data samples--------------------------
    print("==> Getting called samples...")
    
    save_called_samples(DL, sample_type, split, exp, config_id, start_idx, samples_per_job)
    
    print("==> Saved!")            

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
parser.add_argument('--name_start_idx', type=int)
args = parser.parse_args()

get_called_samples(args.split, args.sample_type, args.exp, args.start_idx, args.config_id, args.samples_per_job, args.name_start_idx)
