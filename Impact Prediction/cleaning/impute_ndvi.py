"""
Impute NDVI
"""

"""
Create cloud corrected data samples from the raw samples

Authors: Selene Ledain (selene.ledain@ibm.com)
Feb 1st 2023
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
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from scipy.interpolate import interp1d
from scipy import signal
import pickle
from datetime import datetime, timedelta

def impute_ndvi(sample_type, exp_raw, exp, cp_idx, b2_idx, b8_idx, ndvi_idx, cp_thresh):
    """
    :param sample_type: pixel_data
    :param exp_raw: name of the experiment containing raw/uncleaned samples
    :param exp: name of the experiment with cleaning
    :param cp_idx: index of CP layer in tensor
    :param b2_idx: index of B2 layer in tensor
    :param b8_idx: index of B8 layer in tensor
    :param ndvi_idx: index of NDVI layer in tensor
    :param cp_thresh: threhsold of CP
    """
    
    # ---------------------------- Check that raw/new data folders exist -------------------------------
    print('Checking data...')
    
    path_to_data = f'{sample_type}/{exp_raw}/'
    path_save = f'{sample_type}/{exp}/'
    
    if not os.path.exists(path_to_data):
        raise Exception('No raw data to clean')
    
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        os.makedirs(f'{path_save}train/x/')
        os.makedirs(f'{path_save}train/y/')
        os.makedirs(f'{path_save}val/x/')
        os.makedirs(f'{path_save}val/y/')
        os.makedirs(f'{path_save}test/x/')
        os.makedirs(f'{path_save}test/y/')
        print(f"==> Created new {path_save}")
    else:
        print(f"==> {path_save} folder already exists no need to create")


    # --------------------------- Load and correct data ------------------------------------------------
    print('Starting data cleaning...')
    
    with open('full_date_range.pkl', 'rb') as f:
        full_date_range = pickle.load(f)
        
    with open('all_loc_dates.pkl', 'rb') as f:
        all_loc_dates = pickle.load(f)
        
    sample_list = [[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3]] #th.load(f'{path_to_data}train/called_samples.pt')
    
    process_sample(path_to_data, path_save, sample_type, 'train', cp_idx, b2_idx, b8_idx, ndvi_idx, cp_thresh, sample_list, all_loc_dates, full_date_range)
    #process_sample(path_to_data, path_save, sample_type, 'val',  cp_idx, b2_idx, b8_idx, ndvi_idx, b2_thresh, b8_thresh, ts_len, len_preds)
    #process_sample(path_to_data, path_save, sample_type, 'test',  cp_idx, b2_idx, b8_idx, ndvi_idx, b2_thresh, b8_thresh, ts_len, len_preds)
    
    print('Done!')
        
    return



def process_sample(path_to_data, path_save, sample_type, split, cp_idx, b2_idx, b8_idx, ndvi_idx, cp_thresh, sample_list, all_loc_dates, full_date_range):
    
    samples_dir = f"{path_to_data}{split}/x/"
    
    with ThreadPoolExecutor() as executor:
        future_to_sample = {executor.submit(process_sample_impl, path_to_data, path_save, sample_type, split, cp_idx, b2_idx, b8_idx, ndvi_idx, cp_thresh, sample_list, all_loc_dates, full_date_range, sample): sample for sample in os.listdir(samples_dir)}
        for future in concurrent.futures.as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                future.result()
            except Exception as exc:
                print(f'generated an exception: {exc}')
                

def process_sample_impl(path_to_data, path_save, sample_type, split, cp_idx, b2_idx, b8_idx, ndvi_idx, cp_thresh, sample_list, all_loc_dates, full_date_range, sample):
    sample_nbr = sample.split('cube_')[1]
    sample_x = th.load(f'{path_to_data}{split}/x/{sample}')
    sample_y = th.load(f'{path_to_data}{split}/y/data_cube_{sample_nbr}')
    sample_clean_x, sample_clean_y = impute_noisy_ndvi(sample_x, sample_y, cp_idx, b2_idx, b8_idx, ndvi_idx, cp_thresh, sample_list, sample_nbr, all_loc_dates, full_date_range)
    th.save(sample_clean_x, f'{path_save}{split}/x/data_cube_{sample_nbr}')
    th.save(sample_clean_y, f'{path_save}{split}/y/data_cube_{sample_nbr}')

    




def impute_noisy_ndvi(img, label, cp_idx, b2_idx, b8_idx, ndvi_idx, cp_thresh, sample_list, sample_nbr, all_loc_dates, full_date_range):

    """
    Change the NDVI value if cloud probability is too high

    Author: Selene
    :param sen_image_list: list of sentinel images at a date that have been sampled and read
    :param timestamp: timestamp of data (string)
    """
    
    # Get the date of the sample: idx in saved samples list
    # Can then idx train_ds.all_loc_dates to get list of dates
    sample_idx = sample_list[int(sample_nbr.split('.pt')[0])] #check if order in list is the same as order being called
    list_times = all_loc_dates[sample_idx][1] + all_loc_dates[sample_idx][3]
    
    img_tensor = img.clone().detach() 
    label_tensor = label.clone().detach() 
    
    # Get band data for image and label
    cp_data = th.cat([img_tensor[:,:,cp_idx,:,:] ,label_tensor[:,:,cp_idx,:,:]], axis=1).numpy()
    b2_data = th.cat([img_tensor[:,:,b2_idx,:,:] ,label_tensor[:,:,b2_idx,:,:]], axis=1).numpy()
    b8_data = th.cat([img_tensor[:,:,b8_idx,:,:] ,label_tensor[:,:,b8_idx,:,:]], axis=1).numpy()
    ndvi_data = th.cat([img_tensor[:,:,ndvi_idx,:,:] ,label_tensor[:,:,ndvi_idx,:,:]], axis=1).numpy()
    print('ok', ndvi_idx)

    # Loop through time:
    for t in list_times:
        ts = ts_sample[t]
        
        if np.any(cp_data[:,t,:,:,:]>cp_thresh):
            # Convert to datetime to find location in self.full_date_range
            timestamp = datetime.strptime(ts, '%Y-%m-%d')
            timestamp_idx = full_date_range.index(timestamp)
            ndvi_correct = get_ndvi_at_day(timestamp_idx) #Function in utils
            ndvi_data[cp_data[:,t,:,:,:]>cp_thresh] = ndvi_correct
            img_tensor[:,:,ndvi_idx,:,:] = ndvi_data
            cp_data[cp_data[:,t,:,:,:]>cp_thresh] = 0
            img_tensor[:,:,cp_idx,:,:] = cp_data


    return sen_image_list



def get_ndvi_at_day(timestamp_idx):
    """
    Get the NDVI value for the day (between 2015-08-01 and 2021-12-31 numbered in integers starting from 0)
    
    :param timestamp: index
    """
    
    """
    dayofyear = timestamp.timetuple().tm_yday #datetime.strptime(timestamp, '%j')
    ndvi_signal = np.load('correct_ndvi.npy') #saved as np array
    
    if dayofyear==366: #leap year
        return ndvi_signal[365-1] # The last 2 days will be the same
    
    return ndvi_signal[dayofyear-1] 
    """
    
    ndvi_signal = np.load('correct_ndvi_5.npy') #saved as np array
    
    return ndvi_signal[timestamp_idx] 

    


    
    
########################################################################
# CALL FUNCTION

parser = argparse.ArgumentParser()
parser.add_argument('--sample_type', type=str)
parser.add_argument('--exp_raw', type=str)
parser.add_argument('--exp', type=str)
parser.add_argument('--cp_idx', type=int)
parser.add_argument('--b2_idx', type=int)
parser.add_argument('--b8_idx', type=int)
parser.add_argument('--ndvi_idx', type=int)
parser.add_argument('--cp_thresh', type=float)
args = parser.parse_args()

impute_ndvi(args.sample_type, args.exp_raw, args.exp, args.cp_idx, args.b2_idx, args.b8_idx, args.ndvi_idx, args.cp_thresh)

    
#python impute_ndvi.py --sample_type pixel_data --exp_raw arch_exp --exp clean --cp_idx 14 --b2_idx 6 --b8_idx 12 --ndvi_idx 15 --cp_thresh 30 