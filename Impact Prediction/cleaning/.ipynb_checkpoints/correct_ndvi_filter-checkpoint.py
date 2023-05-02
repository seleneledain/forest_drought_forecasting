"""
Create cloud corrected data samples from the raw samples

Authors: Selene Ledain (selene.ledain@ibm.com)
Feb 1st 2023
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
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from scipy.interpolate import interp1d
from scipy import signal
import pickle
from datetime import datetime, timedelta



def correct_ndvi_filter(sample_type, exp_raw, exp, cp_idx, b2_idx, ndvi_idx, b2_thresh, ts_len, len_preds, split):
    """
    :param sample_type: pixel_data
    :param exp_raw: name of the experiment containing raw/uncleaned samples
    :param exp: name of the experiment with cleaning
    :param cp_idx: index of CP layer in tensor
    :param b2_idx: index of B2 layer in tensor
    :param b8_idx: index of B8 layer in tensor
    :param ndvi_idx: index of NDVI layer in tensor
    :param b2_thresh: threhsold of B2 
    :param b8_thresh: threhsold of B8 
    :param ts_len: length of context tensor
    :param len_preds: legnth of prediction tensor
    """
    
    # ---------------------------- Check that raw/new data folders exist -------------------------------
    print('Checking data...')
    
    path_to_data = f'{sample_type}/{exp_raw}/'
    path_save = f'{sample_type}/{exp}/'
    
    if not os.path.exists(path_to_data):
        raise Exception('No raw data to clean')
    
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    else:
        print(f"==> {path_save} folder already exists no need to create")
        
    if not os.path.exists(path_save+f'{split}'):
        os.makedirs(f'{path_save}{split}/x/')
        os.makedirs(f'{path_save}{split}/y/')
        print(f"==> Created new {path_save}{split}")
    else:
        print(f"==> {path_save}{split} folder already exists no need to create")
        
    
    # ---------------------------- Get metadata -------------------------------
    print('Checking data...')
        
    metadata_path = f'{sample_type}/{exp_raw}/{split}/'

    # Get date/range of pred
    all_loc_dates_files = sorted([f for f in os.listdir(metadata_path) if f.startswith('all_loc_dates')])
    full_date_range_files = sorted([f for f in os.listdir(metadata_path) if f.startswith('full_date_range')])

    all_loc_dates = []
    for f in all_loc_dates_files:
        with open(metadata_path+f, 'rb') as f :
            all_loc_dates += pickle.load(f)

    full_date_range = []

    for f in full_date_range_files:
        with open(metadata_path+f, 'rb') as f :
            full_date_range += pickle.load(f)
    

    # --------------------------- Load and correct data ------------------------------------------------
    print('Starting data cleaning...')
    
    process_sample(path_to_data, path_save, sample_type, split,  cp_idx, b2_idx, ndvi_idx, b2_thresh, ts_len, len_preds, all_loc_dates, full_date_range)
    
    print('Done!')
        
    return



def process_sample(path_to_data, path_save, sample_type, split,  cp_idx, b2_idx, ndvi_idx, b2_thresh, ts_len, len_preds, all_loc_dates, full_date_range):
    
    samples_dir = f"{path_to_data}{split}/x/"
    len_dataset = len([f for f in os.listdir(samples_dir)])
    
    with ThreadPoolExecutor() as executor:
        future_to_sample = {executor.submit(process_sample_impl, path_to_data, path_save, sample_type, split, cp_idx, b2_idx, ndvi_idx, b2_thresh, ts_len, len_preds, all_loc_dates, full_date_range, len_dataset, sample): sample for sample in os.listdir(samples_dir)}
        for future in concurrent.futures.as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                future.result()
            except Exception as exc:
                print(f'generated an exception: {exc}')
                

def process_sample_impl(path_to_data, path_save, sample_type, split, cp_idx, b2_idx, ndvi_idx, b2_thresh, ts_len, len_preds, all_loc_dates, full_date_range, len_dataset, sample):
    sample_nbr = sample.split('cube_')[1]
    sample_x = th.load(f'{path_to_data}{split}/x/{sample}')
    sample_y = th.load(f'{path_to_data}{split}/y/data_cube_{sample_nbr}')
    sample_clean_x, sample_clean_y = correct_noisy_s2(sample_x, sample_y, cp_idx, b2_idx, ndvi_idx, b2_thresh, ts_len, len_preds, sample_nbr, all_loc_dates, full_date_range, len_dataset)
    th.save(sample_clean_x, f'{path_save}{split}/x/data_cube_{sample_nbr}')
    th.save(sample_clean_y, f'{path_save}{split}/y/data_cube_{sample_nbr}')

    



def correct_noisy_s2(img, label, cp_idx, b2_idx, ndvi_idx, b2_thresh, ts_len, len_preds, sample_nbr, all_loc_dates, full_date_range, len_dataset):
    """
    Impute NDVI through linear interpolation if blue band (B2) > b2_thresh and infrared band (B8) < b8_thresh.
    Use both img and label for interpolation.
    
    Author: Selene
    :param img: data tensor
    :param label: label tensor
    :param dataset: dataset
    :param b2_thresh: threhsold for filtering with blue band (0.1 typically)
    :param b8_thresh: threhsold for filtering with infrared band (0.15 typically)
    :param ts_len: number of timestamps in data tensor
    :param len_preds: number of timestamps in label tensor
    """
    
    img_tensor = img.clone().detach() 
    label_tensor = label.clone().detach()[:,0,:,:,:].unsqueeze(1)
    
    # Get band data for image and label
    cp_data = th.cat([img_tensor[:,:,cp_idx,:,:] ,label_tensor[:,:,cp_idx,:,:]], axis=1)
    b2_data = th.cat([img_tensor[:,:,b2_idx,:,:] ,label_tensor[:,:,b2_idx,:,:]], axis=1)

    # Filter: find NDVI that needs to be replaced
    to_rep = ((b2_data>b2_thresh) & (cp_data>0.01)) #((b2_data>b2_thresh) | (b8_data<b8_thresh)) #(cp_data>0)
    to_rep = to_rep.squeeze(0).squeeze(1).squeeze(1) # make it 1D
    
    if to_rep.sum() < ts_len+len_preds-1: # There needs to be at least 2 points for inteprolation
        # Linear interpolation for each S2 band
        for band in range(16):
            if band != cp_idx and band != b2_idx:
                band_data = th.cat([img_tensor[:,:,band,:,:] ,label_tensor[:,:,band,:,:]], axis=1)

                band_vec = band_data.squeeze(0).squeeze(1).squeeze(1) # make it 1D
          
                x = np.arange(band_data.shape[1])[~to_rep] # get the x values that are valid
                #f = interp1d(x, band_vec[~to_rep], fill_value="extrapolate")
                # get the index of the first False value (first that won't be dropped)
                index = (to_rep == False).nonzero()[0]
                f = interp1d(x, band_vec[~to_rep], fill_value=band_vec[index], bounds_error=False)
                interpolated = f(np.arange(ts_len+len_preds))

                # Make sure its between 0 and 1
                interpolated[interpolated < 0] = 0
                interpolated[interpolated > 1] = 1

                # Replace band values
                img_tensor[:,:,band,:,:] = th.from_numpy(interpolated[:ts_len]).unsqueeze(0).unsqueeze(2).unsqueeze(2)
                label_tensor[:,:,band,:,:] = th.from_numpy(interpolated[ts_len:]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

            # Set CP to 0 for all values (or drop CP as feature)
            img_tensor[:,:,cp_idx,:,:] = 0
            label_tensor[:,:,cp_idx,:,:] = 0
            
            img_tensor[:,:,b2_idx,:,:] = b2_thresh
            label_tensor[:,:,b2_idx,:,:] = b2_thresh

    
    # If everything gets dropped
    else:
        while to_rep.sum() >= ts_len+len_preds-1:
            b2_thresh += 0.05
            
            to_rep =  ((b2_data>b2_thresh) & (cp_data>0.01))#((b2_data>b2_thresh+0.1) | (b8_data<b8_thresh-0.05))
            to_rep = to_rep.squeeze(0).squeeze(1).squeeze(1) # make it 1D
        
        # Linear interpolation for each band
        for band in range(16):
            if band != cp_idx and band != b2_idx:
                band_data = th.cat([img_tensor[:,:,band,:,:] ,label_tensor[:,:,band,:,:]], axis=1)

                band_vec = band_data.squeeze(0).squeeze(1).squeeze(1) # make it 1D

                x = np.arange(band_data.shape[1])[~to_rep] # get the x values that are valid
                #f = interp1d(x, band_vec[~to_rep], fill_value="extrapolate")
                # get the index of the first False value (first that won't be dropped)
                index = (to_rep == False).nonzero()[0]
                f = interp1d(x, band_vec[~to_rep], fill_value=band_vec[index], bounds_error=False)
                interpolated = f(np.arange(ts_len+len_preds))

                # Make sure its between 0 and 1
                interpolated[interpolated < 0] = 0
                interpolated[interpolated > 1] = 1

                # Replace band values
                img_tensor[:,:,band,:,:] = th.from_numpy(interpolated[:ts_len]).unsqueeze(0).unsqueeze(2).unsqueeze(2)
                label_tensor[:,:,band,:,:] = th.from_numpy(interpolated[ts_len:]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        # Set CP to 0 for all values
        img_tensor[:,:,cp_idx,:,:] = 0
        label_tensor[:,:,cp_idx,:,:] = 0
        
        img_tensor[:,:,b2_idx,:,:] = b2_thresh
        label_tensor[:,:,b2_idx,:,:] = b2_thresh
        
        
    if to_rep[-1]:
        print(sample_nbr)
        

    return img_tensor, label_tensor



    
    
########################################################################
# CALL FUNCTION

parser = argparse.ArgumentParser()
parser.add_argument('--sample_type', type=str)
parser.add_argument('--exp_raw', type=str)
parser.add_argument('--exp', type=str)
parser.add_argument('--cp_idx', type=int)
parser.add_argument('--b2_idx', type=int)
parser.add_argument('--ndvi_idx', type=int)
parser.add_argument('--b2_thresh', type=float)
parser.add_argument('--ts_len', type=int)
parser.add_argument('--len_preds', type=int)
parser.add_argument('--split', type=str)
args = parser.parse_args()

correct_ndvi_filter(args.sample_type, args.exp_raw, args.exp, args.cp_idx, args.b2_idx, args.ndvi_idx, args.b2_thresh, args.ts_len, args.len_preds, args.split)

                            
# python correct_ndvi_filter.py --sample_type pixel_data --exp_raw filtered --exp filtered_clean --cp_idx 14 --b2_idx 6 --ndvi_idx 15 --b2_thresh 0.1 --ts_len 9 --len_preds 1 --split test2