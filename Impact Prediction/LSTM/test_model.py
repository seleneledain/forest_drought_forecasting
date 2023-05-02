"""
Train model with a hyperparameter configuration

Created:    22 Jan 2023
Authors:    Selene Ledain (selene.ledain@ibm.com)
"""

# ==================================================================================================
# import packages
from pathlib import Path
import torch as th
from torch.utils.data import DataLoader
import time
from tqdm import tqdm # Instantly make your loops show a smart progress meter
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.functional import r2_score
import torch.optim as optim
from torchsummary import summary
from PIL import Image
import uuid
import pandas as pd
from datetime import datetime
import warnings
import argparse
import importlib
warnings.filterwarnings("ignore")

# Import other scripts here 
from drought_impact_dataset import *
from drought_impact_sampler import *
from torch.utils.data import DataLoader
from utils.utils_pixel import *
from model import *

# MLFlow
import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



def test_model(config_id, logged_model, exp_name, ndvi_idx, split):
    """
    Perform training pipeline with a combination of hyperparameters. Assess performance on validation set
    
    Author: Selene
    :param method: str. recursive or oneshot predictions
    :param config_id: configuration file to use
    :param logged_model: logging id in MLflow
    :param split: dataset to use
    """
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(device)
    
    ########################################################################
    # SET UP PARAMETERS FROM CONFIGURATION FILE
    
    module = importlib.import_module(f'utils.{config_id}')
    train_dataset_params = module.train_dataset_params
    train_sampler_params = module.train_sampler_params
    sim_params = module.sim_params
    model_params = module.model_params

    
    # ==================================================================================================
    # Get setup details from configuration file

    feature_set = train_dataset_params['feature_set']
    remove_band = train_dataset_params['remove_bands']
    multiple_labels = train_dataset_params['multiple_labels']
    batch_size_tr =  train_sampler_params['batch_size'] 
    n_batch = sim_params['n_batches']
    n_timesteps_in =  train_dataset_params['ts_len']
    n_timesteps_out =  model_params["output_dim"] #train_dataset_params['len_preds']
    n_feats_in = len(feature_set) - len(remove_band)
    n_feats_out = model_params["output_dim"]

    n_train = batch_size_tr*n_batch
    batch_size_val = sim_params['batch_size_val'] 
    n_batch_val = sim_params['n_batches_val']
    batch_size_te = sim_params['batch_size_te'] 
    n_batch_te = sim_params['n_batches_te']
    exp = sim_params["exp"]
    exp_val = sim_params["exp_val"]
    exp_te = sim_params["exp_test"]
    sample_type = sim_params["sample_type"]
    method = sim_params["method"] # direct vs oneshot
    cp_idx = sim_params["cp_idx"] # used to mask out clouds in loss (cp_id, ndvi_id)
    
    loss_function = nn.MSELoss()
    
    if split=='val':
        n_batch = n_batch_val
        
    if split=='test':
        n_batch = n_batch_te
        
    if split=='test_vaud':
        n_batch = 96
        
    if split=='test_valais':
        n_batch = 63
        
    if split=='train':
        n_batch = 559
    
    
    
    
    ########################################################################
    # LOAD MODEL
    
    model = mlflow.sklearn.load_model(logged_model)
    
    ########################################################################
    # TEST MODEL

    # Use the GPU for computations
    model = model.to(device)
    loss_function = loss_function.to(device)
    
    total_loss = 0
    
    model.eval()
    
    print(f"{split}-ing model...")
    
    with th.no_grad():
    
        for batch_nbr in range(n_batch):
            batch_loss = 0

            X_te = np.empty([batch_size_te, n_timesteps_in, n_feats_in])
            y_te = np.empty([batch_size_te, n_timesteps_out, n_feats_out])


            # Load a batch here 
            img, label = load_batch(batch_size=batch_size_te, batch_nbr=batch_nbr, sample_type=sample_type, split=split, exp=exp, n_timesteps_out=n_timesteps_out)
            # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

            img = remove_features_batch_level(img, remove_band, feature_set)
            label = remove_features_batch_level(label, remove_band, feature_set)
            label = label[:,:,ndvi_idx,:,:]

            X_te[:, :, :] = th.reshape(img, (batch_size_te, img.size(dim=1), img.size(dim=2))).numpy()
            y_te[:, :, :] = th.reshape(label, (batch_size_te, label.size(dim=1), label.size(dim=2))).numpy()

            x, y = th.tensor(X_te, dtype=th.float32, requires_grad=True, device=device), th.tensor(y_te, dtype=th.float32, requires_grad=True, device=device)

            preds = model(x, n_timesteps_out)
            preds = preds.to(device)

            loss = loss_function(preds, y) 
            
            # Save predictions
            th.save(preds, f'{sample_type}/{exp}/preds/{split}/{exp_name}_data_cube_b{batch_nbr}.pt') #_e{epoch}

            batch_loss += loss.item()
            total_loss += batch_loss

        
            # Average MSE per sample in the batch
            avg_batch_loss = batch_loss/batch_size_te
            print(f"Batch nbr {batch_nbr}. Average batch loss: {avg_batch_loss}")
        
    print(f"Total avg loss {total_loss/n_batch_te}")





    
########################################################################
# CALL FUNCTION

parser = argparse.ArgumentParser()
parser.add_argument('--config_id', type=str)
parser.add_argument('--logged_model', type=str)
parser.add_argument('--exp_name', type=str)
parser.add_argument('--ndvi_idx', type=int)
parser.add_argument('--split', type=str)
args = parser.parse_args()

test_model(args.config_id, args.logged_model,args.exp_name,args.ndvi_idx, args.split)

#jbsub -q x86_24h -cores 1x8+1 -mem 128G -out sj_out/testfinal10_filter_0.0001041_2_32.stdout -err sj_out/testfinal10_filter_0.0001041_2_32.stderr python test_model.py --config_id train_final_config_clean_filter --logged_model runs:/306a8e78b83f47a4af4c35bf3f6b6682/model --exp_name final10_filter_26-02-2023_0.0001041_2_32 --ndvi_idx 3 --split test_valais