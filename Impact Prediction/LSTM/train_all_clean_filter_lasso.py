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
from model_all_lasso import *

# MLFlow
import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)

logger = logging.getLogger(__name__)



def train_all(method, num_layers, hidden_dim, lr, output_dim, config_id, feature_imp=None):
    """
    Perform training pipeline with a combination of hyperparameters. Assess performance on validation set
    
    Author: Selene
    :param method: str. recursive or oneshot predictions
    :param num_layers: number of hidden layers
    :param hiidden_dim: dimension of hidden layers (number of neurons)
    :param lr: leaning rate
    :param output_dim: not a hyperparameter, but changes if doing direct or recursive
    :param config_id: configuration file to use
    :param bidirectional: boolean. Whether LSTM should be bidirectional or not
    """
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
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
    remove_bands = train_dataset_params['remove_bands']
    multiple_labels = train_dataset_params['multiple_labels']
    batch_size_tr =  train_sampler_params['batch_size'] 
    n_batch = sim_params['n_batches']
    ts_len =  train_dataset_params['ts_len']
    len_preds =  train_dataset_params['len_preds']

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



    ########################################################################
    # TRAIN MODEL
    
    # ==================================================================================================
    # Prepare checkpointing
    
    # Create folder where checkpoints for model will be saved
    checkpoint_folder = f'checkpoints/{method}_{lr}_{num_layers}_{hidden_dim}/'
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    
    checkpoint_file_prefix = 'checkpoint_' 
    checkpoints = [file for file in os.listdir(checkpoint_folder) if file.startswith(checkpoint_file_prefix) and 'best' not in file]
    sorted_checkpoints = sorted(checkpoints, key=get_ckpt_epoch_batch)
    # Get last checkpoint (latest epoch)
    checkpoint_file = sorted_checkpoints[-1] if len(sorted_checkpoints)!=0 else None
    

    if checkpoint_file is not None:
        checkpoint = th.load(checkpoint_folder+checkpoint_file)
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        epoch_loss = checkpoint['epoch_loss']
        epoch_r2 = checkpoint['epoch_r2']
        optimizer = checkpoint['optimizer']
        dt_string = checkpoint['experiment_name']
        mlflow_run_id = checkpoint['mlflow_run_id']

        if method == 'dir': #direct
            model = LSTM_oneshot(input_dim=len(feature_set)-len(remove_bands), hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, device=device)
        if method == 'rec': #recursive
            model = LSTM_recursive(input_dim=len(feature_set)-len(remove_bands), hidden_dim=hidden_dim, num_layers=num_layers, num_steps=sim_params["num_steps"])
        criterion = select_loss_function(sim_params['loss_function'])

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(checkpoint_file, checkpoint['epoch']+1))

        # Get existing MLflow experiment
        experiment = mlflow.get_experiment_by_name(dt_string)

        # Get run that was just created and its ID to use when tracking
        client = mlflow.tracking.MlflowClient() # Create a MlflowClient object
        runs = client.search_runs(experiment.experiment_id)
        mlflow_run_id = [r.info.run_id for r in runs if r.info.run_name==f'train_{method}'][0]
        mlflow_run_id_val = [r.info.run_id for r in runs if r.info.run_name==f'val_{method}'][0]


    if checkpoint_file is None:
        start_epoch = 0
        start_batch = 0
        total_loss = 0
        total_r2 = 0

        if method == 'dir': #direct
            model = LSTM_oneshot(input_dim=len(feature_set)-len(remove_bands), hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, device=device)
        if method == 'rec': #recursive
            model = LSTM_recursive(input_dim=len(feature_set)-len(remove_bands), hidden_dim=hidden_dim, num_layers=num_layers, num_steps=sim_params["num_steps"])
        criterion = select_loss_function(sim_params['loss_function'])
        optimizer = select_optimizer(sim_params["optimizer"], model.parameters(), lr, sim_params["momentum"])

        #summary(model, (len(train_ds.feature_set)-len(remove_bands), 1, 1))

        # Create new MLflow experiment
        now = datetime.now()
        dt_string = 'lasso_all_'+now.strftime("%d/%m/%Y")+f'_{lr}_{num_layers}_{hidden_dim}'
        #dt_string = 'debug'
        mlflow.create_experiment(name=dt_string) 
        experiment = mlflow.get_experiment_by_name(dt_string)

        with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=f'train_{method}'):
            mlflow.log_param("n_samples training", n_train)
            mlflow.log_param("batch_size training", batch_size_tr)


        with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=f'val_{method}'):
            mlflow.log_param(f"n_samples val", batch_size_val*n_batch_val)

        # Get run that was just created and its ID to use when tracking
        client = mlflow.tracking.MlflowClient() # Create a MlflowClient object
        runs = client.search_runs(experiment.experiment_id)
        mlflow_run_id = [r.info.run_id for r in runs if r.info.run_name==f'train_{method}'][0]
        mlflow_run_id_val = [r.info.run_id for r in runs if r.info.run_name==f'val_{method}'][0]
        
    # ==================================================================================================
    # Train 
    
    best_loss = np.inf

    for ix_epoch in tqdm(range(sim_params["num_epochs"])): 
        if ix_epoch<start_epoch:
            continue

        print(f"Epoch {ix_epoch}\n---------")

        # Train
        total_loss, epoch_r2 = train_model(method=method, model=model, epoch=ix_epoch, loss_function=criterion, optimizer=optimizer, 
                                 batch_size=batch_size_tr, n_batch=n_batch,
                                 n_timesteps_in=ts_len, n_timesteps_out=1, n_feats_in=len(feature_set)-len(remove_bands), n_feats_out=output_dim, 
                                 remove_band=remove_bands, feature_set=feature_set, 
                                 experiment=experiment, checkpoint_folder=checkpoint_folder, dt_string=dt_string, start_batch=start_batch, client=client, run_id=mlflow_run_id, total_loss=total_loss, total_r2=total_r2,
                                 sample_type=sample_type, exp=exp, cp_idx=cp_idx, device=device)


        # Validate

        val_loss, val_r2 = test_model(method=method, model=model, epoch=ix_epoch, loss_function=criterion, 
                                 batch_size=batch_size_val, n_batch=n_batch_val,
                                 n_timesteps_in=ts_len, n_timesteps_out=1, n_feats_in=len(feature_set)-len(remove_bands), n_feats_out=output_dim, 
                                 remove_band=remove_bands, feature_set=feature_set, 
                                 experiment=experiment, split='val', start_batch=start_batch, client=client, run_id=mlflow_run_id_val, checkpoint_folder=checkpoint_folder,
                                 sample_type=sample_type, exp=exp_val, cp_idx=cp_idx, dt_string=dt_string, device=device)

        best_loss = compare_model_for_checkpoint(val_loss, best_loss, model, ix_epoch, checkpoint_folder+'checkpoint_'+dt_string.split(' ')[0].replace('/', '_')+f'_e{ix_epoch}_best.pth.tar') 

    with mlflow.start_run(experiment_id = experiment.experiment_id, run_name='trained'):
        mlflow.sklearn.log_model(model, "model")

    ########################################################################
    # TEST MODEL

    with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=f'test_{method}'):
            mlflow.log_param(f"n_samples test", batch_size_te*n_batch_te)

    # Get run that was just created and its ID to use when tracking
    client = mlflow.tracking.MlflowClient() # Create a MlflowClient object
    runs = client.search_runs(experiment.experiment_id)
    mlflow_run_id_test = [r.info.run_id for r in runs if r.info.run_name==f'test_{method}'][0]

    test_loss, test_r2 = test_model(method=method, model=model, epoch=ix_epoch, loss_function=criterion, 
                                 batch_size=batch_size_te, n_batch=n_batch_te,
                                 n_timesteps_in=ts_len, n_timesteps_out=1, n_feats_in=len(feature_set)-len(remove_bands), n_feats_out=output_dim, 
                                 remove_band=remove_bands, feature_set=feature_set, 
                                 experiment=experiment, split='test', start_batch=start_batch, client=client, run_id=mlflow_run_id_test, checkpoint_folder=checkpoint_folder,
                                 sample_type=sample_type, exp=exp_te, cp_idx=cp_idx, dt_string=dt_string, device=device)


    ########################################################################
    # SAVE TRACKING DATA 
    
    mlflow.end_run(mlflow_run_id)
    mlflow.end_run(mlflow_run_id_val)
    mlflow.end_run(mlflow_run_id_test)

    

    
########################################################################
# CALL FUNCTION

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str)
parser.add_argument('--num_layers', type=int)
parser.add_argument('--hidden_dim', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--output_dim', type=int)
parser.add_argument('--config_id', type=str)
parser.add_argument('--feature_imp', type=int)
args = parser.parse_args()

train_all(args.method, args.num_layers, args.hidden_dim, args.lr, args.output_dim, args.config_id, args.feature_imp)

#jbsub -q x86_24h -cores 1x8+1 -mem 128G -out sj_out/lasso_all_filter_0.0001041_2_32.stdout -err sj_out/lasso_all_filter_0.0001041_2_32.stderr python train_all_clean_filter_lasso.py --method dir --num_layers 2 --hidden_dim 32 --lr 0.0001041 --output_dim 1 --config_id train_all_config_clean_filter

