"""
Hyperparameter tuning with optuna
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
from torchmetrics.functional import r2_score
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
from utils.utils_pixel import *
from model_all import *
import optuna
import concurrent.futures

# MLFlow
import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



########################################################################
# SET UP PARAMETERS FROM CONFIGURATION FILE

config_id = 'train_all_config_clean_filter'
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
output_dim = model_params["output_dim"]
n_epochs = 50 #sim_params["num_epochs"]
loss_function = sim_params['loss_function']
optim = sim_params["optimizer"]
n_feats_in = len(feature_set)-len(remove_bands)
n_feats_out = 1
n_timesteps_in = ts_len
n_timesteps_out = len_preds




# Create new MLflow experiment
now = datetime.now()
dt_string = 'optuna_filter_all_'+now.strftime("%d/%m/%Y")
mlflow.create_experiment(name=dt_string) 
experiment = mlflow.get_experiment_by_name(dt_string)

    
    
def objective(trial):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    
    # Define the search space for the hyperparameters
    num_layers = trial.suggest_int("num_layers", 1, 10)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    hidden_dim = trial.suggest_int("n_hidden", 32, 128)

    # Define the LSTM model
    model = LSTM_oneshot(input_dim=n_feats_in, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, device=device)
    criterion = select_loss_function(loss_function)
    optimizer = select_optimizer(optim, model.parameters(), learning_rate)
    
    # Use the GPU for computations
    model = model.to(device)
    criterion = criterion.to(device)
    
    print(f"Training with num layers {num_layers}, hidden dim {hidden_dim}, lr {learning_rate}")
    
   
    with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=f'{learning_rate}_{num_layers}_{hidden_dim}'):
        mlflow.log_param("batch_size training", batch_size_tr)
    
    # Get run that was just created and its ID to use when tracking
    client = mlflow.tracking.MlflowClient() # Create a MlflowClient object
    runs = client.search_runs(experiment.experiment_id)
    mlflow_run_id = [r.info.run_id for r in runs if r.info.run_name==f'{learning_rate}_{num_layers}_{hidden_dim}'][0]
            
            
    for ix_epoch in range(10): 
        
        # TRAIN
        for batch_nbr in range(n_batch): 

            X_tr = np.empty([batch_size_tr, n_timesteps_in, n_feats_in])
            y_tr = np.empty([batch_size_tr, n_timesteps_out, n_feats_out])

            # Load a batch here 
            img, label = load_batch(batch_size=batch_size_tr, batch_nbr=batch_nbr, sample_type=sample_type, split='train', exp=exp, n_timesteps_out=n_timesteps_out)
            # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

            img = remove_features_batch_level(img, remove_bands, feature_set)
            label = remove_features_batch_level(label, remove_bands, feature_set)
            label = label[:,:,14,:,:]

            X_tr[:, :, :] = th.reshape(img, (batch_size_tr, img.size(dim=1), img.size(dim=2))).numpy()
            y_tr[:, :, :] = th.reshape(label, (batch_size_tr, label.size(dim=1), label.size(dim=2))).numpy()

            x, y = th.tensor(X_tr, dtype=th.float32, requires_grad=True,device=device), th.tensor(y_tr, dtype=th.float32, requires_grad=True,device=device)
            optimizer.zero_grad()

            preds = model.forward(x, n_timesteps_out)
            preds = preds.to(device)
            
            loss = criterion(preds, y) 
            client.log_metric(mlflow_run_id,"train batch loss", loss, step=ix_epoch*n_batch+batch_nbr)

            loss.backward()
            optimizer.step()
            
            r2_batch = 0 
            # Calculate R2 per sample
            for sample in range(batch_size_tr):
                y_true = th.cat([img[sample,:,14,:,:], label[sample,:,:,:]]).squeeze(1).squeeze(1).to(device)
                y_pred_all = th.cat([img[sample,:,14,:,:].squeeze(1).squeeze(1).to(device), preds[sample][0]]).to(device)
                r2_batch += r2_score(y_true, y_pred_all)
            r2 = r2_batch/batch_size_tr
            client.log_metric(mlflow_run_id,"train r2", r2, step=ix_epoch*n_batch+batch_nbr)
            
        
    # VALIDATE
    total_loss = 0
    total_r2 = 0
    
    model.eval()
    

    with th.no_grad():

        for batch_nbr in range(n_batch_val):

            X_te = np.empty([batch_size_val, n_timesteps_in, n_feats_in])
            y_te = np.empty([batch_size_val, n_timesteps_out, n_feats_out])

            # Load a batch here 
            img, label = load_batch(batch_size=batch_size_val, batch_nbr=batch_nbr, sample_type=sample_type, split='val', exp=exp, n_timesteps_out=n_timesteps_out)
            # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

            img = remove_features_batch_level(img, remove_bands, feature_set)
            label = remove_features_batch_level(label, remove_bands, feature_set)
            label = label[:,:,14,:,:]

            X_te[:, :, :] = th.reshape(img, (batch_size_val, img.size(dim=1), img.size(dim=2))).numpy()
            y_te[:, :, :] = th.reshape(label, (batch_size_val, label.size(dim=1), label.size(dim=2))).numpy()

            x, y = th.tensor(X_te, dtype=th.float32, requires_grad=True, device=device), th.tensor(y_te, dtype=th.float32, requires_grad=True, device=device)
            
            preds = model(x, n_timesteps_out)
            preds = preds.to(device)
            
            loss = criterion(preds, y)
            total_loss += loss.item()
            
            r2_batch = 0 
            # Calculate R2 per sample
            for sample in range(batch_size_val):
                y_true = th.cat([img[sample,:,14,:,:], label[sample,:,:,:]]).squeeze(1).squeeze(1).to(device)
                y_pred_all = th.cat([img[sample,:,14,:,:].squeeze(1).squeeze(1).to(device), preds[sample][0]]).to(device)
                r2_batch += r2_score(y_true, y_pred_all)
            total_r2 += r2_batch/batch_size_val
            
                
    print(f"Val loss {total_loss/n_batch_val}")
    client.log_metric(mlflow_run_id,"val loss", total_loss/n_batch_val, step=ix_epoch)
    client.log_metric(mlflow_run_id,"val r2",  total_r2/n_batch_val, step=ix_epoch)
                
    return total_loss/n_batch_val #return validation loss after last epoch



optuna.delete_study(study_name="tuning-15-02-2023-filterall", storage="sqlite:///example-study.db")

study = optuna.create_study(
        storage="sqlite:///example-study.db",  # Specify the storage URL here.
        study_name="tuning-15-02-2023-filterall",
    sampler=optuna.samplers.TPESampler(seed=0),
load_if_exists=True)


study.optimize(objective, n_trials=50)

    

print("Best hyperparameters: ", study.best_params)
print("Best validation loss: ", study.best_value)

    
    
#jbsub -q x86_24h -cores 1x8+1 -mem 128G -out sj_out/optuna_filterall.stdout -err sj_out/optuna_filterall.stderr python hyperparam_optuna_filter_all.py 



    
    
    
    
