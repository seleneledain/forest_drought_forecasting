"""
Pixel-wise LSTM models for drought forecasting

Created:    Dec. 11th 2022
Authors:    Selene Ledain (selene.ledain@ibm.com)
"""

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import numpy as np
from datetime import datetime
from itertools import islice
from tqdm import tqdm
import time
from torchmetrics.functional import r2_score

from utils.utils_pixel import *


# MLFlow
import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Define the LSTM model
class LSTM_oneshot(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super(LSTM_oneshot, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        # Define the LSTM layer
        self.lstm_cell = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, device=self.device)
            
        # Define the output layer (same dim as input)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim, device=self.device)
        
        # Set requires_grad=True for all the parameters in the model
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, prediction_length):
        
        # Initialize the hidden state and cell state with zeros
        h0 = th.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device)
        c0 = th.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device)

        
        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm_cell(x, (h0,c0))
        # process the input sequence and obtain the initial hidden state and cell state. These hidden states and cell states are then used as the initial states for the subsequent time steps in the for loop.

        # Store the predictions in a list
        predictions  = th.zeros(x.size(0), prediction_length, self.output_dim)

        # Iterate over the prediction steps
        for i in range(prediction_length):
            # Use the hidden state from the previous timestep as the initial hidden state
            # for the next timestep
            if i == 0:
                self.hidden = self.hidden
            else:
                self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
                # the hidden state and cell state are detached from their previous values, which is required to avoid backpropagation through time.

            # Forward pass through the LSTM layer
            lstm_out, self.hidden = self.lstm_cell(x, self.hidden)

            # Get the prediction and add it to the list
            y_pred = self.fc(lstm_out[:, -1, :])
            predictions[:,i,:] = y_pred

        return predictions
      
    
    
class LSTM_recursive(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_steps):
        super(LSTM_recursive, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps

        # Deifne the LSTM layer
        self.lstm_cell = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
            
        # Define the output layer (same dim as input)
        self.fc = nn.Linear(self.hidden_dim, self.input_dim)
    
    
    def forward(self, x, n_timesteps_in):
        # Initialize the output sequence with zeros
        out = th.zeros(x.size(0), self.num_steps, self.input_dim)
        
        for t in range(self.num_steps):
            # Initialize the hidden state and cell state with zeros
            h0 = th.zeros(self.num_layers, x.size(0), self.hidden_dim)
            c0 = th.zeros(self.num_layers, x.size(0), self.hidden_dim)
            
            t_in_x = x.size(dim=1) # this will grow at each step as we add predictions to data tensor
            
            # Forward propagate the LSTM cell
            _, (hn, cn) = self.lstm_cell(x[:, t_in_x-n_timesteps_in+t:t_in_x+t,:], (h0, c0))

            # Use the hidden state of the last time step as the input to the output layer
            y_pred = self.fc(hn[-1])
            
            # Store the output of the LSTM cell in the output sequence and for the next input
            out[:,t,:] = y_pred
            x = th.cat([x, th.unsqueeze(y_pred, dim=1)], dim=1)

        return out



def train_model(method, model, epoch, loss_function, optimizer, batch_size, n_batch, n_timesteps_in,  n_timesteps_out, n_feats_in, n_feats_out, remove_band, feature_set, experiment, checkpoint_folder, dt_string, start_batch, client, run_id, total_loss, total_r2, sample_type, exp, cp_idx, device, feature_imp=None):
    """
    Train model in one epoch
    
    Author: Selene
    :param method: str to indicte if 'dir' or 'rec' method
    :param model: model to be trained. Instantiated model class with a forward function
    :param epoch: current training epoch nuber
    :param optimizer: optimizer for the model training
    :param batch_size: number of samples in a batch
    :param n_batch: number of batches in an epoch
    :param n_timeteps_in: number of timesteps in a training data sample
    :param n_timeteps_out: number of timesteps in a training label sample
    :param n_feats_in: number of features in training data
    :param n_feats_out: number of features in label data
    :param remove_band: bands to remove from tensor
    :param feature_set: dataset attribute with original list of features it contains
    :param expriment: experiment tracking for MLFlow
    :param checkpoint_foler: where to store checkpoints
    :param dt_string: used to track experiments and save checkpoints
    :param start_batch: batch tp start at if loaded checkpoint
    :param client: mlflow API
    :param run_id: ID of current run in MLflow
    :param epoch_loss: loss of epoch calcualted so far (if traiing got interrupted).
    :param epoch_r2s: r2 of epoch calcualted so far (if training got interrupted).
    :param sample_type: pixel_data or scene_data
    :param exp: experiment name (for where the data is stored)
    :param cp_idx: tuple containing index of where the cp layer and ndvi is in the label/predicted tensor, so that it can be used to mask results
    :param feature_imp: index of the feature that needs to be permuted. If no feature importance, leave as None.
    """
    
    # Use the GPU for computations
    model = model.to(device)
    loss_function = loss_function.to(device)
    l1_lambda = 0.0562
    l1_loss =  nn.L1Loss()
     
    
    model.train()
    
    n_samples = batch_size*n_batch
    epoch_loss = 0
    epoch_r2 = 0
    
    client.log_metric(run_id,"epoch", epoch)
    
    for batch_nbr in range(n_batch):
        if batch_nbr<start_batch:
            continue
        if batch_nbr>=n_batch:
            # training has been completed
            return

        batch_loss = 0 
        
        X_tr = np.empty([batch_size, n_timesteps_in, n_feats_in])
        y_tr = np.empty([batch_size, n_timesteps_out, n_feats_out] if method=='rec' else [batch_size, n_timesteps_out, 2 if cp_idx else 1])
            
            
        # Load a batch here 
        img, label = load_batch(batch_size=batch_size, batch_nbr=batch_nbr, sample_type=sample_type, split='train', exp=exp, n_timesteps_out=n_timesteps_out)
        # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

        img = remove_features_batch_level(img, remove_band, feature_set)
        label = remove_features_batch_level(label, remove_band, feature_set)
        
        if method=='dir':
            if cp_idx: # Use both cp and ndvi
                to_keep = [i for i in cp_idx]
                label = label[:,:,to_keep,:,:] # keep CP in label, but still just predict NDVI
                #n_feats_out = 2
            if not cp_idx: # only NDVI
                label = label[:,:,14,:,:]

            
        X_tr[:, :, :] = th.reshape(img, (batch_size, img.size(dim=1), img.size(dim=2))).numpy()
        y_tr[:, :, :] = th.reshape(label, (batch_size, label.size(dim=1), label.size(dim=2))).numpy()
        
        if feature_imp:
            # Shuffle the index of the feature
            shuffle = th.randperm(X_tr.shape[1]) # Shuffle the feature in time
            X_tr[:,:,feature_imp] = X_tr[:,shuffle,feature_imp]
            

        x, y = th.tensor(X_tr, dtype=th.float32, requires_grad=True,device=device), th.tensor(y_tr, dtype=th.float32, requires_grad=True,device=device)
        optimizer.zero_grad()
        
        
        if method=='dir':
            preds = model.forward(x, n_timesteps_out)
        if method=='rec':
            preds = model.forward(x, n_timesteps_in)
        preds = preds.to(device)
            
        
        if cp_idx:
            # Select NDVI where CP < 1% in both pred and label
            cp_id = cp_idx[0]
            ndvi_id = cp_idx[1]
            # create a condition tensor
            cp_pred = y[:,:,cp_id]
            condition = cp_pred < 0.01
            preds_ndvi_masked = th.masked_select(preds, condition)
            y_ndvi_masked = th.masked_select(y[:,:,ndvi_id].unsqueeze(1), condition)
            if preds_ndvi_masked.nelement() == 0:
                loss = th.tensor(0.0, dtype=th.float32, requires_grad=True, device=device) # if loss=0 backpropagation will not occur and the model's parameters will not be updated
                # Track r2 anyways so not -inf?
                r2 = None
            else:
                loss = loss_function(preds_ndvi_masked, y_ndvi_masked)
                r2 = r2_score(preds_ndvi_masked.unsqueeze(1), y_ndvi_masked.unsqueeze(1))
            # Save predictions
            dt_correct = dt_string.replace('/', '-')
            th.save(preds, f'{sample_type}/{exp}/preds_cp/train/{dt_correct}_data_cube_b{batch_nbr}.pt') #_e{epoch}
        else:
            loss = loss_function(preds, y) 
            for param in model.parameters():
                loss += l1_lambda * l1_loss(param, th.zeros_like(param))
            r2_batch = 0 
            # Calculate R2 per sample
            for sample in range(batch_size):
                y_true = th.cat([img[sample,:,14,:,:], label[sample,:,:,:]]).squeeze(1).squeeze(1).to(device)
                y_pred_all = th.cat([img[sample,:,14,:,:].squeeze(1).squeeze(1).to(device), preds[sample][0]]).to(device)
                r2_batch += r2_score(y_true, y_pred_all)
            r2 = r2_batch/batch_size
            #r2 = r2_score(preds.squeeze(1), y.squeeze(1))
            # Save predictions
            dt_correct = dt_string.replace('/', '-')
            th.save(preds, f'{sample_type}/{exp}/preds/train/{dt_correct}_data_cube_b{batch_nbr}.pt') #_e{epoch}

        loss.backward()
        optimizer.step()

        # MSE and R2 in a batch
        client.log_metric(run_id,"mse", loss, step=epoch*n_batch+batch_nbr)
        if r2:
            client.log_metric(run_id,"r2", r2, step=epoch*n_batch+batch_nbr)
        batch_loss += loss.item()


        # Avg MSE per sample in the batch
        avg_batch_loss = batch_loss/batch_size
        print(f"Batch nbr {batch_nbr}. Batch loss: {batch_loss}")
        epoch_loss += batch_loss
        epoch_r2 += r2 if r2 else 0
        client.log_metric(run_id,"batch loss", batch_loss, step=epoch*n_batch+batch_nbr)
        client.log_metric(run_id,"avg batch loss", avg_batch_loss, step=epoch*n_batch+batch_nbr)
        
            
    client.log_metric(run_id,"total epoch loss", epoch_loss, step=epoch)
    client.log_metric(run_id,"avg epoch loss", epoch_loss/n_batch, step=epoch)
    client.log_metric(run_id,"total epoch r2", epoch_r2, step=epoch)
    client.log_metric(run_id,"avg epoch r2", epoch_r2/n_batch, step=epoch)
    
    total_loss += epoch_loss/n_batch
    total_r2 += epoch_r2/n_batch
         
    return total_loss, total_r2



  
    

def test_model(method, model, epoch, loss_function, batch_size, n_batch, n_timesteps_in,  n_timesteps_out, n_feats_in, n_feats_out, remove_band, feature_set, experiment, split, start_batch, client, run_id, sample_type, exp, cp_idx, dt_string, device, checkpoint_folder=None):
    """
    Train model in one epoch
    
    Author: Selene
    :param method: str to indicte if 'dir' or 'rec' method
    :param model: model to be trained. Instantiated model class with a forward function
    :param epoch: current training epoch nuber
    :param optimizer: optimizer for the model training
    :param batch_size: number of samples in a batch
    :param n_batch: number of batches in an epoch
    :param n_timeteps_in: number of timesteps in a training data sample
    :param n_timeteps_out: number of timesteps in a training label sample
    :param n_feats_in: number of features in training data
    :param n_feats_out: number of features in label data
    :param remove_band: bands to remove from tensor
    :param feature_set: dataset attribute with original list of features it contains
    :param expriment: experiment tracking for MLFlow
    :param checkpoint_foler: where to store checkpoints
    :param dt_string: used to track experiments and save checkpoints
    :param start_batch: batch tp start at if loaded checkpoint
    :param client: mlflow API
    :param split: string to indicate test or val
    :param run_id: ID of current run in MLflow
    :param epoch_loss: loss of epoch calcualted so fare (if traiing got interrupted). So if starting the epoch
    :param sample_type: pixel_data or scene_data
    :param exp: experiment name (for where the data is stored)
    :param cp_idx: tuple containing index of where the cp layer and ndvi is in the label/predicted tensor, so that it can be used to mask results
    """
    # Use the GPU for computations
    model = model.to(device)
    l1_lambda = 0.0562
    l1_loss =  nn.L1Loss()
    
    
    total_loss = 0
    total_r2 = 0
    
    model.eval()
    
    print(f"{split}-ing model...")
    
    with th.no_grad():
    
        for batch_nbr in range(n_batch):
            if batch_nbr<start_batch:
                continue
            if batch_nbr>=n_batch:
                # training has been completed
                return

            batch_loss = 0

            X_te = np.empty([batch_size, n_timesteps_in, n_feats_in])
            y_te = np.empty([batch_size, n_timesteps_out, n_feats_out] if method=='rec' else [batch_size, n_timesteps_out, 2 if cp_idx else 1])


            # Load a batch here 
            img, label = load_batch(batch_size=batch_size, batch_nbr=batch_nbr, sample_type=sample_type, split=split, exp=exp, n_timesteps_out=n_timesteps_out)
            # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]

            img = remove_features_batch_level(img, remove_band, feature_set)
            label = remove_features_batch_level(label, remove_band, feature_set)

            if method=='dir':
                if cp_idx: # Use both cp and ndvi
                    to_keep = [i for i in cp_idx]
                    label = label[:,:,to_keep,:,:]
                    n_feats_out = 2
                if not cp_idx: # only NDVI
                    label = label[:,:,14,:,:]



            X_te[:, :, :] = th.reshape(img, (batch_size, img.size(dim=1), img.size(dim=2))).numpy()
            y_te[:, :, :] = th.reshape(label, (batch_size, label.size(dim=1), label.size(dim=2))).numpy()

            x, y = th.tensor(X_te, dtype=th.float32, requires_grad=True,device=device), th.tensor(y_te, dtype=th.float32, requires_grad=True,device=device)

            if method=='rec':
                preds = model(x, n_timesteps_in)
            if method=='dir':
                preds = model(x, n_timesteps_out)
            preds = preds.to(device)


            if cp_idx:
                # Select NDVI where CP < 1% in both pred and label
                cp_id = cp_idx[0]
                ndvi_id = cp_idx[1]
                # create a condition tensor
                cp_pred = y[:,:,cp_id]
                condition = cp_pred < 0.01
                preds_ndvi_masked = th.masked_select(preds, condition)
                y_ndvi_masked = th.masked_select(y[:,:,ndvi_id].unsqueeze(1), condition)
                if preds_ndvi_masked.nelement() == 0:
                    loss = th.tensor(0.0, dtype=th.float32, requires_grad=True) # if loss=0 backpropagation will not occur and the model's parameters will not be updated
                    # Track r2 anyways so not -inf?
                    r2 = None
                else:
                    loss = loss_function(preds_ndvi_masked, y_ndvi_masked)
                    r2 = r2_score(preds_ndvi_masked.unsqueeze(1), y_ndvi_masked.unsqueeze(1))
                # Save predictions
                dt_correct = dt_string.replace('/', '-')
                th.save(preds, f'{sample_type}/{exp}/preds_cp/{split}/{dt_correct}_data_cube_b{batch_nbr}.pt') #_e{epoch}
            else:
                loss = loss_function(preds, y) 
                for param in model.parameters():
                    loss += l1_lambda * l1_loss(param, th.zeros_like(param))
                r2_batch = 0 
                # Calculate R2 per sample
                for sample in range(batch_size):
                    y_true = th.cat([img[sample,:,14,:,:], label[sample,:,:,:]]).squeeze(1).squeeze(1).to(device)
                    y_pred_all = th.cat([img[sample,:,14,:,:].squeeze(1).squeeze(1).to(device), preds[sample][0]]).to(device)
                    r2_batch += r2_score(y_true, y_pred_all)
                r2 = r2_batch/batch_size
                #r2 = r2_score(preds.squeeze(1), y.squeeze(1))
                # Save predictions
                dt_correct = dt_string.replace('/', '-')
                th.save(preds, f'{sample_type}/{exp}/preds/{split}/{dt_correct}_data_cube_b{batch_nbr}.pt') #_e{epoch}



            # MSE and R2 in the batch
            if split=='val':
                client.log_metric(run_id,"mse", loss, step=epoch*n_batch+batch_nbr)
            if 'test' in split:
                client.log_metric(run_id,"mse", loss, step=batch_nbr)

            if r2:
                if split=='val':
                    client.log_metric(run_id,"r2", r2, step=epoch*n_batch+batch_nbr)
                if 'test' in split:
                    client.log_metric(run_id,"r2", r2, step=batch_nbr)
                total_r2 += r2
            batch_loss += loss.item()
            total_loss += batch_loss

            """
            # Loss per day in the batch
            loss_day1 = loss_function(output[:,0,:], y[:,0,:])
            client.log_metric(run_id,"loss_day1", loss_function(output[:,0,:], y[:,0,:]), step=batch_nbr)

            if n_timesteps_out > 1:
                loss_day2 = loss_function(output[:,1,:], y[:,1,:])
                loss_day3 = loss_function(output[:,2,:], y[:,2,:])
                client.log_metric(run_id,"loss_day2", loss_function(output[:,1,:], y[:,1,:]), step=batch_nbr)
                client.log_metric(run_id,"loss_day3", loss_function(output[:,2,:], y[:,2,:]), step=batch_nbr)
            """

            # Average MSE per sample in the batch
            avg_batch_loss = batch_loss/batch_size
            print(f"Batch nbr {batch_nbr}. Average batch loss: {avg_batch_loss}")

            if split=='val':
                client.log_metric(run_id,"batch loss", batch_loss, step=epoch*n_batch+batch_nbr)
                client.log_metric(run_id,"avg batch loss", avg_batch_loss, step=epoch*n_batch+batch_nbr)
            if 'test' in split:
                client.log_metric(run_id,"batch loss", batch_loss, step=batch_nbr)
                client.log_metric(run_id,"avg batch loss", avg_batch_loss, step=batch_nbr)


    # Total MSE and avg total MSE (total/n_batch), total and avg R2
    if split=='val':
        client.log_metric(run_id,"total loss", total_loss, step=epoch)
        client.log_metric(run_id,"avg total loss", total_loss/n_batch, step=epoch)
        client.log_metric(run_id,"total R2", total_r2, step=epoch)
        client.log_metric(run_id,"avg total R2", total_r2/n_batch, step=epoch)
    if 'test' in split:
        client.log_metric(run_id,"total loss", total_loss, step=epoch)
        client.log_metric(run_id,"avg total loss", total_loss/n_batch, step=epoch)
        client.log_metric(run_id,"total R2", total_r2, step=epoch)
        client.log_metric(run_id,"avg total R2", total_r2/n_batch, step=epoch)

    return total_loss/n_batch, total_r2/n_batch





