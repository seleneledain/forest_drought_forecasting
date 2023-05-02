"""
Helper functions to support train/test of drought impact models 

Created:    7th of December 2022
Authors:    Selene Ledain (selene.ledain@ibm.com)
"""

import os
import numpy as np
import torch as th
import torch.nn as nn
import pickle
from datetime import datetime 
from torch.autograd import Variable 
from itertools import islice


def get_features_list(sen_path, era_path, static_paths=None):
    """
    Gives a list of features in the same order as they would be added in the data sample tensor
    For static paths, order must be LC, DEM, ENV
    
    Author: Selene Ledain
    :param sen_path: path to Sentinel 2 data
    :param era_path: path to Sentinel 2 data
    :param static_paths: list of files to directories of static layer
    """
    list_features = []
    for folder in os.listdir(sen_path): 
        if '2017-03-11' in folder: #Choose a date wehre knwon to have complete data
            files = [f for f in os.listdir(sen_path+folder) if f.endswith('.tiff')]
            list_features += [f for f in sorted(files)]
            break
    for folder in os.listdir(era_path): 
        if '2017-03-11' in folder:
            files = [f for f in os.listdir(era_path+folder) if f.endswith('.tiff')]
            list_features += [f for f in sorted(files)]
            break
    if static_paths is not None:                     
        for p in static_paths:
            for folder in os.listdir(p):
                if not folder.startswith('.'): 
                    files = [f for f in os.listdir(p+folder) if f.endswith('.tiff')]
                    list_features += [f for f in sorted(files)]
                    break
    
    return list_features



def get_feat_dict(feature_set):
    """
    Create a mapping feature name - index
    
    Author: Selene Ledain
    :param feature_set: list of features
    """
    dict_feat = {}
    for i, feat in enumerate(feature_set):
        dict_feat[feat] = i
    return dict_feat



def remove_features_batch_level(remove_from_tensor, remove_band, feature_set): 
    """
    Removes unwanted bands from the the batch. This is used right before
    training. Used when dataloader is passed for training.

    Author: Didem
    :param remove_from_tensor: torch tensor with dimensions [sample, batch, number_of_channels, width, height] 5D
    :param remove_band: list of bands to remove
    :return: 5D tensor, removed version
    """
    band_remove_list = list()

    for band in remove_band:
        idx_to_remove = feature_set.index(band)
        band_remove_list.append(idx_to_remove)

    remove_from_tensor_np = remove_from_tensor.numpy()
    remove_from_tensor_np = np.delete(remove_from_tensor_np, band_remove_list, axis = 2)
    removed_tensor = th.from_numpy(remove_from_tensor_np)
    return removed_tensor


def remove_features_batch_level_old(remove_from_tensor, remove_band, feature_set): 
    """
    Removes unwanted bands from the the batch. This is used right before
    training. Used when dataloader is passed for training.

    Author: Didem
    :param remove_from_tensor: torch tensor with dimensions [sample, batch, number_of_channels, width, height] 5D
    :param remove_band: list of bands to remove
    :return: 5D tensor, removed version
    """
    band_remove_list = list()

    for band in remove_band:
        idx_to_remove = feature_set.get(band)
        band_remove_list.append(idx_to_remove)

    remove_from_tensor_np = remove_from_tensor.numpy()
    remove_from_tensor_np = np.delete(remove_from_tensor_np, band_remove_list, axis = 2)
    removed_tensor = th.from_numpy(remove_from_tensor_np)
    return removed_tensor


def remove_features_image_level(remove_from_tensor, remove_band, feature_set): 
    """
    Removes unwanted bands from the image. This is used right before
    validation

    Author: Didem
    :param remove_from_tensor: torch tensor with dimensions [batch, number_of_channels, width, height] 4D
    :param remove_band: list of bands to remove
    :return: 4D tensor, removed version
    """
    band_remove_list = list()

    for band in remove_band:
        idx_to_remove = feature_set.get(band)
        band_remove_list.append(idx_to_remove)

    remove_from_tensor_np = remove_from_tensor.numpy()
    remove_from_tensor_np = np.delete(remove_from_tensor_np, band_remove_list, axis = 1)
    removed_tensor = th.from_numpy(remove_from_tensor_np)
    return removed_tensor


def dataset_stats(dataloader, norm_method, bands=[], temporal=True, start_idx=None, samples_per_job=None):
    """
    Get statistics (mean, std) of layers in a dataset for 
    lisation
    Return one value per band (mean of mean & std/mean of min & max)
    
    Author: Selene
    :param dataloader:
    :param bands: optional. list of specific band indexes to calculate stats for (if a list is provided, only those indexes will be normalised)
    :param temporal: boolean. Indicate if wokring with temporal or static bands. If statis, no need to loop over time. Provide corresponding band ids 
    :param method: 'stand' or 'minmax'
    :param start_idx: where to start reading data from the dataloader
    :param samples_per_job: number of smaples in a job, for slicing the dataloader
    """
   
    data_vals = []
    if norm_method=='stand':
        data_means = []
        data_stds = []
    if norm_method=='minmax':
        data_mins = []
        data_maxs = []
    
    for idx, c in enumerate(dataloader):
        n_bands = c[0].shape[2]
        break
    
    if len(bands)==0:
        # Compute for all bands
        bands = np.arange(n_bands).tolist()
        
    if temporal:
        for idx, c in islice(enumerate(dataloader), start_idx, start_idx+samples_per_job):
            img, label = c[0], c[1]
            # Compute stats for each band/selected bands
            for i in range(n_bands):
                if i in bands:
                    # Compute band stat in a tensor
                    band_vals, data_band_mean_or_min, data_band_std_or_max = band_in_tensor_stats(img, i, norm_method)
                    if norm_method=='stand':
                        data_means.append(data_band_mean_or_min)
                        data_stds.append(data_band_std_or_max)
                    if norm_method=='minmax':
                        data_mins.append(data_band_mean_or_min)
                        data_maxs.append(data_band_std_or_max)
                    data_vals.append(np.nanmean(band_vals)) # mean value 

        # Need to compute one value per band across all tensors. Calculat mean of all stats
        data_vals = np.nanmean(np.array(data_vals).reshape((samples_per_job,len(bands))), axis=0)
        if norm_method=='stand':
            data_means = np.nanmean(np.array(data_means).reshape((samples_per_job,len(bands))), axis=0)
            data_stds = np.nanmean(np.array(data_stds).reshape((samples_per_job,len(bands))), axis=0)
        if norm_method=='minmax':
            data_mins = np.nanmin(np.array(data_mins).reshape((samples_per_job,len(bands))), axis=0)
            data_maxs = np.nanmax(np.array(data_maxs).reshape((samples_per_job,len(bands))), axis=0)
                    
        
    if not temporal:
        for idx, c in islice(enumerate(dataloader), start_idx, start_idx+samples_per_job):
            img, label = c[0], c[1]
            # Compute stats for each band/selected bands
            for i in range(n_bands):
                if i in bands:
                    band_vals, data_band_mean_or_min, data_band_std_or_max = band_in_tensor_stats(img, i, norm_method)
                    if norm_method=='stand':
                        data_means.append(data_band_mean_or_min)
                        data_stds.append(data_band_std_or_max)
                    if norm_method=='minmax':
                        data_mins.append(data_band_mean_or_min)
                        data_maxs.append(data_band_std_or_max)
                    data_vals.append(np.nanmean(band_vals))
                        
            break        

    if norm_method=='stand':
        return np.array(data_vals), np.array(data_means), np.array(data_stds)
    if norm_method=='minmax':
        return np.array(data_vals), np.array(data_mins), np.array(data_maxs)
    
    
    
def band_in_tensor_stats(batched_data, band_id, norm_method):
    """
    Compute statistics of a certain band in a batch
    
    Author: Selene Ledain
    :param batched_data: tensor containing a batch
    :param band_id: index of band
    :param norm_method: 'stand' or 'minmax'
    :return: valueso f that band in batch, mean, std
    """
    # Convert a tensor to np array
    np_array_batched_data = batched_data.numpy()
    # Get all images of the bands of interest
    band_vals = np_array_batched_data[:,:, band_id,:,:].flatten() #n_timesteps x sample size
    # Compute stats
    if norm_method=='stand':
        batch_band_mean = np.nanmean(band_vals) #1 x sample size
        batch_band_std = np.nanstd(band_vals) #1 x sample size
        return band_vals, batch_band_mean, batch_band_std 
    if norm_method=='minmax':
        batch_band_min = np.nanmin(band_vals) #1 x sample size
        batch_band_max = np.nanmax(band_vals) #1 x sample size
        return band_vals, batch_band_min, batch_band_max 


    
def normalise(norm_method, tensor, norm_stats, feature_set, remove_bands, multiple_labels):
    """
    Normalise tensor before training
    
    Author: Selene
    :param norm_method: 'stand' or 'minmax'
    :param tensor: tensor to normalize
    :param norm_stats: [means, stds] or [mins, maxs]
    :param feature_set: all features in tensor
    :param remove_bands: bands to drop 
    :return: normalised tensor
    """
    if multiple_labels:
        # Keep only stats about concerned bands
        keep_bands = [b for b in feature_set if b not in remove_bands]
        idx_bands = [feature_set[b] for b in keep_bands]
        band_means_or_mins = [norm_stats[0][i] for i in idx_bands]
        band_stds_or_maxs = [norm_stats[1][i] for i in idx_bands]
    
        tensor = normalise_tensor_general(tensor, norm_method, [band_means_or_mins, band_stds_or_maxs])
    if not multiple_labels: # label has only NDVI as feature to normalise
        idx_ndvi = feature_set['NDVI']
        # Provide only stats to normalise NDVI
        tensor = normalise_tensor_general(tensor, norm_method, [[norm_stats[0][idx_ndvi]], [norm_stats[1][idx_ndvi]]]) 
    
    return tensor

def normalise_tensor_general(t, method, stats):
    """
    Normalise a time series tensor. Each band should be normalised using same mean, std at each timesteps (or min and max)
    TO USE IF DOING NORMALISATION OUTSIDE OF DATALOADER (TENSORS HAVE AN EXTRA DIMENSIONS)
    
    Author: Selene
    :param t: tensor [batch, timesteps, features, width, height]
    :param method: 'stand' or 'minmax'
    :param stats: tuple with [means, stds] or [maxs, mins] for normalisation
    """
    
    n_times = t.size(dim=0) # tensor of shape [times, feats, width, height]
    
    if method == 'stand':
        means = stats[0]
        stds = stats[1]
        for i in range(len(means)):
            for tm in range(n_times):
                try:
                    t[:,tm,i,:,:]  = (t[:,tm,i,:,:]-means[i])/stds[i]
                except: #if only 1 feature
                    t[:,tm,:,:,:]  = (t[:,tm,:,:,:]-means[i])/stds[i]
                    
    if method == 'minmax':
        mins = stats[0]
        maxs = stats[1]
        for i in range(len(mins)):
            for tm in range(n_times):
                if maxs[i]-mins[i]==0:
                    # If the value is 0, leave as is, if not scale it to 1
                    try:
                        val = t[:,tm,i,:,:]
                    except: #if only 1 feature
                        val = t[:,tm,:,:,:]
                    if val==0:
                        continue 
                    else: 
                        try:
                            t[:,tm,i,:,:] = 1
                        except: #if only 1 feature
                            t[:,tm,:,:,:] = 1

                else:
                    try:
                        t[:,tm,i,:,:] = (t[:,tm,i,:,:]-mins[i])/(maxs[i]-mins[i])
                    except: #if only 1 feature
                        t[:,tm,:,:,:] = (t[:,tm,:,:,:]-mins[i])/(maxs[i]-mins[i])
    
    return t
    
    
def normalise_tensor(t, method, stats):
    """
    Normalise a time series tensor. Each band should be normalised using same mean, std at each timesteps (or min and max)
    
    Author: Selene
    :param t: tensor [batch, timesteps, features, width, height]
    :param method: 'stand' or 'minmax'
    :param stats: tuple with [means, stds] or [maxs, mins] for normalisation
    """

    n_times = t.size(dim=0) # tensor of shape [times, feats, width, height]
    
    if method == 'stand':
        means = stats[0]
        stds = stats[1]
        for i in range(len(means)):
            for tm in range(n_times):
                try:
                    t[tm,i,:,:]  = (t[tm,i,:,:]-means[i])/stds[i]
                except: #if only 1 feature
                    t[tm,:,:,:]  = (t[tm,:,:,:]-means[i])/stds[i]
                    
    if method == 'minmax':
        mins = stats[0]
        maxs = stats[1]
        for i in range(len(mins)):
            for tm in range(n_times):
                if maxs[i]-mins[i]==0:
                    # If the value is 0, leave as is, if not scale it to 1
                    try:
                        val = t[tm,i,:,:]
                    except: #if only 1 feature
                        val = t[tm,:,:,:]
                    if val==0:
                        continue 
                    else: 
                        try:
                            t[tm,i,:,:] = 1
                        except: #if only 1 feature
                            t[tm,:,:,:] = 1

                else:
                    try:
                        t[tm,i,:,:] = (t[tm,i,:,:]-mins[i])/(maxs[i]-mins[i])
                    except: #if only 1 feature
                        t[tm,:,:,:] = (t[tm,:,:,:]-mins[i])/(maxs[i]-mins[i])
    
    return t
    

def select_loss_function(lossfunction_string):
    """
    Obtain torch loss function from string
    
    Author: Selene
    :param lossfunction_string: name of pytorch loss function to be used
    :return: the lossfunction itself
    """
    if lossfunction_string == 'MSE':
        lossfunction = nn.MSELoss()
    else:
        print("Invalid Loss Selection: {}! Defaulting to CrossEntropyLoss.".format(lossfunction_string))
        lossfunction = nn.MSELoss()
    return lossfunction


def select_optimizer(optimizer_string, model_parameters, lr=None, momentum=None):
    """
    Code taken from work of Jonas Weiss, Thomas Brunschwiler, Michal Muszynski and Paolo Fraccaro
    on the Flood Detection task.
    
    WARNING: Some optimizers may require a change of the other code - Please verify!
    Author: jwe
    :param optimizer_string: name of pytorch loss function to be used
    :return: the lossfunction itself
    """
    if optimizer_string == 'RAdam':
        optimizer = th.optim.RAdam(model_parameters, lr=lr) 
    elif optimizer_string == 'SGD':
        optimizer = th.optim.SGD(model_parameters, lr=lr) 
    elif optimizer_string == 'AdamW':
        optimizer = th.optim.AdamW(model_parameters, lr=lr) 
    elif optimizer_string == 'Adam':
        optimizer = th.optim.Adam(model_parameters, lr=lr) 
    elif optimizer_string == 'Adagrad':
        optimizer = th.optim.Adagrad(model_parameters, lr=lr) 
    elif optimizer_string == 'Rprop':
        optimizer = th.optim.Rprop(model_parameters, lr=lr) 
    elif optimizer_string == 'RMSprop':
        optimizer = th.optim.RMSprop(model_parameters, lr=lr, momentum=momentum)
    else:
        print("Invalid Optimizer Selection: {}! Defaulting to RAdam.".format(optimizer_string))
        optimizer = nn.th.optim.RAdam(model_parameters, lr=lr, momentum=momentum)
    return optimizer



def save_checkpoint(state, is_best, filename):
    """
    Save checkpoint if a new best is achieved
    """
    if is_best:
        # Track previous best
        ckpt_folder = filename.split('/')[0]
        best_ckpt_prev = [ckpt for ckpt in ckpt_folder if 'best' in ckpt]
        
        print ("=> Saving a new best")
        th.save(state, filename)  # save latest best checkpoint
        
        # Delete previous checkpoint
        if len(best_ckpt_prev):
            os.remove(best_ckpt_prev[0]) 
                
    else:
        print ("=> Model loss did not improve")
        
def save_checkpoint_batch(state, filename):
    """
    Save checkpoint after a batch, and delete previous ckpt if exists
    """
    print ("=> Saving batch checkpoint")
    th.save(state, filename)  # save checkpoint
    
    # Delete previous checkpoint
    curr_batch = int(filename.split('_b')[1].split('.pth')[0])
    curr_epoch = int(filename.split('_e')[1].split('_b')[0])
    if curr_batch==0 and curr_epoch==0:
        # If we are saving b0e0, no previous ckpt to remove
        return
    else:
        prev_batch = int(filename.split('_b')[1].split('.pth')[0]) - 1
        if prev_batch>=0:
            prev_ckpt = filename.split('_b')[0] + f'_b{prev_batch}' + '.pth.tar'
            try:
                os.remove(prev_ckpt) #if training resumed, there might not be a previous ckpt
            except:
                pass
        if prev_batch<0: # last ckpt was in previous epoch
            ckpt_folder = filename.split('checkpoint_')[0]
            # Sort pervious epoch ckpts in ascending --> last batch is the last element
            prev_ckpt = [ckpt for ckpt in os.listdir(ckpt_folder) if 'best' not in ckpt and ckpt.startswith('c')]
            if len(prev_ckpt)>1: #if training resumed, there might not be a previous ckpt
                prev_ckpt = sorted(prev_ckpt, key=get_ckpt_epoch_batch)[-2] # If we do [-1] then its the one we just saved
                curr_epoch = filename.split('_e')[1].split('_b')[0]
                if curr_epoch!=0:
                    os.remove(ckpt_folder+prev_ckpt) 
                    
        
    
        
def compare_model_for_checkpoint(epoch_loss, loss_best, model, epoch, filename):
    """
    After each epoch, checkpoint the model if performance has improved
    
    Author: Selene
    :param epoch_loss: loss over the eopch
    :param loss_best: current best performance
    :param model: model to checkpoint
    :param epoch: epoch number
    :param dt_sting: experiment date and time
    """
    
    # After each epoch, checkpoint if the model improved
    is_better = (epoch_loss < loss_best)
    if is_better:
        loss_best = epoch_loss
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_loss': loss_best
    }, is_better,
   filename)
    
    return loss_best

def get_ckpt_epoch_batch(s):
    """
    Obtain epoch number and batch number from checkpoint filename.
    Used to sort list of filenmes by epoch and batch
    
    Auhtor: Selene
    :param s: string filename
    """
    e_value = int(s.split('_e')[1].split('_b')[0])
    b_value =  int(s.split('_b')[1].split('.pth')[0])
    return (e_value, b_value)


        
def save_val_loss(val_loss, filename):
    """
    Save validation loss after training en epoch, overwrites previous save if exists
    
    Auhtor: Selene
    :param val_loss: list containing total validaiton loss over several epochs 
    :param filename: name of pickle file
    """
    
    with open(filename, 'wb') as f:
        pickle.dump(val_loss, f)
        
def load_val_loss(filename):
    """
    Load validation loss after training an epoch.
    
    Auhtor: Selene
    :param filename: name of pickle file containing total validation loss
    """
    try: # If its the first epoch, there might not be an existing loss
        with open(filename, 'rb') as f:
            return pickle.load(f)[0]
    except:
        return 0 # Then initialise the total val loss to 0
    



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

        
        
def create_datacube(sample_type=None, exp=None, split=None, c_features=None, t_features=None, dataloader=None, remove_bands=None, feature_set=None, size=[None,None],ts_len=None,len_preds=None, start_idx=None, multiple_labels=None, samples_per_job=None, name_start_idx=None):
    """
    Function to create datacubes.
    
    Author:Didem, Selene
    :param sample_type: scene_data or pixel_data (sample_type must match the folder name)
    :param exp: experiment name
    :param dataset: dataset to be created (input must be either of them => train/test/val)
    :param c_features: number of features in the context (x)
    :param t_features: number of features in the target/label (y)
    :param dataloader: dataloader to load data
    :param size: size of the one data cube [height,width]
    :param remove_bands: bands to remove
    :param feature_set: feature set
    :param ts_len: length of timeseries
    :param len_preds: prediction length
    :param start_idx: number at which to start numbering the samples
    :param config_id: config file being used
    :param samples_per_job: number of smaples in a job, for slicing the dataloader
    :param name_start_idx: idx where the naming should begin (name_start_idx+start_idx is the number of the first sample)
    """
    
    for idx, c in islice(enumerate(dataloader), start_idx, start_idx+samples_per_job):

        X_tr = np.empty([1,ts_len, len(feature_set), size[0],size[1]]) 
        y_tr = np.empty([1,len_preds,len(feature_set), size[0],size[1]])

        img, label = c[0], c[1]

        X_tr[:,:,:,:,:] = img.numpy()
        y_tr[:,:,:,:,:] = label.numpy()
        x, y = Variable(th.from_numpy(X_tr).float()), Variable(th.from_numpy(y_tr).float())

        th.save(x, f'{sample_type}/{exp}/{split}/x/data_cube_{idx+name_start_idx}.pt')
        th.save(y, f'{sample_type}/{exp}/{split}/y/data_cube_{idx+name_start_idx}.pt')

        #called_sample = c[2]
        #save_called_samples_old(called_sample, sample_type, split, exp, config_id)

        
def save_called_samples_old(called_sample, sample_type, split, exp, config_id):
    """
    Save list of called samples, append to exiting list if it already exists
    
    :param dl: dataloader
    :param sample_type: pixel or scene
    :param split: train/val/test
    :param exp: experiment name
    """
    
    path_list_samples = f'{sample_type}/{exp}/{split}/called_samples_{config_id}.pt'
    if os.path.exists(path_list_samples):
        list_samples = th.load(path_list_samples)
        list_samples.append(called_sample)
        th.save(list_samples, path_list_samples)
    else:
        th.save(list(called_sample), path_list_samples)
        
    return 


        
def save_called_samples(dl, sample_type, split, exp, config_id, start_idx, samples_per_job):
    """
    Save list of called samples, append to exiting list if it already exists
    
    :param dl: dataloader
    :param sample_type: pixel or scene
    :param split: train/val/test
    :param exp: experiment name
    """
    
    list_samples = list()
    
    for idx, c in islice(enumerate(dl), start_idx, start_idx+samples_per_job):
        called_sample = c[2]
        list_samples.append(called_sample)
    
    path_list_samples = f'{sample_type}/{exp}/{split}/called_samples_{config_id}_{start_idx}.pt'
    th.save(list_samples, path_list_samples)
        
    return 
    

def save_dataset_stats(norm_stats=None, sample_type=None, exp=None, idx=None):
    """
    Save normalisaiton statistics calculated on train set
    
    Author: Selene
    :param norm_stats: list of lists. [all_bands_means_or_mins, all_bands_stds_or_maxs]
    :param sample_type: pixel_data or scene_data
    :param exp: experiment name
    :param idx: idx to track which data the stats are for. Usually contains config file name and idx 
    """
    if idx is not None:
        path_norm_stats =  f'{sample_type}/{exp}/train/norm_stats_{idx}.pt'
    else:
        path_norm_stats =  f'{sample_type}/{exp}/train/norm_stats.pt'
        
    if os.path.exists(path_norm_stats):
        # Load
        [all_bands_min_old, all_bands_max_old] = th.load(path_norm_stats)
        # Recompute stats
        all_bands_min, all_bands_max = norm_stats[0], norm_stats[1]
        new_bands_min = np.stack(all_bands_min_old, all_bands_min).min(axis=0)
        new_bands_max = np.stack(all_bands_max_old, all_bands_max).max(axis=0)
        # Save
        th.save([list(new_bands_min), list(new_bands_max)], path_norm_stats)
    else:
        th.save(norm_stats, path_norm_stats)
    return

def load_dataset_stats(sample_type, exp):
    """
    Load normalisaiton statistics calculated on train set
    
    Author: Selene
    :param sample_type: pixel_data or scene_data
    :param exp: experiment name
    """
    
    path_norm_stats =  f'{sample_type}/{exp}/train/norm_stats.pt'
    return th.load(path_norm_stats)



def load_data_cube(sample_type = None,split = None,idx = None, exp=None):
    """
    Load a single datacube. Returns both data and label.
    
    Author:Didem
    :param sample_type: scene_data or pixel_data
    :param split: dataset to be loaded train/test/val
    :param idx: idx to the datacube
    :param exp: experiment name
    """
    
    datacube_x = f'{sample_type}/{exp}/{split}/x/data_cube_{idx}.pt'
    datacube_y = f'{sample_type}/{exp}/{split}/y/data_cube_{idx}.pt'
    
    #load tensors
    x = th.load(datacube_x)
    y = th.load(datacube_y)

    return x,y

def load_batch(batch_size = None, batch_nbr = None, sample_type = None, split=None, exp=None, n_timesteps_out=None):
    """
    Load a batch.
    
    Author:Didem, Selene
    :param batch_size: number of samples in a batch
    :param batch_nbr: batch number to load
    :param data: load either data or label
    :param exp: experiment name
    """
    x_batch = list()
    y_batch = list()
    
    for i in range(batch_nbr*batch_size, (batch_nbr+1)*batch_size):
        x,y = load_data_cube(sample_type=sample_type, split=split, idx=i, exp=exp)
        x_batch += x
        y_batch += y[:,:n_timesteps_out, :]
    
    x = np.stack(x_batch)
    y = np.stack(y_batch)
    x = th.from_numpy(x)
    y = th.from_numpy(y)
    
    return x,y


