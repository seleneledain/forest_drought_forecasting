"""
Test
"""

from sklearn.metrics import mean_squared_error, r2_score
import torch as th
import os
import numpy as np
import argparse
from utils.utils_pixel import *


def persistance_loss(sample_type, exp, split, len_preds, batch_size, n_batch, ndvi_idx, cp_idx=None):
    """
    :param sample_type: pixel data
    :param exp: name of experiment in which the data is found
    :param len_preds: prediction length
    :param batch_size: numbe rof data samples in a batch
    :param n_batch: number of total batches
    :param ndvi_idx: index of NDVI layer in tensor
    :param cp_idx:  index of CP layer in tensor
    """
    
    total_loss = 0
    total_r2 = 0
    
    all_preds = list()
    all_labels = list()
    
    path_to_data = f'{sample_type}/{exp}/{split}'
    
    for batch_nbr in range(n_batch):
          
        # Load a batch
        img, label = load_batch(batch_size=batch_size, batch_nbr=batch_nbr, sample_type=sample_type, split=split, exp=exp, n_timesteps_out=len_preds)
        # shapes: [batch_size, n_timesteps_in, n_feats_in] and [batch_size, n_timesteps_out, n_feats_out]
        
        X = np.empty([batch_size, img.shape[1], img.shape[2]])
        y = np.empty([batch_size, label.shape[1], label.shape[2]])
        
        X[:, :, :] = th.reshape(img, (batch_size, img.size(dim=1), img.size(dim=2))).numpy()
        y[:, :, :] = th.reshape(label, (batch_size, label.size(dim=1), label.size(dim=2))).numpy()
        
        # Keep only NDVI 
        preds = X[:, -1, ndvi_idx]
        preds = np.repeat(preds,len_preds)
        y = y[:, :len_preds, ndvi_idx]
        
        # Mask with CP
        if cp_idx:
            cp = X[:, -1, cp_idx]
            preds_masked = preds[cp==0]
            y = y[cp==0]
   
        
        if not cp_idx:
            total_loss += mean_squared_error(y, preds)
            all_preds.append(preds)
            all_labels.append(y)
        if cp_idx and len(preds_masked):
            total_loss += mean_squared_error(y, preds_masked)
            all_preds.append(preds_masked)
            all_labels.append(y)

        if not cp_idx:
            r2_batch = 0 
            for sample in range(batch_size):
                y_true = np.concatenate([img[sample,:,ndvi_idx,:,:], label[sample,:,ndvi_idx,:,:]], axis=0).squeeze(1).squeeze(1)
                #print(img[sample,:,ndvi_idx,:,:].squeeze(1).squeeze(1).shape, th.tensor([preds[sample]]).shape)
                y_pred_all = np.concatenate([img[sample,:,ndvi_idx,:,:].squeeze(1).squeeze(1), th.tensor([preds[sample]])], axis=0)
                r2_batch += r2_score(y_true, y_pred_all)
            total_r2 += r2_batch/batch_size
            
        if cp_idx:
            r2_batch = 0 
            for sample in range(batch_size):
                if cp[sample]==0:
                    y_true = np.concatenate([img[sample,:,ndvi_idx,:,:], label[sample,:,ndvi_idx,:,:]], axis=0).squeeze(1).squeeze(1)
                    y_pred_all = np.concatenate([img[sample,:,ndvi_idx,:,:].squeeze(1).squeeze(1), th.tensor([preds[sample]])], axis=0)
                    r2_batch += r2_score(y_true, y_pred_all)
            total_r2 += r2_batch/batch_size
            
    
    # Calculate overall R2
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    r2 = r2_score(all_labels, all_preds)
        
    print(total_loss/n_batch,total_r2/n_batch, r2)
    th.save(total_loss/n_batch, f'{path_to_data}/persistance_mse.pt')
    th.save(total_r2/n_batch, f'{path_to_data}/persistance_r2.pt')
    th.save(r2, f'{path_to_data}/meanbaseline_r2new.pt')

    
########################################################################
# CALL FUNCTION

parser = argparse.ArgumentParser()
parser.add_argument('--sample_type', type=str)
parser.add_argument('--exp', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--len_preds', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--n_batch', type=int)
parser.add_argument('--ndvi_idx', type=int)
parser.add_argument('--cp_idx', type=int)
args = parser.parse_args()

persistance_loss(args.sample_type, args.exp, args.split, args.len_preds, args.batch_size, args.n_batch, args.ndvi_idx, args.cp_idx)

# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_nofilter.stdout -err sj_out/persistance_nofilter.stderr python baseline_persistance.py --sample_type pixel_data --exp nofilter --split val --len_preds 1 --batch_size 40 --n_batch 100 --ndvi_idx 15 --cp_idx 14

# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_nofilter2.stdout -err sj_out/persistance_nofilter2.stderr python baseline_persistance.py --sample_type pixel_data --exp nofilter --split test --len_preds 1 --batch_size 40 --n_batch 100 --ndvi_idx 15 --cp_idx 14

# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_nofilter3.stdout -err sj_out/persistance_nofilter3.stderr python baseline_persistance.py --sample_type pixel_data --exp nofilter --split test_vaud --len_preds 1 --batch_size 40 --n_batch 100 --ndvi_idx 15 --cp_idx 14

# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_nofilter4.stdout -err sj_out/persistance_nofilter4.stderr python baseline_persistance.py --sample_type pixel_data --exp nofilter --split test_valais --len_preds 1 --batch_size 40 --n_batch 100 --ndvi_idx 15 --cp_idx 14

#jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_nofilter5.stdout -err sj_out/persistance_nofilter5.stderr python baseline_persistance.py --sample_type pixel_data --exp nofilter --split test2 --len_preds 1 --batch_size 40 --n_batch 100 --ndvi_idx 15 --cp_idx 14
      
    
    
# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_cleanfilt.stdout -err sj_out/persistance_cleanfilt.stderr python baseline_persistance.py --sample_type pixel_data --exp filtered_clean --split test --len_preds 1 --batch_size 40 --n_batch 42 --ndvi_idx 15 

# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_cleanfilt2.stdout -err sj_out/persistance_cleanfilt2.stderr python baseline_persistance.py --sample_type pixel_data --exp filtered_clean --split test_vaud --len_preds 1 --batch_size 40 --n_batch 62 --ndvi_idx 15 

# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_cleanfilt3.stdout -err sj_out/persistance_cleanfil3t.stderr python baseline_persistance.py --sample_type pixel_data --exp filtered_clean --split test_valais --len_preds 1 --batch_size 40 --n_batch 63 --ndvi_idx 15 

# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_cleanfilt4.stdout -err sj_out/persistance_cleanfilt4.stderr python baseline_persistance.py --sample_type pixel_data --exp filtered_clean --split val --len_preds 1 --batch_size 40 --n_batch 58 --ndvi_idx 15 


# jbsub -q x86_24h -cores 1x1+1 -mem 64G -out sj_out/persistance_cleanfilt5.stdout -err sj_out/persistance_cleanfilt5.stderr python baseline_persistance.py --sample_type pixel_data --exp filtered_clean --split test2 --len_preds 1 --batch_size 40 --n_batch 94 --ndvi_idx 15 


