"""Calculate mean and persistance baseline on test set

Selene Ledain
Aug 24th 2023
"""

from sklearn.metrics import mean_squared_error
import os
import numpy as np

def compute_baseline_score(data_path, ndvi_idx):
    
    # Get a list of all .npz files in the data_path directory
    files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
    
    total_mse_mean = 0
    total_mse_pers = 0
    # Loop through the samples
    for f in files :
        # Load file
        context = np.load(os.path.join(data_path, f))["context"][ndvi_idx]
        target = np.load(os.path.join(data_path, f))["target"][ndvi_idx]
        
        # Compute mean of context 
        context_mean = context.mean()
        # Compute MSE between mean and target
        total_mse_mean += mean_squared_error(target, [context_mean]*len(target))
        
        # Extract last value of context data
        context_pers = context[-1]
        # Compute MSE between persistence and target
        total_mse_pers += mean_squared_error(target, [context_pers]*len(target))
        
    # Compute final metrics
    print(f'MSE with mean prediction (mean baseline metric): {total_mse_mean/len(files)}')
    print(f'RMSE with mean prediction (mean baseline metric): {np.sqrt(total_mse_mean/len(files))}')
    print(f'MSE with persistence prediction (persistence baseline metric): {total_mse_pers/len(files)}')
    print(f'RMSE with persistence prediction (persistence baseline metric): {np.sqrt(total_mse_pers/len(files))}')    
    
    return


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to where ground data is stored')
    parser.add_argument('--ndvi_idx', type=int, default=5, help="Index position of NDVI variable in context tensor")
    
    args = parser.parse_args()
    
    compute_baseline_score(args.data_path, args.ndvi_idx)
    
    # python modelling/compute_baseline.py --data_path /data/scratch/selene/test/ood  --ndvi_idx 5 

                
