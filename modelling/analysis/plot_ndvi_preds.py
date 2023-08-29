"""
Plot NDVI timeseries predictions vs ground-truth.

Author: Selene Ledain
Aug 24th 2023
"""

import os
import numpy as np
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import random


def plot_ndvi_preds(truth_path, pred_path, coord_range, time_range, ndvi_idx, limit_plots):
    
    # Loop through prediction files. Format '{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}_{shift}.npz'
    # For each file, find the corresponding ground truth data. Has the same name in truth_path
    
    # Get a list of all prediction files in the pred_path directory
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.npz')]
    # Shuffle the list of prediction files
    random.shuffle(pred_files)

    num_subplots = len(pred_files)
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots))

    for idx, pred_file in enumerate(pred_files):
        # Extract information from the prediction file
       
        start_yr, start_month, start_day, end_yr, end_month, end_day, lon, lat, width, height, shift = pred_file.split('_')
        # These correspond to the context dates
        start_file_date = datetime(int(start_yr), int(start_month), int(start_day))
        end_file_date = datetime(int(end_yr), int(end_month), int(end_day))
        
        # Determine whether the file meets coordinate and time range criteria
        coord_criteria = (coord_range is None or
                            (coord_range[0] <= float(lon) <= coord_range[2] and coord_range[1] <= float(lat) <= coord_range[3]))
        time_criteria = (time_range is None or
                        (start_file_date >= datetime.strptime(time_range[0], '%Y-%m-%d') and
                            end_file_date <= datetime.strptime(time_range[1], '%Y-%m-%d')))

        if coord_criteria and time_criteria:
            # Form the corresponding truth file name
            truth_file = f"{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}_{shift}" #shift already has the .npz string
            truth_file_path = os.path.join(truth_path, truth_file)
            
            # Load the prediction data and ground truth data
            pred_data = np.load(os.path.join(pred_path, pred_file))['preds']
            truth_data = np.load(truth_file_path)['target'][ndvi_idx] 
            context_data = np.load(truth_file_path)['context'][ndvi_idx] 
            
            # Calculate mean of truth data across time
            mean_baseline = context_data.mean()
            
            target_start_file_date = start_file_date + timedelta(days=5)
            target_end_file_date = start_file_date + timedelta(days=5*len(pred_data))
            # Generate date range with 5-day timestep
            date_range = [target_start_file_date + timedelta(days=5 * i) for i in range(len(pred_data))]
            # Plot the two timeseries as lineplots of different colors
            axs[idx].plot(date_range, pred_data, label='Prediction', color='blue')
            axs[idx].plot(date_range, truth_data, label='Ground Truth', color='red')
            # Formatting x ticks
            axs[idx].set_xticks(date_range)  # Set the x ticks to match the date_range
            axs[idx].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the tick labels
            # Rotate the tick labels for better readability
            axs[idx].tick_params(axis='x', rotation=45)

            # Add mean as ticked line
            axs[idx].plot(date_range, [mean_baseline]*len(truth_data), label='Context mean', color='green', linestyle='--')
            axs[idx].set_title(f"Lon: {lon}, Lat: {lat}, Start: {target_start_file_date}, End: {target_end_file_date}") 
            axs[idx].set_xlabel("Time")
            axs[idx].set_ylabel("NDVI Value")
            axs[idx].legend()

            # Continue with the next prediction file
            continue

    plt.tight_layout()
    # Save the figure to pred_path
    png_path = f'preds_'
    if coord_range is None:
        png_path += 'allcoords_'
    else:
        png_path += f'{coord_range[0]}_{coord_range[1]}_{coord_range[2]}_{coord_range[3]}_'
    if time_range is None:
        png_path += 'alltime.png'
    else:
        png_path += f'{time_range[0]}_{time_range[1]}.png'
    save_path = os.path.join(pred_path, png_path)
    plt.savefig(save_path)


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', type=str, help='Path to where ground truth data is stored')
    parser.add_argument('--pred_path', type=str, help='Path to where predictions are stored')
    parser.add_argument('--coord_range', type=float, nargs=4, help="Coordinate range [minX, minY, maxX, maxY]")
    parser.add_argument('--time_range', type=str, nargs=2, help="Time range ['yyyy-mm-dd', 'yyyy-mm-dd']")
    parser.add_argument('--ndvi_idx', type=int, default=5, help="Index position of NDVI variable in context/target tensor")
    parser.add_argument('--limit_plots', type=int, default=5, help="Limit number of plots to produce")
    
    args = parser.parse_args()
    
    plot_ndvi_preds(args.truth_path, args.pred_path, args.coord_range, args.time_range, args.ndvi_idx, args.limit_plots)
    
    # python modelling/analysis/plot_ndvi_preds.py --truth_path /data/scratch/selene/test/ood --pred_path /data/scratch/selene/test/ood_preds --coord_range 8.55 47.65 8.69 47.77 --time_range 2017-04-01 2018-04-01 --ndvi_idx 5 --limit_plots 30

                
