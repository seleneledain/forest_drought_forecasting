"""
Split data between train and test sets accoridng to coordinates and/or dates.
Using filters, data will be moved to train set. The rest will be part of test.
All cubes will be moved to train.
Train and test will have same .npy normalisation stats, and any usused stats will be saved in train/cubes

Author: Selene Ledain
Date: Aug 18th, 2023
"""

import os
import shutil
from datetime import datetime

def split_train_test(data_paths, dir_path, test_track, coord_range=None, time_range=None):
    """Reshuffle data into train and test splits using coordinate and date filter.

    Args:
        data_paths (list): list of paths where data stored
        dir_path (str): where the train and test folders should be created
        test_track (str): name of test set
        coord_range (array): train set coordinate range [minX, minY, maxX, maxY]
        time_range (array): train set start and end dates ['yyyy-mm-dd', 'yyyy-mm-dd']
    """
    
    if coord_range is None and time_range is None:
        raise Exception('Please provide some time or coordinate filters!')
    
    # Create directories train and test in dir_path
    train_dir = os.path.join(dir_path, 'train')
    test_dir = os.path.join(dir_path, f'test/{test_track}')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create 'cube' folder within train_dir. It will store all of the cubes
    train_cube_dir = os.path.join(train_dir, 'cubes')
    os.makedirs(train_cube_dir, exist_ok=True)
    print('Created new directories')
    
    # Move the 'cube' folders and their contents from data_paths to train_cube_dir
    print('Moving data cubes')
    for data_path in data_paths:
        cube_folder = os.path.join(data_path, 'cubes')
        for item in os.listdir(cube_folder):
            item_path = os.path.join(cube_folder, item)
            if os.path.isfile(item_path):
                shutil.move(item_path, os.path.join(train_cube_dir, item))
    
    # Filenames' format is f'{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon.values}_{lat.values}_{width}_{height}_{shift}.npz'
    # Move the files that have lon.values in [minX, maxX] and lat.values in [minY, maxY] to train and
    # have dates in the date range ['yyyy-mm-dd', 'yyyy-mm-dd'] to train.
    # The rest to test.
    print('Moving .npz files...')
    for data_path in data_paths:
        for root, _, files in os.walk(data_path):
            for filename in files:
                if filename.endswith('.npz'):
                    parts = filename.split('_')
                    lon = float(parts[-5])
                    lat = float(parts[-4])
                    start_file_date = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    end_file_date = datetime(int(parts[3]), int(parts[4]), int(parts[5]))

                    # Determine whether the file meets coordinate and time range criteria
                    coord_criteria = (coord_range is None or
                                    (coord_range[0] <= lon <= coord_range[2] and coord_range[1] <= lat <= coord_range[3]))
                    time_criteria = (time_range is None or
                                    (start_file_date >= datetime.strptime(time_range[0], '%Y-%m-%d') and
                                    end_file_date <= datetime.strptime(time_range[1], '%Y-%m-%d')))

                    if coord_criteria and time_criteria:
                        # Move to train directory
                        shutil.move(os.path.join(root, filename), os.path.join(train_dir, filename))
                        #print('train', lon, lat, start_file_date, end_file_date)
                        #pass
                    else:
                        # Move to test directory
                        shutil.move(os.path.join(root, filename), os.path.join(test_dir, filename))
                        #print('test', lon, lat, start_file_date, end_file_date)
                        #pass
    print('Done')
        
    # Filter the .npy file (of format '{start_yr}_{start_month}_{start_day}_{end_yr}_{end_month}_{end_day}_{lon}_{lat}_{width}_{height}_min/max.npy') as for the .npz files.
    # Move them to BOTH train and test
    # Move any remaining .npy file to train_cube_dir
    print('Moving .npy files...')
    for data_path in data_paths:
        for root, _, files in os.walk(data_path):
            for filename in files:
                if filename.endswith('.npy'):
                    parts = filename.split('_')
                    lon = parts[-5]
                    lat = parts[-4]
                    start_file_date = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    end_file_date = datetime(int(parts[3]), int(parts[4]), int(parts[5]))

                    # Determine whether the file meets coordinate and time range criteria
                    coord_criteria = (coord_range is None or
                                     (coord_range[0] <= float(lon) <= coord_range[2] and coord_range[1] <= float(lat) <= coord_range[3]))
                    time_criteria = (time_range is None or
                                    (start_file_date >= datetime.strptime(time_range[0], '%Y-%m-%d') and
                                     end_file_date <= datetime.strptime(time_range[1], '%Y-%m-%d')))

                    if coord_criteria and time_criteria:
                        # Move to both train and test directories
                        shutil.copy(os.path.join(root, filename), os.path.join(train_dir, filename))
                        shutil.copy(os.path.join(root, filename), os.path.join(test_dir, filename))
                        #print('npy train', lon, lat, start_file_date, end_file_date)
                    else:
                        # Move to train_cube_dir
                        shutil.copy(os.path.join(root, filename), os.path.join(train_cube_dir, filename))
                        #print('npy test', lon, lat, start_file_date, end_file_date)
    print('Done')
                        
    # Delete data_paths and their contents
    print('Deleting old files')
    for data_path in data_paths:
        shutil.rmtree(data_path)
        #pass

    return






if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', type=str, nargs='+', required=True, help="List of data paths")
    parser.add_argument('--dir_path', type=str, default="/data/scratch/selene/")
    parser.add_argument('--test_track', type=str, default="iid")
    parser.add_argument('--coord_range', type=float, nargs=4, help="Coordinate range [minX, minY, maxX, maxY]")
    parser.add_argument('--time_range', type=str, nargs=2, help="Time range ['yyyy-mm-dd', 'yyyy-mm-dd']")
    
    args = parser.parse_args()
    
    split_train_test(args.data_paths, args.dir_path, args.test_track, args.coord_range, args.time_range)
    
    # python data_downloading/split_train_test.py --data_paths /data/scratch/selene/neg /data/scratch/selene/drought --dir_path /data/scratch/selene/ --test_track ood --coord_range 5.9 45.8 8.5 47.64 

                
