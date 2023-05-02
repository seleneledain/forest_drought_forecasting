"""
Custom dataset creation for drought impact prediction. 

This code is adopted from the work of Jonas Weiss, Thomas Brunschwiler, Michal Muszynski and Paolo Fraccaro
on the Flood Detection task.

Created:    3rd of November 2022
Authors:    Emine Didem Durukan (emine.didem.durukan@ibm.com)
            Selene Ledain (selene.ledain@ibm.com)

"""
from drought_impact_sampler import BoundingBox
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot
import os
import glob
import re
from scipy import stats


from tqdm import tqdm 
import time
import pandas as pd
import rasterio as rs
from rasterio.windows import from_bounds
import numpy as np
from matplotlib import pyplot as plt
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset
import pathlib
from pathlib import Path
from datetime import datetime, timedelta
#import albumentations as album  # data augmentation
from itertools import groupby, islice, tee
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor
import re
import cProfile


import math
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from utils.utils_pixel import *



class DroughtImpactDataset(Dataset):


    
    #Dataset object to use with pytorch dataloader
    def __init__(self, s2_path = None, era_path = None, lc_path = None, dem_path = None, env_path = None, data_file_extension = "tiff", keep_out_list=[], focus_list=[], focus_time =[], ts_delta=None, ts_len=None, ratio=0.5, len_preds=1, feature_set = {}, agg_funct_dict={}, nan_handling="mean", norm_stats = [None, None], norm_method=None, agg_time=True, multiple_labels=False, correct_ndvi=None):
        """
        Creates a list of all data files available minus the keep_out_list
        Creates relative file paths for all valid data-files and corresponding label-files each in a separate list
        
        Author: Didem, Selene
        :param s2_paths:   list of list of filename paths to Sentinel 2 dataset (./SENTINEL 2/)
        :param era_paths:   list of list of filename paths to ERA 5 dataset (./ERA5/)
        :param lc_paths:   list of list of filename paths to Copernicus Ladncover dataset (./LC Copernicus/)
        :param dem_paths:   list of list of filename paths to DEM (./DEM/)
        :param env_paths:   list of list of filename paths to soil and forest data (./WSL/)
        :param data_file_extension: Type of files being used (e.g. *.tif, *.tiff, .... )
        :param keep_out_list:   List of "regions" (family of filenames) to NOT include in dataset (coordinates). Format [minlat, minlon, maxlat, maxlon]
        :param focus_list:      Normally empty, if not, only files starting with provided strings are included in DS. Format [minlat, minlon, maxlat, maxlon]
        :param focus_time:     Tuples with start and end timestamps (format "YYYY-MM-DD"). Filters only dates in timeseries, but not forecast dates! Inlcudes start and end dates
        :param ts_delta:     Number of days between each timestamp of the timeseries
        :param ts_len:      Total number of timestamps in the generated timseries
        :param ratio:      Ratio used as condition on wether to use timeseries or not
        :param len_preds:      Number of timestamps that will be forecasted (necesary for stacking weather forecasts). Default is 1
        :param feature_set:   Dictionary of features
        :param agg_funct_dict:   Aggregation method for bands of Sentinel-2 and ERA5. Dictionary where key-value are band-method
        :param nan_handling: Nan handling protocol within the image. 
        :param norm_stats:  tuple with [means, stds] or [maxs, mins] for normalisation
        :param norm_method:  optional. 'stand' or 'minmax'
        :param agg_time: Aggregate era5 features to 10-days for the region besides Jura to make the time-series consistent. Default is True
        :param multiple_labels: boolean. Include all bands in labels and optionally remove. later (useful for recursive LSTM)
        :param correct_ndvi:  Value 0-100. If not None, will replace NDVI if cloud proba > correct_ndvi
        """
        
        
        # --------------------------------------
        # SET UP DATASET ATTRIBUTES
        
        self.s2_path = Path(s2_path)
        self.era_path = Path(era_path)
        self.dem_path = Path(dem_path)

        self.env_path = Path(env_path) if env_path is not None else None
        self.lc_path = Path(lc_path) if lc_path is not None else None

        self.data_file_extension = data_file_extension # can be tif or .tiff
        self.dataset_paths = [s2_path,era_path,lc_path,dem_path,env_path]

        # Timeseries parameters
        self.focus_time = focus_time
        self.ts_start = []
        self.ts_end = []
        if len(self.focus_time):
            self.ts_start =  [t[0] for t in self.focus_time]
            self.ts_end =  [t[1] for t in self.focus_time]
        self.ts_len = ts_len
        self.ts_delta  = ts_delta
        self.ratio = ratio
        self.len_preds = len_preds

        # NaN handling protocol
        self.nan_handling = nan_handling
   
        self.feature_set = get_feat_dict(feature_set)   
        self.agg_funct_dict = agg_funct_dict
        self.agg_time = agg_time
        self.multiple_labels = multiple_labels
        
        self.correct_ndvi = correct_ndvi
        self.full_date_range = list(pd.date_range(start='2015-08-01', end='2021-12-31', freq='D', inclusive="both"))
        
        # Get basic properties of dataset for tensor dimensions to return
        self.data_NOB = 0         
        self.label_NOB = 1 if not self.multiple_labels else len(self.feature_set)
        self.bands_era = 0
        self.bands_s2 = 0
        self.dataset_height = None #height of sample generated by __getitem__
        self.dataset_width = None #width of sample generated by __getitem__
        
        # Normalisation
        self.norm_stats = norm_stats
        self.norm_method = norm_method
        
        # Choose a random day to calculate total number of bands
        for folder in os.listdir(self.s2_path):
            if folder.startswith('2'): #date
                random_date = Path(folder)
                break
        random_data_path_sen = os.path.join(s2_path,random_date)
        random_data_path_era = os.path.join(era_path,random_date)

        #Find number of bands from Sentinel 2 data source
        bands_s2 = os.listdir(random_data_path_sen)
        bands_s2 = [band for band in bands_s2 if ((os.path.splitext(band)[1] in ('.tiff')) & (os.path.splitext(band)[1] not in ''))]

        #Find number of bands from ERA5 data source
        bands_era = os.listdir(random_data_path_era)
        bands_era = [band for band in bands_era if ((os.path.splitext(band)[1] in ('.tiff')) & (os.path.splitext(band)[1] not in ''))]

        #Total number of bands 
        self.bands_era = len(bands_era)
        self.bands_s2 = len(bands_s2) #- 1
        self.data_NOB +=  self.bands_s2  # Subtracting one from s2 because that is the label band. This maybe done in a different way
        self.data_NOB += self.bands_era
        # we would need to add other data sources


        # Location extaction E.g.: 'SENTINEL 2/2015-08-01T00:00:00Z_46.907_7.137_47.407_7' -> 46.907_7.137_47.407_7.637
        self.list_of_all_geolocations = []
        for folder in os.listdir(self.s2_path): # all other data should have same geolocation
            if folder.startswith('2'):
                folder_coords = str(folder.split("Z_")[1])
                self.list_of_all_geolocations.append(folder_coords)
        self.list_of_all_geolocations = list(set(self.list_of_all_geolocations))
    
    
        # --------------------------------------
        # FILTER BY GEOLOCATION
        
        if focus_list: # Keep data that intersects with at least one region in focus_list
            to_remove = list()
            for i, data in enumerate(self.list_of_all_geolocations):
                data_bbox = BoundingBox(*[float(data.split('_')[1]), float(data.split('_')[3]), float(data.split('_')[0]), float(data.split('_')[2])])
                if not any(data_bbox.intersects(BoundingBox(*[float(region[1]), float(region[3]), float(region[0]), float(region[2])])) for region in focus_list):
                    to_remove.append(data)
            for d in to_remove:
                self.list_of_all_geolocations.remove(d) 
        
        if keep_out_list: # Remove areas that are in keep_out_list
            to_remove = list()
            for data in self.list_of_all_geolocations:
                data_bbox = BoundingBox(*[float(data.split('_')[1]), float(data.split('_')[3]), float(data.split('_')[0]), float(data.split('_')[2])])
                if any(data_bbox.intersects(BoundingBox(*[float(region[1]), float(region[3]), float(region[0]), float(region[2])])) for region in keep_out_list):
                    to_remove.append(data)
            for d in to_remove:
                self.list_of_all_geolocations.remove(d)

        
        # --------------------------------------
        # FILTER BY FOCUS TIME 
        
        # Get list of all data Sentinel 2 folders, which will used as reference to create timeseries and to filter other data  
        self.folders =   []
        self.folders += [folder for folder in os.listdir(self.s2_path) if folder.startswith('2')]
            
        # Unique timestamp extraction from Sentinel 2. Keep in same format as in folder name.
        self.df, self.timestamps = self.get_timestamps(self.folders)
        
        if len(self.focus_time): # There coud be multiple periods to filter
            self.timestamps = []
            for i in range(len(self.focus_time)):
                self.timestamps += self.df[(self.df['timestamp']>self.ts_start[i]) & (self.df['timestamp']<self.ts_end[i])].timestamp.tolist()
            self.unique_timestamps = list(set(self.timestamps))
                
                
        if not len(self.focus_time): # If nothing was provided
            self.ts_start[0] = datetime.strftime(self.df.sort_values('timestamp', ascending=True, ignore_index=True)['timestamp'][0], '%Y-%m-%d')
            self.ts_end[0] = datetime.strftime(self.df.sort_values('timestamp', ascending=False, ignore_index=True)['timestamp'][0], '%Y-%m-%d')
            self.focus_time = [[self.ts_start, self.ts_end]]
            self.timestamps = self.df[(self.df['timestamp']>self.ts_start[0]) & (self.df['timestamp']<self.ts_end[0])].timestamp.tolist()
            self.unique_timestamps = list(set(self.timestamps))

        # --------------------------------------
        # OBTAIN POSSIBLE TIMESERIES FOR SAMPLES
        
        # And get all possible timeseries that respect start, end, delta. Get missing dates in raw data
        self.potential_timeseries, self.missing_dates = self.get_regular_dates() 
        
        # Get all timeseries that respect the desired length
        if self.potential_timeseries:
            self.dict_ts_missing_dates = self.get_list_timeseries()
            # add a condition in  getitem if the are no potential timeseries and user needs to redefine input
            self.dict_ts_missing_dates = self.get_forecast_timeseries()
        else:
            raise Exception('Not enough timestamps between these dates to form a timeseries')
            
        
        # --------------------------------------
        # PREPARE DATA FOR __getitem__
        
        # Get all temporal folders (Sentinel 2 and ERA5) that will be filtered in getitem
        self.s2_folders = [folder for folder in os.listdir(self.s2_path) if folder.startswith('2')] 
        self.era_folders = [folder for folder in os.listdir(self.era_path) if folder.startswith('2')]
        
        # Get all static data to be added in getitem
        self.dem_folders = [folder for folder in os.listdir(self.dem_path) if not folder.startswith('.')] 
        self.lc_folders = [folder for folder in os.listdir(self.lc_path) if not folder.startswith('.')] if self.lc_path else []
        self.env_folders = [folder for folder in os.listdir(self.env_path) if not folder.startswith('.')] if self.env_path else []


        # Create all possible combinations of geolocations and timeseries that getitem will index from 
        self.all_loc_dates =  self.combine_loc_dates()
        self.all_loc_dates = sorted(self.all_loc_dates, key=lambda x: x[1][0])
        
        if len(self.all_loc_dates)==0:
            raise Exception('Not enough dates to create timeseries')
            
    
        
        return

    
 
    def __getitem__(self, idx_bbox):
        #START_SAMPLE = time.time()
        """
        Return a data sample (i.e. a tensor of raster timeseries combining all data for each date)ß
        - index a geolocation and a timeseries, then filter temporal folders based on those criteria
        - filter static folders by geolocation
        - Iterate through dates of indexed timeseries to add data for each timestamp
        
        Author: Didem, Selene
        """
        # Flag for tracking missing data folders for sentinel 2 and era5
        self.missing_data_exists = False 
        
        # --------------------------------------
        # PART 0: GET DATA ACCORDING TO SAMPLER
        
        n_samples = idx_bbox[0]
        idx = idx_bbox[1]
        bbox = idx_bbox[2]
    
        # Index a location and timeseries
        if n_samples > len(self.all_loc_dates):
            n_repeat = n_samples/len(self.all_loc_dates) # number of times sampler lenght is bigger than list to index
            idx = int(np.floor(idx/n_repeat))
        
        geoloc, sub_timeseries, sub_missing, sub_forecast = self.all_loc_dates[idx]
        
        sub_timeseries_set = set(sub_timeseries)
        sub_forecast_set = set(sub_forecast)
        
        # First index a geolocation : filter to keep folders that have that geolocation. 
        # Geolocation must be contained in folder name
        s2_at_location = filter(lambda folder: re.search(geoloc, folder), self.s2_folders)
        era_at_location = filter(lambda folder: re.search(geoloc, folder), self.era_folders)
        lc_at_location = filter(lambda folder: re.search(geoloc, folder), self.lc_folders)
        dem_at_location = list(filter(lambda folder: re.search(geoloc, folder), self.dem_folders))
        env_at_location = list(filter(lambda folder: re.search(geoloc, folder), self.env_folders))

        # Second index a timeseries in self.dict_ts_missing_dates and filter previous temporal folders to retrieve those dates
        # Timestamp must be contained in folder name
        sub_timeseries_re = '|'.join(sub_timeseries + sub_forecast)
        s2_filtered = list(filter(lambda folder: re.search(sub_timeseries_re, folder), s2_at_location))
        era_filtered = list(filter(lambda folder: re.search(sub_timeseries_re, folder), era_at_location))
        # For LC: the year must be correct: take the file that corresponds to the year of the first tiemstamp
        curr_yr = datetime.strptime(sub_timeseries[0], '%Y-%m-%d').year
        lc_filtered = list(filter(lambda folder: re.search(str(curr_yr), folder), lc_at_location))
        
        lc_file_list = [os.path.join(str(self.lc_path), f, file) for f in lc_filtered for file in os.listdir(os.path.join(str(self.lc_path), f)) if file.endswith(self.data_file_extension)]
        dem_file_list = [os.path.join(str(self.dem_path), f, file) for f in dem_at_location for file in os.listdir(os.path.join(str(self.dem_path), f)) if file.endswith(self.data_file_extension)]
        env_file_list = [os.path.join(str(self.env_path), f, file) for f in env_at_location for file in os.listdir(os.path.join(str(self.env_path), f)) if file.endswith(self.data_file_extension)]
        
        # Calculate height and width of sample
        if self.dataset_height == None: #Compute only once
            coords = self.all_loc_dates[0][0]
            raster_path = os.path.join(self.era_path, f'**{coords}', f'*{self.data_file_extension}')
            raster_path = next(iter(glob.glob(raster_path)), None)
            if not raster_path:
                print("No raster found for the given coords.")
                return None
            with rs.open(raster_path) as src:
                arr = src.read(1, window=from_bounds(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, src.transform))  #left, bottom, right, top
            self.dataset_height = arr.shape[0]
            self.dataset_width =  arr.shape[1]
            
    
        # m x n x k tensors for data and label object each.
        data_object = th.zeros([self.dataset_height, self.dataset_width, self.data_NOB], dtype=th.int32)
        label_object = th.zeros([self.dataset_height, self.dataset_width, self.label_NOB], dtype=th.int32)
        
        data_list = list() 
        label_list = list()
        
        label_search = '\[' +  'NDVI' + '\]' 

        
        # --------------------------------------
        # PART 1: GENERATE DATA TENSOR (SAMPLED AT A LOCATION)
        
        # Get daily data in parallel
        with ThreadPoolExecutor() as executor:
            args = [(bbox, geoloc, timestamp, list(s2_filtered), list(era_filtered), lc_file_list, dem_file_list, env_file_list, self.dataset_height, self.dataset_width, self.correct_ndvi) for timestamp in sub_timeseries]
            daily_data = list(executor.map(self.get_daily_data, args))
        data_list += daily_data
        
        
        # --------------------------------------
        # PART 2: GENERATE LABEL TENSOR (SAMPLED AT A LOCATION)
        
        # Get daily data in parallel
        with ThreadPoolExecutor() as executor:
            args = [(bbox, geoloc, timestamp, list(s2_filtered), list(era_filtered), lc_file_list, dem_file_list, env_file_list, self.dataset_height, self.dataset_width, self.correct_ndvi) for timestamp in sub_forecast]
            daily_data = list(executor.map(self.get_daily_label_data, args))
        label_list += daily_data
        
        

        # --------------------------------------
        # PART 3: HANDLE MISSING DATES
        prev_day_length = 1
        
        data_object = np.stack(data_list) 
        label_object = np.stack(label_list) 
        data_object = th.from_numpy(data_object)
        label_object = th.from_numpy(label_object)
        

        if self.missing_data_exists:
            data_object = self.handle_missing_data(data_object, prev_day_length, method = "replicate")
            label_object = self.handle_missing_data(label_object, prev_day_length, method = "replicate") 

        
        # Normalization
        if self.norm_method is not None:
            data_object = normalise_tensor(data_object, self.norm_method, self.norm_stats)
            if self.multiple_labels:
                label_object = normalise_tensor(label_object, self.norm_method, self.norm_stats)
            if not self.multiple_labels: # label has only NDVI as feature to normalise
                idx_ndvi = self.feature_set['NDVI']
                # Provide only stats to normalise NDVI
                label_object = normalise_tensor(label_object, self.norm_method, [[self.norm_stats[0][idx_ndvi]], [self.norm_stats[1][idx_ndvi]]]) 

        #END_SAMPLE = time.time()
        #print(f'Time for a sample generation: {END_SAMPLE-START_SAMPLE}')
        return data_object, label_object,  np.array([idx, bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])

        
    
    def __len__(self):
        """
        Returns the number of unique data points (i.e, number of timestamps)
        
        Author: Selene
        """
        #This function only takes into account of number of available sentinel 2 timestamps since temporal resolution is determined by Sentinel2

        return len([folder for folder in os.listdir(self.s2_path) if folder.startswith('2')]) # but this is for all locations combined
        
    def get_timestamps(self, list_files):
        """
        Collects timestamps of respective data in a pd.DataFrame. The data can then be filtered using datetime filtering in pandas
        
        Author: Selene
        :param list_files: list of folder or file names from which to extract timestamps
        :param date_format: format in which dates will be stored
        :return: pd.Dataframe containing folder name and respective date, list of timestamps
        """
        # Get list of folder within path
        folder_list = [x for x in list_files if x.startswith('2')]
        df = pd.DataFrame(folder_list, columns=['folder']) 
        # DEM has different naming format
        df['timestamp'] = df.folder.apply(lambda x: datetime.strptime(x.split('/')[-1].split('Z')[0], '%Y-%m-%dT%H:%M:%S') if 'Z' in x 
                                          else datetime.strptime(x.split('/')[-1].split('_')[0], '%Y-%m-%d'))

        
        return df, df.timestamp.tolist()
    
    def trim_on_timestamp(self, df, upper_threshold = None, lower_threshold = None):
        '''
        This function filters a dataframe deleting rows having timestamp values either above an upper threshold or below a lower threshold.
        Includes start and end dates.
        
        Author: Selene
        :param df: pd.DataFrame returned by get_timestamps
        :param upper_threshold: string of latest date. Format 'YYYY-MM-DD'
        :param lower_threshold: string of earliest date. Format 'YYYY-MM-DD'
        :return: list of folders that are between desired timestamps
        '''
        
        if upper_threshold == None and lower_threshold == None: # no filtering applied
            return df.folder.tolist()
        if upper_threshold == None and lower_threshold != None: 
            return df[df.timestamp>lower_threshold].folder.tolist()
        if upper_threshold != None and lower_threshold == None: 
            return df[df.timestamp<upper_threshold].folder.tolist()
        else:
            return df[(df['timestamp']>lower_threshold) & (df['timestamp']<upper_threshold)].folder.tolist()
        
    def get_temporal_steps_count(self, dir_path):
        '''
        This function gets the total number of timestamps in a dataset
        
        Author: Selene
        :param dir_path: path to data such as Sentinel 2, ERA 5...
        :return: int
        '''
        return len([folder for folder in os.listdir(dir_path) if folder.startswith('2')]) # but consders all locations combined
    
    def get_regular_dates(self): 
        '''
        This function creates all possible lists of dates forming a regular time series between start and end with a timestep of delta.
        Uses pd.DataFrame generated by get_timestamps function which contains info on all available timestamps of the data. All regular timeseries respecting ts_start,
        ts_end and ts_delta will be generated. 
        The dates are compared with the actual data dates, and ratio is used to decide whether this new timeseries could be actually used. For example if ratio = 0.5, 
        the 50% of the regular dates should be present in the data. If not, that timeseries is discarded
        
        Author: Selene
        :return potential_timeseries: list of possible timeseries respecting input constraints
        :return missing_dates: list of missing data dates for each respective potential timeseries
        '''
        
        potentail_timeseries = []
        missing_dates_all = []
        
        for t in range(len(self.focus_time)):
            
            list_dates = self.trim_on_timestamp(self.df, upper_threshold = self.ts_end[t], lower_threshold = self.ts_start[t])
            list_dates = [x.split('T')[0] for x in list_dates]
            if len(list_dates)<1:
                raise Exception('Not enough timestamps between these dates to form a timeseries')

            # Generate all possible timeseries between start and end that have given delta
            all_timeseries = []
            for i in range(self.ts_delta):
                start_date = datetime.strptime(self.ts_start[t], '%Y-%m-%d') + timedelta(days=i)
                all_timeseries.append(pd.date_range(start=start_date, end=self.ts_end[t], freq=f"{self.ts_delta}D").astype(str).tolist())

            # See which are possible with the available data
            to_pop = []
            missing_dates = [] # dates that are not present in data and will need to be filled in
            for idx, timeseries in enumerate(all_timeseries):
                match_dates = len(set(timeseries) & set(list_dates)) # number of existing dates in data
                if match_dates < len(timeseries)*self.ratio: # add a condition to decide whether to keep dates or not
                    to_pop.append(idx)
                if match_dates >= len(timeseries)*self.ratio: # add a condition to decide whether to keep dates or not
                    difference = set(timeseries).difference(set(timeseries))
                    missing_dates.append(list(difference))
            for idx in sorted(to_pop, reverse=True):
                del all_timeseries[idx]
            potentail_timeseries += all_timeseries 
            missing_dates_all += missing_dates

        #Another way of doing it is by exploiting pd.DataFrame.asfreq(). Would also return information of what dates exactly need to be interpolated.
        
        return potentail_timeseries, missing_dates_all

    def consecutive_subseq(self, iterable, length):
        """
        Get all n consecutive elemetts of a list into sublist. In this case, get sub-timeseries of a ertain length from a longer timeseries
        
        Author: Selene
        :param iterable: original list to split into sublists
        :param length: length of sublists of consecutive elements
        :return: tuples of n consecutive elements
        """
        for _, consec_run in groupby(enumerate(iterable), lambda x: x[0] - x[1]):
            k_wise = tee(map(itemgetter(1), consec_run), length)
            for n, it in enumerate(k_wise):
                next(islice(it, n, n), None) # consume n items from it
            yield from zip(*k_wise)

    def get_list_timeseries(self):
        '''
        This function creates a dicitionary with 
            - list of timeseries (a list of list of dates) that are of a certain length ts_len
            - list of missing dates in the raw data for each timeseries
        It wil create subsamples of a longer timeseries, which are generated by get_regular_timeseries and adapt the missing dates
        
        Author: Selene
        :return ts_dict: dictionary with list of timeseries and list of missing dates
        '''
        
        if self.ts_len is None:
            dict({'sub_timeseries': self.potential_timeseries, 'sub_missing': self.missing_dates})

        list_sub_ts = []
        list_sub_missing = []

        for ts_idx, ts in enumerate(self.potential_timeseries):
            # Get a timeseries
            # Get corresponding list of missing dates 
            missing = self.missing_dates[ts_idx]
            df_missing = pd.DataFrame(missing, columns=['date'])
            df_missing.date = pd.to_datetime(df_missing.date, format="%Y-%m-%d")
            # Get index of consecutive dates in subtimeseries of length ts_len
            df_tmp = pd.DataFrame(ts, columns=['date']) # date column is string
            idx_sub = list(self.consecutive_subseq(df_tmp.index, self.ts_len))
            # Get each sub timeseries and append to list
            for group in idx_sub:
                group_idx = list(group)
                group_dates = df_tmp.date[group_idx].tolist()
                list_sub_ts.append(group_dates)
                # Filter missing dates according to new subtimeseries and append to list
                sub_start = datetime.strptime(group_dates[0], "%Y-%m-%d")
                sub_end = datetime.strptime(group_dates[-1], "%Y-%m-%d")
                group_missing = df_missing[(df_missing.date>=sub_start) & (df_missing.date<=sub_end)].date.tolist()
                group_missing = [x.strftime("%Y-%m-%d") for x in group_missing]
                list_sub_missing.append(group_missing)
                
        return dict({'sub_timeseries': list_sub_ts, 'sub_missing': list_sub_missing})
    
    def get_forecast_timeseries(self):
        '''
        This adds the dates for the weather forecast data to include in each data sample, which extend beyond the satellite data
        Will update the dictionary provided by get_list_timeseries (self.dict_ts_missing_dates) and add a corresponding list of dates to consider for the weather data
        These dates can be used when creating the samples, so that forecasted weather date is added to the tensor
        
        Author: Selene
        :return dict_ts_missing_dates: dictionary with weather dates added in 'sub_forecast'
        '''
        forecast_dates = []
        for sub_ts in self.dict_ts_missing_dates['sub_timeseries']:
            # Add as many dates as self.len_preds with self.ts_delta timestep
            start_forecast = sub_ts[-1]
            forecast_ts = list(pd.date_range(start=start_forecast, periods=self.len_preds + 1, freq=str(self.ts_delta)+'D', inclusive="neither")) #  (add +1 since we dont consider start date)
            forecast_dates.append([datetime.strftime(x, format='%Y-%m-%d') for x in forecast_ts])

        self.dict_ts_missing_dates['sub_forecast'] = forecast_dates

        return self.dict_ts_missing_dates
    
    def handle_missing_data(self,tensor,prev_day_length,method = "replicate"):
        """
        Consists of imputation methods for handling missing data

        Author: Didem
        :param: tensor :  tensor to be filled
        :param: prev_day_length : how far in the past I want to go to fill the data
        :param: method : imputation method to be used. --> Default: Replicate the last available date
        """

        if method == "replicate":
            return self.replicate_last_available_date(tensor,prev_day_length)


    
    def replicate_last_available_date(self,tensor,prev_day_length):
        """
        If the current date is missing, fill it via replicating the last available date for that band
        If the first day is missing, fill it with next date if possible, else zeros

        Author: Selene       
        :param: tensor :  tensor to be filled
        :param: prev_day_length : how far in the past I want to go to fill the data
        :return: data list filled with replicated values from the last available data
        """
        for idx, data in enumerate(tensor): # loop through timesteps
            for b in range(tensor.size(dim=1)): # Iterate through bands
                if th.all(th.isnan(tensor[idx,b,:,:])) or -9999 in tensor[idx,b,:,:]:
                    if idx!= 0 : #if it is not the first day
                        tensor[idx,b,:,:] = tensor[idx-prev_day_length,b,:,:]

                    if idx==0: #if its the first day
                        if th.all(th.isnan(tensor[idx+1,b,:,:])) or -9999 in tensor[idx+1,b,:,:]:
                            tensor[idx,b,:,:] = 0
                        else:
                            tensor[idx,b,:,:] = tensor[idx+1,b,:,:]

        return tensor
    
    
    def replicate_last_available_date_old(self,tensor,prev_day_length):
        """
        ASSUMES ALL BANDS FOR S2 AND E5 MISSING
        If the current date is missing, fill it via replicating the last available date.
        If the first day is missing, fill it with zeros
        Author: Didem       
        :param: tensor :  tensor to be filled
        :param: prev_day_length : how far in the past I want to go to fill the data

        :return: data list filled with replicated values from the last available data
        """
        for idx,data in enumerate(tensor):
            if th.any(th.isnan(tensor[idx,:,:,:])):
                #if it is not the first day
                if idx!= 0 :
                    tensor[idx,:,:,:] = tensor[idx-prev_day_length,:,:,:]
                #if its the first day
                else:
                    tensor[idx,:,:,:] = 0

        return tensor
           
    def fill_missing_with_nan(self,number_of_bands,dataset_height,dataset_width):
        """
        Fill missing dates with NaN values

        Author: Didem, Selene
        :param: number_of_bands: total number of sentinel 2 and era5 bands
        :param: dataset_height : height of the data
        :param: dataset_width : width of the data
        :return: list of arrays filled with NaN values
        """    
        filled_arrays = list()
        for n in range(number_of_bands):
            filled_data = np.empty((dataset_height,dataset_width))
            filled_data[:] = np.NaN
            filled_arrays.append(filled_data)
        return filled_arrays

    

    def combine_loc_dates(self):
        """
        Create a list of list, which getitem will index from.
        Combines all possible timeseries from self.dict_ts_missing_dates and locations from self.list_of_all_geolocations
        The output will have the format [[location, sub_timeseries, sub_missing, sub_forecast], ...]
        
        Author: Selene
        :param self.dict_ts_missing_dates: from init
        :param self.list_of_all_geolocations: from init
        :return: list of lists
        """
        sub_ts = list(self.dict_ts_missing_dates.values())[0] # sub timeseries list
        sub_md = list(self.dict_ts_missing_dates.values())[1] # sub missing dates list
        sub_fc = list(self.dict_ts_missing_dates.values())[2] # sub forecast dates list

        return [[i, sub_ts[j], sub_md[j], sub_fc[j]] for j in range(len(sub_ts)) for i in self.list_of_all_geolocations]
    
    
    def handle_nan_in_image(self, image):
        """
        Handle NaN values within an image (Only works if image has more than 1 pixel)
        -9999 is also counted as NaN value
        Author: Didem, Selene
        :param image: image (numpy array) to work on
        :param nan_handling: method to impute nan values
        """

        if self.nan_handling == None:
            return image
        if self.nan_handling == "max":
            image = np.nan_to_num(image, nan = np.amax(image))
            image[image == -9999] = np.amax(image)
        if self.nan_handling == "min":
            image = np.nan_to_num(image, nan = np.amin(image))
            image[image == -9999] = np.amin(image)
        if self.nan_handling == "mean":
            image = np.nan_to_num(image, nan = np.mean(image))
            image[image == -9999] = np.mean(image)
        else:   # Corresponds to the case where a numeric value is passed as argument
            try:
                image = np.nan_to_num(image, float(self.nan_handling))  #
                image[image == -9999] = float(self.nan_handling)
            except:
                raise Exception("Error: Unrecognized nan-protocol option or value!")
        
        return image
    
    
    def agg_dates(self, timestamp, data_file, agg_funct_dict, folder_list, bbox):
        """
        Aggregate a bad over multipe timesteps
        
        Author: Selene
        :param data_file: str. File/band that needs to be aggregated
        :param timestamp: str. Current timestamp
        :param agg_funct_dict: dict. Contains key-value of band-aggregation method.
        :param folder_list: list of folders to work on
        :param bbox: BoundingBox object to use for windowed reading of the data
        """ 
        # Get all necessary dates
        timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d") # in datetime format
        if timestamp_dt>datetime.strptime('2017-06-29', "%Y-%m-%d"):
            start_date = timestamp_dt-timedelta(days=self.ts_delta-1)
            times_to_agg = [datetime.strftime(t, "%Y-%m-%d") for t in pd.date_range(start=start_date, end=timestamp, freq='D')]
        
        # Get folders in dates
        folders = [folder for folder in folder_list if any(t in folder for t in times_to_agg)]
        
        
        # Get file/band
        band = data_file.split('[')[1].split(']')[0] # split to keep what is in [] 
        band_search = '\['+band+'\]' 
        band_list = [] # store same band at different times, that needs to be aggregated
        for folder in folders:
            band_list += [os.path.join(folder, f) for f in os.listdir(folder) if (bool(re.search(f'{band_search}',f)) & f.endswith(self.data_file_extension))] 
        
        # Open files and aggregate according to agg funct
        all_arrays = []
        for file in band_list:
            
            with rs.open(file) as dataset:
                image = dataset.read(1, window=from_bounds(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, dataset.transform))

            # Handle NaNs in image
            if np.isnan(image).any():
                image = self.handle_nan_in_image(image, self.nan_handling)

            # Append array 
            all_arrays.append(image)

        # Aggregate
        image = self.aggregate_band(all_arrays, agg_funct_dict[band])

        return image
    
    
    def aggregate_band(self, all_arrays, agg_funct):
        """
        Aggregate a band at different timesteps
        
        Author: Selene
        :param all_arrays: list of arrays
        :param agg_funct: method for aggregating
        """
        if agg_funct=='sum':
            return sum(all_arrays)
        if agg_funct=='mean':
            return np.nanmean(all_arrays, axis=0)
        if agg_funct=='max':
            return np.nanmax(all_arrays, axis=0)
        if agg_funct=='min':
            return np.nanmin(all_arrays, axis=0)
        if agg_funct=='median':
            return np.nanmedian(all_arrays, axis=0)
        else:
            raise Exception('Method not accepted')
        
        return 
    
    
  
    
    
    def read_data_file_temporal(self, args):
        """
        Read raster files of Sentinel-2 or ERA5
        
        Author: Selene
        args contains
        :param data_file: file path
        :param bbox: bounding box from sampler at which file should be opened
        :param timestamp: current file timestamp 
        :param coords: coordinates (bounds) of data
        :param dataset_height: height of sample
        :param dataset_width: width of sample
        :param root_path: self.s2_path or self.era_path
        :param list_folders: all folders if need to aggregate mutliple
        """
        
        data_file, list_folders, bbox, timestamp, coords, dataset_height, dataset_width, root_path = args
        
        if ((datetime.strptime(timestamp, "%Y-%m-%d") > datetime.strptime('2017-06-29', "%Y-%m-%d")) and (coords!='46.907_7.137_47.407_7.637') and (self.agg_time == True)):
            image = self.agg_dates(timestamp, data_file, self.agg_funct_dict, list_folders, bbox)

        else: 
            with rs.open(data_file) as dataset:
                image = dataset.read(1, window=from_bounds(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, dataset.transform))  #left, bottom, right, top
                # Check if the image contains nan values
                if np.isnan(image).any() or -9999 in image:
                    if (dataset_height==1) and (dataset_width==1):
                        self.missing_data_exists = True
                    else:
                        image = self.handle_nan_in_image(image) 
        return image
    
    
    def read_data_file_static(self, args):
        """
        Read static raster files
        
        Author: Selene
        args contains
        :param data_file: file path
        :param bbox: bounding box from sampler at which file should be opened
        :param coords: coordinates (bounds) of data
        :param dataset_height: height of sample
        :param dataset_width: width of sample
        :param root_path: self.s2_path or self.era_path
        """
        
        data_file, bbox, coords, dataset_height, dataset_width = args
 
        with rs.open(data_file) as dataset:
            image = dataset.read(1, window=from_bounds(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, dataset.transform))  #left, bottom, right, top
            
            # Check if the image contains nan values
            if np.isnan(image).any() or -9999 in image:
                if (dataset_height==1) and (dataset_width==1):
                    self.missing_data_exists = True
                else:
                    image = self.handle_nan_in_image(image)
        
        return image

    
    
    def get_daily_data(self, args):
        """
        Get data/Read raster files for a single timestamp (in parallel)
        
        Author: Selene
        args contains
        :param bbox: bounding box from sampler at which file should be opened
        :param coords: coordinates (bounds) of data
        :param timestamp: current timestamp
        :param s2_filtered: list Sentinel-2 files filtered for the current timestamp
        :param era_filtered:  list ERA5 files filtered for the current timestamp
        :param lc_filtered:  list landcover files filtered for the current timestamp
        :param dem_at_location:  list DEM files filtered for the current timestamp
        :param env_at_location: list environmental data files filtered for the current timestamp
        :param dataset_height: height of sample
        :param dataset_width: width of sample
        """

        bbox, coords, timestamp, s2_filtered, era_filtered, lc_filtered, dem_file_list, env_file_list, dataset_height, dataset_width, correct_ndvi = args
        daily_data = list()

        ### COMMENT THIS SECTION IF NOT USING SHIFTED TIMESERIES FOR ERA/SENTINEL
        e_timestamp = datetime.strftime(datetime.strptime(timestamp, '%Y-%m-%d') + timedelta(self.ts_delta*self.len_preds), '%Y-%m-%d')
        # Era date is Sen date + shift of delta*len_preds

        sen_folders = list(filter(lambda folder: timestamp in folder, s2_filtered))
        era_folders = list(filter(lambda folder: e_timestamp in folder, era_filtered)) # change e_timestamp to timestamp if not using shifted timeseries
        
        if len(sen_folders):
            # Create list of files for Sentinel2
            sen_files = [os.path.join(self.s2_path, f, file) for f in sen_folders for file in os.listdir(os.path.join(self.s2_path,f)) if file.endswith(self.data_file_extension)]
            sen_agg_fold = [os.path.join(self.s2_path,f) for f in s2_filtered]
            with ThreadPoolExecutor() as executor:
                args = [(data_file, sen_agg_fold, bbox, timestamp, coords, dataset_height, dataset_width, self.s2_path) for data_file in sorted(sen_files)]
                sen_image_list = list(executor.map(self.read_data_file_temporal, args))
                if correct_ndvi:
                    sen_image_list = self.correct_ndvi_date(sen_image_list, timestamp)
            daily_data += sen_image_list
            
            
        
        if not len(sen_folders):
            #Fill missing dates with NaN for era and sentinel2
            self.missing_data_exists = True
            sen2_missing_data = self.fill_missing_with_nan(self.bands_s2,dataset_height,dataset_width)
            #if correct_ndvi:
                #sen2_missing_data = self.correct_ndvi_date(sen2_missing_data, timestamp)
            daily_data += sen2_missing_data
            

        if len(era_folders):
            # Create list of files for Sentinel2
            era_files = [os.path.join(self.era_path, f, file) for f in era_folders for file in os.listdir(os.path.join(self.era_path,f)) if file.endswith(self.data_file_extension)]
            era_agg_fold = [os.path.join(self.era_path,f) for f in era_filtered]
            with ThreadPoolExecutor() as executor:
                args = [(data_file, era_agg_fold, bbox, e_timestamp, coords, dataset_height, dataset_width, self.era_path) for data_file in sorted(era_files)]
                era_image_list = list(executor.map(self.read_data_file_temporal, args))
            daily_data += era_image_list
        
        
        if not len(era_folders):
            #concatenate an NaN matrix with the dimension of era5 data files for missing dates
            self.missing_data_exists = True
            era5_missing_data = self.fill_missing_with_nan(self.bands_era,dataset_height,dataset_width)
            daily_data += era5_missing_data


        # Then add all the bands from static data 
        static_files = lc_filtered + dem_file_list + env_file_list 
        with ThreadPoolExecutor() as executor:
            args = [(data_file, bbox, coords, dataset_height, dataset_width) for data_file in sorted(static_files)]
            stat_image_list = list(executor.map(self.read_data_file_static, args))
        daily_data += stat_image_list
        
        daily_data = np.stack(daily_data)
        
     
        return daily_data
    
    
    def get_daily_label_data(self, args):
        """
        Get label data/Read raster files for a single timestamp (in parallel)
        
        Author: Selene
        args contains
        :param bbox: bounding box from sampler at which file should be opened
        :param coords: coordinates (bounds) of data
        :param timestamp: current timestamp
        :param s2_filtered: list Sentinel-2 files filtered for the current timestamp
        :param era_filtered:  list ERA5 files filtered for the current timestamp
        :param lc_filtered:  list landcover files filtered for the current timestamp
        :param dem_at_location:  list DEM files filtered for the current timestamp
        :param env_at_location: list environmental data files filtered for the current timestamp
        :param dataset_height: height of sample
        :param dataset_width: width of sample
        """
            
        bbox, coords, timestamp, s2_filtered, era_filtered, lc_filtered, dem_file_list, env_file_list, dataset_height, dataset_width, correct_ndvi = args
        
        daily_data = list()
        
        label_search = '\[' +  'NDVI' + '\]' 
        cp_search = '\[' +  'CP' + '\]' 

        sen_folders = list(filter(lambda folder: timestamp in folder, s2_filtered))
        
        if len(sen_folders):
            # Create list of files for Sentinel2
            sen_files = [os.path.join(self.s2_path, f, file) for f in sen_folders for file in os.listdir(os.path.join(self.s2_path,f)) if file.endswith(self.data_file_extension)]
            if self.multiple_labels:
                sen_agg_fold = [os.path.join(self.s2_path,f) for f in s2_filtered]
                with ThreadPoolExecutor() as executor:
                    args = [(data_file, sen_agg_fold, bbox, timestamp, coords, dataset_height, dataset_width, self.s2_path) for data_file in sorted(sen_files)]
                    sen_image_list = list(executor.map(self.read_data_file_temporal, args))
                if correct_ndvi:
                    sen_image_list = self.correct_ndvi_date(sen_image_list, timestamp)
                daily_data += sen_image_list
            if not self.multiple_labels:
                sen_files = [f for f in sen_files if bool(re.search(f'{label_search}',f)) or bool(re.search(f'{cp_search}',f))]
                sen_agg_fold = [os.path.join(self.s2_path,f) for f in s2_filtered]
                with ThreadPoolExecutor() as executor:
                    args = [(data_file, sen_agg_fold, bbox, timestamp, coords, dataset_height, dataset_width, self.s2_path) for data_file in sorted(sen_files)]
                    sen_image_list = list(executor.map(self.read_data_file_temporal, args))
                if correct_ndvi:
                    sen_image_list = self.correct_ndvi_date(sen_image_list, timestamp)
                # Drop CP 
                sen_image_list = sen_image_list[-1]
                daily_data += [sen_image_list]

            
            
        if not len(sen_folders):
            #Fill missing dates with NaN for era and sentinel2
            self.missing_data_exists = True
            if not self.multiple_labels:
                sen2_missing_data = self.fill_missing_with_nan(1,dataset_height,dataset_width) 
            if self.multiple_labels:
                sen2_missing_data = self.fill_missing_with_nan(self.bands_s2,dataset_height,dataset_width) 
            daily_data += sen2_missing_data

                     

        if self.multiple_labels:
            # If wokring with multiband output and shifted weather data
            e_timestamp = datetime.strftime(datetime.strptime(timestamp, '%Y-%m-%d') + timedelta(self.ts_delta*self.len_preds), '%Y-%m-%d')
            era_folders = list(filter(lambda folder: e_timestamp in folder, era_filtered))
            
            if len(era_folders):
                # Create list of files for Sentinel2
                era_files = [os.path.join(self.era_path, f, file) for f in era_folders for file in os.listdir(os.path.join(self.era_path,f)) if file.endswith(self.data_file_extension)]
                era_agg_fold = [os.path.join(self.era_path,f) for f in era_filtered]
                with ThreadPoolExecutor() as executor:
                    args = [(data_file, era_agg_fold, bbox, e_timestamp, coords, dataset_height, dataset_width, self.era_path) for data_file in sorted(era_files)]
                    era_image_list = list(executor.map(self.read_data_file_temporal, args))
                daily_data += era_image_list
        
        
            if not len(era_folders):
                #concatenate an NaN matrix with the dimension of era5 data files for missing dates
                self.missing_data_exists = True
                era5_missing_data = self.fill_missing_with_nan(self.bands_era,dataset_height,dataset_width)
                daily_data += era5_missing_data

            # Then add all the bands from static data 
            static_files = lc_filtered + dem_file_list + env_file_list 
            with ThreadPoolExecutor() as executor:
                args = [(data_file, bbox, coords, dataset_height, dataset_width) for data_file in sorted(static_files)]
                stat_image_list = list(executor.map(self.read_data_file_static, args))
            daily_data += stat_image_list
        
        if self.multiple_labels:
            daily_data = np.stack(daily_data) 
            
        return daily_data
    
    
    def correct_ndvi_date(self, sen_image_list, timestamp):
        """
        Change the NDVI value if cloud probability is too high
        
        Author: Selene
        :param sen_image_list: list of sentinel images at a date that have been sampeld and read
        :param timestamp: timestamp of data (string)
        """
        
        if self.multiple_labels:
            # Get CP
            cp_data = sen_image_list[self.feature_set['CP']]
            # For values where CP>self.correrct_ndvi get NDVI at date
            if np.any(cp_data>self.correct_ndvi):
                ndvi_data = sen_image_list[self.feature_set['NDVI']]
                # Convert to datetime to find location in self.full_date_range
                timestamp = datetime.strptime(timestamp, '%Y-%m-%d')
                timestamp_idx = self.full_date_range.index(timestamp)
                ndvi_correct = get_ndvi_at_day(timestamp_idx) #Function in utils
                ndvi_data[cp_data>self.correct_ndvi] = ndvi_correct
                sen_image_list[self.feature_set['NDVI']] = ndvi_data
                cp_data[cp_data>self.correct_ndvi] = 0
                sen_image_list[self.feature_set['CP']] = cp_data
                
        if not self.multiple_labels:
            # Get CP
            cp_data = sen_image_list[0]
            # For values where CP>self.correrct_ndvi get NDVI at date
            if np.any(cp_data>self.correct_ndvi):
                ndvi_data = sen_image_list[-1]
                # Convert to datetime to find location in self.full_date_range
                timestamp = datetime.strptime(timestamp, '%Y-%m-%d')
                timestamp_idx = self.full_date_range.index(timestamp)
                ndvi_correct = get_ndvi_at_day(timestamp_idx) #Function in utils
                ndvi_data[cp_data>self.correct_ndvi] = ndvi_correct
                sen_image_list[-1] = ndvi_data
                cp_data[cp_data>self.correct_ndvi] = 0
                sen_image_list[0] = cp_data
            

        return sen_image_list


            
        
        
        

def tiff_to_tensor(fname):
    """
    Imports a file by full pathname into a torch tensor
    Every band is translated into an additional dimension of the tensor
    :param fname:
    :return:
    """
    tiff_image = rs.open(fname)
    im_band_list = tiff_image.read()            # get different layers of image -> list of numpy arrays
    im_bands = np.stack(im_band_list, axis=0)   # stack so it can be converted to torch tensor
    tensor_image = th.from_numpy(im_bands)      # convert to tensor
    tiff_image.close()
    return tensor_image

def look_inside_tiff(fname):
    """
    The following content is available by specs
    driver: the name of the desired format driver
    width: the number of columns of the dataset
    height: the number of rows of the dataset
    count: a count of the dataset bands
    dtype: the data type of the dataset
    crs: a coordinate reference system identifier or description
    transform: an affine transformation matrix, and
    nodata: a “nodata” value
    :param fname:   file name
    :return:        nothing
    """
    with rs.open(fname) as dataset:

        print("Number of channels: ", dataset.count)
        print("Image Height: ", dataset.height)
        print("Image Width: ", dataset.width)
        print("Image Center Coordinates:", dataset.xy(dataset.height // 2, dataset.width // 2))


        # Read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()

        # Extract feature shapes and values from the array.
        for geom, val in rs.features.shapes(
                mask, transform=dataset.transform):
            # Transform shapes from the dataset's own coordinate
            # reference system to CRS84 (EPSG:4326).
            geom = rs.warp.transform_geom(
                dataset.crs, 'EPSG:4326', geom, precision=6)
            # Print GeoJSON shapes to stdout.
            print(geom)
    return

def create_false_rgb_from_2band_batch(*batch_with_Xbands_arg):
    """
    :param batch_with_2bands: torch tensor with dimension [b 2 w h] to be augmented to [b 3 w h]
    :return: batch_with_3bands - torch tensor [b 3 w h] with 2'nd dimension
    """
    batch_with_Xbands=batch_with_Xbands_arg[0]
    [batch_nb,channel_nb,height,width]=batch_with_Xbands.shape

    twobands_batch_mixture = th.unsqueeze(batch_with_Xbands[:, 0, :, :] - batch_with_Xbands[:, 1, :, :], 1)
    #eps = 0.0001
    #twobands_batch_mixture = th.unsqueeze(th.abs(th.div(batch_with_Xbands[:, 0, :, :], (batch_with_Xbands[:, 1, :, :] + eps))), 1)
    if len(batch_with_Xbands_arg) == 1: # do nothing
        processed_batch_with_Ybands = batch_with_Xbands


    elif len(batch_with_Xbands_arg) >= 2: # add a mixture channel
        create_channel_flag=batch_with_Xbands_arg[1]
        if create_channel_flag==True:
            processed_batch_with_Ybands = th.cat([batch_with_Xbands[:, :2, :, :], twobands_batch_mixture, batch_with_Xbands[:, 2:, :, :]], 1)
        else:
            processed_batch_with_Ybands = batch_with_Xbands
    if len(batch_with_Xbands_arg) == 3:  # remove a channel
        #remove_channel_flag = batch_with_Xbands_arg[2]
        remove_channel = batch_with_Xbands_arg[2]
        processed_batch_with_Ybands = th.cat([processed_batch_with_Ybands[:, :remove_channel, :, :], processed_batch_with_Ybands[:, (remove_channel + 1):, :, :]], 1)


    return processed_batch_with_Ybands

def create_false_rgb_from_2band_image(*image_with_Xbands_arg):
    """
    :param image_with_2bands: torch tensor with dimension [b 2 w h] to be augmented to [b 3 w h]
    :return: batch_with_3bands - torch tensor [b 3 w h] with 2'nd dimension
    """
    image_with_Xbands = image_with_Xbands_arg[0]
    twobands_image_mixture = th.unsqueeze(image_with_Xbands[0, :, :] - image_with_Xbands[1, :, :], 0)
    #eps=0.0001
    #twobands_image_mixture = th.unsqueeze(th.abs(th.div(image_with_Xbands[0, :, :],(image_with_Xbands[1, :, :]+eps))), 0)

    #image_with_3bands = th.cat([image_with_2bands, twobands_image_mixture], 0)

    if len(image_with_Xbands_arg) == 1: # do nothing
        processed_image_with_Ybands = image_with_Xbands

    elif len(image_with_Xbands_arg) >= 2: # add a mixture channel
        create_channel_flag=image_with_Xbands_arg[1]
        if create_channel_flag==True:
            processed_image_with_Ybands = th.cat([image_with_Xbands[:2, :, :], twobands_image_mixture, image_with_Xbands[2:, :, :]], 0)
        else:
            processed_image_with_Ybands = image_with_Xbands
    if len(image_with_Xbands_arg) == 3:  # remove a channel
        # remove_channel_flag = image_with_Xbands_arg[2]
        remove_channel = image_with_Xbands_arg[2]
        processed_image_with_Ybands = th.cat([processed_image_with_Ybands[:remove_channel, :, :],processed_image_with_Ybands[(remove_channel + 1):, :, :]], 0)

    return processed_image_with_Ybands

def normalize(array):

    """
    :param array: numpy array containing single input channel
    :return: normalized to 0 ... 1.0 image
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')













