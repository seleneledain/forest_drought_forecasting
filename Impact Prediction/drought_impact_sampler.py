"""
Custom sampler class for drought impact prediction. 

Created:    24th of November 2022
Authors:    Selene Ledain (selene.ledain@ibm.com)
            Emine Didem Durukan (emine.didem.durukan@ibm.com)
"""

from tqdm import tqdm 
import pandas as pd
import rasterio as rs
from rasterio.windows import from_bounds
import numpy as np
from matplotlib import pyplot as plt
import torch as th
import torch.nn as nn
from torch.utils.data import SequentialSampler
import pathlib
from pathlib import Path
from datetime import datetime, timedelta
#import albumentations as album  # data augmentation
from itertools import groupby, islice, tee
from operator import itemgetter
import math
import glob
import random
import os
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)
th.manual_seed(0)
random.seed(0) 

class DroughtImpactSampler(SequentialSampler):

    def __init__(self, dataset, size, length, replacement, roi=None, mask_dir=None, mask_threshold=0, set_seed=True, static_dir=None):
        """
        Samples n=length elements of given size from dataset
        
        Author:Selene
        :param dataset:   Dataset object
        :param size:   a tuple of 2 float [height, width] in pixels
        :param length:   number of samples to generate
        :param repalcement:  Binary (0 or 1). True=sample with replaceement. False=sample wihtout replacement.
        :param roi: a tuple [minx, maxx, miny, maxy]
        :param mask_dir: path to diectory containing raster mask for pixel-wise sampling. Mask should be boolean.
        :param mask_threshold: float. The sample must contain mask_threshold % of the mask in it.
        :param set_seed: boolean. Fix seed for sample generation.
        :param static_dir: Directory where static files are found. Only if doing point-wise sampling. Check thst value at locaiton is not Nan
        """
        self.dataset = dataset
        self.size = size
        self.length = length 
        self.roi = roi if roi is not None else None # bounding box from which we want to sample
        self.set_seed = set_seed    
        self.replacement = replacement
        self.mask_dir = mask_dir if mask_dir is not None else None 
        self.mask_threshold = mask_threshold
        
        if not self.replacement:
            self.used_samples = list() # to save info on which samples arleady used

        if self.roi:
            # Check if dataset bound are within roi, else raise an exception
            self.bbox_roi = BoundingBox(*self.roi)
            # Atleast one of the geolocations intersects with the roi
            if not any(self.bbox_roi.intersects(BoundingBox(*[float(region.split('_')[1]), float(region.split('_')[3]), float(region.split('_')[0]), float(region.split('_')[2])])) for region in self.dataset.list_of_all_geolocations):
                raise ValueError(
                    "ROI doesn't intersect with any data in the dataset"
                )
        
        # ONLY IF DOING POINT-WISE SAMPLING
        self.static_dir = static_dir if static_dir is not None else None 
        
        
        # Convert pixel size to lat/lon coordinates
        # open a random raster at that geolocation and get transform
        raster_path = os.path.join(self.dataset.era_path, f'**{self.dataset.list_of_all_geolocations[0]}', f'*{self.dataset.data_file_extension}')
        raster_path = next(iter(glob.glob(raster_path)), None)
        if not raster_path:
            print("No raster found for the given coords.")
            return None
        with rs.open(raster_path) as src:
            trf = src.transform
        lons, lats = rs.transform.xy(trf, [0, self.size[1]], [0,self.size[0]])
        self.size_lat = lats[0] - lats[1]
        self.size_lon = lons[1] - lons[0] # input the sizes for the sampling
                

    def __iter__(self):
        """
        Providing a way to iterate over indices of dataset elements. Return an iterable
        Return a combo of an index for dataset.all_loc_dates and a bounding box of appropriate size
        """
        

        if self.set_seed:
            th.manual_seed(0)
            random.seed(0) # for functions that are beyond pytorch

        
        for idx in range(self.length):
            # Index a location and timeseries
            n_repeat = self.length/len(self.dataset.all_loc_dates) # number of times sampler lenght is bigger than list to index
            idx_call = int(np.floor(idx/n_repeat))
            #idx = idx%len(self.dataset.all_loc_dates) 
            coords = self.dataset.all_loc_dates[idx_call][0] # get bounds of data 
            miny = float(coords.split('_')[0]) #min lat
            minx = float(coords.split('_')[1]) #min lon
            maxy = float(coords.split('_')[2]) #max lat
            maxx = float(coords.split('_')[3]) #max lon
            
        
            # Convert to BoundingBox object
            bounds = BoundingBox(minx, maxx, miny, maxy)
            if self.roi:
                # Intersect with ROI
                bounds = bounds.__and__(self.bbox_roi)
            # Choose a random box within that tile
            random_box = get_random_bounding_box(bounds, [self.size_lat, self.size_lon])
            
            # If sampling with replacement, might have to check for mask
            if self.mask_dir is not None:
                # Window read the mask using random box
                MASKPATH = os.path.join(self.mask_dir, f'**{coords}', f'*forest_mask_binary.{self.dataset.data_file_extension}')
                MASKPATH = next(iter(glob.glob(MASKPATH)), None)
                if not MASKPATH:
                    print("No mask found for the given coords.")
                    return None
                with rs.open(MASKPATH) as src:
                    mask_box = src.read(1, window=from_bounds(random_box.minx, random_box.miny, random_box.maxx, random_box.maxy, src.transform))  #left, bottom, right, top
                # Compare ratio forest:total and check with threshold
                ratio_forest = mask_box.sum()/len(mask_box.flatten())
                
    

            # If sampling without replacement, make sure that sample doesn't exist already and optionally that static band isnt -9999
            if not self.replacement:
                #if [idx, random_box.minx, random_box.maxx, random_box.miny, random_box.maxy] in self.used_samples:
                if self.mask_dir is not None:
                    while ([idx, random_box.minx, random_box.maxx, random_box.miny, random_box.maxy] in self.used_samples or ratio_forest<=self.mask_threshold):
                        random_box = get_random_bounding_box(bounds, [self.size_lat, self.size_lon])
                        with rs.open(MASKPATH) as src:
                            mask_box = src.read(1, window=from_bounds(random_box.minx, random_box.miny, random_box.maxx, random_box.maxy, src.transform))
                            ratio_forest = mask_box.sum()/len(mask_box.flatten())
                        # Open a static band and check val
                        if self.static_dir is not None:
                            STATICPATH = os.path.join(self.static_dir, f'**{coords}', f'*{self.dataset.data_file_extension}')
                            STATICPATH = next(iter(glob.glob(STATICPATH)), None)
                            if not STATICPATH:
                                print("No data found for the given coords.")
                                return None
                            with rs.open(STATICPATH) as src:
                                static_val = src.read(1, window=from_bounds(random_box.minx, random_box.miny, random_box.maxx, random_box.maxy, src.transform))
                            if static_val == -9999:
                                ratio_forest = -9999

                if self.mask_dir is None:
                    while ([idx, random_box.minx, random_box.maxx, random_box.miny, random_box.maxy] in self.used_samples):
                        random_box = get_random_bounding_box(bounds, [self.size_lat, self.size_lon])
                        # Open a static band and check val
                        if self.static_dir is not None:
                            STATICPATH = os.path.join(self.static_dir, f'**{coords}', f'*{self.dataset.data_file_extension}')
                            STATICPATH = next(iter(glob.glob(STATICPATH)), None)
                            if not STATICPATH:
                                print("No data found for the given coords.")
                                return None
                            with rs.open(STATICPATH) as src:
                                static_val = src.read(1, window=from_bounds(random_box.minx, random_box.miny, random_box.maxx, random_box.maxy, src.transform))
                            if static_val == -9999:
                                ratio_forest = -9999
                                    
                # once sampled, append to list
                self.used_samples.append([idx, random_box.minx, random_box.maxx, random_box.miny, random_box.maxy])    

                
                
            # If sampling with replacement, might still have to resample if mask/forest_ratio conditions not satisfied
            if self.replacement:
                if self.mask_dir is not None:
                    while ratio_forest<=self.mask_threshold:
                        random_box = get_random_bounding_box(bounds, [self.size_lat, self.size_lon])
                        with rs.open(MASKPATH) as src:
                            mask_box = src.read(1, window=from_bounds(random_box.minx, random_box.miny, random_box.maxx, random_box.maxy, src.transform))
                            ratio_forest = mask_box.sum()/len(mask_box.flatten())
                            # Open a static band and check val
                            if self.static_dir is not None:
                                STATICPATH = os.path.join(self.static_dir, f'**{coords}', f'*{self.dataset.data_file_extension}')
                                STATICPATH = next(iter(glob.glob(STATICPATH)), None)
                                if not STATICPATH:
                                    print("No data found for the given coords.")
                                    return None
                                with rs.open(STATICPATH) as src:
                                    static_val = src.read(1, window=from_bounds(random_box.minx, random_box.miny, random_box.maxx, random_box.maxy, src.transform))
                                if static_val == -9999:
                                    ratio_forest = -9999
          

            yield self.length, idx, random_box
    
    

    def __len__(self):
        """
        Returns the length of the returned iterators.
        """
        return self.length



class BoundingBox():
    """
    Edited from torchgeo.
    Bounding box objects that can be operated between them
    
    Editor: Selene Ledain
    """

    def __init__(self, minx, maxx, miny, maxy):
        """
        :params:
        :minx: float, western boundary (min lon)
        :maxx: float, eastern boundary (max lon)
        :miny: float, southern boundary (min lat)
        :maxy: float ,northern boundary (max lat)
        """

        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

        if self.minx > self.maxx:
            raise ValueError(
                f"Bounding box is invalid: 'minx={self.minx}' > 'maxx={self.maxx}'"
            )
        if self.miny > self.maxy:
            raise ValueError(
                f"Bounding box is invalid: 'miny={self.miny}' > 'maxy={self.maxy}'"
            )
        

    def __contains__(self, other: "BoundingBox") -> bool:
        """Whether or not other is within the bounds of this bounding box.
        Args:
            other: another bounding box
        Returns:
            True if other is within this bounding box, else False
        .. versionadded:: 0.2
        """
        return (
            (self.minx <= other.minx <= self.maxx)
            and (self.minx <= other.maxx <= self.maxx)
            and (self.miny <= other.miny <= self.maxy)
            and (self.miny <= other.maxy <= self.maxy)
        )

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        """The union operator.
        Args:
            other: another bounding box
        Returns:
            the minimum bounding box that contains both self and other
        .. versionadded:: 0.2
        """
        return BoundingBox(
            min(self.minx, other.minx),
            max(self.maxx, other.maxx),
            min(self.miny, other.miny),
            max(self.maxy, other.maxy)
        )

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
        """The intersection operator.
        Args:
            other: another bounding box
        Returns:
            the intersection of self and other
        Raises:
            ValueError: if self and other do not intersect
        .. versionadded:: 0.2
        """
        try:
            return BoundingBox(
                max(self.minx, other.minx),
                min(self.maxx, other.maxx),
                max(self.miny, other.miny),
                min(self.maxy, other.maxy)
            )
        except ValueError:
            raise ValueError(f"Bounding boxes {self} and {other} do not overlap")

    @property
    def area(self) -> float:
        """Area of bounding box.
        Area is defined as spatial area.
        Returns:
            area
        .. versionadded:: 0.3
        """
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def intersects(self, other: "BoundingBox") -> bool:
        """Whether or not two bounding boxes intersect.
        Args:
            other: another bounding box
        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
        )


def get_random_bounding_box(
    bounds: BoundingBox, size: Union[Tuple[float, float], float]
) -> BoundingBox:
    """
    Edited from torchgeo. Returns a random bounding box within a given bounding box.
    
    Author: Selene
    :param size: can either be a single float in which case the same value is used for the height and width dimension or 
    a tuple of two floats in which case, the first is used for the height dimension, and the second for the width dimension
    :param bounds: the larger bounding box to sample from
    :param returns: randomly sampled bounding box from the extent of the input
    """
            
    t_size = _to_tuple(size)

    width = (bounds.maxx - bounds.minx - t_size[1]) 
    height = (bounds.maxy - bounds.miny - t_size[0]) 

    minx = bounds.minx
    miny = bounds.miny

    # random.randrange crashes for inputs <= 0
    if width > 0:
        minx += th.rand(1).item() * width 
    if height > 0:
        miny += th.rand(1).item() * height 

    maxx = minx + t_size[1]
    maxy = miny + t_size[0]

    query = BoundingBox(minx, maxx, miny, maxy)
    return query



def _to_tuple(value: Union[Tuple[float, float], float]) -> Tuple[float, float]:
    """
    From torchgeo.
    Convert value to a tuple if it is not already a tuple.
    Args:
        value: input value
    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (float, int)):
        return (value, value)
    else:
        return value