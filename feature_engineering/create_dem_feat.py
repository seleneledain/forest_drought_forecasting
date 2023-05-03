"""
Create topograpgical features from DEM. 

Saves rasters of new features

Created:    Dec. 1st 2022
Authors:    Selene Ledain (selene.ledain@ibm.com)
"""

import argparse
import os
import numpy as np
import rasterio

import feature_engineer as fe
import geospatial_data_utils as gdu


def remove_nan_border(path_file, out_path):
    """
    Remove nan-valued border in raster and resave to file
    :param path_file: path to file
    :param out_path: path where output will be written
    """
    
    with rasterio.open(path_file) as src:
        trs = src.transform
        kwargs = src.meta
        arr = src.read(1)

    height = arr.shape[0]
    width = arr.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    lons = np.array(xs)[0,:]
    lats = np.array(ys)[:,0]

    gt = src.transform
    pixelSizeX = gt[0]
    pixelSizeY =-gt[4]

    # Remove rows and cols with full Nans
    arr[arr==-9999] = np.nan
    arr_row = arr[~np.isnan(arr).all(axis=1), :]
    arr_new = arr_row[:, ~np.isnan(arr_row).all(axis=0)]

    # Remove corresponding lats/lons
    lats_new = lats[~np.isnan(arr).all(axis=1)]
    lons_new = lons[~np.isnan(arr).all(axis=0)]

    trs_new = rasterio.transform.from_bounds(lons_new[0]-pixelSizeX/2, lats_new[-1]-pixelSizeY/2, lons_new[-1]+pixelSizeX/2, lats_new[0]+pixelSizeY/2, 
                                             arr_new.shape[1], arr_new.shape[0])
    #trs_new = rasterio.transform.from_origin(lons_new[0]-pixelSizeX/2,lats_new[0]+pixelSizeY/2, pixelSizeX, pixelSizeY) # also works

    # Save this as target raster
    kwargs.update(
        transform=trs_new,
        width=arr_new.shape[1],
        height=arr_new.shape[0],
    )

    with rasterio.open(out_path, 'w', **kwargs) as dst:
        dst.write(arr_new, 1)


    return


def create_topo_feats(feat_list, dem_path, out_path, suffix=None):
    """
    :param feat_list: list of features to create. ['slope', 'aspect', 'rugg', 'curv', 'twi']
    :param dem_path: path to DEM file
    :out_path: path to directory where features will be saved (TIF files). Needs to end with '/'
    """
    suffix = '_'+suffix if suffix is not None else ''
    file_end = suffix + '.tiff'
    
    if 'slope' in feat_list:
        fe.get_slope(dem_path, out_path+'slope'+file_end)
    if 'aspect' in feat_list:
        fe.get_aspect(dem_path, out_path+'aspect'+file_end)
        fe.get_northing_easting(out_path+'aspect'+file_end, out_path, 'northing'+file_end, 'easting'+file_end)
        os.remove(out_path+'aspect'+file_end)
    if 'rugg' in feat_list:
        fe.get_ruggedness(dem_path, out_path+'rugg'+file_end)
    if 'curv' in feat_list:
        fe.get_mean_curvature(dem_path, out_path+'curv'+file_end)
    if 'twi' in feat_list:
        fe.get_flow_accumulation(dem_path, out_path+'outdem'+file_end, out_path+'pointer'+file_end, out_path+'accum'+file_end)
        fe.get_wetness_index(out_path+'accum'+file_end, out_path+'slope'+file_end, out_path+'twi'+file_end)
        os.remove(out_path+'accum'+file_end)
        os.remove(out_path+'pointer'+file_end)
        os.remove(out_path+'outdem'+file_end)
        
                
    return

def add_border(file_path, original_path):
    """
    Add NaN-valued border so that raster matches the original raster shape
    :param file_path: path to file that should be modified
    :param oiginal_path: path to file that should be used as exmaple for size
    """
    # Open original file
    with rasterio.open(original_path) as dst:
        trs = dst.transform
        kwargs = dst.meta
        arr = dst.read(1)
        arr[arr==-9999] = np.nan
        
    # Open file
    with rasterio.open(file_path) as src:
        arr_f = src.read(1)
        
    # Nan cols left
    n_left = np.isnan(arr[:, :int(np.floor(arr_f.shape[0]/2))]).all(axis=0).sum()
    # Nan cols right
    n_righ = np.isnan(arr[:, int(np.floor(arr_f.shape[0]/2)):]).all(axis=0).sum()
    # Nan rows above (check upper half of array)
    n_top = np.isnan(arr[:int(np.floor(arr_f.shape[0]/2)),:]).all(axis=1).sum()
    # Nan rows below
    n_bot = np.isnan(arr[int(np.floor(arr_f.shape[0]/2)):,:]).all(axis=1).sum()
    
    # Pad the file to match the original
    padded_f = np.pad(arr_f, ((n_top, n_bot), (n_left, n_righ)), constant_values=((-9999,-9999),(-9999,-9999)))

    # Save new file (replace old one) CAREFUL: WILL OVERWRITE FILE
    with rasterio.open(file_path, 'w', **kwargs) as dest:
        dest.write(padded_f, 1)
    
    return 


def feats_from_dem(dem_path, target, feat_list, out_path, suffix=None):
    """
    Create features at lower resolution DEM and boost/align to higher resolution raster
    :param dem_path: path to DEM
    :param target: path to target DEM/raster (in terms of reoslution and alignment)
    :param feat_list: list of features to create
    :param out_path: path where features should be saved
    :param suffix: optional suffix to add to feature raster names
    """
    
    # Create target: crop Nan borders if there are any
    if not os.path.exists('target_tmp.tiff'):
        remove_nan_border(target, out_path+'target_tmp.tiff')
    
    # Crop DEM: remove Nan border if necessary
    remove_nan_border(dem_path, out_path+'dem_cropped.tiff')
    
    # Create features
    create_topo_feats(feat_list, dem_path, out_path, suffix=suffix)
    
    # Align and boost features to match target (will crop if necessary)
    if 'aspect' in feat_list:
        feat_list += ['northing', 'easting']
    for f in feat_list:
        filename = out_path + f + '_' + suffix + '.tiff'
        target_path = out_path+'target_tmp.tiff'
        raster_matched = gdu.match_raster_to_target(filename, target_path, output_suffix='_matched', resampling_method="near") 

        
    # Re-add border to have original shape
    for f in os.listdir(out_path):
        if suffix+'_matched' in f:
            add_border(out_path+f, target)
            os.rename(out_path+f,out_path+f.split('_matched')[0]+'.tiff')
    

    # Delete temporary files
    os.remove(out_path + 'dem_cropped.tiff')
    os.remove(out_path + 'target_tmp.tiff')
 

    return



def create_from_multiple_dems(list_paths, target, feat_list, out_path, list_suffix=None):
    """
    Iterate feature creation process for mutiple DEMs (at different resolutions)
    :params list_paths: list of DEM paths
    :param target: target that should be followed alignment and final resolution
    :param feat_list: list of features to be created
    :param out_path: where the feature should be saved
    :param list_suffix: list of corresponding suffixes to be used
    """
    
    list_suffix = list_suffix if list_suffix is not None else ['']*len(list_paths)
    
    for i, path in enumerate(list_paths):
        suffix=list_suffix[i]
        feats_from_dem(path, target, feat_list, out_path, suffix=suffix)       
    
    
    return
