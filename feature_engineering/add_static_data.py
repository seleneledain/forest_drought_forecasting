"""
Read raster and create tif to add to minicube.

Author: Selene Ledain
Date: May 26th, 2023
"""

import rasterio
import os
from osgeo import osr

try:
    from osgeo import gdal
except:
    import gdal
    
import os
import rasterio
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio import features
from bands_info import BANDS_DESCRIPTION, dict_static_features


def get_attrs_for_band(band, provider):

        attrs = {}
        attrs["provider"] = provider #"Sentinel 2"
        if provider == 'Sentinel 2':
            attrs["interpolation_type"] = "linear"
        else:
            attrs["interpolation_type"] = "nearest"
        attrs["description"] = BANDS_DESCRIPTION[band]
        
        return attrs
        
    
def get_raster_resolution(raster_file):

    """
    It extracts a raster resolution. It accounts for changes in rasterio >= 1.0
    :param raster_file: path to raster file
    :return: resolution
    """

    transform = get_raster_meta(raster_file)["transform"]

    if isinstance(transform, Affine):

        res = transform[0]

    else:

        res = transform[1]

    return res


def get_raster_meta(raster_file):

    """
    It extracts the raster's metadata
    :param raster_file: path to raster file
    :return: raster metadata
    """

    with rasterio.open(raster_file) as src:

        out_meta = src.meta

    return out_meta


def get_raster_bbox(raster_file):

    """
    It extracts a bbox from a raster file.
    :param raster_file: path to raster file
    :return: bbox (left, bottom, right, up)
    """

    with rasterio.open(raster_file) as src:

        bbox = src.bounds

    return bbox

def get_raster_crs(raster_file):

    """
    It extracts a raster crs.
    :param raster_file: raster_file: path to raster file
    :return: string containing crs
    """
    meta = get_raster_meta(raster_file)
        
    crs = str(meta['crs'])

    return crs

def match_raster_to_target(input_file, target_file, target_crs="epsg:4326", resampling_method="nearest"):
    """
    Matches an input raster to a target raster and returns a NumPy array.
    :param input_file: path to input raster file
    :param target_file: path to target input file
    :param target_crs: CRS of the output/target
    :param resampling_method: resampling method for aligning the rasters
    :return: NumPy array containing the aligned raster values.
    """
    target_dataset = gdal.Open(target_file)
    if target_dataset is None:
        print("Failed to open target dataset.")
        return None

    target_geo_transform = target_dataset.GetGeoTransform()
    target_projection = target_dataset.GetProjection()
    
    # Get bounds of target (minX, minY, maxX, maxY)
    minX, xres, xskew, maxY, yskew, yres  = target_dataset.GetGeoTransform()
    maxX = minX + (target_dataset.RasterXSize * xres)
    minY = maxY + (target_dataset.RasterYSize * yres) #yres is negative

    input_dataset = gdal.Open(input_file)
    if input_dataset is None:
        print("Failed to open input dataset.")
        return None

    input_projection = input_dataset.GetProjection()

    output_geo_transform = target_geo_transform
    output_projection = target_projection
    output_width = target_dataset.RasterXSize
    output_height = target_dataset.RasterYSize

    output_dataset = gdal.Warp('', input_dataset, format='MEM', dstSRS=target_crs,
                               outputBounds=(minX, minY, maxX, maxY),
                               xRes=target_geo_transform[1], yRes=target_geo_transform[5],
                               resampleAlg=resampling_method)
    
    if output_dataset is None:
        print("Failed to create output dataset.")
        return None

    output_array = output_dataset.ReadAsArray()
    output_dataset = None  # Close the dataset to release resources
    input_dataset = None

    return output_array




def match_raster_to_minicube(input_file, minicube, target_crs="epsg:4326", resampling_method="nearest"):
    """
    Matches an input raster to a target raster and returns a NumPy array.
    :param input_file: path to input raster file
    :param minicube: xarray dataset to which to add data to.
    :param target_crs: CRS of the output/target
    :param resampling_method: resampling method for aligning the rasters
    :return: NumPy array containing the aligned raster values.
    """
    
    output_width = len(minicube['lon'])
    output_height = len(minicube['lat'])
    xRes = (minicube['lon'][1]- minicube['lon'][0]).values #longitudinal
    yRes = (minicube['lat'][1]- minicube['lat'][0]).values #latitudinal

    # Bounds
    minX, maxX = minicube['lon'].min().item(), minicube['lon'].max().item()
    minY, maxY = minicube['lat'].min().item(), minicube['lat'].max().item()

    # Transform
    transform = rasterio.transform.from_bounds(minX, minY, maxX, maxY, output_width, output_height)
    
    input_dataset = gdal.Open(input_file)
    if input_dataset is None:
        print("Failed to open input dataset.")
        return None

    input_projection = input_dataset.GetProjection()
    output_projection = target_crs
    
    output_dataset = gdal.Warp('', input_dataset, format='MEM', dstSRS=target_crs,
                           outputBounds=(minX-xRes/2, minY+yRes/2, maxX+xRes/2, maxY-yRes/2),
                           xRes=xRes, yRes=yRes,
                           resampleAlg=resampling_method, outputType=gdal.GDT_Float32, dstNodata=-9999)


    if output_dataset is None:
        print("Failed to create output dataset.")
        return None

    output_array = output_dataset.ReadAsArray()
    output_dataset = None  # Close the dataset to release resources
    input_dataset = None
    
    return output_array



def add_static_to_minicube(list_features, static_dir, minicube, target_crs="epsg:4326", resampling_method="nearest"):
    """
    Add time-invariant (stored locally) to the minicube that matches a target raster.
    
    :param list_features: list of features to add using short names.
    :param static_dir: directory where static features are stored.
    :param minicube: 
    :param target_crs: CRS to convert the data to.
    :param resampling_method: GDAL resampling method when changing data resolution. Nearest neighbor by default
    
    :return minicube: list of feature arrays in the order of the list of features
    """
    
    list_features_paths = [os.path.join(static_dir, dict_static_features[f]) for f in list_features]
    
    for i, feat in enumerate(list_features_paths):
        
        # Resampling method depends on file
        tmp_arr = match_raster_to_minicube(feat, minicube, target_crs="epsg:4326", resampling_method="nearest")

        # Create a new xarray data array with the time-invariant array
        tmp_ds = xr.DataArray(tmp_arr,
                              dims=['lat', 'lon'],
                              coords={'lat': minicube['lat'], 'lon': minicube['lon']})
        
        # Reindex the coordinates of ds2 to match the order of ds1
        #tmp_ds = tmp_ds.reindex_like(minicube)
        
        minicube = xr.merge([minicube, tmp_ds.rename(list_features[i])])
        minicube[list_features[i]].attrs = get_attrs_for_band(list_features[i], 'Local data')

    return minicube


def create_mask_in_bbox(shp_path, bbox, target_path, out_path, crs):
    """
    :param shp_path: path to forest shapefile
    :param bbox: (xmin, ymin, xmax, ymax) where the forest mask should be contained.
    :param target_path: path to raster file to use as reference
    :param out_path: path name of generated mask
    :param crs: CRS to use for the created mask. Format is 'EPSG:4326'
    """
    
    # Load the shapefile
    shapefile = gpd.read_file(shp_path)
    
    # Reproject the shapefile to the target CRS
    reprojected_shapefile = shapefile.to_crs(crs)
    
    # Clip the shapefile to the bounding box
    bounding_box = box(*bbox)
    clipped_shapefile = reprojected_shapefile.intersection(bounding_box)
    
    # Filter out empty geometries
    non_empty_geoseries = clipped_shapefile[~clipped_shapefile.is_empty]
    
    # Get metadata of target raster
    source_dataset = rasterio.open(target_path)
    transform = source_dataset.transform
    width = source_dataset.width 
    height = source_dataset.height
    dtype = source_dataset.dtypes[0]
    count = source_dataset.count

    # Rasterize the clipped shapefile
    shapes = ((geom, 1) for geom in non_empty_geoseries.geometry)
    burned = features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform)

    return burned


def create_mask_in_bbox(minicube, shp_path, crs="epsg:4326"):
    """
    :param shp_path: path to forest mask file
    :param minicube: minicube to use as reference
    :param crs: CRS to use for the created mask. Format is 'EPSG:4326'
    """
    
    # Load the shapefile
    shapefile = gpd.read_file(shp_path)
    
    # Reproject the shapefile to the target CRS
    reprojected_shapefile = shapefile.to_crs(crs)
    
    # Get metadata of target raster
    width = len(minicube['lon'])
    height = len(minicube['lat'])

    # Bounds
    minX, maxX = minicube['lon'].min().item(), minicube['lon'].max().item()
    minY, maxY = minicube['lat'].min().item(), minicube['lat'].max().item()
    bbox = (minX, minY, maxX, maxY)
    
    # Transform
    transform = rasterio.transform.from_bounds(minX, minY, maxX, maxY, width, height)
    
    # Clip the shapefile to the bounding box
    bounding_box = box(*bbox)
    clipped_shapefile = reprojected_shapefile.intersection(bounding_box)
    
    # Filter out empty geometries
    non_empty_geoseries = clipped_shapefile[~clipped_shapefile.is_empty]
    
    # Rasterize the clipped shapefile
    shapes = ((geom, 1) for geom in non_empty_geoseries.geometry)
    burned = features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform)

    return burned



def add_mask_to_minicube(static_dir, minicube, target_crs="epsg:4326", resampling_method="nearest"):
    """
    Add time-invariant (stored locally) to the minicube that matches a target raster.
    
    :param list_features: list of features to add using short names.
    :param static_dir: directory where static features are stored.
    :param minicube: 
    :param target_crs: CRS to convert the data to.
    :param resampling_method: GDAL resampling method when changing data resolution.
    
    :return minicube: list of feature arrays in the order of the list of features
    """
    
    shp_path = os.path.join(static_dir, dict_static_features['FOREST_MASK'])
    
    tmp_arr = create_mask_in_bbox(minicube, shp_path, target_crs)

    # Create a new xarray data array with the time-invariant array
    tmp_ds = xr.DataArray(tmp_arr,
                          dims=['lat', 'lon'],
                          coords={'lat': minicube['lat'], 'lon': minicube['lon']})

    minicube = xr.merge([minicube, tmp_ds.rename('FOREST_MASK')])

    return minicube
