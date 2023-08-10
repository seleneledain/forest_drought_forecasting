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


"""
BANDS_DESCRIPTION = {
    "CLAY0_5": "Clay content [%] 0-5 cm",
    "CLAY100_200": "Clay content [%] 100-200 cm",
    "CLAY15_30": "Clay content 15-30 [%] cm",
    "CLAY30_60": "Clay content 30-60 [%] cm",
    "CLAY5_15": "Clay content 5-15 [%] cm", 
    "CLAY60_100": "Clay content 60-100 [%] cm",
    "FED0_5": "Fine earth density [g/cm3] 0-5 cm", 
    "FED100_200": "Fine earth density [g/cm3] 100-200 cm",
    "FED15_30": "Fine earth density [g/cm3] 15-30 cm", 
    "FED30_60": "Fine earth density [g/cm3] 30-60 cm", 
    "FED5_15": "Fine earth density [g/cm3] 5-15 cm",
    "FED60_100": "Fine earth density [g/cm3] 60-100 cm",
    "FC": "Forest composition",
    "FH": "Vegetation height in meters",
    "FOREST_MASK" : "Binary forest mask",
    "GRAV0_5": "Gravel content [%] 0-5 cm", 
    "GRAV100_200": "Gravel content [%] 0-5 cm", 
    "GRAV15_30": "Gravel content [%] 15-30 cm", 
    "GRAV30_60": "Gravel content [%] 30-60 cm", 
    "GRAV5_15": "Gravel content [%] 5-15 cm", 
    "GRAV60_100": "Gravel content [%] 60-100 cm", 
    "SAND0_5": "Sand content [%] 0-5 cm",
    "SAND100_200": "Sand content [%] 100-200 cm", 
    "SAND15_30": "Sand content [%] 15-30 cm", 
    "SAND30_60": "Sand content [%] 30-60 cm", 
    "SAND5_15": "Sand content [%] 5-15 cm",   
    "SAND60_100": "Sand content [%] 60-100 cm", 
    "SDEP": "Soil depth [cm]",
    "CARB0_5": "Organic carbon content [g/kg] 0-5 cm",
    "CARB100_200": "Organic carbon content [g/kg] 100-200 cm",
    "CARB15_30": "Organic carbon content [g/kg] 15-30 cm",
    "CARB30_60": "Organic carbon content [g/kg] 30-60 cm",
    "CARB5_15": "Organic carbon content [g/kg] 5-15 cm",
    "CARB60_100": "Organic carbon content [g/kg] 60-100 cm",
    "PH0_5": "pH 0-5 cm",  
    "PH100_200": "pH 100-200 cm", 
    "PH15_30": "pH 15-30 cm",
    "PH30_60": "pH 30-60 cm", 
    "PH5_15": "pH 5-15 cm", 
    "PH60_100": "pH 60-100 cm",
    "NDVI": "Normalized Difference Vegetation Index",
    "DEM": "Digital elevation model",
    "DROUGHT_MASK": "Occurence of drought event in pixel",
}
"""
    
def get_attrs_for_band(band, provider):

        attrs = {}
        attrs["provider"] = provider #"Sentinel 2"
        if provider == 'Sentinel 2':
            attrs["interpolation_type"] = "linear"
        else:
            attrs["interpolation_type"] = "nearest"
        attrs["description"] = BANDS_DESCRIPTION[band]
        
        return
        
    
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





"""
dict_static_features = {
    "CLAY0_5": "ton-final-0_5-rf.tif", 
    "CLAY100_200": "ton-final-100_200-rf.tif",
    "CLAY15_30": "ton-final-15_30-rf.tif", 
    "CLAY30_60": "ton-final-30_60-rf.tif",
    "CLAY5_15": "ton-final-5_15-rf.tif", 
    "CLAY60_100": "ton-final-60_100-rf.tif",
    "DEM": "DEM.tif",
    "DROUGHT_MASK": "polys_brun.tif",
    "FED0_5": "trd-feinerde-final-0_5-rf.tif", 
    "FED100_200": "trd-feinerde-final-100_200-rf.tif", 
    "FED15_30": "trd-feinerde-final-15_30-rf.tif",  
    "FED30_60": "trd-feinerde-final-30_60-rf.tif",  
    "FED5_15": "trd-feinerde-final-5_15-rf.tif",  
    "FED60_100": "trd-feinerde-final-60_100-rf.tif", 
    "FC": "Waldmischungsgrad_2018_10m_2056.tif", 
    "FH": "Vegetationshoehenmodell_2021_10m_LFI_ID164_19_LV95.tif", 
    "FOREST_MASK" : "forest_mask_reproj.tif",
    "GRAV0_5": "skelettgehalt-0_5-rf.tif",  
    "GRAV100_200": "skelettgehalt-100_200-rf.tif",  
    "GRAV15_30": "skelettgehalt-15_30-rf.tif",  
    "GRAV30_60": "skelettgehalt-30_60-rf.tif",  
    "GRAV5_15": "skelettgehalt-5_15-rf.tif", 
    "GRAV60_100": "skelettgehalt-60_100-rf.tif", 
    "SAND0_5": "sand-final-0_5-rf.tif", 
    "SAND100_200": "sand-final-100_200-rf.tif",  
    "SAND15_30": "sand-final-15_30-rf.tif",  
    "SAND30_60": "sand-final-30_60-rf.tif", 
    "SAND5_15": "sand-final-5_15-rf.tif",  
    "SAND60_100": "sand-final-60_100-rf.tif", 
    "SDEP": "mittlere-gruendigkeit-p-rf.tif", 
    "CARB0_5": "corg-gehalt_final-0_5-rf.tif", 
    "CARB100_200": "corg-gehalt_final-100_200-rf.tif",  
    "CARB15_30": "corg-gehalt_final-15_30-rf.tif",  
    "CARB30_60": "corg-gehalt_final-30_60-rf.tif",  
    "CARB5_15": "corg-gehalt_final-5_15-rf.tif",  
    "CARB60_100": "corg-gehalt_final-60_100-rf.tif", 
    "PH0_5": "ph-calc2-0_5-rf.tif",  
    "PH100_200": "ph-calc2-100_200-rf.tif",  
    "PH15_30": "ph-calc2-0_5-rf.tif", 
    "PH30_60": "ph-calc2-30_60-rf.tif", 
    "PH5_15": "ph-calc2-5_15-rf.tif",  
    "PH60_100": "ph-calc2-60_100-rf.tif"   
}
"""


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
