"""
Create forest mask from forest shapefile in a certain bounding box. 
Match resolution and pixel alignment with another raster.

Author: Selene Ledain
Date: May 17th, 2023
"""

import rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box
import geopandas as gpd
import xarray as xr
import argparse


def create_mask_in_bbox(minicube, static_dir, shp_path, crs="epsg:4326"):
    """
    :param shp_path: path to forest shapefile
    :param bbox: (xmin, ymin, xmax, ymax) where the forest mask should be contained.
    :param target_path: path to raster file to use as reference
    :param out_path: path name of generated mask
    :param crs: CRS to use for the created mask. Format is 'EPSG:4326'
    """
    
    # Load the shapefile
    shapefile = gpd.read_file(static_dir+shp_path)
    
    # Reproject the shapefile to the target CRS
    reprojected_shapefile = shapefile.to_crs(crs)
    
    # Get metadata of target raster
    """
    source_dataset = rasterio.open(target_path)
    transform = source_dataset.transform
    width = source_dataset.width 
    height = source_dataset.height
    """
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

    # Create the raster dataset
    #with rasterio.open(out_path, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype, crs=crs, transform=transform) as dataset:

        
        
        # Write the burned raster to the dataset
        #dataset.write_band(1, burned)
        #print('Done!')
        
        #return burned
        
        
