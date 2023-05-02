import rasterio
import fiona
import rasterio.mask
import rasterio.fill
import json
import shapely
import numpy as np
import pandas as pd
import os
from pyproj import Proj, transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import Affine
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.geometry.polygon import LinearRing
from itertools import chain
from osgeo import osr
import glob
import argparse
from typing import List
import geopandas as gpd
import shutil

try:
    from osgeo import gdal
except:
    import gdal
    
    
    
def reproject_raster(
    raster_file_in,
    raster_file_out=None,
    dst_crs="EPSG:4326",
    res=None,
    bbox=None,
    width=None,
    height=None,
    resampling=Resampling.nearest,
):

    """
    `it reprojects a raster to a new coordinate system or resolution. Adapted from
    https://rasterio.readthedocs.io/en/latest/topics/reproject.html
    :param raster_file_in: path to input raster file
    :param raster_file_out: path to output raster file
    :param dst_crs: destination crs
    :param res: destination resolution
    :param bbox: bbox of output (left, bottom, right, up)
    :param width: width of output
    :param height: height of output
    :param resampling: resampling method to use (from rasterio.warp.Resampling)
    :return:
    """

    if raster_file_out is None:
        raster_file_out = raster_file_in

    with rasterio.open(raster_file_in) as src:

        bbox = bbox if not bbox is None else src.bounds
        dst_crs = rasterio.crs.CRS.from_string(dst_crs)

        if not res is None:

            transform, width_, height_ = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *bbox, resolution=res
            )

        else:

            transform, width_, height_ = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *bbox
            )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width_ if width is None else width,
                "height": height_ if height is None else height,
            }
        )

        print(kwargs)

        with rasterio.open(raster_file_out, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )


                
# For folder of files
import os

FOLDER_PATH = 'downloads/'
REPROJ_PATH = 'dem/'

for file in os.listdir(FOLDER_PATH):
    print(f'Reprojection file {file}...')
    # Create copy to edit (otherwise original input file will be edited)
    source = file
    target = file.split('.tif')[0] + '_reprojected.tiff'
    shutil.copy(FOLDER_PATH + source, FOLDER_PATH + target)
    raster_src = FOLDER_PATH + target
    raster_reprojected = REPROJ_PATH + file
    reproject_raster(raster_src, raster_reprojected, 'epsg:4326')
    print('Done')