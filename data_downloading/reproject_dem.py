import rasterio
#import fiona
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
    
    
    
def reproject_resample_raster(
    raster_file_in,
    raster_file_out=None,
    dst_crs="EPSG:4326",
    res=None,
    bbox=None,
    width=None,
    height=None,
    resampling=Resampling.bilinear,
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



def reproject_dem(folder_path, reproj_path):
    """Reproject DEM tiles and change resolution to 20m

    Args:
        folder_path (str): path to DEM tiles
        reproj_path (str): path to store reprojected tiles
    """
    

    for file in os.listdir(folder_path):
        print(f'Reprojection file {file}...')
        # Create copy to edit (otherwise original input file will be edited)
        source = file
        target = file.split('.tif')[0] + '_reprojected.tiff'
        shutil.copy(folder_path + source, reproj_path + target)
        raster_src = reproj_path + target
        raster_reprojected = reproj_path + file
        reproject_resample_raster(raster_src, raster_reprojected, 'epsg:4326', res=20)
        print('Done')


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='/data/scratch/selene/dem/')
    parser.add_argument('--reproj_path', type=str, default='/data/scratch/selene/dem_reproj_resamp/')

    args = parser.parse_args()
    
    reproject_dem(args.folder, args.n_sub, args.output_folder)
