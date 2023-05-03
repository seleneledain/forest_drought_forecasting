"""
Functions for manipulating geospatial data. Edited from Paolo Fraccaro's work.

Created:    Dec. 5th 2022
Authors:    Selene Ledain (selene.ledain@ibm.com)
"""


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
#import folium
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

def match_raster_to_target(input_file, target_file, output_suffix="_padded", resampling_method="bilinear"):

    """
    It matches an input raster to a target raster (e.g. to be used to compare rasters pixel-to-pixel)
    :param input_file: path to input raster file
    :param target_file: path to target input file
    :return: It returns the name of the output file.
    """

    target_meta = get_raster_meta(target_file)

    res = get_raster_resolution(target_file)
    bbox = get_raster_bbox(target_file)
    crs = get_raster_crs(target_file)
    padded_input_file = input_file.replace(".tif", output_suffix + ".tif")

    if os.path.isfile(padded_input_file):
        os.system("rm " + padded_input_file)

    os.system(
        "gdalwarp -t_srs {} -te_srs {} -te {} {} {} {} -ts {} {} -r {} {} {}".format(
            crs,
            crs,
            str(bbox[0]),
            str(bbox[1]),
            str(bbox[2]),
            str(bbox[3]),
            target_meta["width"],
            target_meta["height"],
            resampling_method,
            input_file,
            padded_input_file,
        )
    )

    return padded_input_file

def match_raster_to_target_nocrop(input_file, target_file, output_suffix="_padded", resampling_method="bilinear"):

    """
    It matches an input raster to a target raster (e.g. to be used to compare rasters pixel-to-pixel)
    :param input_file: path to input raster file
    :param target_file: path to target input file
    :return: It returns the name of the output file.
    """

    target_meta = get_raster_meta(target_file)

    res = get_raster_resolution(target_file)
    bbox = get_raster_bbox(input_file)
    crs = get_raster_crs(target_file)
    padded_input_file = input_file.replace(".tif", output_suffix + ".tif")

    if os.path.isfile(padded_input_file):
        os.system("rm " + padded_input_file)

    os.system(
        "gdalwarp -t_srs {} -te_srs {} -te {} {} {} {} -r {} {} {} -tr {} {} -tap".format(
            crs,
            crs,
            str(bbox[0]),
            str(bbox[1]),
            str(bbox[2]),
            str(bbox[3]),
            resampling_method,
            input_file,
            padded_input_file,
            target_meta['transform'][0],
            target_meta['transform'][0]
        )
    )

    return padded_input_file



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

def match_raster_to_target(input_file, target_file, output_suffix="_padded", resampling_method="bilinear"):

    """
    It matches an input raster to a target raster (e.g. to be used to compare rasters pixel-to-pixel)
    :param input_file: path to input raster file
    :param target_file: path to target input file
    :return: It returns the name of the output file.
    """

    target_meta = get_raster_meta(target_file)

    res = get_raster_resolution(target_file)
    bbox = get_raster_bbox(target_file)
    crs = get_raster_crs(target_file)
    padded_input_file = input_file.replace(".tif", output_suffix + ".tif")

    if os.path.isfile(padded_input_file):
        os.system("rm " + padded_input_file)

    os.system(
        "gdalwarp -t_srs {} -te_srs {} -te {} {} {} {} -ts {} {} -r {} {} {}".format(
            crs,
            crs,
            str(bbox[0]),
            str(bbox[1]),
            str(bbox[2]),
            str(bbox[3]),
            target_meta["width"],
            target_meta["height"],
            resampling_method,
            input_file,
            padded_input_file,
        )
    )

    return padded_input_file

def match_raster_to_target_nocrop(input_file, target_file, output_suffix="_padded", resampling_method="bilinear"):

    """
    It matches an input raster to a target raster (e.g. to be used to compare rasters pixel-to-pixel)
    :param input_file: path to input raster file
    :param target_file: path to target input file
    :return: It returns the name of the output file.
    """

    target_meta = get_raster_meta(target_file)

    res = get_raster_resolution(target_file)
    bbox = get_raster_bbox(input_file)
    crs = get_raster_crs(target_file)
    padded_input_file = input_file.replace(".tif", output_suffix + ".tif")

    if os.path.isfile(padded_input_file):
        os.system("rm " + padded_input_file)

    os.system(
        "gdalwarp -t_srs {} -te_srs {} -te {} {} {} {} -r {} {} {} -tr {} {} -tap".format(
            crs,
            crs,
            str(bbox[0]),
            str(bbox[1]),
            str(bbox[2]),
            str(bbox[3]),
            resampling_method,
            input_file,
            padded_input_file,
            target_meta['transform'][0],
            target_meta['transform'][0]
        )
    )

    return padded_input_file