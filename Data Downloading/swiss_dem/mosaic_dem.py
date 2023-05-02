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
    
    

    
# modified from https://github.com/aerospaceresearch/DirectDemod/blob/Vladyslav_dev/directdemod/merger.py
def add_pixel_fn(filename: str, resample_name: str, output_type='Float32', nodata=-9999.0) -> None:
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        resample_name (:obj:`string`): name of resampling method
    """

    header = f"""  <VRTRasterBand dataType="{output_type}" band="1" subClass="VRTDerivedRasterBand">"""
    contents = """
    <PixelFunctionType>{0}</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionArguments {2}/>
    <PixelFunctionCode><![CDATA[{1}]]>
    </PixelFunctionCode>"""

    lines = open(filename, 'r').readlines()
    lines[3] = header  # FIX ME: 3 is a hand constant
    lines.insert(4, contents.format(resample_name,
                                    get_resample(resample_name),
                                    f'nodata="{nodata}"'))
    open(filename, 'w').write("".join(lines))

def get_resample(name: str) -> str:
    """retrieves code for resampling method
    Args:
        name (:obj:`string`): name of resampling method
    Returns:
        method :obj:`string`: code of resample method
    """

    methods = {
        "first":
        """
import numpy as np
import pandas as pd
def first(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    df = pd.DataFrame(np.argwhere(~np.isnan(in_ar)), columns=['Tile', 'Row', 'Column'])
    indexes_output = df.groupby(['Row', 'Column']).min().reset_index()[['Tile', 'Row', 'Column']]
    tiles = indexes_output['Tile'].unique()
    for tile in tiles:
        tile_indexes = indexes_output[indexes_output['Tile'].values == tile][['Row', 'Column']]
        out_ar[tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)] = in_ar[tile][tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)]
""",
        "last":
        """
import numpy as np
import pandas as pd
def last(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    df = pd.DataFrame(np.argwhere(~np.isnan(in_ar)), columns=['Tile', 'Row', 'Column'])
    indexes_output = df.groupby(['Row', 'Column']).max().reset_index()[['Tile', 'Row', 'Column']]
    tiles = indexes_output['Tile'].unique()
    for tile in tiles:
        tile_indexes = indexes_output[indexes_output['Tile'].values == tile][['Row', 'Column']]
        out_ar[tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)] = in_ar[tile][tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)]
""",
        "max":
        """
import numpy as np
def max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmax(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
""",
        "min":
        """
import numpy as np
def min(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmin(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
""",
        "median":
        """
import numpy as np
import pickle
def median(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmedian(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
""",
        "average":
        """
import numpy as np
import pickle
def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmean(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
"""}

    if name not in methods:
        raise ValueError(
            "ERROR: Unrecognized resampling method (see documentation): '{}'.".
            format(name))

    return methods[name]


def get_raster_meta(raster_file):

    """
    It extracts the raster's metadata
    :param raster_file: path to raster file
    :return: raster metadata
    """

    with rasterio.open(raster_file) as src:

        out_meta = src.meta

    return out_meta

    
def create_mosaic(output_file_prefix: str,
                  tile_prefix: str, 
                  in_dir: str,
                  out_dir: str=None,
                  method="average",
                  delete_tiles=False,
                  output_type='Float32',
                  nodata=-9999) -> str:
    """
    It creates a mosaic of raster files using gdalbuildvrt and gdal_translate. Overlap is taken car of by using the mean.
    :param output_file_prefix: prefix of the files to be tiled. If '' every .tif file is merged together. This is also used to
            save the final result
    :param tile_prefix: prefix of the tile files. If '' every .tif file is merged together
    :param in_dir: input dir (with '/' at the end)
    :param out_dir: output dir (without '/' at the end)
    :param method: gdalbuildvrt method to merge
    :param delete_tiles: whether to delete the tiles after having merged or not
    :param output_type: dtype of output, allowed types are UInt8, UInt16, Int16, UInt32, Int32, Float32, Float64
    :param output_type: no data value to use
    :return: path to output file
    """
    
    # get all files to merge
    files = glob.glob(f'{in_dir}{output_file_prefix}*{tile_prefix}*tif*')
    
    
    if len(files) > 0:
        
        print(f"Files to merge: {str(len(files))}")
        
        # get tiles type (it assumes all of the same type)
        tile_meta = get_raster_meta(files[0])
        input_dtype = tile_meta['dtype']

        # if type different than required one
        if output_type.lower() != input_dtype:

            print(f'Converting tiles dtype from {input_dtype} to output type {output_type}')
            for file in files:

                change_raster_dtype(file, output_type.lower())

        # create file names and vrt dataset
        vrt_tmp_file = f'{in_dir}tmp.vrt'
        os.system(
                f"gdalbuildvrt -vrtnodata {nodata} {vrt_tmp_file} {in_dir}{output_file_prefix}*{tile_prefix}*.tif*"
            )

        output_file = f'{in_dir}{output_file_prefix}.tif' if output_file_prefix != '' else f'{in_dir}output.tif'

        # add required pixel function
        add_pixel_fn(vrt_tmp_file, method, output_type, nodata)

        # compute mosaic to GeoTiff
        os.environ['GDAL_VRT_ENABLE_PYTHON'] = 'Yes'
        os.system('gdal_translate -of GTiff -co "TILED=YES" {} {}'.format(vrt_tmp_file, output_file))
        os.environ['GDAL_VRT_ENABLE_PYTHON'] = 'None'

        # delete vrt dataset
        if os.path.isfile(vrt_tmp_file):
            os.remove(vrt_tmp_file)

        if delete_tiles:

            os.system("rm {}{}*{}*.tif*".format(in_dir, output_file_prefix, tile_prefix))

        else:

            # if want to keep tiles check whether need to change dtype back to original one
            if output_type.lower() != input_dtype:

                print(f'Converting tiles dtype back to original type {input_dtype}')
                for file in files:

                    change_raster_dtype(file, input_dtype)

        # move file if needed
        if out_dir is not None:

            output_file = output_file.replace(in_dir, out_dir)

            os.system("mv {}{}.tif {}".format(in_dir, output_file_prefix, out_dir))
    
    else:
        
        print("No files to merge")
        output_file=""
        
    return output_file




################
# Iterative process
###############

# Count number of files
FILE_PATH = 'dem/'
_, _, files = next(os.walk(FILE_PATH))
file_count = len(files)
n_files = 10

nodata=-9999
method="average"
delete_tiles=False
output_type='Float32'

# Iterate over batches of files
for batch in np.arange(0, file_count, n_files):
    
    # Get batch files
    files_batch = os.listdir(FILE_PATH)[batch:batch+n_files]
    files_to_merge = [file for file in files_batch if '.tif' in file] # Keep only tif files
    
    if os.path.exists(FILE_PATH + "paths.txt"):
        os.remove(FILE_PATH + "paths.txt")
        
    with open(FILE_PATH + r"paths.txt", "w") as f:
        for file in files_to_merge:
            f.write("%s\n" % (FILE_PATH + file))
    
    # Create mosiac from list of files in txt file
    text_file_path = FILE_PATH + "paths.txt"
    vrt_tmp_file = f'{FILE_PATH}merged/tmp.vrt'
    os.system(
        f"gdalbuildvrt -input_file_list {text_file_path} {vrt_tmp_file}"
    )
    
    output_file = f'{FILE_PATH}merged/output.tif'

    # add required pixel function
    add_pixel_fn(vrt_tmp_file, method, output_type, nodata)
    
    import os, psutil
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)  # in bytes 

    # compute mosaic to GeoTiff
    os.environ['GDAL_VRT_ENABLE_PYTHON'] = 'Yes'
    os.system('gdal_translate -of GTiff -co "TILED=YES" {} {}'.format(vrt_tmp_file, output_file))
    os.environ['GDAL_VRT_ENABLE_PYTHON'] = 'None'

    # delete vrt dataset
    if os.path.isfile(vrt_tmp_file):
        os.remove(vrt_tmp_file)
        
    # Get bounding box and include it in output file name
    src = rasterio.open(FILE_PATH + "merged/output.tif")
    bbox = list(src.bounds)
    old_name = FILE_PATH + "merged/output.tif"
    new_name = FILE_PATH + f"merged/dem_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.tif"
    os.rename(old_name, new_name)
    os.remove(text_file_path)
    
    break

