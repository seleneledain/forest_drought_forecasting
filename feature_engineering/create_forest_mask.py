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


def create_mask_in_bbox(shp_path, bbox, target_path, out_path, crs):
    """
    :param shp_path: path to forest shapefile
    :param bbox: (xmin, ymin, xmax, ymax) where the forest mask should be contained.
    :param target_path: path to raster file to use as reference
    :param out_path: path name of generated mask
    :param crs: CRS to use for the created mask. Format is 'EPSG:4326'
    """
    
    """
    # Read args
    shp_path = args.shp_path
    bbox = tuple(args.bbox)
    target_path = args.target_path
    out_path = args.out_path
    crs = args.crs
    """
    
    print(f'Generating mask...')
    
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

    print('opening dataset')
    # Create the raster dataset
    with rasterio.open(out_path, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype, crs=crs, transform=transform) as dataset:

        # Rasterize the clipped shapefile
        shapes = ((geom, 1) for geom in non_empty_geoseries.geometry)
        burned = features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform)

        # Write the burned raster to the dataset
        dataset.write_band(1, burned)
        print('Done!')
        

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shp_path', type=str)
    parser.add_argument('--bbox', nargs='+')
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--crs', type=str)

    args = parser.parse_args()

    create_mask_in_bbox(args)
    
# python create_mask_in_bbox --shp_path str --bbox float float float float --target_path str --out_path str --crs str
"""