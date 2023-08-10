"""
Remove artifacts in DEM and downsample to different resolutions

9 Aug. 2023
Selene Ledain
"""
    
import os
import rasterio
from rasterio.fill import fillnodata
from osgeo import gdal, osr, ogr


res_conversion = {20: 0.00018,
                  100: 0.0009,
                  500: 0.0045}

def resample_raster(dem_path, new_res, outpath):
    
    # Convert reoslution to degrees EPSG:4326
    new_res = res_conversion[new_res]
    
    # Open the input raster
    input_raster = gdal.Open(dem_path)
    
    # Get the original raster's geotransform and dimensions
    original_geotransform = input_raster.GetGeoTransform()
    original_width = input_raster.RasterXSize
    original_height = input_raster.RasterYSize

    # Compute the new dimensions based on the desired resolution
    new_width = int(original_width * original_geotransform[1] / new_res)
    new_height = int(original_height * abs(original_geotransform[5]) / new_res)

    # Create a new raster dataset with the desired resolution
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(
        outpath,
        new_width,
        new_height,
        1,  # number of bands
        gdal.GDT_Float32  # data type of the raster (change if needed)
    )

    # Compute the new geotransform
    new_geotransform = (
        original_geotransform[0],
        new_res,
        original_geotransform[2],
        original_geotransform[3],
        original_geotransform[4],
        -new_res
    )

    # Set the projection and geotransform of the new raster dataset
    output_raster.SetProjection(input_raster.GetProjection())
    output_raster.SetGeoTransform(new_geotransform)
    
    # Set the nodata value for the output raster
    output_raster.GetRasterBand(1).SetNoDataValue(-9999)
    
    # Resample the input raster to match the new resolution
    gdal.ReprojectImage(
        input_raster,
        output_raster,
        input_raster.GetProjection(),
        input_raster.GetProjection(),
        gdal.GRA_Bilinear 
    )

    # Close the datasets
    output_raster = None
    input_raster = None


    return


def smooth_dem(dem_path, smooth_dem_path):
    """Fill missing values in DEM raster

    Args:
        dem_path (str): path to DEM raster
    """

    # Open the input raster for reading
    with rasterio.open(dem_path) as src:
        # Read the raster data
        data = src.read(1)  
        # Update the source profile for the output raster
        profile = src.profile
        
    # Create a mask where nodata values are True
    mask = data != src.nodata # Code fills values indicated with 0

    # Apply fillnodata to fill missing values
    filled_data = fillnodata(data, mask=mask, max_search_distance=10)

    # Save the filled raster to the output path
    with rasterio.open(smooth_dem_path, 'w', **profile) as dst:
        dst.write(filled_data, 1)  # Assuming you're working with a single-band raster

    return



def smooth_and_resample(dem_path, smooth_dem_path, resolutions, output_folder):
    """First deal with missing values in DEM. The resample the DEM to different resolutions

    Args:
        dem_path (str): path to DEM raster
        resolutions (list): list of float values indicating degrees of resolution in meters
    """
    
    # Smooth/fill in missing values in DEM
    smooth_dem(dem_path, smooth_dem_path)
    print('Filled missing values.')
    
    # Resample to different resolutions
    if resolutions is not None:
        for res in resolutions:
            outpath = os.path.join(output_folder, f"DEM_{res}.tif")
            resample_raster(smooth_dem_path, res, outpath)
            print(f'Resampled to {res}.')

    return
    


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dem_path', type=str, default="/data/scratch/selene/static_data/DEM.tif")
    parser.add_argument('--smooth_dem_path', type=str, default="/data/scratch/selene/static_data/DEM_smooth.tif")
    parser.add_argument('--resolutions', type=int, nargs='+') # --resolutions 100 500
    parser.add_argument('--output_folder', type=str, default="/data/scratch/selene/static_data/")

    args = parser.parse_args()
    
    smooth_and_resample(args.dem_path, args.smooth_dem_path, args.resolutions, args.output_folder)
