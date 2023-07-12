"""
Rasterise shapefiles, upsample, reproject and normalise rasters.

Author: Selene Ledain
Date: July 7, 2023
"""

from osgeo import gdal, osr, ogr

def rasterise_shapefile(shapefile_path,output_raster_path, pixel_size):
    """
    Binary rasterisation of a shapefile. CRS of output is same as input shapefile.
    
    :param shapefile_path: path to shapefile
    :param output_raster_path: path to output
    :param pixel_size: Pixel size in the units of the shapefile CRS.
    """
    
    # Open the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()

    # Get the extent and spatial reference of the shapefile
    extent = layer.GetExtent()
    spatial_ref = layer.GetSpatialRef()

    # Create a new raster dataset
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(
        output_raster_path,
        int((extent[1] - extent[0]) / pixel_size),  # width
        int((extent[3] - extent[2]) / pixel_size),  # height
        1,  # number of bands
        gdal.GDT_Byte  # data type of the raster (you can change this if needed)
    )

    # Set the projection and geotransform of the raster dataset
    output_raster.SetProjection(spatial_ref.ExportToWkt())
    output_raster.SetGeoTransform((extent[0], pixel_size, 0, extent[3], 0, -pixel_size))

    # Rasterize the shapefile
    gdal.RasterizeLayer(output_raster, [1], layer, burn_values=[1])

    # Close the datasets
    output_raster = None
    shapefile = None
    
    return


def rasterise_reproject_shapefile(shapefile_path,output_raster_path, pixel_size, target_crs):
    """
    Binary rasterisation of a shapefile. CRS of output raster can be modified.
    
    :param shapefile_path: path to shapefile
    :param output_raster_path: path to output
    :param pixel_size: Pixel size in the units of the shapefile CRS.
    :param target_crs: output raster CRS e.g. 'EPSG:4326'
    """

    # Open the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()

    # Get the extent and spatial reference of the shapefile
    extent = layer.GetExtent()
    spatial_ref = layer.GetSpatialRef()

    # Create a new raster dataset
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(
        output_raster_path,
        int((extent[1] - extent[0]) / pixel_size),  # width
        int((extent[3] - extent[2]) / pixel_size),  # height
        1,  # number of bands
        gdal.GDT_Byte  # data type of the raster (you can change this if needed)
    )

    # Set the projection and geotransform of the raster dataset
    output_raster.SetProjection(spatial_ref.ExportToWkt())
    output_raster.SetGeoTransform((extent[0], pixel_size, 0, extent[3], 0, -pixel_size))

    # Rasterize the shapefile
    gdal.RasterizeLayer(output_raster, [1], layer, burn_values=[1])

    # Reproject the raster to the target CRS using gdal.Warp()
    reprojected_raster = gdal.Warp(output_raster_path, output_raster, dstSRS=target_crs)

    # Close the datasets
    output_raster = None
    shapefile = None
    
    return


def upsample_sum_raster(input_raster_path, output_raster_path, new_pixel_size):
    """
    Upsample a raster, by summing pixels together to obtain value of coarser pixel.
    
    :param input_raster_path: path to raster
    :param output_raster_path: path to output
    :param new_pixel_size: output pixel size in the units of the input raster CRS.
    """

    # Open the input raster
    input_raster = gdal.Open(input_raster_path)

    # Get the original raster's geotransform and dimensions
    original_geotransform = input_raster.GetGeoTransform()
    original_width = input_raster.RasterXSize
    original_height = input_raster.RasterYSize

    # Compute the new dimensions based on the desired resolution
    new_width = int(original_width * original_geotransform[1] / new_pixel_size)
    new_height = int(original_height * abs(original_geotransform[5]) / new_pixel_size)

    # Create a new raster dataset with the desired resolution
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(
        output_raster_path,
        new_width,
        new_height,
        1,  # number of bands
        gdal.GDT_Float32  # data type of the raster (change if needed)
    )

    # Compute the new geotransform
    new_geotransform = (
        original_geotransform[0],
        new_pixel_size,
        original_geotransform[2],
        original_geotransform[3],
        original_geotransform[4],
        -new_pixel_size
    )

    # Set the projection and geotransform of the new raster dataset
    output_raster.SetProjection(input_raster.GetProjection())
    output_raster.SetGeoTransform(new_geotransform)

    # Resample the input raster to match the new resolution
    gdal.ReprojectImage(
        input_raster,
        output_raster,
        input_raster.GetProjection(),
        input_raster.GetProjection(),
        gdal.GRA_Sum
    )

    # Close the datasets
    output_raster = None
    input_raster = None


    return



def upsample_normalise_raster(input_raster_path, output_raster_path, new_pixel_size):
    """
    Upsample a raster, by summing pixels together to obtain value of coarser pixel. Then normalise values.
    
    :param input_raster_path: path to raster
    :param output_raster_path: path to output
    :param new_pixel_size: output pixel size in the units of the input raster CRS.
    """

    # Open the input raster
    input_raster = gdal.Open(input_raster_path)

    # Get the original raster's geotransform and dimensions
    original_geotransform = input_raster.GetGeoTransform()
    original_width = input_raster.RasterXSize
    original_height = input_raster.RasterYSize

    # Compute the new dimensions based on the desired resolution
    new_width = int(original_width * original_geotransform[1] / new_pixel_size)
    new_height = int(original_height * abs(original_geotransform[5]) / new_pixel_size)

    # Create a new raster dataset with the desired resolution
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(
        output_raster_path,
        new_width,
        new_height,
        1,  # number of bands
        gdal.GDT_Float32  # data type of the raster (change if needed)
    )

    # Compute the new geotransform
    new_geotransform = (
        original_geotransform[0],
        new_pixel_size,
        original_geotransform[2],
        original_geotransform[3],
        original_geotransform[4],
        -new_pixel_size
    )

    # Set the projection and geotransform of the new raster dataset
    output_raster.SetProjection(input_raster.GetProjection())
    output_raster.SetGeoTransform(new_geotransform)

    # Resample the input raster to match the new resolution
    gdal.ReprojectImage(
        input_raster,
        output_raster,
        input_raster.GetProjection(),
        input_raster.GetProjection(),
        gdal.GRA_Sum
    )

    
    # Get the upsampled raster band
    output_band = output_raster.GetRasterBand(1)
    # Read the band values as an array
    output_array = output_band.ReadAsArray()

    # Normalize the values of the reprojected array to a range of 0-1
    min_value = output_array.min()
    max_value = output_array.max()
    normalized_array = (output_array - min_value) / (max_value - min_value)
    
    # Write the normalized array to the output raster band
    normalized_band = output_raster.GetRasterBand(1)
    normalized_band.WriteArray(normalized_array)
    
    # Close the datasets
    output_raster = None
    

    return




def reproject_normalise_raster(input_raster_path, output_raster_path, target_crs):
    """
    Reproject and normalise pixels of a raster.
    
    :param input_raster_path: path to raster
    :param output_raster_path: path to output
    :param target_crs: CRS of output raster  e.g. 'EPSG:4326'
    """
    
    # Open the input raster
    input_raster = gdal.Open(input_raster_path)

    # Get the input raster's geotransform and projection
    original_geotransform = input_raster.GetGeoTransform()
    original_projection = input_raster.GetProjection()

    # Reproject the input raster to the target CRS
    reprojected_raster = gdal.Warp(
        output_raster_path,
        input_raster,
        dstSRS=target_crs
    )

    # Get the reprojected raster band
    reprojected_band = reprojected_raster.GetRasterBand(1)

    # Read the band values as an array
    reprojected_array = reprojected_band.ReadAsArray()

    # Normalize the values of the reprojected array to a range of 0-1
    min_value = reprojected_array.min()
    max_value = reprojected_array.max()
    normalized_array = (reprojected_array - min_value) / (max_value - min_value)

    # Create a new output raster with the normalized values
    driver = gdal.GetDriverByName('GTiff')
    normalized_raster = driver.Create(
        output_raster_path,
        reprojected_raster.RasterXSize,
        reprojected_raster.RasterYSize,
        1,  # number of bands
        gdal.GDT_Float32  # data type of the raster (change if needed)
    )

    # Set the geotransform and projection of the reprojected raster
    normalized_raster.SetGeoTransform(reprojected_raster.GetGeoTransform())
    normalized_raster.SetProjection(reprojected_raster.GetProjection())

    # Write the normalized array to the output raster band
    normalized_band = normalized_raster.GetRasterBand(1)
    normalized_band.WriteArray(normalized_array)

    # Close the datasets
    reprojected_raster = None
    normalized_raster = None

    return 


def upsample_reproject_normalise_raster(input_raster_path, output_raster_path, target_crs, new_pixel_size_x, new_pixel_size_y):
    """
    Upsample (by sum), reproject and normalise a raster.
    
    :param input_raster_path: path to raster
    :param output_raster_path: path to output
    :param target_crs: CRS of output raster  e.g. 'EPSG:4326'
    :param new_pixel_size_x:  output pixel size x in the units of the target CRS.
    :param new_pixel_size_y: output pixel size y in the units of the target CRS.
    """
    
    # Open the input raster
    input_raster = gdal.Open(input_raster_path)

    # 1. Reproject
    # Create a new reprojected raster to the target CRS
    reprojected_raster = gdal.Warp(
        output_raster_path,
        input_raster,
        dstSRS=target_crs
    )

    # 2. Upsample

    # Get the original raster's geotransform and dimensions
    original_geotransform = reprojected_raster.GetGeoTransform()
    original_width = reprojected_raster.RasterXSize
    original_height = reprojected_raster.RasterYSize

    # Compute the new dimensions based on the desired resolution
    new_width = int(original_width * original_geotransform[1] / new_pixel_size_x)
    new_height = int(original_height * abs(original_geotransform[5]) / new_pixel_size_y)

    # Create a new raster dataset with the desired resolution
    driver = gdal.GetDriverByName('GTiff')
    upsampled_raster = driver.Create(
        output_raster_path,
        new_width,
        new_height,
        1,  # number of bands
        gdal.GDT_Float32  # data type of the raster (change if needed)
    )

    # Compute the new geotransform
    new_geotransform = (
        original_geotransform[0],
        new_pixel_size_x,
        original_geotransform[2],
        original_geotransform[3],
        original_geotransform[4],
        -new_pixel_size_y
    )

    # Set the projection and geotransform of the new raster dataset
    upsampled_raster.SetProjection(reprojected_raster.GetProjection())
    upsampled_raster.SetGeoTransform(new_geotransform)

    # Resample the input raster to match the new resolution
    gdal.ReprojectImage(
        reprojected_raster,
        upsampled_raster,
        reprojected_raster.GetProjection(),
        reprojected_raster.GetProjection(),
        gdal.GRA_Sum
    )



    # 3. Normalise
    # Get the upsampled raster band
    upsampled_band = upsampled_raster.GetRasterBand(1)

    # Read the band values as an array
    upsampled_array = upsampled_band.ReadAsArray()

    # Normalize the values of the reprojected array to a range of 0-1
    min_value = upsampled_array.min()
    max_value = upsampled_array.max()
    normalized_array = (upsampled_array - min_value) / (max_value - min_value)

    # Create a new output raster with the normalized values
    driver = gdal.GetDriverByName('GTiff')
    normalized_raster = driver.Create(
        output_raster_path,
        upsampled_raster.RasterXSize,
        upsampled_raster.RasterYSize,
        1,  # number of bands
        gdal.GDT_Float32  # data type of the raster (change if needed)
    )

    # Set the geotransform and projection of the reprojected raster
    normalized_raster.SetGeoTransform(upsampled_raster.GetGeoTransform())
    normalized_raster.SetProjection(upsampled_raster.GetProjection())

    # Write the normalized array to the output raster band
    normalized_band = normalized_raster.GetRasterBand(1)
    normalized_band.WriteArray(normalized_array)

    # Close the datasets
    reprojected_raster = None
    normalized_raster = None
    
    return




def rasterise_kml_to_existing_raster(input_kml_file, output_raster_file, forest_mask):
    """
    Rasterise polygons in shapefile, creating a raster with same extent and pixel size as forest mask raster.
    Assumes that CRS of input_kml_file and output_raster_file are the same.
    
    :param input_kml_file: path to KML file with polygons to rasterise
    :param output_raster_file: path to output raster
    :param forest_mask: path to exsiting raster to use as reference
    """
    
    gdal.UseExceptions()
    ogr.RegisterAll()

    kml_driver = ogr.GetDriverByName("KML")
    kml_dataset = kml_driver.Open(input_kml_file, 0)
    if kml_dataset is None:
        print("Failed to open the KML file.")
        exit(1)

    layer = kml_dataset.GetLayer()

    # Set raster extent to that of forest mask
    existing_raster_dataset = gdal.Open(forest_mask)
    existing_transform = existing_raster_dataset.GetGeoTransform()
    extent = (
        existing_transform[0],
        existing_transform[0] + existing_transform[1] * existing_raster_dataset.RasterXSize,
        existing_transform[3] + existing_transform[5] * existing_raster_dataset.RasterYSize,
        existing_transform[3]
    )

    # Set pixel resoltuion to that of forest mask
    pixel_size_x = existing_transform[1]
    pixel_size_y = existing_transform[5]
    width = existing_raster_dataset.RasterXSize
    height = existing_raster_dataset.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_raster_file, width, height, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(layer.GetSpatialRef().ExportToWkt())
    output_dataset.SetGeoTransform(existing_transform) 
    band = output_dataset.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()

    gdal.RasterizeLayer(output_dataset, [1], layer, burn_values=[1])

    output_dataset = None
    kml_dataset = None