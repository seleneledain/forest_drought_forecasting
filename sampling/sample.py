"""
Sample scenes (2.56km^2) according to forest and drought mask

Author: Selene Ledain
Date: July 13th 2023
"""

from osgeo import gdal
import random
import numpy as np

def sample_drought(forest_mask_256, drought_labels_256, thresh_drought, thresh_forest, output_file_path, N):
    """
    Get center coordinates of scenes of 2.56km where % drought > drought_thresh and % forest > thresh_forest
    
    :param forest_mask_256: path to forest mask at 2.56 km resolution
    :param drought_labels_256: path to drought mask at 2.56 km resolution
    :param thresh_drought: threshold for drought 
    :param thresh_forest: threshold for forest 
    :param output_file_path: where output text file should be written
    :param N: number of scenes to sample
    """
    # Open the input raster
    forest_dataset = gdal.Open(forest_mask_256)
    drought_dataset = gdal.Open(drought_labels_256)

    # Get the raster bands
    forest_mask = forest_dataset.GetRasterBand(1)
    drought_mask = drought_dataset.GetRasterBand(1)

    # Read the band values as an array
    forest_array = forest_mask.ReadAsArray()
    drought_array = drought_mask.ReadAsArray()

    # Get the geotransform to calculate the coordinates
    geotransform = forest_dataset.GetGeoTransform()

    # Get the pixel size
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    # Get the upper-left coordinate of the raster
    ulx = geotransform[0]
    uly = geotransform[3]

    # Open the output file for writing
    output_file = open(output_file_path, 'w')

    # Iterate over the array and write the coordinates of the pixels with values greater than 0.4 to the output file
    for row in range(forest_mask.YSize):
        for col in range(forest_mask.XSize):
            value_forest = forest_array[row][col]
            value_drought = drought_array[row][col]
            if value_forest > thresh_forest and value_drought > thresh_drought:
                # Calculate the center coordinates of the pixel
                x = ulx + (col + 0.5) * pixel_width
                y = uly + (row + 0.5) * pixel_height
                output_file.write("{}, {}\n".format(x, y))
                
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
    num_lines = len(lines)
    if num_lines > N:
        # Randomly select N lines from the file
        random_lines = random.sample(lines, N)
        # Write the randomly selected lines to a new file
        with open(output_file, 'w') as f:
            f.writelines(random_lines)
    else:
        ('There are less than N samples')

    # Close the output file
    output_file.close()

    # Close the dataset
    input_raster = None
    
    
def get_coords_forest_drought_neg(forest_array, drought_array, thresh, ulx, uly, pixel_width, pixel_height):
    """
    Get center coordinates of pixels on the mask whose value  >= thresh. 
    There are 2 rasters: forest mask and drought labels
    """
    
    coord_list = []
    # Iterate over the array and write the coordinates of the pixels with values greater than 0.4 to the output file
    for row in range(forest_array.shape[0]):
        for col in range(forest_array.shape[1]):
            value_forest = forest_array[row][col]
            value_drought = drought_array[row][col]
            if value_forest > thresh and value_drought == 0:
                # Calculate the center coordinates of the pixel
                x = ulx + (col + 0.5) * pixel_width
                y = uly + (row + 0.5) * pixel_height
                coord_list.append((x,y))
    
    return coord_list


def sample_negatives(forest_mask_256, drought_labels_256, thresh_forest, output_file, N):
    """
    Get center coordinates of scenes of 2.56km where no drought and % forest > thresh_forest
    
    :param forest_mask_256: path to forest mask at 2.56 km resolution
    :param drought_labels_256: path to drought mask at 2.56 km resolution
    :param thresh_forest: threshold for forest 
    :param N: number of samples to get
    :param output_file: where output text file should be written
    """
    
    with open(output_file, 'w') as file:
        pass

    # Open the raster file
    forest_dataset = gdal.Open(forest_mask_256)
    drought_dataset = gdal.Open(drought_labels_256)

    # Get the total width and height of the raster
    width = forest_dataset.RasterXSize #idk why but they are inverted here
    height = forest_dataset.RasterYSize

    # Calculate the size of 1/6th of the raster
    segment_width = width // 3
    segment_height = height // 2

    # Get the geotransform to calculate the coordinates
    geotransform = forest_dataset.GetGeoTransform()

    # Get the pixel size
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    # Get the upper-left coordinate of the raster
    ulx = geotransform[0]
    uly = geotransform[3]

    # Read and process 1/6th of the raster at a time
    for i in range(3):
        for j in range(2):
            # Calculate the starting position of the current section            
            start_x = i * segment_width
            start_y = j * segment_height
            ulx_new = ulx + i * segment_width * pixel_width
            uly_new = uly + j * segment_height * pixel_height

            # Read the section of the raster
            forest_data = forest_dataset.ReadAsArray(start_x, start_y, segment_width, segment_height)
            drought_data = drought_dataset.ReadAsArray(start_x, start_y, segment_width, segment_height)

            # Get coordinates that satisfy condition
            coord_list = get_coords_forest_drought_neg(forest_data, drought_data, thresh_forest, ulx_new, uly_new, pixel_width, pixel_height)
            # If more than enough, randomly pick N/6
            if len(coord_list)>N/6:
                coord_list = random.sample(coord_list, int(np.ceil(N/6)))

            # Save to text file
            with open(output_file, 'a') as file:
                # Iterate over each element in the list
                for item in coord_list:
                    # Write the element to a new line in the file
                    file.write("{}, {}\n".format(item[0], item[1]))


    # Close the raster dataset
    dataset = None
    
    # Close the output file
    file.close()