"""
Add features that cannot be obtained online (are stored locally or need to be computed from raw bands)

Author: Selene Ledain
Date: June 9th, 2023
"""

from add_static_data import *

def compute_ndvi(cube):
    
    if ('s2_B04' not in cube.data_vars) or ('s2_B08' not in cube.data_vars):
        raise Exception("Cannot compute NDVI without B04 and/or B08")
    else:
        red_band = cube['s2_B04']  # Red band
        nir_band = cube['s2_B08']  # Near-infrared band
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        cube['s2_ndvi'] = ndvi
    
    return cube
        
    
def get_additional_bands(specs_add_bands, cube):
    """
    Add static features and bands that are not on Planetary Computer (computed from existing bands).
    :param specs_add_bands: dictionary with list of bands to add, storage location of local bands.
    """
    
    to_add = specs_add_bands["bands"]
    if 'NDVI' in to_add:
        to_add.remove('NDVI')
        cube = compute_ndvi(cube)
    if 'FOREST_MASK' in to_add:
        to_add.remove('FOREST_MASK')
        cube = add_mask_to_minicube(specs_add_bands["static_dir"], cube)
    if len(to_add):
        cube = add_static_to_minicube(to_add, specs_add_bands["static_dir"], cube)
        
    return cube
