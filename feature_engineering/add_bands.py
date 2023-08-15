"""
Add features that cannot be obtained online (are stored locally or need to be computed from raw bands)

Author: Selene Ledain
Date: June 9th, 2023
"""

from add_static_data import *
from bands_info import BANDS_DESCRIPTION, dict_static_features


def get_attrs_for_band(band, provider):

        attrs = {}
        attrs["provider"] = provider #"Sentinel 2"
        if provider == 'Sentinel 2':
            attrs["interpolation_type"] = "linear"
        else:
            attrs["interpolation_type"] = "nearest"
        attrs["description"] = BANDS_DESCRIPTION[band]


        return attrs
    
    
def compute_ndvi(cube):
    
    if ('s2_B04' not in cube.data_vars) or ('s2_B08' not in cube.data_vars):
        raise Exception("Cannot compute NDVI without B04 and/or B08")
    else:
        red_band = cube['s2_B04']  # Red band
        nir_band = cube['s2_B08']  # Near-infrared band
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        cube['s2_ndvi'] = ndvi
    
    return cube

def compute_ndwi(cube):
    
    if ('s2_B03' not in cube.data_vars) or ('s2_B08' not in cube.data_vars):
        raise Exception("Cannot compute NDVI without B03 and/or B08")
    else:
        green_band = cube['s2_B03']  # Red band
        nir_band = cube['s2_B08']  # Near-infrared band
        ndvi = (nir_band - green_band) / (nir_band + green_band)
        cube['s2_ndwi'] = ndwi
    
    return cube
        
    
def get_additional_bands(specs_add_bands, cube):
    """
    Add static features and bands that are not on Planetary Computer (computed from existing bands).
    :param specs_add_bands: dictionary with list of bands to add, storage location of local bands.
    """
    
    to_add = specs_add_bands["bands"][:]
    if 'NDVI' in to_add:
        to_add.remove('NDVI')
        cube = compute_ndvi(cube)
        cube["s2_ndvi"].attrs = get_attrs_for_band('NDVI', 'Sentinel 2')
    if 'NDVW' in to_add:
        to_add.remove('NDWI')
        cube = compute_ndwi(cube)
        cube["s2_ndwi"].attrs = get_attrs_for_band('NDWI', 'Sentinel 2')
    """
    if 'FOREST_MASK' in to_add:
        to_add.remove('FOREST_MASK')
        cube = add_mask_to_minicube(specs_add_bands["static_dir"], cube)
        cube["FOREST_MASK"].attrs = get_attrs_for_band('FOREST_MASK', 'Local data')
    """
    if len(to_add):
        cube = add_static_to_minicube(to_add, specs_add_bands["static_dir"], cube)
        
    return cube
