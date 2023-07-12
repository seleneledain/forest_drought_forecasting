"""
Add features that cannot be obtained online (are stored locally or need to be computed from raw bands)

Author: Selene Ledain
Date: June 9th, 2023
"""
#import sys
#sys.path.append('..') 
from feature_engineering.add_static_data import *


BANDS_DESCRIPTION = {
    "CLAY0_5": "Clay content [%] 0-5 cm",
    "CLAY100_200": "Clay content [%] 100-200 cm",
    "CLAY15_30": "Clay content 15-30 [%] cm",
    "CLAY30_60": "Clay content 30-60 [%] cm",
    "CLAY5_15": "Clay content 5-15 [%] cm", 
    "CLAY60_100": "Clay content 60-100 [%] cm",
    "FED0_5": "Fine earth density [g/cm3] 0-5 cm", 
    "FED100_200": "Fine earth density [g/cm3] 100-200 cm",
    "FED15_30": "Fine earth density [g/cm3] 15-30 cm", 
    "FED30_60": "Fine earth density [g/cm3] 30-60 cm", 
    "FED5_15": "Fine earth density [g/cm3] 5-15 cm",
    "FED60_100": "Fine earth density [g/cm3] 60-100 cm",
    "FC": "Forest composition",
    "FH": "Vegetation height in meters",
    "FOREST_MASK" : "Binary forest mask",
    "GRAV0_5": "Gravel content [%] 0-5 cm", 
    "GRAV100_200": "Gravel content [%] 0-5 cm", 
    "GRAV15_30": "Gravel content [%] 15-30 cm", 
    "GRAV30_60": "Gravel content [%] 30-60 cm", 
    "GRAV5_15": "Gravel content [%] 5-15 cm", 
    "GRAV60_100": "Gravel content [%] 60-100 cm", 
    "SAND0_5": "Sand content [%] 0-5 cm",
    "SAND100_200": "Sand content [%] 100-200 cm", 
    "SAND15_30": "Sand content [%] 15-30 cm", 
    "SAND30_60": "Sand content [%] 30-60 cm", 
    "SAND5_15": "Sand content [%] 5-15 cm",   
    "SAND60_100": "Sand content [%] 60-100 cm", 
    "SDEP": "Soil depth [cm]",
    "CARB0_5": "Organic carbon content [g/kg] 0-5 cm",
    "CARB100_200": "Organic carbon content [g/kg] 100-200 cm",
    "CARB15_30": "Organic carbon content [g/kg] 15-30 cm",
    "CARB30_60": "Organic carbon content [g/kg] 30-60 cm",
    "CARB5_15": "Organic carbon content [g/kg] 5-15 cm",
    "CARB60_100": "Organic carbon content [g/kg] 60-100 cm",
    "PH0_5": "pH 0-5 cm",  
    "PH100_200": "pH 100-200 cm", 
    "PH15_30": "pH 15-30 cm",
    "PH30_60": "pH 30-60 cm", 
    "PH5_15": "pH 5-15 cm", 
    "PH60_100": "pH 60-100 cm",
    "NDVI": "Normalized Difference Vegetation Index",
    "NDWI": "Normalized Difference Water Index"
}

def get_attrs_for_band(band, provider):

        attrs = {}
        attrs["provider"] = provider #"Sentinel 2"
        attrs["interpolation_type"] = "linear" # if forest mask, nearest?
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
    
    to_add = specs_add_bands["bands"]
    if 'NDVI' in to_add:
        to_add.remove('NDVI')
        cube = compute_ndvi(cube)
        cube["s2_ndvi"].attrs = get_attrs_for_band('NDVI', 'Sentinel 2')
    if 'NDVW' in to_add:
        to_add.remove('NDWI')
        cube = compute_ndwi(cube)
        cube["s2_ndwi"].attrs = get_attrs_for_band('NDWI', 'Sentinel 2')
    if 'FOREST_MASK' in to_add:
        to_add.remove('FOREST_MASK')
        cube = add_mask_to_minicube(specs_add_bands["static_dir"], cube)
        cube["FOREST_MASK"].attrs = get_attrs_for_band('FOREST_MASK', 'Local data')
    if len(to_add):
        cube = add_static_to_minicube(to_add, specs_add_bands["static_dir"], cube)
        
    return cube
