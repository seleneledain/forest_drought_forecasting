"""
Script to run topographic feature engineering.

Created:    May. 3rd 2023
Authors:    Selene Ledain (selene.ledain@ibm.com)
"""

from create_dem_feat import *

list_paths = ['/dccstor/cimf/drought_impact/downloads/DEM_ch/17/46.85_7_47.45_7.7/Drought-DEM Switzerland-01_01_2022T00_00_00.tiff']
target = '/dccstor/cimf/drought_impact/downloads/DEM_ch/21/46.907_7.137_47.407_7.637/Drought-DEM Switzerland-01_01_2022T00_00_00.tiff'
feat_list = ['slope', 'aspect', 'rugg', 'curv', 'twi']
out_path = '/dccstor/cimf/drought_impact/downloads/DEM_ch/21/46.907_7.137_47.407_7.637/'

print('Creating topographic features...')

create_from_multiple_dems(list_paths=list_paths, target=target, feat_list=feat_list, out_path=out_path, list_suffix=None)

print('Done!')