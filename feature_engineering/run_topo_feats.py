"""
Script to run topographic feature engineering.

Created:    May. 3rd 2023
Authors:    Selene Ledain (selene.ledain@ibm.com)
"""

from create_dem_feat import *

list_paths = ['/data/scratch/selene/static_data/DEM_smooth.tif', '/data/scratch/selene/static_data/DEM_100.tif', '/data/scratch/selene/static_data/DEM_500.tif']
target = '/data/scratch/selene/static_data/DEM_smooth.tif'
feat_list = ['slope', 'aspect', 'rugg', 'curv', 'twi']
out_path = '/data/scratch/selene/static_data/'
list_suffix = ['20', '100', '500']

print('Creating topographic features...')

create_from_multiple_dems(list_paths=list_paths, target=target, feat_list=feat_list, out_path=out_path, list_suffix=list_suffix)

print('Done!')