"""
Automatically upload DEM tiles using urls.
URLS retrieved from swisstopo swissalti3D

Author: Selene Ledain
Nov. 21st 2022

How to use:
- provide list of urls
- provide layer id
- provide path to downloaded dem tiles
"""

import os
import time


while True:

    # get other variables
    URLS_PATH = 'swiss_dem_urls/bern.txt'
    DOWNLOADS_PATH = 'dem_reprojected/'
    LAYER_ID = "51726"

    # get latest file --> can read last number from log txt
    f = open("log.txt", "r")
    try:
        FILE_NBR_START = f.readlines()[-1]
    except:
        FILE_NBR_START = 0
    print("Last uploaded file", FILE_NBR_START)

    # check that latest file < max file 
    f = open(URLS_PATH, "r") 
    max_files = 1539 #str(len(f.readlines()))
    #print(max_files)
    

    if int(FILE_NBR_START) >= int(max_files):
        print('Upload complete for this URL PATH. Max files to upload', max_files)
        break

    if int(FILE_NBR_START) < int(max_files):
        # send job
        print("Starting upload")
        starttime = time.time()
        os.system(f'jbsub -name dem -q x86_6h -cores 1x1+1 -mem 16G python upload_dem.py --urls_path {URLS_PATH} --downloads_path {DOWNLOADS_PATH} --layer_id {LAYER_ID} --file_nbr_start {FILE_NBR_START}')

        
        # Kill job after 5 hours
        time.sleep(18000 - ((time.time() - starttime) % 18000)) # 18000s = 5hrs
        os.system('bkill -J dem')

    
    
   
