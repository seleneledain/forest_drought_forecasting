"""
This script downloads Landcover Copernicus data from IBM PAIRS.
Data is available yearly, download every year available. One layer of Sentinel 2 is also
downloaded in order to match resolutions. It is deleted afterwards.

10 Oct. 2022
Selene Ledain & Didem Durukan
"""


import pandas as pd
from ibmpairs import paw, authentication
import json
import numpy
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import re
import numpy as np
import traceback
import rasterio
import os
from datetime import date
import traceback
import zipfile
import glob

### Connect to PAIRS API ###

logging.basicConfig(level=logging.INFO)
pawLogger = logging.getLogger("ibmpairs.paw")
pawLogger.setLevel(logging.INFO)

with open("ibmpairskey.txt", "r") as f:
    pairs_key = f.read()
    
def authenticate(pairs_key):
    
    pairs_credentials = authentication.OAuth2(api_key=pairs_key)
    auth_header = {"Authorization": f"Bearer {pairs_credentials.jwt_token}"}
    PAIRS_SERVER   = "https://pairs.res.ibm.com"
    PAIRS_CREDENTIALS = authentication.OAuth2(api_key=pairs_key)
    
    return PAIRS_SERVER, PAIRS_CREDENTIALS


### Define temporal parameters ###

landcover_years = [date(2015, 1, 1), date(2016, 1, 1), date(2017, 1, 1), date(2018, 1, 1), date(2019, 1, 1)] #Landcover is only available for these timestamps. 
date_red = date(2016,8,3) #Random date for downloading Red Band from Sentinel. This is used for matching the resolutions between two different layeres. It is deleted afterwards.
date = None
iso8601 = "%Y-%m-%dT%H:%M:%SZ"


### Define spatial parameters ### 

regions = {
    "jura" : ["46.907", "7.137", "47.407", "7.637" ]
}
bbox = None


### Define Query for Landcover Copernicus ###

#Layer IDs
landcoverID = "51353"
redID = "49360"

queryLCJson = {
    "layers" : [
        {            
            "type" : "raster", "id" : landcoverID,
            "temporal" : {"intervals" : [{"snapshot" : date}]} 
        },
        {   
            "type" : "raster", "id" : redID,
            "temporal" : {"intervals" : [{"snapshot" : date_red.strftime(iso8601)}]}
            #, "output" : False
        }
    ],
    "spatial" : {"type" : "square",  "coordinates" : bbox },
    "temporal" : {"intervals" : [{"snapshot" : date }]},
     "processor" : [{
        "order" : 1,
        "type" : "coarse-grain",
        "options" : [
            {"name" : "levelsUp", "value" : "2"},
            {"name" : "aggregation", "value" : "bilinear"}
        ]
    }] 
}

root_dir = '/dccstor/cimf/drought_impact/'


### Download Landcover Copernicus ### 


for bbox, coord in regions.items():
    for year in landcover_years:

        print(f"Trying for date {year}")

        #After every date re-authenticate PAIRS credentials
        PAIRS_SERVER, PAIRS_CREDENTIALS = authenticate(pairs_key)

        date = year.strftime(iso8601)
        queryLCJson["layers"][0]["temporal"]["intervals"][0]["snapshot"] = date
        queryLCJson["temporal"]["intervals"][0]["snapshot"] = date
        queryLCJson["spatial"]["coordinates"] = coord

        
        try:    
            queryLC = paw.PAIRSQuery(queryLCJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType="api-key", overwriteExisting=False)
            queryLC.submit()

            queryLC.poll_till_finished()
            queryLC.download()
            queryLC.create_layers()

            #Check if the LC Copernicus folder exists. If not, create.
            path_LC = root_dir+"downloads/LC Copernicus/"
            isExistLC = os.path.exists(path_LC)                          

            if isExistLC == True : 
                pass
            else:
                os.makedirs(path_LC)

            #Rename File
            old_name = "downloads/" + str(queryLC.zipFilePath)
            new_name = "downloads/" + str(year) + "_" + str(coord[0]) + "_" + str(coord[1]) + "_" + str(coord[2]) + "_" + str(coord[3]) + ".zip"
            directory_name = path_LC + str(year) + "_" + str(coord[0]) + "_" + str(coord[1]) + "_" + str(coord[2]) + "_" + str(coord[3])
            os.rename(old_name, new_name)
            
             #extract zip file
            with zipfile.ZipFile(new_name, "r") as zip_ref:
                zip_ref.extractall(directory_name)
        
            #delete the zip file
                os.remove(new_name)
        
            #remove Sentinel 2 bands from the directory
            for f in glob.glob(directory_name + "/High*.*"):
                os.remove(f)
            
            #remove original files
            for f in os.listdir(directory_name + "/original/"):
                os.remove(directory_name + "/original/" + f)
            os.rmdir(directory_name + "/original/")
            
            print("Landcover data downloaded")
            
        except:
            print("No Landcover data for this year")


    


