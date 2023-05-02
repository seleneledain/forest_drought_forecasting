"""
This script downloads NASA SRTM Digital Elevation Model from IBM PAIRS.
Data is time invariant. One layer of Sentinel 2 is also
downloaded in order to match resolutions. It is deleted afterwards.

6 Dec. 2022
Selene Ledain
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

date_dem = date(2022,1,1)
iso8601 = "%Y-%m-%dT%H:%M:%SZ"

### Define spatial parameters ### 

regions = {
    "jura" : ["46.907", "7.137", "47.407", "7.637" ]#,
    #"schaffhausen": ["47.586", "8.397", "47.807", "8.756"],
    #"sarraz" : ["46.630", "6.379", "47.700", "6.507"],
    #"wallis" : ["46.000", "7.02", "46.372", "7.435"]
}
bbox = None

regions_larger = {
    "jura" :  ["46.85", "7", "47.45", "7.7" ]
}

# Define the query for DEM --> no time dependence

#Layer IDs
demID = "51726"

lvl_up = "2"
dem_resolutions = [21, 19, 17]


queryDemJson = {
    "layers" : [
        {            
            "type" : "raster", "id" : demID,
            "temporal" : {"intervals" : [{"snapshot" : date_dem.strftime(iso8601)}]} 
        }
    ],
    "spatial" : {"type" : "square",  "coordinates" : bbox },
    "temporal" : {"intervals" : [{"snapshot" : date_dem.strftime(iso8601)}]},
    "processor" : [{
        "order" : 1,
        "type" : "coarse-grain",
        "options" : [
            {"name" : "levelsUp", "value" : lvl_up},
            {"name" : "aggregation", "value" : "bilinear"}
        ]
    }] 
}

root_dir = '/dccstor/cimf/drought_impact/'
# Download DEM data

for bbox, coord in regions.items():
    for lvl in dem_resolutions:
    
        PAIRS_SERVER, PAIRS_CREDENTIALS = authenticate(pairs_key)
        queryDemJson["spatial"]["coordinates"] = coord
        if lvl<21:
            # Download larger area   
            coord  =  regions_larger[bbox]
            queryDemJson["spatial"]["coordinates"] = coord    
            

        try:    
            queryDemJson["processor"][0]["options"][0]["value"] = str(23-lvl) #23 is Sentinel level/Original DEM 
            queryDem = paw.PAIRSQuery(queryDemJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType="api-key", overwriteExisting=False)
            queryDem.submit()

            queryDem.poll_till_finished()
            queryDem.download()
            queryDem.create_layers()

            #Check if the NASA DEM folder exists. If not, create.
            path_Dem = root_dir+f"downloads/DEM_ch/{lvl}/"
            isExistDem = os.path.exists(path_Dem)                          

            if isExistDem == True : 
                pass
            else:
                os.makedirs(path_Dem)

            #Rename File
            old_name = "downloads/" + str(queryDem.zipFilePath)
            new_name = "downloads/"  + str(coord[0]) + "_" + str(coord[1]) + "_" + str(coord[2]) + "_" + str(coord[3]) + ".zip"
            directory_name = path_Dem + str(coord[0]) + "_" + str(coord[1]) + "_" + str(coord[2]) + "_" + str(coord[3])
            os.rename(old_name, new_name)
            
            #extract zip file
            with zipfile.ZipFile(new_name, "r") as zip_ref:
                zip_ref.extractall(directory_name)
            
            #delete the zip file
            os.remove(new_name)


            #remove original files
            for f in os.listdir(directory_name + "/original/"):
                os.remove(directory_name + "/original/" + f)
            os.rmdir(directory_name + "/original/")
            
            
            print("DEM data downloaded")


        except:
            print("Could not download DEM")