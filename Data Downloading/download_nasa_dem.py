"""
This script downloads NASA SRTM Digital Elevation Model from IBM PAIRS.
Data is time invariant. One layer of Sentinel 2 is also
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

date_dem =  datetime(2013, 1, 1, 12)
date_red = date(2017,6,29)
iso8601 = "%Y-%m-%dT%H:%M:%SZ"

### Define spatial parameters ### 

regions = {
    "jura" : ["46.907", "7.137", "47.407", "7.637" ]
}
bbox = None

# Define the query for DEM --> no time dependence

#Layer IDs
demID "49506"
redID = "49360"

queryDemJson = {
    "layers" : [
        {            
            "type" : "raster", "id" : demID,
            "temporal" : {"intervals" : [{"snapshot" : date_dem.strftime(iso8601)}]} 
        },
        {   
            "type" : "raster", "id" : redID,
            "temporal" : {"intervals" : [{"snapshot" : date_red.strftime(iso8601)}]}
            #, "output" : False
        }
    ],
    "spatial" : {"type" : "square",  "coordinates" : bbox },
    "temporal" : {"intervals" : [{"snapshot" : date_dem.strftime(iso8601)}]},
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

# Download DEM data

for bbox, coord in regions.items():
    
    PAIRS_SERVER, PAIRS_CREDENTIALS = authenticate(pairs_key)
    queryDemJson["spatial"]["coordinates"] = coord
    

    try:    
        queryDem = paw.PAIRSQuery(queryDemJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType="api-key", overwriteExisting=False)
        queryDem.submit()

        queryDem.poll_till_finished()
        queryDem.download()
        queryDem.create_layers()

        #Check if the NASA DEM folder exists. If not, create.
        path_Dem = root_dir+"downloads/NASA DEM/" #"downloads/NASA DEM/"
        isExistDem = os.path.exists(path_Dem)                          

        if isExistDem == True : 
            pass
        else:
            os.makedirs(path_Dem)

        #Rename File
        old_name = "downloads/" + str(queryDem.zipFilePath)
        new_name = "downloads/" + str(coord[0]) + "_" + str(coord[1]) + "_" + str(coord[2]) + "_" + str(coord[3]) + ".zip"
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
        
        #remove Sentinel 2 bands from the directory
        for f in glob.glob(directory_name + "/High*.*"):
            os.remove(f)
        
        print("NASA DEM data downloaded")

    except:
        print("Could not download DEM")