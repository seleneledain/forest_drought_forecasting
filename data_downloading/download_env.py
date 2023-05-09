"""
This script downloads soil and forest data layers from IBM PAIRS.
Data is time invariant. One layer of Sentinel 2 is also
downloaded in order to match resolutions. It is deleted afterwards.

1 Dec. 2022
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

date_soil = date(2022,8,19)
date_comp = date(2018,1,1)
date_height = date(2022,1,1)
date_red = date(2017,6,29) # for making resolutions match
iso8601 = "%Y-%m-%dT%H:%M:%SZ"

### Define spatial parameters ### 

regions = {
    "jura" : ["46.907", "7.137", "47.407", "7.637" ],
    "schaffhausen": ["47.586", "8.397", "47.807", "8.756"],
    "sarraz" : ["46.630", "6.379", "47.700", "6.507"],
    "wallis" : ["46.000", "7.02", "46.372", "7.435"]
}
bbox = None

# Define the query for DEM --> no time dependence

#Layer IDs
redID = "49360"
comp_ID = "51702"
height_ID = "51702"

# Environmental data layers
soilLayers = {
    "oc0_5" : "51693", #organic carbon
    "oc5_15" : "51694", 
    "oc15_30" : "51695",
    "oc30_60" : "51696", 
    "oc60_100" : "51697", 
    "oc100_200" : "51698", 
    "ph0_5" : "51658", #pH
    "ph5_15" : "51659", 
    "ph15_30" : "51660", 
    "ph30_60" : "51661", 
    "ph60_100" : "51662", 
    "ph100_200" : "51663", 
    "sand0_5" : "51664", #sand
    "sand5_15" : "51665", 
    "sand15_30" : "51666", 
    "sand30_60" : "51667", 
    "sand60_100" : "51668", 
    "sand100_200" : "51669", 
    "grav0_5" : "51670", #gravel
    "grav5_15" : "51671", 
    "grav15_30" : "51672", 
    "grav30_60" : "51673", 
    "grav60_100" : "51674", 
    "grav100_200" : "51675", 
    "clay0_5" : "51676", #clay
    "clay5_15" : "51677", 
    "clay15_30" : "51682", 
    "clay30_60" : "51683", 
    "clay60_100" : "51684", 
    "clay100_200" : "51685", 
    "dens0_5" : "51686", #fine earth density
    "dens5_15" : "51687", 
    "dens15_30" : "51688", 
    "dens30_60" : "51689", 
    "dens60_100" : "51690", 
    "dens100_200" : "51691", 
    "soil_depth" : "51692" #soil depth
}

    


queryEnvJson = {
    "layers" : [
        {
            
            "type" : "raster", "id" : soilLayers[lKey], "alias": lKey,
            "temporal" : {"intervals" : [{"snapshot" : date_soil.strftime(iso8601)}]} 
        }
        for lKey in soilLayers
    ] +
    [
        {
            
            "type" : "raster", "id" : comp_ID, "alias": "forest_comp",
            "temporal" : {"intervals" : [{"snapshot" : date_comp.strftime(iso8601)}]} 
        },
        {
            
            "type" : "raster", "id" : height_ID, "alias": "forest_height",
            "temporal" : {"intervals" : [{"snapshot" : date_height.strftime(iso8601)}]} 
        },
        {   
            "type" : "raster", "id" : redID,
            "temporal" : {"intervals" : [{"snapshot" : date_red.strftime(iso8601)}]}
        }
    ] ,
    "spatial" : {"type" : "square",  "coordinates" : bbox },
    "temporal" : {"intervals" : [{"snapshot" : date_red.strftime(iso8601)}]},
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
    queryEnvJson["spatial"]["coordinates"] = coord
    

    try:    
        queryEnv = paw.PAIRSQuery(queryEnvJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType="api-key", overwriteExisting=False)
        queryEnv.submit()

        queryEnv.poll_till_finished()
        queryEnv.download()
        queryEnv.create_layers()

        #Check if the NASA DEM folder exists. If not, create.
        path_env = root_dir+"downloads/ENV_DATA/"
        isExistEnv = os.path.exists(path_env)                          

        if isExistEnv == True : 
            pass
        else:
            os.makedirs(path_env)

        #Rename File
        old_name = "downloads/" + str(queryEnv.zipFilePath)
        new_name = "downloads/" + str(coord[0]) + "_" + str(coord[1]) + "_" + str(coord[2]) + "_" + str(coord[3]) + ".zip"
        directory_name = path_env  + str(coord[0]) + "_" + str(coord[1]) + "_" + str(coord[2]) + "_" + str(coord[3])
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
 
        print("Environmental data downloaded")


    except:
        print("Could not download data")