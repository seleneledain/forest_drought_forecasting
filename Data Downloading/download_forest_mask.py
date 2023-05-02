"""
Download high-resolution forest mask (single date).

15 Nov. 2022
Selene Ledain
"""

import pandas as pd
from ibmpairs import paw, authentication
import json
import numpy
from datetime import datetime, timedelta, date
import logging
import matplotlib.pyplot as plt
import re
import numpy as np
import traceback
import rasterio
import os
import zipfile
import traceback
import zipfile
import glob


###############################
# IBM PAIRS Authentication
###############################

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


###############################
# Define Parameters
###############################

# Layers
Layers = {
    "forest" : "51716", #continuous forest mask
    "red" : "49360" # red band from Sentinel 2
}


# Define date and space parameters

from datetime import date
date_mask = date(2022, 11, 3)
date_red = date(2017,6,29)
bbox = None
iso8601 = "%Y-%m-%dT%H:%M:%SZ"


# Define areas
areas = {
    "jura" : ["46.907", "7.137", "47.407", "7.637" ]
}


# Define the query 

queryJson = {
    "layers" : [
        {            
            "type" : "raster", "id" : Layers["forest"],
            "temporal" : {"intervals" : [{"snapshot" : date_mask.strftime(iso8601)}]}
        },
        {   
            "type" : "raster", "id" : Layers["red"],
            "temporal" : {"intervals" : [{"snapshot" : date_red.strftime(iso8601)}]}
            #, "output" : False
        }
    ],
    "spatial" : {"type" : "square",  "coordinates" : bbox }, 
    "temporal" : {"intervals" : [{"snapshot" : date_mask.strftime(iso8601)}]},
    "processor" : [{
        "order" : 1,
        "type" : "coarse-grain",
        "options" : [
            {"name" : "levelsUp", "value" : "2"},
            {"name" : "aggregation", "value" : "bilinear"}
        ]
    }] 
}


root_dir = '/dccstor/cimf/drought_impact/ 
        
###############################
# Download data
###############################


for bbox, coord in areas.items():

        
    PAIRS_SERVER, PAIRS_CREDENTIALS = authenticate(pairs_key)
    queryJson["spatial"]["coordinates"] = coord
   
    try:    
        query = paw.PAIRSQuery(queryJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType='api-key', overwriteExisting=False)
        query.submit()

        query.poll_till_finished()
        query.download()
        query.create_layers()

        #Check if the folder exists
        path_mask = root_dir+'downloads/forest_mask/'
        isExistMask = os.path.exists(path_mask)                          

        if isExistMask == True : 
            pass
        else:
            os.makedirs(path_mask)

        #extract zip file
        file_name = 'downloads/' + str(query.zipFilePath)
        directory_name = path_mask
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(directory_name)
        os.remove(file_name)
        
        #remove Sentinel 2 bands from the directory
        for f in glob.glob(directory_name + "/High*.*"):
            os.remove(f)
            
        #remove original files
        for f in os.listdir(directory_name + "/original/"):
            os.remove(directory_name + "/original/" + f)
        os.rmdir(directory_name + "/original/")

        print('Forest mask downloaded')

    except:
        print('Download failed') 
        
