"""
Download Sentinel2 and ERA5 data from IBM PAIRS
Test for available dates for Sentinel2. 
Download daily aggregates of ERA5 data.

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
import zipfile
import traceback
import zipfile
import glob
import shutil


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

match_dates = True # True: download ERA5 for dates on Sen2. False: download all possible ERA5 dates


# ERA Layers
eraLayers = {
        "Red" : "49360",
        "MinT" : "49429"
    }


vpd_layers = [
            {
                # Min T convert kelvin to celcius
                "alias": "tminc",
                "expression": "$MinT - 273.15",
                "output" : "false"
            },
            {            
                # Max Rel humidity
                "type" : "raster", "id" : "48760",
                "aggregation" : "Max",
                "alias": "relativehummax",
                "output" : "false"
            },
            {
                # calculate saturation water vapour pressure in Pa at Tmax
                "alias": "esatmin",
                "expression": "611*math:exp((17.27 * $tminc)/($tminc + 237.3))",
                "output" : "false"
            },
            {
                # Calculate vpd min
                "alias": "vpdmin",
                "expression": "$esatmin * (1 - $relativehummax)",
                "output" : "false"
            },
            {
                # Calculate vpd min
                "alias": "vpd_min",
                "expression": "math:min(0, $vpdmin)"
            }
    ]




# Define date and space parameters
date_start, date_end = None, None #Needed for defining the query for ERA5
date = None #Needed for defining the query for Sentinel 2
bbox = None


# Define areas
areas = {
    "jura" :["46.907", "7.137", "47.407", "7.637" ]
}


# Define the query for ERA5
queryEraJson = {
    "layers" : [
        {            
            "type" : "raster", "id" : eraLayers[lKey], "alias": lKey,
            "aggregation" : None
        }
        for lKey in eraLayers    
    ] + vpd_layers,
    "spatial" : {"type" : "square",  "coordinates" : bbox }, 
    "temporal" : {"intervals" : [
    {"start" : date_start, "end" : date_end}
]},
    "processor" : [{
        "order" : 1,
        "type" : "coarse-grain",
        "options" : [
            {"name" : "levelsUp", "value" : "2"},
            {"name" : "aggregation", "value" : "near"}
        ]
    }]
}

# Define   aggregation type for each ERA 5 layer
agg_mean = ["49360", "49439", "49454", "49423", "49458"] #Red, Surface pressure, Total cloud cover, Temperature, AWVC
agg_sum = ["49459", "49440"] # Total precipitation, Solar radiation
agg_max = ["49430"] # Max temperature
agg_min = ["49429"] # Min temperature


for idx, layer in enumerate(queryEraJson["layers"]):
    if "id" in layer:
        if layer["id"] in agg_mean:
            queryEraJson["layers"][idx]["aggregation"] = 'Mean' #'Median' doesnt exist
        if layer["id"] in agg_sum:
            queryEraJson["layers"][idx]["aggregation"] = 'Sum'
        if layer["id"] in agg_max:
            queryEraJson["layers"][idx]["aggregation"] = 'Max'
        if layer["id"] in agg_min:
            queryEraJson["layers"][idx]["aggregation"] = 'Min'
    else:
        continue

        
        
###############################
# Download data
###############################


# Function that sets the date range
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
from datetime import date
# Define start and end date
start_date = date(2017, 6, 29)
end_date = date(2021, 12, 31)

iso8601 = "%Y-%m-%dT%H:%M:%SZ"

root_dir = '/dccstor/cimf/drought_impact/'

if match_dates:

    # Authenticate before every query because of the time-out problem
    for bbox, coord in areas.items():

        for single_date in daterange(start_date, end_date):
            PAIRS_SERVER, PAIRS_CREDENTIALS = authenticate(pairs_key)

            # Check if file with that date exists
            date = single_date.strftime(iso8601)
            path_date = root_dir+'downloads/ERA5/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3])
            if os.path.exists(path_date):
                print(date)
                # Download and just add vpd min to folder
                
                date_start = single_date.strftime(iso8601)
                date_end = (single_date + timedelta(1)).strftime(iso8601)
                queryEraJson["temporal"]["intervals"][0]["start"] = date_start
                queryEraJson["temporal"]["intervals"][0]["end"] = date_end
                queryEraJson["spatial"]["coordinates"] = coord
                
                queryEra = paw.PAIRSQuery(queryEraJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType='api-key', overwriteExisting=False)
                queryEra.submit()

                queryEra.poll_till_finished()
                queryEra.download()
                queryEra.create_layers()

                ##Rename zip file
                old_name = 'downloads/' + str(queryEra.zipFilePath)
                new_name = 'downloads/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]) + '.zip'
                os.rename(old_name, new_name)

                # Extract zip file
                directory_name = root_dir+'downloads/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3])
                with zipfile.ZipFile(new_name, 'r') as zip_ref:
                    zip_ref.extractall(directory_name)

                # Delete the zip file
                os.remove(new_name)

                # Move vpd min file 
                vpd_file = list(glob.glob(directory_name + '/*vpd_min*.tiff*'))
                for f in vpd_file:
                    shutil.move(f, path_date+'/'+f.split('/')[-1])

                # Delete the original folder 
                for f in os.listdir(directory_name + "/original/"):
                    os.remove(directory_name + "/original/" + f)
                os.rmdir(directory_name + "/original/")
                for f in os.listdir(directory_name ):
                    os.remove(directory_name + '/' + f)
                os.rmdir(directory_name)

                print('ERA-5 vpd data downloaded')
