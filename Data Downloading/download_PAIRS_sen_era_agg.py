"""
Download Sentinel2 and ERA5 data from IBM PAIRS
Test for available dates for Sentinel2. 
Download daily aggregates of ERA5 data.

18 Nov. 2022
Selene Ledain & Didem Durukan

params:
:match_dates: bool
    If you want to have ERA5 aggregates downloaded for dates of Sentinel 2 (match_dates=True)
    Leave delta=None
    If you want to have regular ERA5 aggregates downloaded regardless of Sentinel 2 (match_dates=False)
    Input delta. Careful, if you have several perioicities, download only in intervals (specfiy in start and end dates) with same periodicity and insert respective delta
:senLayers: json
    Sentinel-2 layer names and IDs to download
:index_layers: json
    Layers to compute from Sentinel 2 layers
:eraLayers: json
    ERA5 names and IDs to download
:vpd_layers: json
    computation of vapor pressure deficit from ERA5 layers
:areas: json
    bounding box with coordinate of regions to download
:date_start: datetime
    first date for downloads
:end_date: datetime
    last date for downloads
:delta: int
    number of days over which to aggregate ERA5 data (if using match_dates=False)
    make sure that start_date + delta will match your dsired timestamp
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
import argparse


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
def download_data(start_date):

    match_dates = False # True: download ERA5 aggregates only when Sen2 available (using last Sen2 available date to start ERA aggregation). False: download ERA5 aggregate at regular dates, even when Sen2 not availble

    # Sentinel 2 Layers
    senLayers = {
        "B04" : "49360", #red
        "B03" : "49681", #green
        "B02" : "49680", #blue
        "B08": "49361", #nir
        "NDVI" : "49464",
        "SCL": "49362",
        "B8a": "49685", #Narrow IR
        "B12": "49687", # SWIR 2200 nm
        "B11" : "49686", # SWIR 1610 nm
        "B05" : "49682", # vegetation red edge
        "B06" : "49683", # vegetation red edge
        "B07" : "49684", # vegetation red edge
        "CP" : "50250" # cloud prob map
    }

    # Extra derived layers for Sentinel 2
    index_layers = [
        {
        "alias": "NDWI",
        "expression": "($B03 - $B08)/($B03 + $B08)"
        },
        {
        "alias": "NDMI",
        "expression": "($B08 - $B11)/($B08 + $B11)"
        },
        {
        "alias": "MSI",
        "expression": "$B11/$B8a"
        },
        {
        "alias": "NDVIre",
        "expression": "($B8a-$B05)/($B8a+$B05)"
        }    
    ]


    # ERA Layers
    eraLayers = {
        "Red" : "49360",
        "Total Prec" : "49459",
        "SP" : "49439",
        "Temp" : "49423",
        "AWVC" : "49458",
        "SR" : "49440",
        "TCC" : "49454",
        "MinT" : "49429",
        "MaxT" : "49430"
    }

    vpd_layers = [
            {
                # Max T convert kelvin to celcius
                "alias": "tmaxc",
                "expression": "$MaxT - 273.15",
                "output" : "false"
            },
            {
                # Min T convert kelvin to celcius
                "alias": "tminc",
                "expression": "$MinT - 273.15",
                "output" : "false"
            },
            {            
                # Min Rel humidity
                "type" : "raster", "id" : "48760",
                "aggregation" : "Min",
                "alias": "relativehummin",
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
                "alias": "esatmax",
                "expression": "611*math:exp((17.27 * $tmaxc)/($tmaxc + 237.3))",
                "output" : "false"
            },
            {
                # calculate saturation water vapour pressure in Pa at Tmax
                "alias": "esatmin",
                "expression": "611*math:exp((17.27 * $tminc)/($tminc + 237.3))",
                "output" : "false"
            },
            {
                # Calculate vpd max
                "alias": "vpdmax",
                "expression": "$esatmax * (1 - $relativehummin)",
                "output" : "false"
            },
            {
                # Calculate vpd min
                "alias": "vpdmin",
                "expression": "$esatmin * (1 - $relativehummax)",
                "output" : "false"
            },
            {
                # Calculate vpd max
                "alias": "vpd_max",
                "expression": "math:max(0, $vpdmax)"
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
        #"jura" : ["46.907", "7.137", "47.407", "7.637" ],
        #"schaffhausen": ["47.586", "8.397", "47.807", "8.756"] #,
        "sarraz" : ["46.630", "6.379", "47.700", "6.507"]#,  in Vaud
       # "wallis" : ["46.000", "7.02", "46.372", "7.435"]
    }

    # Define the query for Sentinel 2
    querySentinelJson = {
        "layers" : [
            {
                
                "type" : "raster", "id" : senLayers[lKey], "alias": lKey
            }
            for lKey in senLayers
        ] + index_layers,
        "spatial" : {"type" : "square",  "coordinates" : bbox }, 
        "temporal" : {"intervals" : [{"snapshot" : date}]},
        "processor" : [{
            "order" : 1,
            "type" : "coarse-grain",
            "options" : [
                {"name" : "levelsUp", "value" : "2"},
                {"name" : "aggregation", "value" : "bilinear"}
            ]
        }] 
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
            
    root_dir = '/dccstor/cimf/drought_impact/' 
    
    ###############################
    # Download data
    ###############################

            
    from datetime import date
    # Define start and end date
    #start_date = date(2015, 8, 9) is the function input, convert to date
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = date(2021, 12, 31)

    delta = 5 # Number of days over which to aggregate ERA 5 data. Or the expected timestep
    last_sen2date = None # use if aggregating ERA5 over irregular dates (using last downloaded Sentinel 2 date)

    iso8601 = "%Y-%m-%dT%H:%M:%SZ"


    if match_dates:

        # Authenticate before every query because of the time-out problem
        for bbox, coord in areas.items():

            for single_date in pd.date_range(start_date, end_date, freq=str(delta)+'D', inclusive='both'):
                print(f'Trying for date {single_date}')
                PAIRS_SERVER, PAIRS_CREDENTIALS = authenticate(pairs_key)

                # Try Sentinel 2 first

                date = single_date.strftime(iso8601)
                querySentinelJson["temporal"]["intervals"][0]["snapshot"] = date
                querySentinelJson["spatial"]["coordinates"] = coord
                sentinel_check = 0 # to check if there is Sentinel 2 data for this date or not

                # Check if date at this location already downloaded or not
                existsFile = os.path.exists(root_dir+'downloads/SENTINEL 2/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]))
                if existsFile:
                    print('Data already downloaded')
                    sentinel_check = 1

                if not existsFile:

                    try:    
                        querySentinel = paw.PAIRSQuery(querySentinelJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType='api-key', overwriteExisting=False)
                        querySentinel.submit()

                        querySentinel.poll_till_finished()
                        querySentinel.download()
                        querySentinel.create_layers()

                        #Check if the Sentinel folder exists
                        path_sentinel = root_dir+'downloads/SENTINEL 2/'
                        isExistSentinel = os.path.exists(path_sentinel)                          

                        if isExistSentinel == True : 
                            pass
                        else:
                            os.makedirs(path_sentinel)


                        #Rename File
                        old_name = 'downloads/' + str(querySentinel.zipFilePath)
                        new_name = 'downloads/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]) + '.zip'
                        os.rename(old_name, new_name)

                        #extract zip file
                        directory_name = path_sentinel + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3])
                        with zipfile.ZipFile(new_name, 'r') as zip_ref:
                            zip_ref.extractall(directory_name)
                        os.remove(new_name)

                        #remove original files
                        for f in os.listdir(directory_name + "/original/"):
                            os.remove(directory_name + "/original/" + f)
                        os.rmdir(directory_name + "/original/")

                        print('Sentinel-2 data downloaded')

                        sentinel_check = 1
                    except:
                        print('No Sentinel-2 data for this date') 


                if sentinel_check:

                    # Now download ERA 5 aggregated btween last date of Sentienl and present date
                    # Find last Sentinel 2 date that was downloaded
                    if last_sen2date:
                        date_start = (last_sen2date + timedelta(1)).strftime(iso8601) # the day after the last downloaded Sen2 date
                    if not last_sen2date:
                        # There is no previous Sen2date (i.e. download just started). Then consider only daily aggregate
                        date_start  = single_date.strftime(iso8601)
                    date_end = (single_date + timedelta(1)).strftime(iso8601)
                    queryEraJson["temporal"]["intervals"][0]["start"] = date_start
                    queryEraJson["temporal"]["intervals"][0]["end"] = date_end
                    queryEraJson["spatial"]["coordinates"] = coord

                    # Check if date at this location already downloaded or not
                    existsFile = os.path.exists(root_dir+'downloads/ERA5/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]))
                    if existsFile: 
                        print('Data already downloaded')
                    
                    if not existsFile:

                        try:    
                            queryEra = paw.PAIRSQuery(queryEraJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType='api-key', overwriteExisting=False)
                            queryEra.submit()

                            queryEra.poll_till_finished()
                            queryEra.download()
                            queryEra.create_layers()

                            #Check if the ERA folder exists
                            path_era = root_dir+'downloads/ERA5/'
                            isExistEra = os.path.exists(path_era)                           

                            if isExistEra == True : 
                                pass
                            else:
                                os.makedirs(path_era)

                            #Rename File
                            old_name = 'downloads/' + str(queryEra.zipFilePath)
                            new_name = 'downloads/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]) + '.zip'
                            os.rename(old_name, new_name)


                            # Delete Sentinel 2 files in ERA folder            
                            #extract zip file
                            directory_name = path_era + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3])
                            with zipfile.ZipFile(new_name, 'r') as zip_ref:
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
            

                            print('ERA-5 data downloaded')

                            last_sen2date = single_date 


                        except:
                            print('No ERA 5 data for this date')

                        
    if not match_dates:

        # DOWNLOADING ERA5 regular aggregates even when sen2 is not available

        # Authenticate before every query because of the time-out problem
        for bbox, coord in areas.items():

            for single_date in pd.date_range(start_date, end_date, freq=str(delta)+'D', inclusive='both'):
                print(f'Trying for date {single_date}')
                PAIRS_SERVER, PAIRS_CREDENTIALS = authenticate(pairs_key)

                # Try Sentinel 2 first

                date = single_date.strftime(iso8601)
                querySentinelJson["temporal"]["intervals"][0]["snapshot"] = date
                querySentinelJson["spatial"]["coordinates"] = coord
                sentinel_check = 0 # to check if there is Sentinel 2 data for this date or not

                # Check if date at this location already downloaded or not
                existsFile = os.path.exists(root_dir+'downloads/SENTINEL 2/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]))
                if existsFile:
                    print('Sentinel data already downloaded')
                
                if not existsFile:

                    try:   
                        print(f'Trying Sentinel...')
                        querySentinel = paw.PAIRSQuery(querySentinelJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType='api-key', overwriteExisting=False)
                        querySentinel.submit()

                        querySentinel.poll_till_finished()
                        querySentinel.download()
                        querySentinel.create_layers()

                        #Check if the Sentinel folder exists
                        path_sentinel = root_dir+'downloads/SENTINEL 2/'
                        isExistSentinel = os.path.exists(path_sentinel)                          

                        if isExistSentinel == True : 
                            pass
                        else:
                            os.makedirs(path_sentinel)

                        #Rename File
                        old_name = 'downloads/' + str(querySentinel.zipFilePath)
                        new_name = 'downloads/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]) + '.zip'
                        os.rename(old_name, new_name)

                        #extract zip file
                        directory_name = path_sentinel + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3])
                        with zipfile.ZipFile(new_name, 'r') as zip_ref:
                            zip_ref.extractall(directory_name)
                        os.remove(new_name)

                        #remove original files
                        for f in os.listdir(directory_name + "/original/"):
                            os.remove(directory_name + "/original/" + f)
                        os.rmdir(directory_name + "/original/")

                        print('Sentinel-2 data downloaded')

                        sentinel_check = 1
                    except:
                        print('No Sentinel-2 data for this date')



                # Now download ERA 5
                date_start = (single_date - timedelta(delta)).strftime(iso8601)
                date_end = (single_date + timedelta(1)).strftime(iso8601)
                queryEraJson["temporal"]["intervals"][0]["start"] = date_start
                queryEraJson["temporal"]["intervals"][0]["end"] = date_end
                queryEraJson["spatial"]["coordinates"] = coord

                # Check if date at this location already downloaded or not
                existsFile = os.path.exists(root_dir+'downloads/ERA5/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]))
                if existsFile: 
                    print('ERA data already downloaded')
                        
                if not existsFile:

                    try:   
                        print(f'Trying ERA between {date_start} and {date_end}')
                        queryEra = paw.PAIRSQuery(queryEraJson, PAIRS_SERVER, PAIRS_CREDENTIALS, authType='api-key', overwriteExisting=False)
                        queryEra.submit()

                        queryEra.poll_till_finished()
                        queryEra.download()
                        queryEra.create_layers()

                        #Check if the ERA folder exists
                        path_era = root_dir+'downloads/ERA5/'
                        isExistEra = os.path.exists(path_era)                          

                        if isExistEra == True : 
                            pass
                        else:
                            os.makedirs(path_era)

                        #Rename File
                        old_name = 'downloads/' + str(queryEra.zipFilePath)
                        new_name = 'downloads/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3]) + '.zip'
                        os.rename(old_name, new_name)


                        # Delete Sentinel 2 files in ERA folder            
                        #extract zip file
                        directory_name = root_dir+'downloads/ERA5/' + str(date) + '_' + str(coord[0]) + '_' + str(coord[1]) + '_' + str(coord[2]) + '_' + str(coord[3])
                        with zipfile.ZipFile(new_name, 'r') as zip_ref:
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

                        print('ERA-5 data downloaded')

                    except:
                        print('No ERA 5 data for this date')





parser = argparse.ArgumentParser()
parser.add_argument('--start_date', type=str)
args = parser.parse_args()

download_data(args.start_date)