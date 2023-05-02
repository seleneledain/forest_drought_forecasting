import pandas as pd
from ibmpairs import paw, authentication
import json
import numpy as np
import logging
import traceback
import os
import glob
import requests
import sys, traceback, asyncio, aiohttp, warnings
import uploads_key_auth as uploads #local script
from pathlib import Path
import argparse


import logging

#logging.basicConfig(level=logging.DEBUG)
logging.getLogger('ibm_s3transfer').setLevel(logging.INFO)
logging.getLogger('ibm_botocore').setLevel(logging.INFO)
logging.getLogger('ibm_boto3').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)

pawLogger = logging.getLogger('ibmpairs.paw')
pawLogger.setLevel(logging.INFO)

with open("ibmpairskey.txt", "r") as f:
    pairs_key = f.read()
    
# Append successful uploads to a log file

    
def authenticate(pairs_key):
    
    pairs_credentials = authentication.OAuth2(api_key=pairs_key)
    auth_header = {'Authorization': f'Bearer {pairs_credentials.jwt_token}'}
    PAIRS_SERVER   = 'https://pairs.res.ibm.com'
    PAIRS_CREDENTIALS = authentication.OAuth2(api_key=pairs_key)
    
    return PAIRS_SERVER, PAIRS_CREDENTIALS


############################
# UPLOADING TO IBM PAIRS: upload
############################


""" UPLOAD """

async def upload_to_pairs(urls_path, downloads_path, layer_id, file_nbr_start):

    data = pd.read_csv(urls_path, header=None)
    data.columns = ["link"]
    
    # Upload n files per job
    file_count = 0

    for idx, row in data.iterrows():
        
        uploadFail = 1 # initiate as "failed uploading". Leave the while loop when not failed (uploadFail = not None)
        
        if idx >= file_nbr_start:

            while uploadFail:
                print('File', idx)
                url = row["link"]
                name = url.split('/')[-1].split('\n')[0] 

                ### Reauthenticate ###

                apikey = "FzmaPxVOTb6_LOH1AqNggpFwER7n61EISx1_TqrQugKU"
                resourceInstanceID = "crn:v1:bluemix:public:cloud-object-storage:global:a/045077ab9f1a4dcfafa2c58389d3d639:11a6c883-43a0-4758-a264-6568f2b5ad9a::"
                authEndpoint = "https://iam.eu-de.bluemix.net/oidc/token"
                endpoint = "https://s3.eu-de.cloud-object-storage.appdomain.cloud"
                bucketName = "drought"
                accessKeyID = "b9f5af1d0bae45e4bbd9a58420a6190c"
                secretAccessKey = "53faf26559acb6e0752d16d869cabff2016a1991b97d216a"


                AUTH_PROVIDER_BASE = 'https://auth-b2b-twc.ibm.com'
                PAIRS_API_KEY = pairs_key

                auth_response = requests.post(    
                    AUTH_PROVIDER_BASE + '/connect/token',
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    data=[
                        ('client_id', 'ibm-pairs'),
                        ('grant_type', 'apikey'),
                        ('apikey', PAIRS_API_KEY)])

                auth_obj = auth_response.json()
                access_token = auth_obj.get('access_token')
                refresh_token = auth_obj.get('refresh_token')

                pairsAuthHeader = {
                        'Authorization':'Bearer {}'.format(access_token),
                        'Content-Type': 'application/json'
                    }


                ### Create meta.json ###
                meta_json = {
                    "datalayer_id": [layer_id],
                    "pairsdatatype": "2draster",
                    "datainterpolation": "bilinear",
                    "timestamp": "20220101T000000Z"
                }


                json_object = json.dumps(meta_json, indent=4)

                with open(downloads_path + f"{name}.meta.json", "w") as outfile:
                    outfile.write(json_object)

                ### Upload to COS and PAIRS ###
                cosBucket = uploads.IBMCosBucket(
                    apikey, resourceInstanceID, authEndpoint, endpoint,
                    bucketName, accessKeyID, secretAccessKey
                )

                filesToUpload = [downloads_path + name]

                uploadFromLocal = uploads.PAIRSUpload.fromLocal(
                    cosBucket, pairsAuthHeader, False, fileList=filesToUpload
                )

                await uploadFromLocal.run(1)

                # add a check to see if uploaded correctly
                uploadFail = {v.remoteObject for v in uploadFromLocal.completedUploads if not v.status == 'SUCCEEDED'}
                

            ### delete meta.json ###

            fileList = glob.glob(downloads_path + '*.meta.json', recursive=True)
            for filePath in fileList:
                os.remove(filePath)

            cosBucket.delete(name)
            
            
            with open("log.txt", "a") as f:
                f.write(f'{file_nbr_start + file_count}\n')
                
            file_count +=1

            """
            # Terminate after 500 uploads 
            if file_count == 500:
                return file_count 
            """





parser = argparse.ArgumentParser()
parser.add_argument('--urls_path', type=str)
parser.add_argument('--downloads_path',  type=str) 
parser.add_argument('--layer_id', type=str)
parser.add_argument('--file_nbr_start', type=int)
args = parser.parse_args()
print(args)

asyncio.run(upload_to_pairs(args.urls_path, args.downloads_path, args.layer_id, args.file_nbr_start))