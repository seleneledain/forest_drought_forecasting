"""
Automatically launch jobs for downloading data using function in download_PAIRS_sen_era_agg.py
Will check which was the last date downloaded for a certain region and resatrt the donwloading from that date


Author: Selene Ledain
Nov. 21st, 2022

How to use:
- edit region of interest in line 21
- Edit end date for download in line 32
"""

import os
import time
import pandas as pd
from datetime import datetime, date


while True:

    # get last downloaded date: use ERA dates
   
 
    ERA_PATH = '/dccstor/cimf/drought_impact/downloads/ERA5'
    folder_list = [x for x in os.listdir(ERA_PATH) if "47.586_8.397_47.807_8.756" in x]
    df = pd.DataFrame(folder_list, columns=['folder']) 
    df['timestamp'] = df.folder.apply(lambda x: datetime.strptime(x.split('/')[-1].split('Z')[0], '%Y-%m-%dT%H:%M:%S') if 'Z' in x else datetime.strptime(x.split('/')[-1].split('_')[0], '%Y-%m-%d'))
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    last_downloaded_date = df['timestamp'][0].to_pydatetime()

    #last_downloaded_date = datetime(2017, 3, 1)

    # check that last downloaded date < last date desired
    last_date = datetime.strptime('2021-12-31', '%Y-%m-%d')
    print(type(last_downloaded_date))
    print(type(last_date))
    
    if last_downloaded_date>last_date:
        print("All dates already downloaded")
        break
    
    if last_downloaded_date<last_date:

        # send job
        print('Will start download from', last_downloaded_date)
        last_downloaded_date = datetime.strftime(last_downloaded_date, '%Y-%m-%d') # convert to string
        starttime = time.time()
        os.system(f'jbsub -name download -q x86_6h -cores 1x1+1 -mem 16G python download_PAIRS_sen_era_agg.py --start_date {last_downloaded_date}') # convert to approriate format in function

        # Kill job after 5 hours
        time.sleep(18000 - ((time.time() - starttime) % 18000)) # 18000s = 5hrs
        os.system('bkill -J download')

    
