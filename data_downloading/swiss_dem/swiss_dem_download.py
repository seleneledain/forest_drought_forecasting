"""
Script to download tiles of swssiALTI3D DEM from SwissTopo.
URLs are collectd from the following website: https://www.swisstopo.admin.ch/fr/geodata/height/alti3d.html

20 Sep. 2022
Selene Ledain
"""

import requests
from PIL import Image
import pandas as pd


""" VARIABLES """
URLS_PATH = 'urls_all.txt'
DOWNLOADS_PATH = 'downloads/dem/'


""" DOWNLOAD """

data = pd.read_csv(URLS_PATH, header=None)
data.columns = ["link"]

for idx, row in data.iterrows():
    url = row["link"]
    print(url)
    name = url.split('/')[-1].split('\n')[0]
    print(name)
    r = requests.get(url)
    print('Download successful')
    open(DOWNLOADS_PATH + name, "wb").write(r.content)