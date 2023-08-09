"""
Script to download tiles of swssiALTI3D DEM from SwissTopo.
URLs are collectd from the following website: https://www.swisstopo.admin.ch/fr/geodata/height/alti3d.html

20 Sep. 2022
Selene Ledain
"""

import requests
import pandas as pd


def swiss_dem_download(urls_path, downloads_path):
    """Download DEM tiles from swissAlti3D

    Args:
        urls_path (str): path to text file containing URLS to download
        downloads_path (str): path to store downloaded files
    """

    data = pd.read_csv(urls_path, header=None)
    data.columns = ["link"]

    for idx, row in data.iterrows():
        url = row["link"]
        print(url)
        name = url.split('/')[-1].split('\n')[0]
        print(name)
        r = requests.get(url)
        print('Download successful')
        open(downloads_path + name, "wb").write(r.content)
        
    
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--urls_path', type=str, default='urls_all.txt')
    parser.add_argument('--downloads_path', type=str, default='/data/scratch/selene/dem/')

    args = parser.parse_args()
    
    swiss_dem_download(args.urls_path, args.downloads_path)
