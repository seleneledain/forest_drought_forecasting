# Forest drought forecasting: a spatially distributed approach using LSTMs
Forecasting of forest drought impacts in Switzerland from satellite imagery, weather reanalysis and remote sensing data. Pixel-wise forecasts with feature engineering to provide model explainability.

## Repository structure

```.
├── README.md                               > This file contains a description of what is the repository used for and how to use it.
├── data_downloading                        > Folder for scripts concering downloading of data.
    ├── swiss_dem                           > Specific to downloading the DEM from Swisstopo.
        ├── swiss_dem_download.py           > Download DEM tiles from swisstopo
        ├── reproject_dem.py                > Reproject the DEM tiles to CRS EPSG:4326
        ├──  mosaic_dem.py                  > Mosaic the tiles together into one single raster file.
    ├── pairs                               > Specific to downloading remote sensing data from IBM PAIRS.
        ├── download_PAIRS_sen_era.py       > Download Sentinel-2 and ERA-5 data. 
        ├── download_PAIRS_sen_era_agg.py   > Download Sentinel-2 and ERA-5 data aggregated over multiple days.
        ├── download_swiss_dem.py           > Download the swisstopo DEM that was uploaded to PAIRS.
        ├── download_env.py                 > Download the environemntal (forest, soil) data that was uploaded to PAIRS.
        ├── VaporPressureDeficit_udf.ipynb  > Notebook containing example of how user-defined-functions (UDFs) were made to compute vapor pressure deficit from aother atmospheric variables.
        ├── launch_downloads.py             > Automate multiple downloads.
        ├── ibmpairskey.txt                 > Where you would provide your account credentials for API use.
├── feature_engineering                     > Folder containing scripts for the creation of topographic features from DEM(s).
    ├── create_dem_feat.py                  > Functions to create multiple features from DEM(s) and adjust generated raster files.
    ├── feature_engineer.py                 > Functions to extract individual properties from DEM (based on WhiteboxTools package).
    ├── geospatial_data_utils.py            > Helper functions for manipulation raster data in python.
    ├── run_topo_feaets.py                  > Final script for topo feature engineering.
    ├── topo_features.ipynb                 > Example notebook. 

```
    
## Data Downloading 
### Digital Elevation Model (DEM)

Download the DEM from Swisstopo (at 0.5 or 2 m resolution). The DEM is split into tiles that need to be downloaded individually.\
The scripts are found in `data_downloading/swiss_dem/`

How to use:
- Go to https://www.swisstopo.admin.ch/fr/geodata/height/alti3d.html and request the whole dataset of tiles. This will provide you with a list of URLs that you can call to download.
- Copy the URLs and save them to a `.txt` file. The URLs for the DEM at 2 m resolution (in 2022) is provided (`urls_all.txt`).
- Using the `swiss_dem_download.py` script, provide the path to this `.txt` file and the path to where you want the DEM tiles to be downloaded. Run the script from terminal by calling `python swiss_dem_download.py`.
- The DEM tiles need to be reprojected to a global CRS (EPSG:4326) from local CRS (MN95 NF02). Use `python reproject_dem.py` to create new rasters named `*_reprojected.tiff`.
- Create a single DEM raster by mosaicking the (reprojected) tiles together using `python mosaic_dem.py`. Ensure that the paths are correct in the script.

### PAIRS
These scripts are provided if downloading from IBM PAIRS. An account is needed to use the API and credentials must be provided in a text file (`ibmpairskey.txt`)

## Topographic feature engineering

Create features from a DEM. Multiple DEMs at different resolutions can be used, features will be computed at different resolutions then matched to a reference (such that all final tiff files have same resolution and shape although representing different data). Resampling is done using nearest interpolation method (pixel splitting).

Supported features are slope, apsect (will automatically generate northing and easting), ruggedness index, curvature, terrain wetness index. These are cmputed using the WhiteboxTools package. Additional features can be computed by adding functions to the `feature_engineer.py` script. For more information on WhiteboxTools and its functions: https://www.whiteboxgeo.com/manual/wbt_book/preface.html

How to use:
- Edit and call the script in terminal as `python run_topo_feats.py`.
- For examples, including using multiple DEMs at different resolutions, refer to the `topo_features.ipynb` example notebook in the `feature_engineering` folder.
- Ensure that the paths used in the script/notebook for feature generation relative to the script/notebook.
- Provide
  - `list_paths`: list of paths to DEM(s). List will be of length 1 if using only 1 DEM
  - `target`: all generated features will be matched (in final resolution and shape) to this DEM tiff (usually the original DEM)
  - `feat_list`: list of features among [`slope`, `aspect`, `rugg`, `curv`, `twi`]
  - `out_path`: folder where to store generated features
  - `list_suffix`: if using multiple DEMs, provide a list of suffixes to differentiate the features generated from various DEMs
