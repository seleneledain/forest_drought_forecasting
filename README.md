# Forest drought forecasting: a spatially distributed approach using LSTMs
Forecasting of forest drought impacts in Switzerland from satellite imagery, weather reanalysis and remote sensing data. Pixel-wise forecasts with feature engineering to provide model explainability.

## Repository structure

```.
├── README.md                               > This file contains a description of what is the repository used for and how to use it.
├── data_downloading                        > Folder for scripts concering downloading of data.
    ├── swiss_dem_download.py           > Download DEM tiles from swisstopo
    ├── reproject_dem.py                > Reproject the DEM tiles to CRS EPSG:4326
    ├──  mosaic_dem.py                  > Mosaic the tiles together into one single raster file.
    ├──  swiss_dem_urls                 > Folder containing download URLs by canton.
        ├── *.txt                       > URLs for a given canton.
├── feature_engineering                     > Folder containing scripts for the creation of topographic features from DEM(s).
    ├── create_dem_feat.py                  > Functions to create multiple features from DEM(s) and adjust generated raster files.
    ├── feature_engineer.py                 > Functions to extract individual properties from DEM (based on WhiteboxTools package).
    ├── geospatial_data_utils.py            > Helper functions for manipulation raster data in python.
    ├── run_topo_feaets.py                  > Final script for topo feature engineering.
    ├── topo_features.ipynb                 > Example notebook. 
    ├── create_forest_mask.py               > Generate binary forest mask from forest shapefile in Switzerland.

```
    
## Data Downloading 
### Digital Elevation Model (DEM)

Download the DEM from Swisstopo (at 0.5 or 2 m resolution). The DEM is split into tiles that need to be downloaded individually.\
The scripts are found in `data_downloading/`.

**Disclaimer**: This will produce a high resolution DEM for the whole of Switzerland, ensure you have enough storage space (min 265GB for the 2 m resolution DEM). Furthermore, the data processing might require a lot of RAM. It is advised to only download and process the DEM for your region of interest. 

**How to use**:
- Go to https://www.swisstopo.admin.ch/fr/geodata/height/alti3d.html and request the whole dataset of tiles. This will provide you with a list of URLs that you can call to download.
- Copy the URLs and save them to a `.txt` file. The URLs for the DEM at 2 m resolution (in 2022) is provided (`urls_all.txt`). The URLs have also been seperated by canton and are provided in `data_downloading/swiss_dem/swiss_dem_urls/`.
- Using the `swiss_dem_download.py` script, provide the path to this `.txt` file and the path to where you want the DEM tiles to be downloaded. Run the script from terminal by calling
```
python swiss_dem_download.py
```
- The DEM tiles need to be reprojected to a global CRS (EPSG:4326) from local CRS (MN95 NF02). New rasters named `*_reprojected.tiff` will be created. 
```
python reproject_dem.py
```
- Create a single DEM raster by mosaicking the (reprojected) tiles together. Ensure that the paths are correct in the script.
```
python mosaic_dem.py
```


### Vegetation and Soil Data 
Additional environmental data can be downloaded. Refer to https://github.com/geco-bern/data_management for data availability.
These are time-invariant data and can be stored locally (~10GB). For ideal use, store all "static data" in a same folder.
Relevant data includes [Name, (repository)]:
- Forest mask for Switzerland (forestmaskswiss_waser_2015)
- Vegetation height Switzerland (vegheight_ginzler_2015)
- Swiss forest composition (forestcompositionswiss_waser_2017)
- High-resolution soil maps for the Swiss forest (somach_baltensweiler_2021)



## Feature engineering

### Topographic features

Create features from a DEM. Multiple DEMs at different resolutions can be used, features will be computed at different resolutions then matched to a reference (such that all final tiff files have same resolution and shape although representing different data). Resampling is done using nearest interpolation method (pixel splitting).

Supported features are slope, apsect (will automatically generate northing and easting), ruggedness index, curvature, terrain wetness index. These are cmputed using the WhiteboxTools package. Additional features can be computed by adding functions to the `feature_engineer.py` script. For more information on WhiteboxTools and its functions: https://www.whiteboxgeo.com/manual/wbt_book/preface.html

**How to use**:
- Edit and call the script in terminal as 
```
python run_topo_feats.py
```
- For examples, including using multiple DEMs at different resolutions, refer to the `topo_features.ipynb` example notebook in the `feature_engineering` folder.
- Ensure that the paths used in the script/notebook for feature generation relative to the script/notebook.
- Provide
  - `list_paths`: list of paths to DEM(s). List will be of length 1 if using only 1 DEM
  - `target`: all generated features will be matched (in final resolution and shape) to this DEM tiff (usually the original DEM)
  - `feat_list`: list of features among [`slope`, `aspect`, `rugg`, `curv`, `twi`]
  - `out_path`: folder where to store generated features
  - `list_suffix`: if using multiple DEMs, provide a list of suffixes to differentiate the features generated from various DEMs


### Vegetation and soil features

The static data that has been downloaded (forest mask, forest height/composition, soil maps) can be added to the minicubes. The data will be added in the same resolution, projection and within the same bounds as the existing data in the minicube. 


**How to use**:
- To add raster data:
```
list_features = ['SAND0_5', 'FED100_200'] # The layers to add
static_dir = # Directory where your data is stored
cube = add_static_to_minicube(list_features, static_dir, cube, target_crs="epsg:4326", resampling_method="bilinear")
```
The possible features that can be added and their meaning are detailed in `feature_engineering/feature_list.txt`.


- To add the forest mask (vector data):
```
static_dir = # Directory where your data is stored
cube = add_mask_to_minicube(static_dir, cube)
```
