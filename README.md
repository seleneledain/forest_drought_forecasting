# Forest drought forecasting: a spatially distributed approach using LSTMs
Forecasting of forest drought impacts in Switzerland from satellite imagery, weather reanalysis and remote sensing data. Pixel-wise forecasts with feature engineering to provide model explainability.

## Repository structure

```.
├── README.md                               > This file contains a description of what is the repository used for and how to use it.
├── sampling                                > Forlder for scripts concerning scene sampling in Switzerland.
    ├── rasterise.py                        > Rasterise, upsample and normalise shapefiles to use as masks in sampling.
    ├── sample.py                           > Sampling algorithms.
    ├── sampling.ipynb                      > Example notebook.
├── data_downloading                        > Folder for scripts concering downloading of data.
    ├── swiss_dem_download.py               > Download DEM tiles from swisstopo
    ├── reproject_dem.py                    > Reproject the DEM tiles to CRS EPSG:4326
    ├──  mosaic_dem.py                      > Mosaic the tiles together into one single raster file.
    ├──  swiss_dem_urls                     > Folder containing download URLs by canton.
        ├── *.txt                           > URLs for a given canton.
    ├──  create_dataset.py                  > Generate a dataset of minicubes.
    ├──  cloud_cleaning.py                  > Remove and replace cloud coevred values in Sentinel-2 data.
    ├──  config.py                          > Configuration file for dataset creation.
├── feature_engineering                     > Folder containing scripts for the creation of topographic features from DEM(s).
    ├── create_dem_feat.py                  > Functions to create multiple features from DEM(s) and adjust generated raster files.
    ├── feature_engineer.py                 > Functions to extract individual properties from DEM (based on WhiteboxTools package).
    ├── geospatial_data_utils.py            > Helper functions for manipulation raster data in python.
    ├── run_topo_feaets.py                  > Final script for topo feature engineering.
    ├── topo_features.ipynb                 > Example notebook. 
    ├── create_forest_mask.py               > Generate binary forest mask from forest shapefile in Switzerland.
    ├── add_static_data.py                  > Add time-invariant layers to the data cubes.
    ├── add_bands.py                        > Add additional features to the data cube (either computed or using add_static_data.py)
    ├── feature_list.txt                    > List of locally stored features and their description.
```


## 0. Scene Sampling

The first step is to identify locations around Switzerland. In particular, we collect the center coordinates of scenes of 2.56km x 2.56 km that will then be called during Data Downloading.
The approach here uses a shapefile of the forests in Switzerland, as well as some polygons that label droughts (obtained from Brun et al., 2020. https://doi.org/10.1111/gcb.15360). We want to ensure that we sample scenes containing drought events, as well as scenes across all the country.

**How to sample coordinates**
1. Rasterise your shapefiles, reproject, and upsample to a 2.56km resolution where each pixel is a continuous value between 0-1 (e.g. proportion of forest contained in each pixel). Use `sampling/rasterise.py` and refer to `sampling/sampling.ipynb`.
2. Sample scenes with containing drought labels
```
from sampling.sample import *

forest_mask_256 = ... # Path to forest mask with 2.56km resolution
drought_labels_256 = ... # Path to drought mask with 2.56km resolution
thresh_drought = 0 # Sample scenes with > thresh_drought
thresh_forest = 0.4  # Sample scenes with > thresh_forest
output_file_path = 'coords_drought.txt' # Define the output text file path
N = 100 # Number of scenes to sample. Will randomly keep N scenes

sample_drought(forest_mask_256, drought_labels_256, thresh_drought, thresh_forest, output_file_path, N)
```

3. Sample scenes that have NO positive drought labels. Split Switzerland in 6 subregions. For each subregion, get randomly N/6 scenes in each subregions.
```
from sampling.sample import *

forest_mask_256 = ... # Path to forest mask with 2.56km resolution
drought_labels_256 = ... # Path to drought mask with 2.56km resolution
thresh_forest = 0.4  # Sample scenes with > thresh_forest
output_file_path = 'coords_negative_drought.txt' # Define the output text file path
N = 100 # Number of scenes to sample

sample_negatives(forest_mask_256, drought_labels_256, thresh_forest, output_file, N)
```

## 1. Data Downloading and Dataset creation

Make sure that you have gone through step 2. Feature Engineering and that your local feature are ready to be added to the dataset!

### 1.1 Create datasets from minicubes

To create pixel timeseries from multiple data sources, we use the earthnet-minicuber package (https://github.com/geco-bern/earthnet-minicuber) to obtain spatio-temporal data (time-series of a scene for multiple bands). A minicube contains spatio-temporal data ready for machine learning applications. It usually contains satellite imagery and additional remote sensing data, all regridded to a common grid.\
To this cube, custom/local data can be added. Cloud removal on Sentinel-2 data can be done and pixel timeseries are sampled from the scene using a forest mask.

**Setup**\
The format of the data that will be downloaded and the samples that will be created are set up in the `data_downloading/config.py`. The parameters to modify are:
- `coord_list_paths`: list of paths to text files from which to read coordinates (generated duirng sampling).
- `specs`: Specifications passed to earthnet-minicuber, defining region, time and bands of interest. For more details of the possible cube specifications, check the `earthnet-minicuber` code. 
- `specs_add_bands`:  Specifications for adding locally stored data (including forest mask). The data will be added in the same resolution, projection and within the same bounds as the existing data in the minicube. There are two types of features that can be added to the minicube.
    - temporal: bands computed using the raw data in the cube, among ['NDVI']
    - static/local features: include topographic, vegetation and soil features. The possible features that can be added and their meaning are detailed in `feature_engineering/feature_list.txt`.
- `root_dir`: Where generated data should be stored.
- `split`: train/validation/test
- `context`: Number of context frames
- `target`: Number of target frames
- `shift`: Shift between Sentinel-2 and ERA5. ERA5 will be shifted n steps forward with respect to Sentinel-2, but the timestamps used for naming files are those of Sentinel-2.
- `cloud_cleaning`: maximum number of consecutive missing values (in NDVI timeseries) for which cloud cleaning will be performed. If > cloud_cleaning, the pixel wil not be used as a data sample. If 0/None no cloud cleaning is done.
- `normalisation`: If True, compute min/max for each band in the training set

**How to create dataset**
1. Download the required code to generate minicubes: https://github.com/geco-bern/earthnet-minicuber
2. In `data_downloading.create_dataset.py` add the path to the downloaded repository
```
import sys
# Add the path to the repository containing the file
sys.path.insert(0, 'Path_to_code/earthnet-minicuber/')
# Import the module
from earthnet_minicuber.minicuber import *
```
3. Also edit the dictionaries in `feature_engineering/add_bands.py` and `feature_engineering/add_static_data.py` to include all of you filenames and path to local features you may have.
4. Edit `data_downloading/config.py`
5. Launch the dataset creation
```
from data_downloading.create_dataset import *
import data_downloading.config as config

for path in config.coord_list_paths:
    # Open text file and read coordinates one by one
    with open(path) as f:
        lines = f.readlines()
        for coord in lines:
            print(f"Downloading cube and generating samples at {coord}")
            # Change the coordinate in config.specs
            config.specs["lon_lat"] = (float(coord.split(',')[0]), float(coord.split(',')[1]))
            # Generate sample for that coordinate
            generate_samples(config)
```

### 1.2 Digital Elevation Model (DEM)

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


### 1.3 Vegetation and Soil Data 
Additional environmental data can be downloaded. Refer to https://github.com/geco-bern/data_management for data availability.
These are time-invariant data and can be stored locally (~10GB). For ideal use, store all "static data" in a same folder.
Relevant data includes [Name, (repository)]:
- Forest mask for Switzerland (forestmaskswiss_waser_2015)
- Vegetation height Switzerland (vegheight_ginzler_2015)
- Swiss forest composition (forestcompositionswiss_waser_2017)
- High-resolution soil maps for the Swiss forest (somach_baltensweiler_2021)



## 2. Feature engineering

### 2.1 Topographic features

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
  - `out_path`: folder where to store generated features. Ideally, store these in the same folder as the vegetation/soil static features.
  - `list_suffix`: if using multiple DEMs, provide a list of suffixes to differentiate the features generated from various DEMs
