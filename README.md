# Forest drought forecasting: a spatially distributed approach using LSTMs
Forecasting of forest drought impacts in Switzerland from satellite imagery, weather reanalysis and remote sensing data. Pixel-wise forecasts with feature engineering to provide model explainability.


## Topographic feature engineering

Create features from a DEM. Multiple DEMs at different resolutions can be used, features will be computed at different resolutions then matched to a reference (such that all final tiff files have same resolution and shape although representing different data). Resampling is done using nearest interpolation method (pixel splitting).

Supported features are slope, apsect (will automatically generate northing and easting), ruggedness index, curvature, terrain wetness index. These are cmputed using the WhitebocTools package. Additional features can be computed by adding functions to the `feature_engineer.py` script. For more information on WhiteboxTools and its functions: https://www.whiteboxgeo.com/manual/wbt_book/preface.html

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




