"""
Functions to create topographic features

Selene Ledain
Oct. 20th, 2022
"""

import numpy as np
import rasterio
import whitebox as wbt
import os
import pkg_resources
import whitebox
wbt = whitebox.WhiteboxTools()


def get_ruggedness(dem_path, outpath):
    """
    Calculates ruggedness index from DEM. Root-mean-square-deviation (RMSD) between a grid cell and its eight neighbours.
    Parameters
    -----------
    dem_path: str
        path to the input DEM
    outpath: str
        where output.tif should be stored
    """
    wbt.ruggedness_index(
        dem_path, 
        outpath)
    
def get_slope(dem_path, outpath):
    """
    Calculates slope from DEM. For geographic coord systems aka angular units (projected systems) it does a Talyor polynomial fitting in a 3x3 (5x5) neighborhood.
    Parameters
    -----------
    dem_path: str
        path to the input DEM
    outpath: str
        where output.tif should be stored
    zfactor: fl
        scale factor if horizontal/vertical units not the same
    units: str
        units of measurment of the slope
    """

    wbt.slope(dem_path, 
              outpath, 
              zfactor=None, 
              units="degrees")
    

def get_aspect(dem_path, outpath):
    """
    Calculates aspect from DEM. For geographic coord systems aka angular units (projected systems) it does a Taylor polynomial fitting in a 3x3 (5x5) neighborhood.
    Parameters
    -----------
    dem_path: str
        path to the input DEM
    outpath: str
        where output.tif should be stored
    zfactor: fl
        scale factor if horizontal/vertical units not the same
    """

    wbt.aspect(dem_path, 
              outpath, 
              zfactor=None)
    
    
def get_plan_curvature(dem_path, outpath):
    """
    Calculates plan curvature from DEM. For geographic coord systems aka angular units (projected systems) it  does a polynomial fitting in a 3x3 (5x5) neighborhood.
    Parameters
    -----------
    dem_path: str
        path to the input DEM
    outpath: str
        where output.tif should be stored
    log: bool
        if output resulst on log scale
    zfactor: fl
        scale factor if horizontal/vertical units not the same
    """

    wbt.plan_curvature(dem_path, 
              outpath, 
              log=False,
              zfactor=None)
    

def get_profile_curvature(dem_path, outpath):
    """
    Calculates profile curvature from DEM. For geographic coord systems aka angular units (projected systems) it  does a polynomial fitting in a 3x3 (5x5) neighborhood.
    Parameters
    -----------
    dem_path: str
        path to the input DEM
    outpath: str
        where output.tif should be stored
    log: bool
        if output resulst on log scale
    zfactor: fl
        scale factor if horizontal/vertical units not the same
    """

    wbt.profile_curvature(dem_path, 
              outpath, 
              log=False,
              zfactor=None)
    
    
def get_mean_curvature(dem_path, outpath):
    """
    Calculates mean (mean of plan and profile) curvature from DEM. For geographic coord systems aka angular units (projected systems) it  does a polynomial fitting in a 3x3 (5x5) neighborhood.
    Parameters
    -----------
    dem_path: str
        path to the input DEM
    outpath: str
        where output.tif should be stored
    log: bool
        if output resulst on log scale
    zfactor: fl
        scale factor if horizontal/vertical units not the same
    """

    wbt.mean_curvature(dem_path, 
              outpath, 
              log=False,
              zfactor=None)

    
def get_flow_accumulation(dem_path, output_path, pointer_path, accum_path):
    """
    Calculates flow accumulation/surface catchment area from DEM
    Parameters
    -----------
    dem_path: str
        path to the input DEM
    output_path: str
        where output.tif should be stored
    pointer_path: str
        where flow pointer .tif should be stored
    accum_path: str
        where flow accumulation .tif should be stored
    """

    wbt.flow_accumulation_full_workflow(
        dem_path, 
        output_path, 
        pointer_path, 
        accum_path, 
        out_type="cells", 
        log=False, 
        clip=False, 
        esri_pntr=False
    )



def get_wetness_index(accum_path, slope_path, outpath):
    """
    Calculates wetness index from DEM
    Parameters
    -----------
    accum_path: str
        where flow accumulation .tif is stored
    slope_path: str
        where slope .tif is stored
    outpath: str
        where output.tif should be stored
    """

    wbt.wetness_index(
        accum_path, 
        slope_path,
        outpath
    )
    
    
def get_relative_position(dem_path, filterx, filtery, outpath):
    """
    Calculates relative position to mean in a filterx x filtery region.
    Parameters
    -----------
    dem_path: str
         path to the input DEM
    filterx: int
        number of cells to consider in x-direction
    filtery: int
        number of cells to consider in y-direction
    outpath: str
        where output.tif should be stored
    """

    wbt.relative_topographic_position(
        dem_path, 
        outpath,
        filterx=filterx, 
        filtery=filtery
    )
    
    
def depth_in_sink(dem_path, outpath, zero_background):
    """
    Calculates, for cells in a depression, their depth.
    Parameters
    -----------
    dem_path: str
         path to the input DEM
    outpath: str
        where output.tif should be stored
    zero_background: bool
        If cells are not in sink put 0 as value (True) or NoData (False).
    """

    wbt.depth_in_sink(
        dem_path, 
        outpath,
        zero_background=True
    )
    
def get_northing_easting(dem_path, dir_name, northing_name, easting_name):
    """
    Calculates, for cells in a depression, their depth.
    Parameters
    -----------
    dem_path: str
         path to the aspect DEM
    dir_name: str
        folder where output northing and easting tifs
    """
    
    aspect = rasterio.open(dem_path)
    arr = aspect.read(1)
    northing = np.cos(arr*np.pi/180)
    easting = np.sin(arr*np.pi/180)
    profile = aspect.profile
    
    with rasterio.open(dir_name + northing_name, 'w', **profile) as dst:
        dst.write(northing, 1)
        
    with rasterio.open(dir_name + easting_name, 'w', **profile) as dst:
        dst.write(easting, 1)
        

def get_deviation(filepath, outpath, filterx, filtery):
    """
    Calculates the devaition of each cell from the mean in a filterx x filtery box (differnece normalized by standard deviation, so z-score)
    Parameters
    -----------
    filepath: str
         path to file for which we wawnt to compute windowed deviation
    outpath: str
        name of output file
    filterx: int
        number of cells to consider on the x-dimension
    filtery: int
        number of cells to consider on the y-dimension
    """
    
    wbt.dev_from_mean_elev(
    filepath, 
    outpath, 
    filterx=11, 
    filtery=11
    )