"""
interpolation.py
HILARIOUS ATTEMPT TO MAKE THIS A GIANT PROGRAM

Author....Ryan Poulsen
Created...11/04/2025
Updated...12/22/2025

Made with <3, fueled by mint tea and chocolate chips.
"""

# data
import pandas as pd
import numpy as np

# raster
import xarray as xr
import rioxarray as rxr

# file management
import os
import glob

# scipy
from scipy import interpolate, optimize, signal

# visualization
import matplotlib.pyplot as plt

# Raster Functions
def temporal_stack(data_src: str) -> xr.DataArray:
    """
    Reads in MODIS data from a directory, then creates a new `DataArray` on the dimension *doy*.
    ### Parameters
    * **data_src**: `str`
        * Path to the directory with the data.
    ### Returns
    * `DataArray`
        * The new `DataArray` stacked by *doy*.
    """
    rasters = sorted(glob.glob(os.path.expanduser(data_src))) # Get all paths to files in data_src
    crs_src = rxr.open_rasterio(rasters[0], masked=True).squeeze("band", drop=True).rio.crs # Get the crs of the first raster in the directory to use as template for the others to ensure all rasters match.

    rsts, doys = [], []
    year = 0
    for r in rasters:
        rio_r = rxr.open_rasterio(r, masked=True).squeeze("band", drop=True)
        year, doy = get_date(r)
        rio_r.rio.reproject(crs_src) # Might not be necessary, but is probably good practice to ensure that all rasters are in the same crs
        rsts.append(rio_r)
        doys.append(doy)

    data = xr.Dataset({d: r for d, r in zip(doys, rsts)}).to_array(dim="doy") # Creates a new DataArray with each band stacked.
    data.attrs.update({"year":year})
    return data

def get_date(path: str) -> tuple:
    """
    Gets information about the date of a raster from filepath. ("doyYEARDOY" must be present in filepath)
    ### Parameters
    * **path**: `str`
    ### Returns
    * `tuple`
        * two ints, year and doy
    """
    index = path.find("doy") + 3
    year = path[index:index+4]
    doy = path[index+4:index+7]
    return int(year), int(doy)

def export_data(rs:xr.DataArray, path:str, prefix:str="rs") -> None:
    """
    Exports a raster to tiff file with the naming convention <prefix>_doyYEARDOY.tif
    ### Parameters
    * **rs**: `DataArray`
        * Raster dataset to be exported.
    * **path**: `str`
        * Output location for the tiff file.
    * **prefix**: `str`
        * Prefix to be appended to the file's name.
    ### Returns
    * `None`
    """
    for r in rs:
        out = f"{os.path.expanduser(path)}/{prefix}_doy{rs.attrs["year"]}{int(r.doy):03d}.tif"
        r.rio.to_raster(out)


# Interpolation Functions
def interpolate_pixel(rs: xr.DataArray, x:int, y:int, display:bool=False) -> np.array:
    """
    Exports a raster to tiff file with the naming convention <prefix>_doyYEARDOY.tif
    ### Parameters
    * **rs**: `DataArray`
        * Input raster dataset with the pixel to be interpolated.
    * **y**: `int`
        * y-coordinate of the pixel to be interpolated.
    * **x**: `int`
        * x-coordinate of the pixel to be interpolated.
    * **display**: `bool`, *optional*
        * If `True`, the timeseries of the pixel value will be displayed.
        * `False` by default.
    ### Returns
    * `array`
        * A new array of the interpolated values.
    """

    doys = np.array([i for i in range(int(rs[0].doy), len(rs)+int(rs[0].doy))]) # Get a list of the doys
    start, stop = doys[0], doys[-1]
    values = np.array([rs.sel({'doy':t})[y][x] for t in doys])
    dv = pd.DataFrame({"doy": doys, "value": values}).set_index("doy", drop=True)

    
    dv_nonan = dv.dropna().reset_index()

    values = np.array(dv_nonan["value"])
    doys = np.array(dv_nonan["doy"])

    f = interpolate.interp1d(doys, values, fill_value="extrapolate")
    doy_new = np.linspace(start=start, stop=stop, num=stop)
    
    print(f"Pixel at y{y}, x{x} interpolated.")

    if display:
        plt.plot(doy_new, f(doy_new), "--", color="#2679B4", label="Interpolated Values")  #f_cubic(temp_new) tells it to estimate the enzyme activity at all 200 temps
        plt.plot(doys, values, 'o', color="#d06f1e", label="Original Data")
        plt.ylim(0, 1)
        plt.xlabel("Day of Year")
        plt.ylabel("NDVI Value")
        plt.title(f"NDVI Values in {rs.attrs["year"]} at x{x}, y{y}")
        plt.legend()
        plt.show()

    return f(doy_new)

def temporal_interpolation(rs:xr.DataArray) -> xr.DataArray:
    """
    Temporally interpolates raster data by the *doy* dimension. Iterates through the raster pixel-by-pixel.\n
    NOTE: This does work, however it takes. so. LONG.
    ### Parameters
    * **rs**: `DataArray`
        * Input raster dataset to be interpolated.
    ### Returns
    * `DataArray`
        * New interpolated raster.
    """
    rs_new = rs.copy() # Create a copy of the original raster to update with interpolated data
    for y in range(rs.shape[1]):
        for x in range(rs.shape[2]):
            new_vals = interpolate_pixel(rs=rs, y=int(y), x=int(x)) # New, interpolated, value for each doy in a pixel
            for ind, val in enumerate(new_vals):
                rs_new.sel({'doy':ind+1})[y][x] = val # Sets the values of the current pixel to be the new_vals.
    return rs_new

# NDVI Functions
def ndvi(red: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
    """
    Calculates the Normalized Difference Vegetation Index given red and nir bands.
    ### Parameters
    * **red**: `DataArray`
    * **nir**: `DataArray`
    ### Returns
    * `DataArray`
        * Normalized Difference Vegetation Index
    """
    ndvi = (nir - red) / (nir + red)
    ndvi.attrs.update(red.attrs) 
    return ndvi

def smooth_ndvi(ndvi: list, window_width: int = 32, order: int = 0) -> xr.DataArray:
    """
    Uses a Savitzky-Golay smoothing filter to smooth the ndvi data.
    ### Parameters
    * **ndvi**: `array`
        * The ndvi sequence to smooth.
    * **window_width**: `int`
        * Integer specifying the number of data points to smooth.
        * A window_width produces a smoother result at the expense of flattening sharp peaks.
        * Default is 32.
    * **order**: `int`\n
        * An integer specifying the order of the derivative desired.
        * Default for smoothing is order 0.
    ### Returns
    * `array`
        * The smoothed ndvi data.
    """
    return doy_dim(xr.DataArray(signal.savgol_filter(ndvi, window_width, order)))

# NBI Functions
def logistic_fall(x, a: float, b: float, c: float, d: float, x0 = 186):
    """
    Adapted from Zhang et al., 2011 and Spera et al., 2023\n
    ### Parameters
    * **x**: `array`
        * Doy
    * **x0**: `float`
        * Starting x value; offset.
    * **a**: `float`
        * Leaf development parameter a
    * **b**: `float`
        * Leaf development parameter b
    * **c**: `float`
        * base NDVI value
    * **d**: `float`
        * Minimum NDVI value (c+d = maxNDVI value)
    * **t**: `int`\n
        * Time in days
    ### Returns
    * `array`
        * Sigmoidal function values
    """
    #IT WORKKKEKDD! LET'S FUCKING GOOOOOOOOOOOOOO
    return (c/(1+np.exp(a+b*(x-x0))))+d

def nbi(x, pA, pB, x0: int = 186):
    """
    (Zhang et al., 2011)\n
    ### Parameters
    * **x**: `array`\n
        * The doys
    * **pA**: `DataArray`
        * Leaf development parameter a (from logistic_fall)
    * **pB**: `DataArray`
        * Leaf development parameter b (from logistic_fall)
    ### Returns
    * `DataArray`
        * Normalized Difference Brownnness Index
    """
    return 1-1/(1+np.exp(pA+(x-x0)*pB))

def logistic_pixel(ndvi_rs: xr.DataArray, x:int, y:int, start:int=180, end:int=366) -> list:
    """
    Fits the logistic fall function to the autumn ndvi data.\n
    ### Parameters
    * **ndvi_rs**: `xr.DataArray`\n
        * The data to calculate the curve for.
    * **x**: `int`\n
        * The x coordinate of the pixel
    * **y**: `int`\n
        * The y coordinate of the pixel
    ### Returns
    * `list`
        * List of the logistic fall equation parameters in order a, b, c, d.
    """

    doys = xr.DataArray([i for i in range(start, end)])
    ndvi_rs_p = xr.DataArray([ndvi_rs.sel({"doy":t})[x][y] for t in range(start, end)])

    maxNDVI = ndvi_rs_p.max()
    minNDVI = ndvi_rs_p.min()
    baseNDVI = maxNDVI - minNDVI

    pIn = [baseNDVI, 0.5, baseNDVI, minNDVI] # Initial guess at the parameters to feed into optimize

    param = optimize.curve_fit(f=logistic_fall, xdata=doys, ydata=ndvi_rs_p, p0=pIn, sigma=1) # Find the logistic fall equation that fits the NDVI data

    return param # Parameters of the logistic fall curve from optimize.curve_fit

def calculate_peak_pixel(ndvi_rs: xr.DataArray, start:int=180, end:int=366):
    """
    Fits the logistic fall function to the autumn ndvi data.\n
    ### Parameters
    * **ndvi_rs**: `xr.DataArray`\n
        * The data to calculate the curve for.
    * **start**: `int`
        * Start doy of fall season, alternatively, provide the doy of the max ndvi value.
        * Default `185`.
    * **end**: `int`
        * End doy. 
        * Default `365`.
    * **display**: `bool`, *optional*
        * Whether or not to display a graph of the senescence curve.
        * Default `False`.
    ### Returns
    * `list`
        * List of the logistic fall equation parameters in order a, b, c, d.
    """
    ndvi_autumn_sm = smooth_ndvi(ndvi_rs) # Smooth the ndvi values using Savisky-Golay filter

    for y in range(ndvi_rs.shape[1]):
        for x in range(ndvi_rs.shape[2]):
            params = logistic_pixel(ndvi_autumn_sm, x, y, start, end)
            print(params)

def doy_dim(rs: xr.DataArray):
    return rs.expand_dims({"doy": [doy for doy in range(1, len(rs)+1)]})
    