#!/sr/bin/env python

###############################################################################
# Program : 
# Authors  : Jesus Lizana
# Date	  : 25 April 2025
# Purpose : Descriptive statistics of climate variability – 10th percentile, median and 90th percentile per subregion
##############################################################################

#%%


import os
import glob

print("....importing libraries")

import netCDF4
from netCDF4 import Dataset,num2date # or from netCDF4 import Dataset as NetCDFFile
import numpy as np
import shutil
import dateutil.parser

import xarray as xr

import matplotlib.pyplot as plt    
import seaborn as sns

import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend

from statsmodels.distributions.empirical_distribution import StepFunction

from sys import exit

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'

import numpy as np
import numpy.ma as ma


import rasterio
from rasterio.plot import show

import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from cartopy.io import shapereader

import geopandas as gpd

from rasterstats import zonal_stats

from gridfill import fill

print("All libraries imported")


#%%

"""

Data directory 

(a) Identify all data files and create a table with names and path

"""   
##########################################################################
##########################################################################

#INPUT DATA 

# get folder location of script
cwd = os.path.dirname(__file__) 

#go into folder cwd + DATA
folder = cwd+"/DATA"

#create a list with all files inside subfolders
os.chdir(folder)
files = glob.glob(folder+"/*/*.nc")
print("files :",files)

#create a table with file name (without extension), and file path

files_table = pd.DataFrame(files, columns=["file_path"])
files_table["file_name"] = files_table["file_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

#new column to classify the data by CDD or HDD
files_table["data_type"] = files_table["file_name"].apply(lambda x: "CDD" if "CDD" in x else "HDD")

#first column file name, second column file path
files_table = files_table[["data_type","file_name", "file_path"]]

#sort by file name
files_table = files_table.sort_values(by="file_name").reset_index(drop=True)

#show table
print("Files table:")
print(files_table)


  
#%%

##########################################################################
##########################################################################

# gap filling using INTERPOLATION with GRIDFILL: https://ocefpaf.github.io/python4oceanographers/blog/2014/10/20/gridfill/
# Convert NetCDF files to GeoTIFF 

##########################################################################
##########################################################################


kw = dict(eps=1e-4, relax=0.6, itermax=1e5, initzonal=False,
          cyclic=False, verbose=True)


#create new folder for temporal files
temp_folder = os.path.join(cwd, "temporal_files")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    
# Get the list of NetCDF files for HDD
nc_filesHDD = files_table[
    (files_table["data_type"] == "HDD") &
    (files_table["file_path"].str.contains("mean", case=False, na=False))
]["file_path"].tolist()

nc_filesCDD = files_table[
    (files_table["data_type"] == "CDD") &
    (files_table["file_path"].str.contains("mean", case=False, na=False))
]["file_path"].tolist()

  
#%%

#create list with out output files:
for nc_file in nc_filesHDD:
    # Get the base name of the file without extension
    base_name = os.path.splitext(os.path.basename(nc_file))[0]
    output_file = os.path.join(temp_folder, f"{base_name}_raster_filling.tif") 
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping.")
    else:
        print(f"Output file {output_file} will be created.")

        # Loop over each file
        for nc_file in nc_filesHDD:
            print(f"Processing file: {nc_file}")
            # Load NetCDF dataset
            ds = xr.open_dataset(nc_file)

            # --- Select variable ---
            if "HDD_total" not in ds:
                print(f"Variable 'HDD_total' not found in {nc_file}. Skipping.")
                continue

            # Select the variable of interest
            da = ds["HDD_total"]

            da_values = ds["HDD_total"][:]

            da_masked = ma.masked_invalid(da_values)  

            filled, converged = fill(da_masked, 1, 0, **kw)


            if not converged:
                print(f"Warning: fill did not converge for {nc_file}")

            # --- Create a filled DataArray ---
            da_filled = da.copy(deep=True)
            da_filled.values = filled

            # --- Rename coordinates if needed ---
            if "longitude0" in da.coords and "latitude0" in da.coords:
                da_filled = da_filled.rename({"longitude0": "x", "latitude0": "y"})

            print("ready to save")
            # --- Attach CRS and export ---
            da_filled.rio.write_crs("EPSG:4326", inplace=True)

            # --- Save to GeoTIFF ---
            nc_file_name = os.path.basename(nc_file)
            base_name = os.path.splitext(nc_file_name)[0]
            output_file = os.path.join(temp_folder, f"{base_name}_raster_filling.tif")

            da_filled.rio.to_raster(output_file)
            print(f"Saved filled raster to: {output_file}")

            # --- Close dataset ---
            ds.close()

#create list with out output files:
for nc_file in nc_filesCDD:
    # Get the base name of the file without extension
    base_name = os.path.splitext(os.path.basename(nc_file))[0]
    output_file = os.path.join(temp_folder, f"{base_name}_raster_filling.tif") 
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping.")
    else:
        print(f"Output file {output_file} will be created.")

        # Loop over each file
        for nc_file in nc_filesCDD:
            print(f"Processing file: {nc_file}")
            # Load NetCDF dataset
            ds = xr.open_dataset(nc_file)

            # --- Select variable ---
            if "CDD_total" not in ds:
                print(f"Variable 'CDD_total' not found in {nc_file}. Skipping.")
                continue

            # Select the variable of interest
            da = ds["CDD_total"]

            da_values = ds["CDD_total"][:]

            da_masked = ma.masked_invalid(da_values)  

            filled, converged = fill(da_masked, 1, 0, **kw)


            if not converged:
                print(f"Warning: fill did not converge for {nc_file}")

            # --- Create a filled DataArray ---
            da_filled = da.copy(deep=True)
            da_filled.values = filled

            # --- Rename coordinates if needed ---
            if "longitude0" in da.coords and "latitude0" in da.coords:
                da_filled = da_filled.rename({"longitude0": "x", "latitude0": "y"})

            print("ready to save")
            # --- Attach CRS and export ---
            da_filled.rio.write_crs("EPSG:4326", inplace=True)

            # --- Save to GeoTIFF ---
            nc_file_name = os.path.basename(nc_file)
            base_name = os.path.splitext(nc_file_name)[0]
            output_file = os.path.join(temp_folder, f"{base_name}_raster_filling.tif")

            da_filled.rio.to_raster(output_file)
            print(f"Saved filled raster to: {output_file}")

            # --- Close dataset ---
        ds.close()


#%%

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


LAND = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='face',
                                    facecolor=cfeature.COLORS['land'])


OCEAN = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                    color="white",zorder=1)

def make_map(bbox, projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw=dict(projection=projection))
    ax.set_extent(bbox)
    ax.add_feature(OCEAN) #, facecolor='0.75'
    #ax.add_feature(cfeature.OCEAN, facecolor='0.75',color='white',zorder=1) #default scale (1:110m)
    ax.coastlines(resolution='50m',zorder=1)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


#%%

##########################################################################
##########################################################################

# BOUNDARIES FOR SPATIAL ANALYSIS

##########################################################################
##########################################################################

# Load country boundaries
shpfilename = shapereader.natural_earth(
    resolution='10m',  # use '10m' for higher detail
    category='cultural',
    name='admin_0_countries'
)
countries = gpd.read_file(shpfilename)

# Dissolve by continent
#continents = countries.dissolve(by="SUBREGION", as_index=False) #"REGION_UN" "CONTINENT" "SUBREGION"

#go to temp_folder
os.chdir(temp_folder)

# Save to shapefile
output_dir = "SUBREGION_shapefiles"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "COUNTRIES.shp")
countries.to_file(output_path)

      
#%%

###plot map
# Set up the map projection and plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_title("World Countries (Admin Level: country)")

# Add the country borders from the shapefile
countries.plot(ax=ax, facecolor='whitesmoke', edgecolor='grey', linewidth=0.5, transform=ccrs.PlateCarree())

#list columns names in countries


# Add Cartopy features for context (optional)
#ax.coastlines(edgecolor='grey')
#ax.gridlines(draw_labels=True)

# --- Add subregion names at centroid ---
for idx, row in countries.iterrows():
    if pd.notnull(row['ADMIN']):
        centroid = row['geometry'].centroid
        ax.text(
            centroid.x, centroid.y, row['ADMIN'],
            fontsize=5, ha='center', color='darkblue',transform=ccrs.PlateCarree(), alpha=1
        )

plt.tight_layout()
plt.show()

      
#%%

# Get the list of unique continents
country_list = countries["ADMIN"].unique().tolist()

# Sort for readability (optional)
country_list.sort()

# Print the list
print("List of continents in the dataset:")
print(country_list)

      

      
#%%

##########################################################################
##########################################################################

# FINAL SPATIAL ANALYSIS - Spatial analysis by country (1ºC, 1.5ºC, 2ºC)

##########################################################################
##########################################################################

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


LAND = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='face',
                                    facecolor=cfeature.COLORS['land'])


OCEAN = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                    color="white",zorder=1)

def make_map(bbox, projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw=dict(projection=projection))
    ax.set_extent(bbox)
    ax.add_feature(OCEAN) #, facecolor='0.75'
    #ax.add_feature(cfeature.OCEAN, facecolor='0.75',color='white',zorder=1) #default scale (1:110m)
    ax.coastlines(resolution='50m',zorder=1)
    #gl = ax.gridlines(draw_labels=True)
    #gl.xlabels_top = gl.ylabels_right = False
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


#%%

##########################################################################
##########################################################################

# DEFINE COUNTRY TO PLOT  - example with UK

##########################################################################
##########################################################################

country_to_plot = "United Kingdom"

#check if country is in the list
if country_to_plot not in country_list:
    print(f"Country '{country_to_plot}' not found in the dataset.")
    exit()

uk = countries[countries['ADMIN'] == country_to_plot]

if uk.empty:
    raise ValueError(f"{country_to_plot} not found in countries GeoDataFrame.")

bbox_uk = uk.total_bounds  # [minx, miny, maxx, maxy]

# Convert to [min_lon, max_lon, min_lat, max_lat] for make_map
bbox_uk_reference = [bbox_uk[0], bbox_uk[2], bbox_uk[1], bbox_uk[3]]

print("Bounding box:", bbox_uk)


#%%

##########################################################################
##########################################################################

# CHECK AREA OF INTEREST

# ##########################################################################
##########################################################################

#define bbox_uk
bbox_uk =  [bbox_uk[0]+2.5, bbox_uk[2]+1, bbox_uk[1]-0.5, bbox_uk[3]+0.5]  # [min_lon, max_lon, min_lat, max_lat]
print("Adjusted:", bbox_uk)
fig, ax = make_map(bbox=bbox_uk)

# Plot
fig, ax = make_map(bbox=bbox_uk)
plt.title('area of interest')
plt.show()


# %%

##########################################################################
##########################################################################

# define min and max values for HDD and CDD

# ##########################################################################
##########################################################################

vmin_HDD = 0
vmax_HDD = 3500
vmin_CDD = 0
vmax_CDD = 200

import rasterio
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#go to temp_folder
os.chdir(temp_folder)

# --- File names and titles ---
hdd_files = [
    "HDD_historical_mean_v1_raster_filling.tif",
    "HDD_scenario15_mean_v1_raster_filling.tif",
    "HDD_scenario20_mean_v1_raster_filling.tif"
]
cdd_files = [
    "CDD_historical_mean_v1_raster_filling.tif",
    "CDD_scenario15_mean_v1_raster_filling.tif",
    "CDD_scenario20_mean_v1_raster_filling.tif"
]
titles_hdd = [
    "a1, Scenario 1.0°C (2006-2016)",
    "b1, Scenario 1.5°C",
    "c1, Scenario 2.0°C"
]

titles_cdd = [
    "a2, Scenario 1.0°C (2006-2016)",
    "b2, Scenario 1.5°C",
    "c2, Scenario 2.0°C"
]

bbox_uk = bbox_uk  # [min_lon, max_lon, min_lat, max_lat]

fig, axes = plt.subplots(3, 2, figsize=(11, 14), subplot_kw={'projection': ccrs.PlateCarree()})
cs_hdd = cs_cdd = None

for i in range(3):
    # --- HDD ---
    ax = axes[i, 0]
    with rasterio.open(hdd_files[i]) as src:
        member_value = src.read(1)
        transform = src.transform
        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        member_long = np.array(xs[0])
        member_lat = np.array([y[0] for y in ys])
    value = ma.masked_invalid(member_value)
    x, y = np.meshgrid(member_long, member_lat)
    ax.set_extent(bbox_uk)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', zorder=2)
    ax.coastlines(resolution='50m', zorder=2)
    cs_hdd = ax.pcolormesh(x, y, value, vmin=vmin_HDD, vmax=vmax_HDD, zorder=1, cmap='plasma_r')
    ax.set_title(titles_hdd[i], loc='left', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- CDD ---
    ax = axes[i, 1]
    with rasterio.open(cdd_files[i]) as src:
        member_value = src.read(1)
        transform = src.transform
        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        member_long = np.array(xs[0])
        member_lat = np.array([y[0] for y in ys])
    value = ma.masked_invalid(member_value)
    x, y = np.meshgrid(member_long, member_lat)
    ax.set_extent(bbox_uk)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', zorder=2)
    ax.coastlines(resolution='50m', zorder=2)
    cs_cdd = ax.pcolormesh(x, y, value, vmin=vmin_CDD, vmax=vmax_CDD, zorder=1, cmap='turbo')
    ax.set_title(titles_cdd[i], loc='left', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

# --- Colorbars ---
fig.subplots_adjust(bottom=0.13, top=0.95, left=0.08, right=0.92, hspace=0.25, wspace=0.10)
cbar_ax_hdd = fig.add_axes([0.06, 0.06, 0.40, 0.025])
cbar_ax_cdd = fig.add_axes([0.53, 0.06, 0.40, 0.025])
cbar_hdd = fig.colorbar(cs_hdd, cax=cbar_ax_hdd, orientation='horizontal', label='Heating Degree Days (HDD)')
cbar_cdd = fig.colorbar(cs_cdd, cax=cbar_ax_cdd, orientation='horizontal', label='Cooling Degree Days (CDD)')

cbar_hdd.ax.tick_params(labelsize=11)
cbar_cdd.ax.tick_params(labelsize=11)

cbar_hdd.set_ticks(np.arange(0, vmax_HDD+1, 500))
cbar_cdd.set_ticks(np.arange(0, vmax_CDD+1, 25))

for cbar in [cbar_hdd, cbar_cdd]:
    cbar.ax.tick_params(
        bottom=False,
        labelbottom=True,
        top=False,
        labeltop=False
    )
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

plt.tight_layout(rect=[0, 0.1, 1, 1])

#save in cwd + output folder
output_folder = cwd + "/output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)

#plt.savefig("Figure1_HDD_maps_and_boxplots.svg", format='svg', dpi=600, bbox_inches='tight')
plt.savefig("Figure_country_maps_UK.jpg", format='jpg', dpi=600, bbox_inches='tight')

plt.show()
# %%




#%%

##########################################################################
##########################################################################

# DEFINE COUNTRY TO PLOT - example with Spain

##########################################################################
##########################################################################

country_to_plot = "Spain"

#check if country is in the list
if country_to_plot not in country_list:
    print(f"Country '{country_to_plot}' not found in the dataset.")
    exit()

uk = countries[countries['ADMIN'] == country_to_plot]

if uk.empty:
    raise ValueError(f"{country_to_plot} not found in countries GeoDataFrame.")

bbox_cr1 = uk.total_bounds  # [minx, miny, maxx, maxy]

# Convert to [min_lon, max_lon, min_lat, max_lat] for make_map
bbox_cr1_reference = [bbox_cr1[0], bbox_cr1[2], bbox_cr1[1], bbox_cr1[3]]

print("Bounding box:", bbox_uk)


#%%

##########################################################################
##########################################################################

# CHECK AREA OF INTEREST

# ##########################################################################
##########################################################################

#define bbox_uk
bbox_cr =  [bbox_cr1[0]+8, bbox_cr1[2], bbox_cr1[1]+7, bbox_cr1[3]+1]  # [min_lon, max_lon, min_lat, max_lat]
print("Adjusted:", bbox_cr)
fig, ax = make_map(bbox=bbox_cr)

# Plot
fig, ax = make_map(bbox=bbox_cr)
plt.title('area of interest')
plt.show()


# %%

##########################################################################
##########################################################################

# define min and max values for HDD and CDD

# ##########################################################################
##########################################################################

vmin_HDD = 0
vmax_HDD = 2500
vmin_CDD = 0
vmax_CDD = 1500

import rasterio
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#go to temp_folder
os.chdir(temp_folder)


# --- File names and titles ---
hdd_files = [
    "HDD_historical_mean_v1_raster_filling.tif",
    "HDD_scenario15_mean_v1_raster_filling.tif",
    "HDD_scenario20_mean_v1_raster_filling.tif"
]
cdd_files = [
    "CDD_historical_mean_v1_raster_filling.tif",
    "CDD_scenario15_mean_v1_raster_filling.tif",
    "CDD_scenario20_mean_v1_raster_filling.tif"
]
titles_hdd = [
    "a1, Scenario 1.0°C (2006-2016)",
    "b1, Scenario 1.5°C",
    "c1, Scenario 2.0°C"
]

titles_cdd = [
    "a2, Scenario 1.0°C (2006-2016)",
    "b2, Scenario 1.5°C",
    "c2, Scenario 2.0°C"
]

bbox_cr4 = bbox_cr  

fig, axes = plt.subplots(3, 2, figsize=(11, 14), subplot_kw={'projection': ccrs.PlateCarree()})
cs_hdd = cs_cdd = None

for i in range(3):
    # --- HDD ---
    ax = axes[i, 0]
    with rasterio.open(hdd_files[i]) as src:
        member_value = src.read(1)
        transform = src.transform
        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        member_long = np.array(xs[0])
        member_lat = np.array([y[0] for y in ys])
    value = ma.masked_invalid(member_value)
    x, y = np.meshgrid(member_long, member_lat)
    ax.set_extent(bbox_cr4)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', zorder=2)
    ax.coastlines(resolution='50m', zorder=2)
    cs_hdd = ax.pcolormesh(x, y, value, vmin=vmin_HDD, vmax=vmax_HDD, zorder=1, cmap='plasma_r')
    ax.set_title(titles_hdd[i], loc='left', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- CDD ---
    ax = axes[i, 1]
    with rasterio.open(cdd_files[i]) as src:
        member_value = src.read(1)
        transform = src.transform
        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        member_long = np.array(xs[0])
        member_lat = np.array([y[0] for y in ys])
    value = ma.masked_invalid(member_value)
    x, y = np.meshgrid(member_long, member_lat)
    ax.set_extent(bbox_cr4)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', zorder=2)
    ax.coastlines(resolution='50m', zorder=2)
    cs_cdd = ax.pcolormesh(x, y, value, vmin=vmin_CDD, vmax=vmax_CDD, zorder=1, cmap='turbo')
    ax.set_title(titles_cdd[i], loc='left', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

# --- Colorbars ---
fig.subplots_adjust(bottom=0.13, top=0.95, left=0.08, right=0.92, hspace=0.25, wspace=0.10)
cbar_ax_hdd = fig.add_axes([0.05, 0.06, 0.42, 0.025])
cbar_ax_cdd = fig.add_axes([0.53, 0.06, 0.42, 0.025])
cbar_hdd = fig.colorbar(cs_hdd, cax=cbar_ax_hdd, orientation='horizontal', label='Heating Degree Days (HDD)')
cbar_cdd = fig.colorbar(cs_cdd, cax=cbar_ax_cdd, orientation='horizontal', label='Cooling Degree Days (CDD)')

cbar_hdd.ax.tick_params(labelsize=11)
cbar_cdd.ax.tick_params(labelsize=11)

cbar_hdd.set_ticks(np.arange(0, vmax_HDD+1, 500))
cbar_cdd.set_ticks(np.arange(0, vmax_CDD+1, 250))

for cbar in [cbar_hdd, cbar_cdd]:
    cbar.ax.tick_params(
        bottom=False,
        labelbottom=True,
        top=False,
        labeltop=False
    )
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

plt.tight_layout(rect=[0, 0.1, 1, 1])

#save in cwd + output folder
output_folder = cwd + "/output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)

#plt.savefig("Figure1_HDD_maps_and_boxplots.svg", format='svg', dpi=600, bbox_inches='tight')
plt.savefig("Figure_country_maps_Spain.jpg", format='jpg', dpi=600, bbox_inches='tight')

plt.show()
# %%
