#!/sr/bin/env python

###############################################################################
# Program : 
# Authors  : Jesus Lizana
# Date	  : 25 April 2025
# Purpose : Descriptive statistics of a dataset
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

# Convert NetCDF files to GeoTIFF

##########################################################################
##########################################################################

#create new folder for temporal files
temp_folder = os.path.join(cwd, "temporal_files")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    
# Get the list of NetCDF files for HDD
nc_filesHDD = files_table[files_table["data_type"] == "HDD"]["file_path"].tolist()
# Get the list of NetCDF files for CDD (if needed)
nc_filesCDD = files_table[files_table["data_type"] == "CDD"]["file_path"].tolist()

# Loop over each file
for nc_file in nc_filesHDD:
    # Load NetCDF dataset
    ds = xr.open_dataset(nc_file)

    # Select the variable of interest
    da = ds["HDD_total"]

    # Rename coordinates if needed
    da = da.rename({"longitude0": "x", "latitude0": "y"})

    # Set CRS
    da.rio.write_crs("EPSG:4326", inplace=True)

    # Define output filename
    #save in temporal folder
    nc_file_name = os.path.basename(nc_file)
    temp_nc_file = os.path.join(temp_folder, nc_file_name)
    base_name = os.path.splitext(temp_nc_file)[0]
    output_file = f"{base_name}_raster.tif"

    # Export to GeoTIFF
    da.rio.to_raster(output_file)

      
#%%

##########################################################################
##########################################################################

# BOUNDARIES FOR SPATIAL ANALYSIS

##########################################################################
##########################################################################

# Load country boundaries
shpfilename = shapereader.natural_earth(
    resolution='110m',  # use '10m' for higher detail
    category='cultural',
    name='admin_0_countries'
)
countries = gpd.read_file(shpfilename)

# Dissolve by continent
continents = countries.dissolve(by="SUBREGION", as_index=False) #"REGION_UN" "CONTINENT" "SUBREGION"

#go to temp_folder
os.chdir(temp_folder)

# Save to shapefile
output_dir = "SUBREGION_shapefiles"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "SUBREGION.shp")
continents.to_file(output_path)

      
#%%

###plot map
# Set up the map projection and plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_title("World Countries (Admin Level: Subregion)")

# Add the country borders from the shapefile
continents.plot(ax=ax, facecolor='whitesmoke', edgecolor='grey', linewidth=0.5, transform=ccrs.PlateCarree())

# Add Cartopy features for context (optional)
#ax.coastlines(edgecolor='grey')
#ax.gridlines(draw_labels=True)

# --- Add subregion names at centroid ---
for idx, row in continents.iterrows():
    if pd.notnull(row['SUBREGION']):
        centroid = row['geometry'].centroid
        ax.text(
            centroid.x, centroid.y, row['SUBREGION'],
            fontsize=8, ha='center', color='darkblue',transform=ccrs.PlateCarree(), alpha=1
        )

plt.tight_layout()
plt.show()

      
#%%

# Get the list of unique continents
continent_list = continents["SUBREGION"].unique().tolist()

# Sort for readability (optional)
continent_list.sort()

# Print the list
print("List of continents in the dataset:")
print(continent_list)

      
#%%

##########################################################################
##########################################################################

# set the mean file names for each scenario

##########################################################################
##########################################################################

#go to temp_folder
os.chdir(temp_folder)

mean1 = "HDD_historical_mean_v1_raster.tif"

mean15 = "HDD_scenario15_mean_v1_raster.tif"

mean20 = "HDD_scenario20_mean_v1_raster.tif"

      
#%%

##########################################################################
##########################################################################

# FINAL SPATIAL ANALYSIS - CDD maps and boxplots

##########################################################################
##########################################################################

# --- Load subregion shapefile ---
shapefile_path = "SUBREGION_shapefiles/SUBREGION.shp"
subregions = gpd.read_file(shapefile_path)

# --- Raster files and titles ---
raster_files = [
    ("a, Scenario 1.0°C (2006-2016)", mean1),
    ("b, Scenario 1.5°C", mean15),
    ("c, Scenario 2.0°C", mean20),
]
"""
# --- Prepare colormap with white for 0 ---
cmap = cm.get_cmap('YlGnBu').copy()
cmap.set_under('white')
vmin, vmax = 0, 5000
norm = colors.Normalize(vmin=vmin, vmax=vmax)
"""
# --- Prepare discrete colormap with 6 intervals ---
vmin, vmax = 0, 5000
bounds = np.linspace(vmin, vmax, 21)  # 6 intervals = 7 edges
cmap = cm.get_cmap('plasma_r', 20)      # 6 discrete colors YlGnBu  YIGnBu 'crest' PuBu
cmap = ListedColormap(cmap(np.arange(20)))
#cmap.set_under('white')             # white for values < vmin
norm = BoundaryNorm(boundaries=bounds, ncolors=20)

# --- Subregions to analyze ---
selected_continents = [
    'Australia and New Zealand', 'Caribbean', 'Central America', 'Central Asia', 'Eastern Africa',
    'Eastern Asia', 'Eastern Europe', 'Melanesia', 'Middle Africa', 'Northern Africa',
    'Northern America', 'Northern Europe', 'South America', 'South-Eastern Asia', 'Southern Africa',
    'Southern Asia', 'Southern Europe', 'Western Africa', 'Western Asia', 'Western Europe'
]
gdf1 = subregions[subregions['SUBREGION'].isin(selected_continents)].reset_index(drop=True)

# --- Extract raster values by subregion ---
long_df = pd.DataFrame()
for title, raster_path in raster_files:
    stats = zonal_stats(
        vectors=gdf1,
        raster=raster_path,
        stats=None,
        raster_out=True,
        all_touched=True,
        nodata=None
    )

    for i, item in enumerate(stats):
        values = item["mini_raster_array"].compressed()
        df = pd.DataFrame({
            "value": values,
            "region": gdf1.iloc[i]["SUBREGION"],
            "raster": title
        })
        long_df = pd.concat([long_df, df], ignore_index=True)
        
        
# --- Create the 3x2 figure layout ---
fig = plt.figure(figsize=(12, 12))  # square layout for symmetry
gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 0.6], wspace=0.14, hspace=0.14)

# Loop through maps and boxplots
for i, (title, raster_path) in enumerate(raster_files):
    
    # --- MAP (left column) ---
    ax_map = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        data_masked = data.copy()
        data_masked[data_masked == 0] = -9999
        show(data_masked, ax=ax_map, transform=src.transform, cmap=cmap, norm=norm)

    subregions.boundary.plot(ax=ax_map, edgecolor='white', linewidth=0.8, transform=ccrs.PlateCarree())
    ax_map.set_title(title, fontsize=9, loc='left')
    #ax_map.coastlines(color='white',linewidth=0.5)
    ax_map.set_global()
    
    cmap.set_under('white')
    
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    

    # --- BOXPLOT (right column) ---
    ax_box = fig.add_subplot(gs[i, 1])
    df = long_df[long_df["raster"] == title]

    # Compute median values for each region
    region_medians = df.groupby("region")["value"].median()

    # Normalize median values and map to turbo colors
    norm_box = colors.Normalize(vmin=region_medians.min(), vmax=region_medians.max())
    turbo = cm.get_cmap('plasma_r',20)
    palette = {region: turbo(norm_box(median)) for region, median in region_medians.items()}

    sns.boxplot(
        x="region", y="value", data=df,
        ax=ax_box,
        palette=palette,  # Apply dynamic coloring based on median
        linewidth=0.5,
        flierprops=dict(marker='x', markerfacecolor='black', markeredgecolor='black', markersize=1, linestyle='none')
    )

    ax_box.set_title(f"{title} - Subregion distribution", fontsize=9, loc='left')
    ax_box.set_ylabel("HDD", fontsize=8)
    ax_box.set_ylim(0, 17500)
    ax_box.tick_params(axis='y', labelsize=8)
    ax_box.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='grey', alpha=0.6)
    ax_box.tick_params(axis='both', which='both', length=0)
    
    # Remove axis box lines (spines)
    for spine in ax_box.spines.values():
        spine.set_visible(False)
    

    # Remove x-axis labels for upper plots
    if i < 2:
        ax_box.set_xticklabels([])
        ax_box.set_xlabel("")
    else:
        ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=90, fontsize=7)
        ax_box.set_xlabel("Subregion", fontsize=8)

# --- Colorbar below only the maps (column 1) ---
cbar_ax = fig.add_axes([0.13, 0.08, 0.44, 0.015])  # only under left column
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Heating Degree Days (HDD)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Set ticks every 500
cbar.set_ticks(np.arange(0, 5001, 500))

# Remove colorbar ticks
cbar.ax.tick_params(
    bottom=False,
    labelbottom=True,
    top=False,
    labeltop=False
)

for spine in cbar.ax.spines.values():
    spine.set_visible(False)
    
# --- Final layout adjustments ---
plt.tight_layout(rect=[0, 0.08, 1, 1])

#save in cwd + output folder
output_folder = cwd + "/output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)

#plt.savefig("Figure1_HDD_maps_and_boxplots.svg", format='svg', dpi=600, bbox_inches='tight')
plt.savefig("Figure1_HDD_maps_and_boxplots.jpg", format='jpg', dpi=600, bbox_inches='tight')

plt.show()

      
#%%