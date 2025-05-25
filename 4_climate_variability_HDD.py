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

mean_10 = "HDD_historical_mean_v1_raster.tif"
median_10 ="HDD_historical_median_v1_raster.tif"
perc10_10  ="HDD_historical_quantile10_v1_raster.tif"
perc90_10 ="HDD_historical_quantile90_v1_raster.tif"
stdv_10 = "HDD_historical_stdv_v1_raster.tif"

mean_15 = "HDD_scenario15_mean_v1_raster.tif"
median_15 ="HDD_scenario15_median_v1_raster.tif"
perc10_15  ="HDD_scenario15_quantile10_v1_raster.tif"
perc90_15 ="HDD_scenario15_quantile90_v1_raster.tif"
stdv_15 = "HDD_scenario15_stdv_v1_raster.tif"

mean_20 = "HDD_scenario20_mean_v1_raster.tif"
median_20 ="HDD_scenario20_median_v1_raster.tif"
perc10_20  ="HDD_scenario20_quantile10_v1_raster.tif"
perc90_20 ="HDD_scenario20_quantile90_v1_raster.tif"
stdv_20 = "HDD_scenario20_stdv_v1_raster.tif"


raster_files_10 = [perc10_10, median_10,  perc90_10]
raster_files_15 = [perc10_15, median_15,  perc90_15]
raster_files_20 = [perc10_20, median_20,  perc90_20]
      

#%%

##########################################################################
##########################################################################

# FINAL SPATIAL ANALYSIS - Climate Variability in HDD under 3 scenarios (1ºC, 1.5ºC, 2ºC)

##########################################################################
##########################################################################

# --- Select specific continents to include ---
selected_continents = ['Australia and New Zealand', 'Caribbean', 'Central America', 'Central Asia', 'Eastern Africa', 'Eastern Asia', 'Eastern Europe', 'Melanesia', 'Middle Africa', 'Northern Africa', 'Northern America', 'Northern Europe', 'South America', 'South-Eastern Asia', 'Southern Africa', 'Southern Asia', 'Southern Europe', 'Western Africa', 'Western Asia', 'Western Europe']  # ← Customize this list

# --- Filter the GeoDataFrame ---
gdf1 = continents[continents['SUBREGION'].isin(selected_continents)].reset_index(drop=True)

#go to temp_folder
os.chdir(temp_folder)

rcParams['font.family'] = 'Arial'

raster_groups = [
    ("a, Scenario 1.0°C (2006-2016)", raster_files_10),
    ("b, Scenario 1.5°C", raster_files_15),
    ("c, Scenario 2.0°C", raster_files_20)
]

fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True, sharey=True)  # 3 rows, 1 column

for ax, (label, raster_files) in zip(axes, raster_groups):
    long_df = pd.DataFrame()
    for raster in raster_files:
        print(f"Processing: {raster}")
        stats = zonal_stats(
            vectors=gdf1,
            raster=raster,
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
                "raster": raster.split("_GIS_")[0]
            })
            long_df = pd.concat([long_df, df], ignore_index=True)

    n_rasters = long_df["raster"].nunique()
    palette = sns.color_palette("Blues", n_colors=n_rasters)
    sns.boxplot(x="region", y="value", 
                hue="raster", data=long_df, palette=palette, ax=ax,
                flierprops=dict(marker='x', markerfacecolor='black', 
                markeredgecolor='black', markersize=1, linestyle='none'),
                boxprops=dict(edgecolor='black'),
                 whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='black'),
                linewidth=0.5)
    
    ax.set_title(f"{label}",loc='left')
    ax.set_ylabel("Heating Degree Days (HDD)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylim(0, 18000)
    #ax.legend_.remove()  # Remove legend from individual plots

    #Add horizontal dashed grid lines
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add custom legend to each subplot
    handles, _ = ax.get_legend_handles_labels()
    custom_labels = ["10th percentile", "Median", "90th percentile"]
    ax.legend(
        handles,
        custom_labels,
        title="",
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        fontsize=10,
        title_fontsize=10
    )

#save in cwd + output folder
output_folder = cwd + "/output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)
plt.savefig("Figure3_HDD_climate_variability.jpg", format='jpg', dpi=600, bbox_inches='tight')


# Adjust layout to make room for the bottom legend
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.show()







