#!/sr/bin/env python

###############################################################################
# Program : 
# Authors  : Jesus Lizana
# Date	  : 25 April 2025
# Purpose : Work with NetCDF4 files
##############################################################################

#%%



#%%
import os
import glob

print("....importing libraries")

import netCDF4
from netCDF4 import Dataset,num2date # or from netCDF4 import Dataset as NetCDFFile

import xarray

import numpy as np
import numpy.ma as ma
import pandas as pd

import dateutil.parser

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend

from statsmodels.distributions.empirical_distribution import StepFunction

import matplotlib.pyplot as plt    
import seaborn as sns


###########################################################

#Basic function for netCDF4 files: 
    
###########################################################

def print_variables(data):
    """prints variables in netcdf Dataset
    only works if have assigned attributes!"""
    for i in data.variables:
        print(i, data.variables[i].units, data.variables[i].shape)
        
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

"""

Data directory 

(b) Select one file to read from the data directory

"""   

##########################################################################
##########################################################################

#SELECT DATA AND VARIABLE TO READ

#Select file name
file_name = "CDD_historical_mean_v1"
variable = "Cooling Degree Days (CDD)" #OR "Heating Degree Days (HDD)"

#variables
lat = "latitude0"
long = "longitude0"
value = "CDD_total" #OR "HDD_total"

#get palth from table
file_path = files_table[files_table["file_name"] == file_name]["file_path"].values[0]	
print("File path:", file_path)

#*.nc file     
file = file_path



#%%   

"""
Working with NetCDF4 files of CDD and HDD:

(1) Read, analysis and visualise netCDF4 file. 

"""   

####################################################

##code 1 - read netCDF4 file and close

####################################################

#READ - take core of long system (from 0-360 or -180-180)
data = Dataset(file, mode='r', format="NetCDF")

#summary of characteristics
print_variables(data)
print("")

#all details
print(data)



#%%   

####################################################

##code 2 - read netCDF4 file, and analysis data

####################################################

#Open using netCDF library: 
    
data = Dataset(file, mode='r', format="NetCDF")

data_value = data.variables[value][:]
data_lat = data.variables[lat][:]
data_long = data.variables[long][:]

print(data_value.max())
print(data_value.mean())
print(data_value.min())


#%%   

####################################################

##code 3 - statistics

####################################################

# Flatten and remove NaNs for histogram
flat_data = np.array(data_value).flatten()
flat_data = flat_data[~np.isnan(flat_data)]

plt.figure(figsize=(8, 5))
plt.hist(flat_data, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of values')
plt.xlabel(variable)
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(bottom=0)
plt.xlim(left=0)  # Set x-axis limit to start from 0
plt.tight_layout()
plt.show()

mean_val = np.mean(flat_data)
std_val = np.std(flat_data)
median_val = np.median(flat_data)
percentile_90 = np.percentile(flat_data, 90)
percentile_10 = np.percentile(flat_data, 10)
min_val = np.min(flat_data)
max_val = np.max(flat_data)

print("Statistics for data_value:")
print(f"Mean: {mean_val:.2f}")
print(f"Standard deviation: {std_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"90th percentile: {percentile_90:.2f}")
print(f"10th percentile: {percentile_10:.2f}")
print(f"Range: min={min_val:.2f}, max={max_val:.2f}")



#%%   

####################################################

##code 4 - visualise map with cartopy

####################################################

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

# --- Fix longitudes from 0-360 to -180 to 180 if needed ---
data_long_fixed = np.where(data_long > 180, data_long - 360, data_long)

# --- Prepare data for plotting (2D or extract 1st level of 3D) ---
if data_value.ndim == 3:
    plot_data = data_value[0, :, :]
elif data_value.ndim == 2:
    plot_data = data_value
else:
    raise ValueError("data_value must be a 2D or 3D array")

lons, lats = np.meshgrid(data_long_fixed, data_lat)

# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
#ax.coastlines(color='white', linewidth=0.8)
ax.add_feature(cfeature.BORDERS, color='white',linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor="#FFFFFF")

for spine in ax.spines.values():
    spine.set_visible(False)

# --- Color map and levels ---
vmin, vmax = 0, 3500
levels = np.linspace(vmin, vmax, 15)  # 14 bins
cmap = get_cmap('turbo', len(levels)-1)
norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N)

# --- Plot with consistent normalization ---
mesh = ax.contourf(lons, lats, plot_data, levels=levels, 
                   transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, extend='max')

# --- Colorbar ---
cb = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=50, ax=ax)
cb.set_label(variable, fontsize=12)
cb.set_ticks(np.arange(0, 3501, 500))
cb.outline.set_visible(False)
cb.ax.tick_params(left=False, right=False, top=False, bottom=False)  # Hide ticks and labels

plt.title('Global gridded map of ' + file_name)
plt.tight_layout()

#save in folder cwd + output as jpg
output_folder = cwd + "/OUTPUT"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = os.path.join(output_folder, file_name + ".jpg")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")

# Show the plot
plt.show()




#%% 