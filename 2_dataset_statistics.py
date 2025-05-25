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
Descriptive statistics of the CDD and HDD dataset

(1) Table with summary of the dataset 

"""   

stats_table = files_table[["data_type","file_name"]].copy()

# Add columns for statistics
stats_table["min"] = np.nan
stats_table["max"] = np.nan
stats_table["mean"] = np.nan
stats_table["std"] = np.nan
stats_table["median"] = np.nan
stats_table["90th_percentile"] = np.nan
stats_table["10th_percentile"] = np.nan
stats_table["range"] = np.nan

# Loop through each file to calculate statistics
for index, row in stats_table.iterrows():
    # Get the file path from files_table based on the file name
    file_name = row["file_name"]
    print(f"Processing file: {file_name}")
    file_path = files_table.loc[files_table["file_name"] == file_name, "file_path"].values[0]
    data = Dataset(file_path, mode='r', format="NetCDF")
    
    # Assuming the variable of interest is 'CDD_total' or 'HDD_total'
    value = "CDD_total" if row["data_type"] == "CDD" else "HDD_total"
    
    data_value = data.variables[value][:]
    
    # Flatten and remove NaNs for statistics
    flat_data = np.array(data_value).flatten()
    flat_data = flat_data[~np.isnan(flat_data)]
    
    stats_table.at[index, "min"] = np.min(flat_data)
    stats_table.at[index, "max"] = np.max(flat_data)
    stats_table.at[index, "mean"] = np.mean(flat_data)
    stats_table.at[index, "std"] = np.std(flat_data)
    stats_table.at[index, "median"] = np.median(flat_data)
    stats_table.at[index, "90th_percentile"] = np.percentile(flat_data, 90)
    stats_table.at[index, "10th_percentile"] = np.percentile(flat_data, 10)
    stats_table.at[index, "range"] = f"{stats_table.at[index, 'min']} - {stats_table.at[index, 'max']}"

    print(f"Statistics for {file_name}:", " - done!")

    data.close()

# Show the statistics table
print("Statistics table:")
print(stats_table)


  
#%%

#save the statistics table to a CSV file in cwd + OUTPUT
output_folder = cwd + "/output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, "CDD_HDD_statistics.csv")
stats_table.to_csv(output_file, index=False)
print(f"Statistics table saved to {output_file}")

print("All done!")


