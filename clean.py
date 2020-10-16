#%%
# load libraries
import os
import csv
import pandas as pd
import numpy as np
import glob
from sqlalchemy import create_engine
from config import db_password

# %%
# read over data folder and open each csv

# set dir location
year = 2013  # set range of years to look at for loop
dirYear = 'data/2013'

# make a location to store only tex data by year
path = "data/tex"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed; it already exists." % path)
else:
    print ("Successfully created the directory %s" % path)

#%%
# make this into f(x) and call in loop iterating through year
year_data = pd.DataFrame()
for f in glob.glob("data/2013/*.csv"):
    df = pd.read_csv(f)
    year_data = year_data.append(df,ignore_index=True)
year_data.shape

# get right columns for each year
year_data = year_data[['gameId', 'year', 'week', 'homeAbbr', 'awayAbbr', 'offenseAbbr', 'defenseAbbr', 'homeScore', 'awayScore', 'quarter', 'clock', 'type', 'down', 'distance', 'yardLine', 'yardsGained']]

# get only Texas data for both home and away games
Tex_data = year_data[(year_data.homeAbbr == 'TEX') | (year_data.awayAbbr == 'TEX')]

# write to csv
Tex_data.to_csv('data/tex/Tex_data2013', sep= ',', index= False)
Tex_data
# %%
# check out values in columns
column_list = list(Tex_data.columns.tolist())
Tex_data[column_list].nunique() 
# %%
# see list of values in play type column
sorted(Tex_data.type.unique().tolist())
# %%
# get pass and rush plays only

# %%

# %%

# %%
Tex_data.shape
# %%

#%%
year_data.shape
# %%
list(year_data)

# iterate through loop for year

# %%
# read over data folder and open each csv
# merge into single csv for data cleaning
csv_destdir = ('/data/')
AllData = "data/AllData.csv"
allFiles = glob.glob("**/*.csv", recursive = True)
data = []

for i, fname in enumerate(allFiles):
    with open(fname, 'r') as infile:
        read_lines = infile.readlines()
        read_lines = ['{0},{1}'.format(line.rstrip('\n'),fname) for line in read_lines]
        data += read_lines

with open(AllData, 'w') as outfile:
    outfile.write('\n'.join(data))
# adds file name to last column 2014+
# keeps reading header

#%%
# merge data by year
path = 'data/2013'
dfs = glob.glob('path/*.csv')

result = pd.concat([pd.read_csv(df) for df in dfs], ignore_index=True)

result.to_csv('path/merge.csv', ignore_index=True)

# %%
# not all years are the same
# iterating the columns to get column names

# read each year separately & clean columns
# then merge complete years together
# %%
# evaluate data
print(
    f'Tex DB data types: {Tex_data.dtypes}\n\n'
    f'Nulls: {Tex_data.isnull().sum}\n\n'
    f'Not Nulls: {Tex_data.notnull().sum()}\n\n'
)

for column in Tex_data.columns:
    print(f'Column {column} has {Tex_data[column].isnull().sum()} null values.')

print(f'Duplicate entries: {Tex_data.duplicated().sum()}')

# %%
