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
start_year = 2015
end_year = 2018
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
# get pass and rush plays only
# no regex, runs faster
df_Tex_plays = Tex_data[(Tex_data.type.str.contains('Pass', regex=False)) | (
    Tex_data.type.str.contains('Rush', regex=False))]

# write to csv
df_Tex_plays.to_csv('data/tex/Tex_data2013.csv', sep= ',', index= False)
df_Tex_plays
# %%
# check out values in columns
column_list = list(Tex_data.columns.tolist())
Tex_data[column_list].nunique() 
# %%
# see list of values in play type column
sorted(Tex_data.type.unique().tolist())
# %%
df_Tex_plays.shape
# %%
# make this into f(x) and call in loop iterating through year
def combine_years(year): 
    path_string = f"data/{year}/*.csv"
    year_data = pd.DataFrame()
    for f in glob.glob(path_string):
        df = pd.read_csv(f)
        year_data = year_data.append(df,ignore_index=True)
    # year_data.shape

    # get right columns for each year
    year_data = year_data[['gameId', 'year', 'week', 'homeAbbr', 'awayAbbr', 'offenseAbbr', 'defenseAbbr', 'homeScore', 'awayScore', 'quarter', 'clock', 'type', 'down', 'distance', 'yardLine', 'yardsGained']]

    # get only Texas data for both home and away games
    Tex_data = year_data[(year_data.homeAbbr == 'TEX') | (year_data.awayAbbr == 'TEX')]
    
    # get pass and rush plays only
    # no regex, runs faster
    df_Tex_plays = Tex_data[(Tex_data.type.str.contains('Pass', regex=False)) | (
        Tex_data.type.str.contains('Rush', regex=False))]

    # write to csv
    # df_Tex_plays.to_csv('data/tex/Tex_data2013', sep= ',', index= False)
    
    csv_path = f"data/tex/Tex_data{year}.csv"
    df_Tex_plays.to_csv(csv_path, sep= ',', index= False)
    return df_Tex_plays

# %%
# create for loop to iterate through rows
for i in range (start_year, end_year):
    combine_years(i)
# %%
path_Tex = 'data/tex'
# %%

# %%
Tex_data.shape
# %%
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
