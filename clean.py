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
start_year = 2013
end_year = 2018

# make a location to store only tex data by year
path = "data/tex"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed; it already exists." % path)
else:
    print ("Successfully created the directory %s" % path)

# %%
def evaluate_data(df):
    # evaluate data
    print(
        f'Tex DB data types: {df.dtypes}\n\n'
        f'Nulls: {df.isnull().sum}\n\n'
        f'Not Nulls: {df.notnull().sum()}\n\n'
    )

    for column in df.columns:
        print(f'Column {column} has {df[column].isnull().sum()} null values.')

    print(
        f'Duplicate entries: {df.duplicated().sum()}'
        f'Dataframe shape: {df.shape}'
        )
    return
# %%
# make this into f(x) and call in loop iterating through year
def combine_years(year): 
    path_string = f"data/{year}/*.csv"
    year_data = pd.DataFrame()
    for f in glob.glob(path_string):
        df = pd.read_csv(f, error_bad_lines=False)
        year_data = year_data.append(df,ignore_index=True)

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

    # evaluate the dataframes, verify working
    print(
        f'\n---------------------------------------------------\n'
        f'            ***** For year {year} *****   '
        f'\n---------------------------------------------------\n'
    )
    evaluate_data(Tex_data)

    evaluate_data(df_Tex_plays)

    return df_Tex_plays

# %%
# create for loop to iterate through rows
for i in range (start_year, end_year+1):
    combine_years(i)

# %%
# combine tex folder files into one and save in main directory
df_combined = pd.DataFrame()
for f in glob.glob("data/tex/*.csv"):
    df = pd.read_csv(f)
    df_combined = df_combined.append(df,ignore_index=True)

df_combined.to_csv('Texas_combined.csv', index=False)
df_combined.shape
# %%
