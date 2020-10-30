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
# set end and start values at the beginning
# no need to make this a user input
start_year = 2016
end_year = 2018

# set up data folder to temp hold intermediate data processing step
path = "dataRaw/tex"  # make a location to store only tex data by year

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed; it already exists." % path)
else:
    print ("Successfully created the directory %s" % path)
# %%
def evaluate_data(df):
    # evaluate data by year; check data types and find nulls if any
    print(
        f'Tex DB data types: {df.dtypes}\n\n'
        f'Nulls: {df.isnull().sum}\n\n'
        f'Not Nulls: {df.notnull().sum()}\n\n'
    )

    # after checking nulls in the dataframe, is any appear, check it by column
    for column in df.columns:
        print(f'Column {column} has {df[column].isnull().sum()} null values.')

    # check for duplicate entries and see the dataframe shape
    print(
        f'Duplicate entries: {df.duplicated().sum()}'
        f'Dataframe shape: {df.shape}'
        )
    return
# %%
# make this into f(x) and call in loop iterating through year
# f(x) takes in year value
def combine_years(year): 
    path_string = f"dataRaw/{year}/*.csv"
    year_data = pd.DataFrame()
    for f in glob.glob(path_string):
        df = pd.read_csv(f, error_bad_lines=False)
        year_data = year_data.append(df,ignore_index=True)

    # get right columns for each year - dropping doesn't work because 2014 has issues
    # specfically found week 6 to be problematic; dropped one line of data & it works
    year_data = year_data[['gameId', 'driveIndex', 'playIndex', 'year', 'week', 'homeAbbr', 'awayAbbr', 
    'offenseAbbr', 'defenseAbbr', 'homeScore', 'awayScore', 'quarter', 'clock', 
    'type', 'down', 'distance', 'yardLine', 'yardsGained']]

    # get only Texas data for both home and away games
    Tex_data = year_data[(year_data.homeAbbr == 'TEX') | (year_data.awayAbbr == 'TEX')]
    
    # get pass and rush plays only; no regex, so it runs faster
    df_Tex_plays = Tex_data[(Tex_data.type.str.contains('Pass', regex=False)) | (
        Tex_data.type.str.contains('Rush', regex=False))]
    
    csv_path = f"dataRaw/tex/Tex_data{year}.csv"
    df_Tex_plays.to_csv(csv_path, sep= ',', index= False)

    # evaluate the dataframes, verify working
    print(
        f'\n---------------------------------------------------\n'
        f'            ***** For year {year} *****   '
        f'\n---------------------------------------------------\n'
    )

    # make sure that when pulling in only Texas data, it was fine
    evaluate_data(Tex_data)

    # verfying that filtering down to only offensive plays worked
    evaluate_data(df_Tex_plays)

    return df_Tex_plays
# %%
# create for loop to iterate through the different folders organized by year
for i in range (start_year, end_year+1):
    combine_years(i)
# %%
# combine tex folder files into one and save in main directory 
# that will be published to github
df_combined = pd.DataFrame()  # initialize dataframe
for f in glob.glob("dataRaw/tex/*.csv"):
    df = pd.read_csv(f)
    df_combined = df_combined.append(df,ignore_index=True)

df_combined.to_csv('Texas_combined_small.csv', index=False)
df_combined.shape # check that dataframe shape makes sense after years are combined
# %%
