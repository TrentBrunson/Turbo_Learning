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
# drop columns to make file size more manageable
inputFile = AllData
outputFile = 'data/selectData.csv'
# colsToRemove = [['homeId'], ['homeTeam'], ['awayId'], ['awayTeam'], ['driveIndex'], 
#             ['playIndex'], ['offenseId'], ['offenseTeam'], ['defenseId'], ['defenseTeam'], 
#             ['defenseAbbr'], ['wallclock'], ['endYardLine'], ['description']]

colsToRemove = [0,3,4,6,7,9,10,11,12,14,15,16,21,27,28]
# reverse rows to remove end first otherwise indexes keep shifting left
colsToRemove = sorted(colsToRemove, reverse=True)
rowCount = 0

with open(inputFile, 'r') as source:
    reader = csv.reader(source)
    with open(outputFile, 'w', newline='') as result:
        writer = csv.writer(result)
        for row in reader:
            rowCount += 1
            # check progress thru file
            print('\r{0}'.format(rowCount), end='')
            for colIndex in colsToRemove:
                del row[colIndex]
            writer.writerow(row)
#%%
# merge data by year
path = 'data/2013'
dfs = glob.glob('path/*.csv')

result = pd.concat([pd.read_csv(df) for df in dfs], ignore_index=True)

result.to_csv('path/merge.csv', ignore_index=True)
#%%
# make into DF
df_CFB = pd.read_csv(AllData, sep=",", error_bad_lines=False)
df_CFB
#%%
df_CFB.shape
# %%
# not all years are the same
# iterating the columns to get column names
list(df.columns)
# read each year separately & clean columns
# then merge complete years together
# %%
# evaluate data
print(
    f'CFB DB data types: {df_CFB.dtypes}\n\n'
    f'Nulls: {df_CFB.isnull().sum}\n\n'
    f'Not Nulls: {df_CFB.notnull().sum()}\n\n'
)

for column in df_CFB.columns:
    print(f'Column {column} has {df_CFB[column].isnull().sum()} null values.')

print(f'Duplicate entries: {df_CFB.duplicated().sum()}')

#%%
# Generate categorical variable list
colList = df_CFB.index.tolist()

# column_drop_list
#%%
# check number of unique values in each column
df_CFB[df_CFB].nunique()
#%%
# drop of columns
df_CFB_small = df_CFB.drop(df_CFB.columns['homeId', 'homeTeam', 'awayId',
'awayTeam', 'driveIndex', 'playIndex', 'offenseId', 'offenseTeam',
'defenseId', 'defenseTeam', 'defenseAbbr', 'wallclock', 'endYardLine',
'description'], axis = 1) 
  
df_CFB_small
#%%
# create 2 df: 1 with all TEX defense, 2nd with all TEX off (2nd not needed)

# remove columns before importing csv
with open(AllData,"r") as source:
    rdr= csv.reader(source)
    with open("result","w") as result:
        dest= csv.writer(result)
        colDel= ((
            ['homeId'], ['homeTeam'], ['awayId'], ['awayTeam'], ['driveIndex'], 
            ['playIndex'], ['offenseId'], ['offenseTeam'], ['defenseId'], ['defenseTeam'], 
            ['defenseAbbr'], ['wallclock'], ['endYardLine'], ['description']
            ) for r in rdr )
        dest.writerows(colDel)
# %%
print(dest)
dest
# %%
#  get list of play types, keep plays, drop rest