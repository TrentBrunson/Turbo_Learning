#%%
import pandas as pd

#%%
#Bring in combined csv
cfb_df = pd.read_csv('Texas_combined_small.csv')

#%%
#Create a unique ID with the ID and indexes provided
cols = ['gameId', 'driveIndex', 'playIndex']
cfb_df['playId'] = cfb_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

#%%
#Drop duplicate rows from dataset
cfb_df = cfb_df.drop_duplicates()

#%%
#Use newly created ID as the index of the dataset
cfb_df = cfb_df.set_index("playId", drop = True)
cfb_df.index.name = "playID"

#%%
#Replaces Bowl in Week column to integer 16
cfb_df = cfb_df.replace("Bowl",16)

#%%
# Creates game clock in mm:ss format
cfb_df['clock'] = pd.to_datetime(cfb_df['clock'], format='%M:%S').dt.time

#%%
#Creates a new data frame to store score data for transformation in SQL
cfb_score_df = cfb_df[["homeAbbr","awayAbbr","homeScore","awayScore"]].copy()

#Drops columns used for unique ID and scoreing
cfb_df = cfb_df.drop(["driveIndex","playIndex","homeAbbr","awayAbbr","homeScore","awayScore"], axis=1)

#%%
#Saves new tables as csv files
cfb_df.to_csv('Resources/Texas_combined_formatted.csv', index=True)
cfb_score_df.to_csv('Resources/Texas_scores.csv', index=True)