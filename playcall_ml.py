#%%
# Import dependencies
import pandas as pd
from sqlalchemy import create_engine
from config import db_password
import matplotlib as plt
import sklearn as skl
import tensorflow as tf
from scipy import stats

#%%
# Initialize framework to import clean df from postgres db
#db_string = 'f”postgres://postgres:{postgres}@127.0.0.1:5432/CFB_DB'
#engine = create_engine(db_string)
#pd.read_sql_table('table_name', 'postgres:///db_name')

#%%
# Import clean dataset from raw file
data_df = pd.read_csv('Resources/Texas_combined.csv')

#%%
# Creating a 'time remaining in quarter' column 
def time_convert(x):
    m,s = map(int,x.split(':'))
    return (m*60)+s

data_df['seconds_in_quarter_remaining'] = data_df.clock.apply(time_convert)

#%%
# Utilizing 'time remaining in quarter' column to generate 'time remaining in half' a criteria that should be a better indicator of run vs pass
data_df['seconds_in_half_remaining'] = data_df['seconds_in_quarter_remaining']
data_df.loc[data_df.quarter== 1,'seconds_in_half_remaining':]*=2
data_df.loc[data_df.quarter== 3,'seconds_in_half_remaining':]*=2

#%%
# Create 'half' feature using the 'quarter' column
data_df.loc[data_df.quarter == 1, 'half'] = 1
data_df.loc[data_df.quarter == 2, 'half'] = 1
data_df.loc[data_df.quarter == 3, 'half'] = 2
data_df.loc[data_df.quarter == 4, 'half'] = 2

#%%
# Bucketing 'time remaining in half'
time_remaining_bins = [0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800]
labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min', '10-12 min', '12-14 min', '14-16 min', '16-18 min', '18-20 min', '20-22 min', '22-24 min', '24-26 min', '26-28 min', '28-30 min']
data_df['time_remaining_binned'] = pd.cut(data_df['seconds_in_half_remaining'], bins=time_remaining_bins, labels=labels)

#%%
# Convert all pass outcomes to "pass" to create a true binary outcome
data_df.loc[data_df['type'].str.contains('Pass'), 'type'] = 'Pass'

#%%
# Drop Output label into separate object
output_df = data_df.type
features_df = data_df[['year','week','homeAbbr','awayAbbr', 'offenseAbbr', 'defenseAbbr', 'homeScore','awayScore','quarter','down','distance','yardLine','half', 'time_remaining_binned']]

#%%
# Check categorical columns of feature df and check the number of unique values in each column
data_cat = features_df.dtypes[features_df.dtypes == "object"].index.tolist()
data_cat.append('time_remaining_binned')
features_df[data_cat].nunique()

#%%
# Encode categorical data


#%%
#Split into testing and training groups
# Try and 50/50 and then 50/50 the testing data again to then apply to the final validationb. Don't let model see validation until the very end.

#%%
#Scale data based on testing group and apply to testing and training
# Create a StandardScaler instance


# Fit the StandardScaler


# Scale the data


#%%
# Create model architecture

#%%
# Add density to model architecture


#%%
#Compile model


#%%
# Fit the model to the training data


#%%
# Evaluate the model using the test data
