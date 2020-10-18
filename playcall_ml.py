#%%
# Import dependencies
import pandas as pd
from sqlalchemy import create_engine
from config import db_password
import matplotlib as plt
import sklearn as skl
import tensorflow as tf

#%%
# Import clean dataset from raw file
data_df = pd.read_csv('Texas_combined.csv')

#%%
# Initialize framework to import clean df from postgres db
#db_string = 'f”postgres://postgres:{postgres}@127.0.0.1:5432/CFB_DB'
#engine = create_engine(db_string)
#pd.read_sql_table('table_name', 'postgres:///db_name')

#%%
# Check categorical columns
data_cat = data_df.dtypes[data_df.dtypes == "object"].index.tolist()

#%%
# Check the number of unique values in each column
data_df[data_cat].nunique()

#%%
# Drop Output label into separate object
output_df = data_df.type
features_df = data_df.drop(columns='type')

#%%
# Encode categorical data


#%%
#Split into testing and training groups

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
