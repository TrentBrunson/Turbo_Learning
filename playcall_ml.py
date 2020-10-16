#%%
# Import dependencies
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_blobs
import sklearn as skl
import tensorflow as tf

#%%
# Import clean dataset
data_df = pd.read_csv('./Resources/Texas_combined.csv')

#%%
# Check categorical columns
data_cat = data_df.dtypes[data_df.dtypes == "object"].index.tolist()

#%%
# Check the number of unique values in each column
data_df[data_cat].nunique()

#%%
# Drop Output label into separate object


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
