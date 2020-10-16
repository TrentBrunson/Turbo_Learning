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
