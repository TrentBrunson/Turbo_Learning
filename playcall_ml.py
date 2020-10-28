#%%
# Import dependencies
import pandas as pd

#DB management
from sqlalchemy import create_engine
from config import db_password

# ML dependencies
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from scipy import stats

#%%
# Initialize framework to import clean df from postgres db
#db_string = 'f”postgres://postgres:{postgres}@127.0.0.1:5432/CFB_DB'
#engine = create_engine(db_string)
#pd.read_sql_table('table_name', 'postgres:///db_name')

#%%
# Import clean dataset from raw file
data_df = pd.read_csv('Resources/Texas_combined_final.csv')

#%%
# Creating a 'time remaining in quarter' column 
# This allows us to bin easily but also treat "time" as a continuous feature more easily should we choose
def time_convert(x):
    h,m,s = map(int,x.split(':'))
    return (m*60)+s

data_df['seconds_in_quarter_remaining'] = data_df.clock.apply(time_convert)

#%%
# Utilizing 'time remaining in quarter' column to generate 'time remaining in half' a criteria 
# which should be a better indicator of run vs pass as strategies change not at the end of the
# quarter but at the half.
data_df['seconds_in_half_remaining'] = data_df['seconds_in_quarter_remaining']
data_df.loc[data_df.quarter== 1,'seconds_in_half_remaining':]*=2
data_df.loc[data_df.quarter== 3,'seconds_in_half_remaining':]*=2


#%%
# Create 'half' feature using the 'quarter' column
# This allows easier calculation for "time left in half" but also provides us with another feature
data_df.loc[data_df.quarter == 1, 'half'] = 1
data_df.loc[data_df.quarter == 2, 'half'] = 1
data_df.loc[data_df.quarter == 3, 'half'] = 2
data_df.loc[data_df.quarter == 4, 'half'] = 2
# Quick workaround to account for OT
data_df.loc[data_df.quarter == 5, 'half'] = 3
data_df.loc[data_df.quarter == 6, 'half'] = 3

#%%
# Bucketing 'time remaining in half'
# In deciding bin size: the two-minute mark is likely where strategies are going to change, 
# so a bin size larger than that would likely obscure the potential effect of the feature.
time_remaining_bins = [-1, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800]
labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min', '10-12 min', '12-14 min', '14-16 min', '16-18 min', '18-20 min', '20-22 min', '22-24 min', '24-26 min', '26-28 min', '28-30 min']
data_df['time_remaining_binned'] = pd.cut(data_df['seconds_in_half_remaining'], bins=time_remaining_bins, labels=labels)

#%%
# Convert all pass outcomes to "pass" to create a true binary outcome
data_df.loc[data_df['type'].str.contains('Pass'), 'type'] = 'Pass'

#%%
# Drop Output label into separate object
output_df = data_df.type
features_df = data_df[['year', 'offenseabbr','defenseabbr', 'texscore','oppscore','quarter','down','distance','yardline','half', 'time_remaining_binned']]

#%%
# Check categorical columns of feature df and check the number of unique values in each column
data_cat = features_df.dtypes[features_df.dtypes == "object"].index.tolist()
data_cat.append('time_remaining_binned')
features_df[data_cat].nunique()

#%%
# Encode categorical data

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(features_df[data_cat]))

# Add the encoded variable names to the DataFrame
encode_df.columns = enc.get_feature_names(data_cat)

#%%
# Merge encoded DataFrame back into the original feature df and drop original object/category columns
encoded_features_df = features_df.merge(encode_df,left_index=True, right_index=True)
encoded_features_df = encoded_features_df.drop(data_cat,1)

#%%
#Split into testing and training groups
X = encoded_features_df
y = output_df
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)

#%%
#Scale data based on testing group and apply to testing and training
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

#%%
# 1st Random Forest Model
# Create a random forest classifier.
rf_model = RandomForestClassifier(n_estimators=128, random_state=42)

# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
print(f" Random forest predictive accuracy: {accuracy_score(y_test,y_pred):.4f}")

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
