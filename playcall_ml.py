#%%
# Import dependencies
import pandas as pd
from datetime import datetime
import numpy as np

# Python SQL toolkit and Object Relational Mapper
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from config import DB_String
#from config import DB_String

# ML dependencies
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import pickle
from joblib import dump, load

#%%
engine = create_engine(DB_String)

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

#%%
# We can view all of the classes that automap found
Base.classes.keys()

#%%
# Create our session (link) from Python to the DB
session = Session(engine)

#%%
data_df = pd.read_sql_table('tex_combined_final', engine)

#%%
data_df = data_df.set_index("playid", drop = True)
data_df.index.name = "playID"

#%%
data_df['clock'] = data_df['clock'].astype(str)

#%%
# Creating a 'time remaining in quarter' column 
# This allows us to bin easily but also treat "time" as a continuous feature more easily should we choose
def time_convert(x):
    h,m,s = map(int,x.split(':'))
    return (h*360)+(m*60)+s

data_df['seconds_in_quarter_remaining'] = data_df.clock.apply(time_convert)

#%%
# Utilizing 'time remaining in quarter' column to generate 'time remaining in half' a criteria 
# which should be a better indicator of run vs pass as strategies change not at the end of the
# quarter but at the half.
data_df['seconds_in_half_remaining'] = data_df['seconds_in_quarter_remaining']
data_df.loc[data_df.quarter== 1,'seconds_in_half_remaining':]*=2
data_df.loc[data_df.quarter== 3,'seconds_in_half_remaining':]*=2

#%%
# Create score differential column
data_df['tex_lead'] = data_df['texscore'] - data_df['oppscore']
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
data_df.loc[data_df['type'].str.contains('Rushing'), 'type'] = 'Rush'

#%%
# Drop Output label into separate object
output_df = data_df.type
features_df = data_df[['half','down','distance', 'time_remaining_binned']].reset_index()
features_df = features_df.drop(columns = ['playID'])
#%%
# Check categorical columns of feature df and check the number of unique values in each column
data_cat = features_df.dtypes[features_df.dtypes == "object"].index.tolist()
data_cat.append('time_remaining_binned')
features_df[data_cat].nunique()

#%%
# Encode categorical data

# Create a OneHotEncoder instance
#enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
#encode_df = pd.DataFrame(enc.fit_transform(features_df[data_cat]))

# Add the encoded variable names to the DataFrame
#encode_df.columns = enc.get_feature_names(data_cat)

# Fit and transform with cat code
encode_df = features_df.copy()
encode_df['time_remaining_binned'] = encode_df['time_remaining_binned'].cat.codes

#%%
# Merge encoded DataFrame back into the original feature df and drop original object/category columns
#encoded_features_df = features_df.merge(encode_df,left_index=True, right_index=True)
#encoded_features_df = encoded_features_df.drop(data_cat,1)

#%%
#Split into feature and output
X = encode_df
y = output_df
#%%
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)

#%%
#Scale data based on testing group and apply to testing and training
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)
dump(X_scaler, 'rf_std_scaler.bin', compress=True)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

#%%
# Random Forest Model
# Create a random forest classifier.
rf_model = RandomForestClassifier(n_estimators=128, random_state=42)

# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
print(f" Random forest predictive accuracy: {accuracy_score(y_test,y_pred):.4f}")
print(classification_report(y_test,y_pred))

#%%
feature_list = [1,2,1,10]
new_data = np.array([feature_list])
scaled_input = X_scaler.transform(new_data)
#%%
rf_model.predict(scaled_input)

#%%
#Save the model
filename = 'finalized_rf_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))