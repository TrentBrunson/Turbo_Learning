#%%
# Import dependencies
import pandas as pd
from datetime import datetime

# ML dependencies
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import tensorflow as tf
from scipy import stats

#%%
data_df = pd.read_csv('Texas_combined_formatted.csv')
#%%
data_df['clock'] = data_df['clock'].astype(str)
data_df.dtypes
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

data_df
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
data_df
# #%%
# # Bucketing 'time remaining in half'
# # In deciding bin size: the two-minute mark is likely where strategies are going to change, 
# # so a bin size larger than that would likely obscure the potential effect of the feature.
# time_remaining_bins = [-1, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800]
# labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min', '10-12 min', '12-14 min', '14-16 min', '16-18 min', '18-20 min', '20-22 min', '22-24 min', '24-26 min', '26-28 min', '28-30 min']
# data_df['time_remaining_binned'] = pd.cut(data_df['seconds_in_half_remaining'], bins=time_remaining_bins, labels=labels)

#%%
# Convert all pass outcomes to "pass" to create a true binary outcome
data_df.loc[data_df['type'].str.contains('Pass'), 'type'] = 'Pass'
data_df.loc[data_df['type'].str.contains('Rush'), 'type'] = 'Rush'
data_df
#%%
# Drop Output label into separate object
output_df = data_df.type
features_df = data_df[['down','distance', 'half', 'seconds_in_half_remaining']].reset_index()
features_df = features_df.drop("index", axis=1)
features_df
# output_df
# %%
# Splitting into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    features_df, output_df, random_state=78, train_size= 0.5)

# %%
# scaling the data so if want to compare this decision tree
# model to other best fit models, can do so quicky

# Creating a StandardScaler instance.
scaler = StandardScaler()

# Fitting the Standard Scaler with the training data.
X_scaler = scaler.fit(X_train)

# Scaling the data.
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

X_train_scaled
# %%
# Fitting the Decision Tree Model

# Random Forest
# Create a random forest classifier.
rf_model = RandomForestClassifier(n_estimators=500, random_state=42) 

# Fitting the random forest model
rf_model = rf_model.fit(X_train_scaled, y_train)
# %%
# Making predictions using the random forest testing data.
predictions = rf_model.predict(X_test_scaled)
# %%
# Model Evaluation
# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual Good (0)", "Actual Bad (1)"], columns=["Predicted Good (0)", "Predicted Bad(1)"])

# add the columns and the rows
total_column = cm_df.sum(axis = 1)
total_row = cm_df.sum(axis = 0)

total_column
total_row

# add the new totals to the DF
cm_df['Column Totals'] = total_column
cm_df.loc['Row Totals'] = cm_df.sum()

# cm_df = cm_df.drop(columns='Column Totals')
cm_df
# %%
# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)
acc_score
# %%
# rank order the features in the random forest
importances = rf_model.feature_importances_
# sort the features by their importance.
sorted(zip(rf_model.feature_importances_, features_df.columns), 
reverse=True)
# %%
# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
print("Confusion Matrix")
display(features_df)
print(f"Random forest predictive accuracy: {accuracy_score(y_test,y_pred):.4f}")
print(classification_report(y_test,y_pred))

# %%
import pickle
pickle.dump(rf_model, open('rfPickle.pkl', 'wb'))

# %%
test_model = pickle.load(open('rfPickle.pkl', 'rb'))
result = test_model.score(X_test, y_test)
print(result)


# %%

# %%
