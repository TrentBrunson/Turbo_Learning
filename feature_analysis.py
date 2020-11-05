#%%
# Import dependencies
import pandas as pd

# ML dependencies
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from graphviz import Source
import tensorflow as tf
from scipy import stats

# Feature plotting dependencies
import matplotlib.pyplot as plt

#%%
# Import clean dataset from raw file
data_df = pd.read_csv('Resources/Texas_combined_final.csv')

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
features_df = data_df[['texscore','oppscore','quarter','down','distance','yardline','half']]

#%%
le = preprocessing.LabelEncoder()
#%%
le.fit(output_df)
#%%
output_df = le.transform(output_df)

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
# %%
X = features_df
y = output_df

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y)
# %%
classifier = LogisticRegression(solver='lbfgs', random_state=24)
classifier
# %%
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='warn', n_jobs=None, penalty='12', random_state=1, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
# %%
classifier.fit(X_train, y_train)

#%%
predictions = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions, "Actual": y_test})
# %%
accuracy_score(y_test, predictions)

#%%
# Linear Regression for feature importance
# Define the linear regression model
model = LinearRegression()
# fit the model
model.fit(X, output_df)
# Set variable for importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.xlabel(['texscore','oppscore','quarter','down','distance','yardline','half'])
plt.savefig('./images/linearFeatureImportance.png')
# %%
# CART Feature Importance
tree_model = DecisionTreeRegressor()
# %%
tree_model.fit(X, y)
# %%
tree_importance = tree_model.feature_importances_
# summarize feature importance
for i,v in enumerate(tree_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(tree_importance))], tree_importance)
plt.xlabel(['texscore','oppscore','quarter','down','distance','yardline','half'])
plt.savefig('./images/CARTFeatureImportance.png')

# %%
# Random Forest Regression
random_forest_model = RandomForestRegressor()

random_forest_model.fit(X, y)

forest_importance = random_forest_model.feature_importances_

for i,v in enumerate(forest_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(forest_importance))], forest_importance)
plt.xlabel(['texscore','oppscore','quarter','down','distance','yardline','half'])
plt.savefig('./images/RF_Regression_FeatureImportance.png')
# %%
# Random Forest Classification
rf_class_model = RandomForestClassifier()
# fit the model
rf_class_model.fit(X, y)
# get importance
rf_importance = rf_class_model.feature_importances_
# summarize feature importance
for i,v in enumerate(rf_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(rf_importance))], rf_importance)
plt.xlabel(['texscore','oppscore','quarter','down','distance','yardline','half'])
plt.savefig('./images/RF_Classification_FeatureImportance.png')
# %%
