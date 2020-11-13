#%%
# Import dependencies
import pandas as pd
from datetime import datetime
import os
import hvplot.pandas
import plotly.express as px

# ML dependencies
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
#%%
data_df = pd.read_csv('Texas_combined_formatted.csv')
data_df
# %%
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
# Convert all pass outcomes to "pass" to create a true binary outcome
data_df.loc[data_df['type'].str.contains('Pass'), 'type'] = 'Pass'
data_df.loc[data_df['type'].str.contains('Rush'), 'type'] = 'Rush'
data_df
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
time_remaining_bins = [-1, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800]
labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min', '10-12 min', '12-14 min', '14-16 min', '16-18 min', '18-20 min', '20-22 min', '22-24 min', '24-26 min', '26-28 min', '28-30 min']
data_df['time_remaining_binned'] = pd.cut(data_df['seconds_in_half_remaining'], bins=time_remaining_bins, labels=labels)

# %%
le = LabelEncoder()
kmodel_df = data_df.copy()
kmodel_df['type'] = le.fit_transform(kmodel_df['type'])
kmodel_df
# 1 = rush; 0 = pass
#%%
kmodel_df = kmodel_df.drop([
    'playID', 'gameId', 'year', 'week', 'offenseAbbr', 
    'defenseAbbr', 'seconds_in_half_remaining', 
    'time_remaining_binned', 'clock'
    ], axis=1)
kmodel_df

# %%
kmodel_df.hvplot.scatter(x="yardLine", y="seconds_in_quarter_remaining", by="type")
# %%
# Plotting the clusters with three features
fig = px.scatter_3d(
    kmodel_df, 
    x="yardLine", 
    y="down", 
    z="distance", 
    color="type", 
    symbol="type", 
    # size="yardsGained",
    width=800
    )
fig.update_layout(legend=dict(x=0,y=1), title = "1 = Rush 0 = Pass") 
fig.show()
# %%
fig.write_html("kmeans.html")
# %%
fig.write_image(file='../static/images/kmeans.png', format='png', engine = 'kaleido')
#%%

# %%
# Function to cluster and plot dataset
def test_cluster_amount(df, clusters):
    model = KMeans(n_clusters=clusters, random_state=5)   
    model
    # Fitting model
    model.fit(df)
    # Add a new class column
    df["class"] = model.labels_
    predictions = model.predict(kmodel_df)
    print(predictions)
#%%
inertia = []
k = list(range(1, 11))

# Looking for the best K
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(kmodel_df)
    inertia.append(km.inertia_)

# Define a DataFrame to plot the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)

#%%








# Model Evaluation
# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual Pass (0)", "Actual Rush (1)"], columns=["Predicted Pass (0)", "Predicted Rush(1)"])

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
