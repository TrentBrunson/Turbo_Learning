#%%
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# %%
df = pd.read_csv('dataRaw/Texas_combined_final.csv')
df
# %%
# drop irrelevant columns
df.drop(['gameid','year','week','playid', 'offenseabbr', 'defenseabbr'], axis=1, inplace=True)
df

# Creating a 'time remaining in quarter' column 
# This allows us to bin easily but also treat "time" as a continuous feature more easily should we choose
def time_convert(x):
    h,m,s = map(int,x.split(':'))
    return (h*360)+(m*60)+s

df['seconds_in_quarter_remaining'] = df.clock.apply(time_convert)

# Utilizing 'time remaining in quarter' column to generate 'time remaining in half' a criteria 
# which should be a better indicator of run vs pass as strategies change not at the end of the
# quarter but at the half.
df['seconds_in_half_remaining'] = df['seconds_in_quarter_remaining']
df.loc[df.quarter== 1,'seconds_in_half_remaining':]*=2
df.loc[df.quarter== 3,'seconds_in_half_remaining':]*=2

# Create 'half' feature using the 'quarter' column
# This allows easier calculation for "time left in half" but also provides us with another feature
df.loc[df.quarter == 1, 'half'] = 1
df.loc[df.quarter == 2, 'half'] = 1
df.loc[df.quarter == 3, 'half'] = 2
df.loc[df.quarter == 4, 'half'] = 2
# Quick workaround to account for OT
df.loc[df.quarter == 5, 'half'] = 3
df.loc[df.quarter == 6, 'half'] = 3
# Bucketing 'time remaining in half'
# In deciding bin size: the two-minute mark is likely where strategies are going to change, 
# so a bin size larger than that would likely obscure the potential effect of the feature.
time_remaining_bins = [-1, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800]
labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min', '10-12 min', '12-14 min', '14-16 min', '16-18 min', '18-20 min', '20-22 min', '22-24 min', '24-26 min', '26-28 min', '28-30 min']
df['time_remaining_binned'] = pd.cut(df['seconds_in_half_remaining'], bins=time_remaining_bins, labels=labels)

# convert play type to all rush or pass
# Convert all pass outcomes to "pass" to create a true binary outcome
df.loc[df['type'].str.contains('Pass'), 'type'] = 'Pass'
df.loc[df['type'].str.contains('Rushing'), 'type'] = 'Rush'

# drop temporary columns
df.drop(['seconds_in_quarter_remaining','seconds_in_half_remaining', 'clock'], axis=1, inplace=True)
df
# %%
# define target
# LabelEncode y 
y = df["type"]

# Encode categorical data
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)
labels = LabelEncoder()
labels.fit(y)
encoded_y = labels.transform(y)

# add column for encoded y values
en_y = encoded_y
df["code"]=en_y

output_df = df.type
features_df = df[['texscore','oppscore','quarter','down','distance','yardline','half', 'time_remaining_binned']]
# %%
fig, ax = plt.subplots(figsize= (10,6))
corrImage = sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap= 'BrBG', annot=True)
fig = corrImage.get_figure()
fig.savefig('images/correlationHeatMap.png')
# %%
np.triu(np.ones_like(df.corr()))
# %%
plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
triangleHeatMap = sns.heatmap(df.corr(), mask=mask, vmin=-0.5, vmax=1, annot=True, cmap='BrBG')

fig = triangleHeatMap.get_figure()
fig.savefig('images/TriangleHeatMap.png')
# %%
df.corr()[['code']].sort_values(by='code', ascending=False)
plt.figure(figsize=(8, 12))
yheatmap = sns.heatmap(df.corr()[['code']].sort_values(by='code', ascending=False), vmin=-0.5, vmax=1, annot=True, cmap='BrBG')
fig = yheatmap.get_figure()
fig.savefig('images/PlayTypeHeatMap.png')
# %%