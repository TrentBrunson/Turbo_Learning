#%%
# import pickle
import numpy as np
from flask import Flask, render_template, request, redirect
#%%
# convert strings to numbers
quarter = int(1)
clock = int(560)
down = int(1)
distance = int(11)

# change quarters to halves
if quarter <= 2:
    half = 1
elif quarter == 5:
    half = 3
else:
    half = 2

# convert clock to seconds so the ML model can take it in
if half == 1:
    clockSeconds = clock * 60
elif half == 2:
    clockSeconds = clock * 60 * 2
else:
    clockSeconds = clock * 60 * 3

# take inputs and put into array, ready for ML model
feature_list = [half, clockSeconds, down, distance]
features = [np.array(feature_list)]

# load model from saved file
model = pickle.load(open('rfPickle.pkl', 'rb'))
prediction = model.predict(features)
output = prediction[0]
# %%
output
#%%
# binary output for pass or rush call
if output == 1:
    result = 'Rush'
else:
    result = 'Pass'
print(result)
# %%
