# coding: utf-8

import pickle
import numpy as np
from joblib import dump, load
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# start home page
@app.route('/')
def index():
    return render_template('index.html')

# create prediction module
@app.route('/predict', methods=['POST'])
def predict():
    # get user inputs
    quarter = request.form['quarter']
    clock = request.form['clock']
    down = request.form['down']
    distance = request.form['distance']

    # convert strings to numbers
    quarter = int(quarter)
    clock = int(clock)
    down = int(down)
    distance = int(distance)

    # change quarters to halves
    if quarter <= 2:
        half = 1
    elif quarter == 5:
        half = 3
    else:
        half = 2

    # convert clock input to fit the model
    if clock == '14-15':
        time_in_quarter = 7
    elif clock == '12-14':
        time_in_quarter = 6
    elif clock == '10-12':
        time_in_quarter = 5
    elif clock == '8-10':
        time_in_quarter = 4
    elif clock == '6-8':
        time_in_quarter = 3
    elif clock == '4-6':
        time_in_quarter = 2
    elif clock == '2-4':
        time_in_quarter = 1
    else:
        time_in_quarter = 0
    
    # taking into account half

    if quarter == 1:
        time_remaining_binned = time_in_quarter
    elif quarter == 3:
        time_remaining_binned = time_in_quarter
    else:
        time_remaining_binned = time_in_quarter + 7

    # Load the saved scaler from the input data
    scaler = load('rf_std_scaler.bin')

    # take inputs and put into array, ready for ML model
    feature_list = [half, down, distance, time_remaining_binned]
    features = [np.array(feature_list)]
    scaled_features = scaler.transform(features)

    # call the play
    # load model from saved file
    model = pickle.load(open('finalized_rf_model.sav', 'rb'))
    prediction = model.predict(scaled_features)
    output = prediction[0]

    # binary output for pass or rush call
    if output == 'Pass':
        result = 'Pass'
    else:
        result = 'Rush'

    return render_template('index.html', call = f'{result} defense!')

@app.route('/findings')
def findings():
    return render_template('findings.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/kmeans')
def kmeans():
    return render_template('kmeans.html')

if __name__ == "__main__":
    app.run()