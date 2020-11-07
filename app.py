# coding: utf-8

import pickle
import numpy as np
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

    # call the play

    # load model from saved file
    model = pickle.load(open('ML_models/random_forest/rfPickle.pkl', 'rb'))
    prediction = model.predict(features)
    output = prediction[0]

    # binary output for pass or rush call
    if output == 1:
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

if __name__ == "__main__":
    app.run()

import pdb; pdb.set_trace()
