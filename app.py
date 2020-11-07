# coding: utf-8

import pickle
import numpy as np
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
# start home page
@app.route("/")
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


    # take inputs and put into array, ready for ML model
    feature_list = [half, clockSeconds, down, distance]
    features = [np.array(feature_list)]

    # call the play

    # use pickle or H5?????
    # load H5 from sklearn/pandas pd.read_hdf
    model = pickle.load(open('???model.pkl'))
    prediction = model.predict(features)
    output = prediction[0]

    # binary output for pass or rush call
    if output == 0:
        result = 'Pass'
    else:
        result = 'Rush'

    return render_template('index.htnml', call = result)

@app.route('/findings')
def findings():
    return render_template('findings.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

if __name__ == "__main__":
    app.run()
