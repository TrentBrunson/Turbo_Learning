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
    time_in_quarter = int(clock)
    down = int(down)
    distance = int(distance)

    # change quarters to halves
    if quarter <= 2:
        half = 1
    elif quarter == 5:
        half = 3
    else:
        half = 2
    
    # binning time into halves
    if quarter == 2:
        time_remaining_binned = time_in_quarter + 7
    elif quarter == 4:
        time_remaining_binned = time_in_quarter + 7
    else: # for quarters 1, 3, main cases & OT edge cases
        time_remaining_binned = time_in_quarter

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