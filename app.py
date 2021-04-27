import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load("diabetes_prediction.pkl", "r")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in form.values()]
    prediction = model.predict([data])
    output = prediction[0]
    if(output == 1.0):
        return render_template('index.html', prediction_text='Patient Has Diabetes')
    else:

        return render_template('index.html', prediction_text='Patient Does Not Have Diabetes')

if __name__ == "__main__":
    app.run(debug=True)