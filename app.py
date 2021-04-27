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
    data1 = request.form['pregnancies']
    data2 = request.form['glucose']
    data3 = request.form['bloodpressure']
    data4 = request.form['skinthickness']
    data5 = request.form['insulin']
    data6 = request.form['bmi']
    data7 = request.form['diabetespedigreefunction']
    data8 = request.form['age']
    data = np.array([data1,data2,data3,data4,data5,data6,data7,data8])
    prediction = model.predict([data])
    output = prediction[0]
    if (output == 1.0):
        return render_template('index.html', prediction_text='Patient Has Diabetes')
    else:

        return render_template('index.html', prediction_text='Patient Does Not Have Diabetes')

if __name__ == "__main__":
    app.run(debug=True)