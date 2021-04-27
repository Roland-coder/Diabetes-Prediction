# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import re
import math
import pickle

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")



@app.route("/predict", methods=['POST'])
def predict():
    
    model = pickle.load(open("diabetes_prediction_model.sav", "rb"))
    

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']

    
    
    
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, inputQuery8]]
    #print('data is: ')
    #print(data)
    #016.14, 74.00, 0.01968, 0.05914, 0.1619
    
    # Create the pandas DataFrame 
    new_df = pd.DataFrame(data, columns = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age'])
    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:,1]
    print(probability)
    if single==1:
        o1 = "The patient has Diabetes"
        o2 = "Confidence: {}".format(probability*100)
    else:
        o1 = "The patient does not have Diabetes"
        o2 = ""
    
    return render_template('home.html', output1=o1, output2=o2, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'],query6 = request.form['query6'],query7 = request.form['query7'],query8 = request.form['query8'])
    
if __name__ == "__main__":
    app.run(debug=True)