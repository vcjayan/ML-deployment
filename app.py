# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:40:00 2021

@author: vcjayan
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,6)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ['POST'])
def result():
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        result = ("%.2f" % result)
        return render_template('index.html', prediction_text = 'Purchase made by the person is approximately INR {}'.format(result))

if __name__ == "__main__":
    app.run(debug = True)

#For creating a procfile run below command from anaconda prompt
#echo web: run this thing >Procfile