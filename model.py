# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 00:09:33 2021

@author: vcjayan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('C:/Python/regression.csv')

df['Age'].fillna(df['Age'].mean(), inplace=True)

#seperate cat and num
#perform label encoding for changing categorical to numerical

df_num= df[['Age','Signed in since(Days)']]
df_cat= df[['Job Type','Marital Status','Education','Metro City']]     

labelencoder = LabelEncoder()

mapping = {}
for i in df_cat:
    df[i] = labelencoder.fit_transform(df[i])
    le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    mapping[i]=le_name_mapping

#splitting
x=df.drop(['Purchase made'], axis =1)
y=df['Purchase made']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8 , random_state=100)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)

#saving the model
pickle.dump(model, open('model.pkl', 'wb'))

#loading the model and compare the prediction
model = pickle.load(open('model.pkl', 'rb'))
#check the prediction on a test data
print(model.predict([[25,0,1,0,1,87]]))