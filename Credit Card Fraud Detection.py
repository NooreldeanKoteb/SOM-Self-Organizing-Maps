# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:40:57 2020

@author: Nooreldean Koteb
"""

#Mega Case Study - make a hybrid learning model

#Part 1 - Identify the frauds with the self organizing map

#Self Organizing Map

#Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
#Normalization
sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)

#Training the SOM
#Importing minisom code from py file in folder
from minisom import MiniSom
#learning_rate = 0.5  increase this for faster convergence
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

#Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, j in enumerate(x):
    w = som.winner(j)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#Finding the frauds
mappings = som.win_map(x)
#Input the coordinates of the box you want the ids from
frauds = np.concatenate((mappings[(6, 4)], mappings[(7, 2)], mappings[(9, 2)] ), axis = 0)
frauds = sc.inverse_transform(frauds)


#Part 2 - Going from unsupervised to supervised deep learning

#Creating the matrix of features
customers = dataset.iloc[:, 1:].values

#Creating the dependent variable
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1


#Feature scaling (Standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

#Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
#Used to improve ANN by reducing overfitting
from keras.layers import Dropout

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 7, input_dim = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.2))

#Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Fitting the ANN to the training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)


#Making the predictions and evaluating the model

# #Predicting the Test set results
y_pred = classifier.predict(customers)
#Adding back customer ids
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

#Rank customers by probabilities
y_pred = y_pred[y_pred[:, 1].argsort()]























