# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:38:26 2020

@author: Noorealdean Koteb
"""

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
frauds = np.concatenate((mappings[(7, 7)], mappings[(7, 8)]), axis = 0)
frauds = sc.inverse_transform(frauds)




