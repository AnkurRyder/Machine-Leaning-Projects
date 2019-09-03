# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:23:40 2019

@author: Ankur Jaiswal
"""


import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from	sklearn.datasets	import	fetch_mldata
mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='C:/Users/Ankur Jaiswal/scikit_learn_data/')

X,	y	=	mnist["data"],	mnist["target"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/7)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_neighbors' :[3 , 4] , 'metric': ['minkowski' , 'manhattan'] , 'weights': ["uniform", "distance"] }]

knn_cls = KNeighborsClassifier(n_neighbors = 4 , n_jobs=-1)
knn_cls.fit(X_train , y_train)
y_pred = knn_cls.predict(X_test)


grid_search = GridSearchCV(knn_cls , param_grid , cv= 3 , scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train , y_train)
