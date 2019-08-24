# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:16:09 2019

@author: Ankur Jaiswal
"""

import numpy as np
import pandas as pd

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

X_train = train.iloc[: , 1:]
y_train = train.iloc[: , 0]

from sklearn.neighbors import KNeighborsClassifier

knn_cls = KNeighborsClassifier(n_neighbors = 4 , n_jobs=-1)
knn_cls.fit(X_train , y_train)
y_pred = knn_cls.predict(test)

index = pd.DataFrame(np.arange(1,28001))
y_pred = pd.DataFrame(y_pred)

y_pred[1] = y_pred[0]
y_pred[0] = index

y_pred.to_csv('sample.csv' , header = ['ImageId', 'Label']  , index = False)