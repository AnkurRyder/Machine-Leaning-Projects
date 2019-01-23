# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:52:24 2019

@author: Pavilion
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('iris.csv')
X = data.iloc[: ,0:4].values
Y = data.iloc[: , -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
Y[:] = labelencoder_X_1.fit_transform(Y[:])

Y=Y.astype('int')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

cor = (y_pred == y_test)
acc = cor.sum()/30