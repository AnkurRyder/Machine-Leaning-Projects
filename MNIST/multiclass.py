# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:08:39 2019

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

from sklearn.linear_model import SGDClassifier
from	sklearn.multiclass	import	OneVsOneClassifier 

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train , y_train)

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state= 42)
forest_clf.fit(X_train , y_train)

from sklearn.model_selection import cross_val_score , cross_val_predict

cross_val_score(ovo_clf,	X_train,	y_train,	cv=3,	scoring="accuracy")

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42) 
cross_val_score(sgd_clf,	X_train,	y_train,	cv=3,	scoring="accuracy") 

from	sklearn.preprocessing	import	StandardScaler
scaler	=	StandardScaler() 
X_train_scaled	=	scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf,	X_train_scaled,	y_train,	cv=3,	scoring="accuracy")

from sklearn.metrics import confusion_matrix

y_train_pred	=	cross_val_predict(sgd_clf,	X_train_scaled,	y_train,	cv=3)  
conf_mx	=	confusion_matrix(y_train,	y_train_pred)
 
import matplotlib.pyplot as plt

plt.figure()
plt.matshow(conf_mx,	cmap=plt.cm.gray) 

row_sums	=	conf_mx.sum(axis=1,	keepdims=True) 
norm_conf_mx	=	conf_mx	/	row_sums

np.fill_diagonal(norm_conf_mx,	0)
plt.figure() 
plt.matshow(norm_conf_mx,	cmap=plt.cm.gray) 
plt.show()