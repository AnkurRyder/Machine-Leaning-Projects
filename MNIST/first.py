# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:35:09 2019

@author: Ankur Jaiswal
"""
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from	sklearn.datasets	import	fetch_mldata
mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='C:/Users/Ankur Jaiswal/scikit_learn_data/')

X,	y	=	mnist["data"],	mnist["target"] 
 
import	matplotlib 
import	matplotlib.pyplot	as	plt
some_digit	=	X[36000] 
some_digit_image	=	some_digit.reshape(28,	28)
plt.imshow(some_digit_image,	cmap	=	matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/7)

y_test_5  = (y_test == 5)
y_train_5  = (y_train == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train , y_train_5)

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3 , random_state=42)

for train_index, test_index in skfolds.split(X_train , y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_fold = X_train[train_index]
    y_train_fold = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_fold , y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
    
from sklearn.model_selection import	cross_val_score
cross_val_score(sgd_clf,	X_train,	y_train_5,	cv=3,	scoring="accuracy")

from sklearn.model_selection import	cross_val_predict
y_train_pred = cross_val_predict(sgd_clf , X_train , y_train_5 , cv=3)

from sklearn.metrics import confusion_matrix, f1_score
confusion_matrix(y_train_5 , y_train_pred)  
f1_score(y_train_5 , y_train_pred)

y_score = cross_val_predict(sgd_clf, X_train , y_train_5 , cv = 3 , method='decision_function') 

from sklearn.metrics import roc_curve
fpr , tpr , threshold = roc_curve(y_train_5 , y_score)

def plot_roc_curve(fpr , tpr , lable = None):
    plt.plot(fpr , tpr , linewidth = 2)
    plt.plot([0,1] , [0 ,1] , 'k--')
    plt.axis([0 , 1 , 0 , 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(fpr , tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5 , y_score)

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf , X_train , y_train_5 , cv= 3 , method='predict_proba')

y_scores_forest	=	y_probas_forest[:,	1]
fpr_forest,	tpr_forest,	thresholds_forest	=	roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr,	tpr,	"b:",	label="SGD") 
plot_roc_curve(fpr_forest,	tpr_forest,	"Random	Forest") 
plt.legend(loc="bottom	right") 
plt.show()

cross_val_score(sgd_clf,	X_train,	y_train,	cv=3,	scoring="accuracy") 