# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:07:36 2019

@author: Pavilion
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:13:52 2019

@author: Pavilion
"""

import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_original = pd.read_csv('test.csv')

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#-------------------Outlier Treatment-------------------#
train['LoanAmount_log'] = np.log(train['LoanAmount'])
test['LoanAmount_log'] = np.log(test['LoanAmount'])

train = train.drop('Loan_ID' , axis = 1)
test=test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status',1)
Y = train.Loan_Status
x_test = test
X = pd.get_dummies(X)
x_test = pd.get_dummies(x_test)

from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
Y[:] = labelencoder_X_1.fit_transform(Y[:])
Y = Y.iloc[:].values

X = X.iloc[: , :].values
x_test = x_test.iloc[: , :].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
x_test = sc.transform(x_test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,Y, test_size =0.3)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_cv)

from sklearn.metrics import accuracy_score
accuracy_score(y_cv,y_pred)

submission=pd.read_csv("sample_submision.csv")
y_pred = classifier.predict(x_test)
submission['Loan_Status']= y_pred
submission['Loan_ID']= test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('sample_submision.csv' , index = False)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X, Y)