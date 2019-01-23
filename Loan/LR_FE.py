# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:36:25 2019

@author: Pavilion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

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

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
train['Balance Income']=train['Total_Income']-(train['EMI']*1000)
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

#-------------------Outlier Treatment-------------------#

train['Total_Income_log'] = np.log(train['Total_Income'])
test['Total_Income_log'] = np.log(test['Total_Income'])

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

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

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,Y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X[train_index],X[test_index]
     ytr,yvl = Y[train_index],Y[test_index]
    
     model = LogisticRegression(random_state=1)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
y_pred = model.predict(x_test)
pred=model.predict_proba(xvl)[:,1]

submission=pd.read_csv("sample_submision.csv")

submission['Loan_Status']= y_pred
submission['Loan_ID']= test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('sample_submision.csv' , index = False)
