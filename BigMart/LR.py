# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:19:13 2019

@author: Pavilion
"""

import numpy as np
import pandas as pd

train = pd.read_csv('Train.csv')
Y = train['Item_Outlet_Sales']
test = pd.read_csv('Test.csv')
test_original = pd.read_csv('Test.csv')

test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace=True)
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace=True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)

#-----------------------Just for Exp---------------------#

#Determine average visibility of a product in differnt outlets
visibility_avg = train.pivot_table(values='Item_Visibility', index='Outlet_Size')
miss_bool = (train['Item_Visibility'] == 0)
train.loc[miss_bool,'Item_Visibility'] = train.loc[miss_bool,'Outlet_Size'].apply(lambda x: visibility_avg.loc[x])

visibility_avg = test.pivot_table(values='Item_Visibility', index='Outlet_Size')
miss_bool = (test['Item_Visibility'] == 0)
test.loc[miss_bool,'Item_Visibility'] = test.loc[miss_bool,'Outlet_Size'].apply(lambda x: visibility_avg.loc[x])

#----------------------Feature Eng.----------------------#

visibility_avg = train.pivot_table(values='Item_Visibility', index='Outlet_Size')
train['Item_Visibility_MeanRatio'] = train.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Outlet_Size']], axis=1)
visibility_avg = test.pivot_table(values='Item_Visibility', index='Outlet_Size')
test['Item_Visibility_MeanRatio'] = test.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Outlet_Size']], axis=1)

train['Item_Type_Combined'] = train['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
train['Item_Type_Combined'] = train['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})

test['Item_Type_Combined'] = test['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
test['Item_Type_Combined'] = test['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})

train['Outlet_Years'] = 2013 - train['Outlet_Establishment_Year']
test['Outlet_Years'] = 2013 - test['Outlet_Establishment_Year']

train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
#-------------------------Droping Var ------------------------#
train = train.drop('Item_Identifier' , axis = 1)
test = test.drop('Item_Identifier' , axis = 1)
train = train.drop('Item_Visibility' , axis = 1)
test = test.drop('Item_Visibility' , axis = 1)
train = train.drop('Outlet_Establishment_Year' , axis = 1)
test = test.drop('Outlet_Establishment_Year' , axis = 1)
train = train.drop('Item_Type' , axis = 1)
test = test.drop('Item_Type' , axis = 1)
train = train.drop('Item_Outlet_Sales' , axis = 1)
#---------------------Label Encoding -----------------------#

from sklearn.preprocessing import LabelEncoder
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet_Identifier']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    
le = LabelEncoder()
for i in var_mod:
    test[i] = le.fit_transform(test[i])

train = pd.get_dummies(train, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet_Identifier'])
test = pd.get_dummies(test, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet_Identifier'])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
x_test = sc.transform(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(train,Y, test_size = 0.3)

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(x_train , y_train)
regresor.score(x_train , y_train)

y_pred = regresor.predict(x_cv)

submission=pd.read_csv("SampleSubmission.csv")
y_pred = regresor.predict(x_test)
submission['Item_Outlet_Sales']= y_pred

pd.DataFrame(submission, columns=['Item_Identifier','Outlet_Identifier' , 'Item_Outlet_Sales']).to_csv('sample_submision.csv' , index = False)