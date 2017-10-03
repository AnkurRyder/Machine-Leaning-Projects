# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:51:17 2019

@author: Ankur Jaiswal
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf)
# Importing the dataset
dataset = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_test = pd.read_csv('gender_submission.csv')
y_test = y_test.iloc[:,0].values
y_test = y_test.reshape(-1,1)
X_rem = [2,4,5,6,7,9,11]
X_test_rem = [1,3,4,5,6,8,10]
X = dataset.iloc[:, X_rem].values
X_test = test_data.iloc[:, X_test_rem].values
y = dataset.iloc[:, 1].values
y_data = pd.DataFrame({'Results' : y})

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan , strategy = "mean")
ch =dataset.iloc[:, 5].values
te =test_data.iloc[:, 4].values
ch = ch.reshape(-1,1)
te = te.reshape(-1,1)
imputer = imputer.fit(ch)
ch=imputer.transform(ch)
ch = np.ravel(ch)
X[:,2]=ch
X[61][6]='S'
X[829][6]='S'
X[61][6]='S'
x_data = pd.DataFrame({'Pclass':X[:,0],'Sex':X[:,1],'Age':X[:,2],'Sibsp':X[:,3],'Parch':X[:,4],'Fare':X[:,5],'Enbarked':X[:,6]})
imputer = imputer.fit(te)
te=imputer.transform(te)
te = np.ravel(te)
X_test[:,2]=te
#import seaborn as sns
#ax = sns.stripplot(X[:,6],y);
#plt.show()
#X_test[152][7]=20


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
ch = onehotencoder.fit_transform(X[:,[1,6]]).toarray()
ch = onehotencoder.fit_transform(X[:,[1,6]]).toarray()
ch = ch[:,1:4]
X=X[:,[0,2,3,4,5]]
X = np.concatenate((ch,X),axis=1)

te = onehotencoder.fit_transform(X_test[:,[1,6]]).toarray()
te = onehotencoder.fit_transform(X_test[:,[1,6]]).toarray()
te = te[:,1:4]
X_test=X_test[:,[0,2,3,4,5]]
X_test = np.concatenate((te,X_test),axis=1)
X_test[152][7]=20
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_test,y_test)
print(model.feature_importances_)

# Fitting the Decision Tree Regression Model to the dataset
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(criterion = "entropy",random_state = 0)

selector = RFE(regressor,step = 1)
selector = selector.fit(X,y);
selector.ranking_
selector.support_
regressor.fit(X,y)


# Predicting a new result

y_pred = regressor.predict(X_test)
y_pred = y_pred.reshape(-1,1)
y_pred = np.concatenate((y_test,y_pred),axis = 1)
np.savetxt("ans.csv",y_pred, delimiter=",",fmt = '%0.0f' )
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)






