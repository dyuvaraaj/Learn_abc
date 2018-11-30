# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 05:54:31 2018

@author: v-yudura
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Split into training and test dataset
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]

import statsmodels.formula.api as sm
X= np.append(arr=np.ones((50,1)).astype(int) , values=X,axis=1)
X_Opt = X[:,[0,1,2,3,4,5]]
regressor_BackwardElimination = sm.OLS(endog=Y,exog=X_Opt).fit()
regressor_BackwardElimination.summary()

X_Opt = X[:,[0,1,3,4]]
regressor_BackwardElimination = sm.OLS(endog=Y,exog=X_Opt).fit()
regressor_BackwardElimination.summary()

X_Opt = X[:,[0,3,4]]
regressor_BackwardElimination = sm.OLS(endog=Y,exog=X_Opt).fit()
regressor_BackwardElimination.summary()


X_Opt = X[:,[0,3]]
regressor_BackwardElimination = sm.OLS(endog=Y,exog=X_Opt).fit()
regressor_BackwardElimination.summary()

#split train and test dataset
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Opt ,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)
Y_Pred = regressor.predict(X_Test)


