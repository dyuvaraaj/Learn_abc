# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:35:38 2018

@author: v-yudura
"""

#Simple Linear Regression
import pandas as pd
dataset = pd.read_csv("Salary_Data.csv")

#Seperate dependent and independent variables
X_Train = dataset.iloc[0:30,:-1].values
Y_Train = dataset.iloc[0:30,1].values


X_Test = dataset.iloc[30:33,:-1].values
Y_Test = dataset.iloc[30:33,1].values

from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_Train, Y_Train)
Y_Pred = linearRegression.predict(X_Train)

import matplotlib.pyplot as plt
plt.scatter(X_Test, Y_Test,color="red")
plt.plot(X_Train,Y_Pred, color="blue")