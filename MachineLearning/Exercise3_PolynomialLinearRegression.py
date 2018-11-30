# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 07:59:47 2018

@author: v-yudura
"""

import pandas as pd
import numpy as np


dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X,Y)
Y_Pred = regression.predict(X)
import matplotlib.pyplot as plt
plt.scatter(X,Y,color="red")
plt.plot(X,Y_Pred, color="blue")


from sklearn.preprocessing import PolynomialFeatures

polynomialRegression = PolynomialFeatures(degree=2)
X_Poly = polynomialRegression.fit_transform(X)
polynomialRegression.fit(X_Poly, Y)

linearReg2 = LinearRegression()
linearReg2.fit(X_Poly, Y)
plt.scatter(X,Y, color="red")
plt.plot(X,Y,color="blue")
￼