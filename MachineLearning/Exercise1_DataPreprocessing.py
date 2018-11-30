# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:55:10 2018

@author: v-yudura
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Data.csv")

#Split into independent and dependent variables.
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Remove missing values using sklearn.imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])

#Encode values
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()

np.set_printoptions(threshold=np.NaN)

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X_Train = standardScaler.fit_transform(X_Train)
X_Test = standardScaler.transform(X_Test)