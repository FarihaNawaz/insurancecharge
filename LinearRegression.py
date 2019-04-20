# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:14:59 2018

@author: Prithila
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#reading the dataset
df = pd.read_csv('insurance.csv')
df['region'] = df['region'].map({'northwest': 1, 'southeast': 2,'northeast':3,'southwest':4})
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
#taking the x and y parameter
X = df.drop('charges', axis=1)
y = df[['charges']]
#Splitting the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=500)
regression_model =LinearRegression()
regression_model.fit(X_train, y_train)
for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))

intercept = regression_model.intercept_[0]
print("The intercept for our model is {}".format(intercept))
print("The accuracy is",regression_model.score(X_test, y_test)*100)
#for plotting

y_pred = regression_model.predict(X_test)
xplot=y_pred
yplot=y_test
xplt=np.asarray(xplot)
yplt=np.asarray(yplot)
plt.scatter(xplt,yplt, marker= 'o', s=10, alpha=0.8)
plt.title('Least-squares linear regression')
plt.xlabel('Predicted result')
plt.ylabel('Test result')
#import seaborn as sns
#sns.boxplot(x=df['charges'])