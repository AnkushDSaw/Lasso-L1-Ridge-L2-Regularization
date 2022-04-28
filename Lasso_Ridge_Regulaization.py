# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:40:55 2022

@author: ankus
"""
#******************Scale data and apply Linear, Ridge & Lasso Regression with Regularization
# import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Data set Miles Per Gallon dataset from UCI repository
dataset=pd.read_csv(r'C:\Users\ankus\OneDrive\Desktop\Naresh IT\April\26th_April\car-mpg.csv')
#************************EDA
#Drop car name
#Replace origin into 1,2,3.. dont forget get_dummies
#Replace ? with nan
#Replace all nan with median
dataset=dataset.drop(['car_name'],axis=1)

dataset['origin'] = dataset['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
dataset = pd.get_dummies(dataset,columns = ['origin']) # Add 3 column 
dataset = dataset.replace('?', np.nan)

dataset = dataset.apply(lambda x: x.fillna(x.median()), axis = 0)

# ********************            Model building
# We have to predict the mpg column given the features.
X = dataset.drop(['mpg'], axis = 1) # independent variable ( Remove mpg from X data)
y = dataset[['mpg']] #dependent variable

#************************** Scaling the data

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

#Scaling the data
X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X_s, columns = X.columns) #converting scaled data into dataframe
y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns = y.columns) #ideally train, test data should be in columns


from sklearn.model_selection import train_test_split
#Split into train, test set
X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.20, random_state = 0)
X_train.shape


#*******************    1. Simple Linear Model
from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

for idx, col_name in enumerate(X_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
    
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))


y_pred=regression_model.predict(X_test)
y_pred

# *********************      Regularized Lasso (L1)  Regression
from sklearn.linear_model import Lasso
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)

print('Lasso model coef: {}'.format(lasso_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here 


# ********************************    Regularized Ridge(L2) Regression
from sklearn.linear_model import Ridge, Lasso
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(X_train, y_train)

print('Ridge model coef: {}'.format(ridge_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here  


#**********************     Score Comparison
#Model score - r^2 or coeff of determinant
#r^2 = 1-(RSS/TSS) = Regression error/TSS 

#Simple Linear Model
print(regression_model.score(X_train, y_train))
print(regression_model.score(X_test, y_test))

print('*************************')
#Ridge
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))

print('*************************')
#Lasso
print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test))

# Model Parameter Tuning

data_train_test = pd.concat([X_train, y_train], axis =1)
import statsmodels.formula.api as smf

ols1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin_america+origin_europe+origin_asia',
              data = data_train_test).fit()
ols1.params

print(ols1.summary())

















