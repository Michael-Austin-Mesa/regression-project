#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

import wrangle as w
import explore as e
import env

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = w.wrangle_zillow()


# In[3]:


train, validate, test = w.split_data(df)


# In[4]:


train_scaled, validate_scaled, test_scaled = w.scale_data(train, validate, test)


# In[5]:


x_train_scaled = train_scaled[['bedrooms', 'bathrooms', 'sq_feet']]
y_train = train[['tax_value']]

x_validate_scaled = validate_scaled[['bedrooms', 'bathrooms', 'sq_feet']]
y_validate = validate[['tax_value']]

x_test_scaled = test_scaled[['bedrooms', 'bathrooms', 'sq_feet']]
y_test = test[['tax_value']]


# In[6]:


def get_baseline():
#def get_baseline():
    y_train['tax_value_pred_mean']= y_train['tax_value'].mean()
    y_validate['tax_value_pred_mean']= y_validate['tax_value'].mean()

    y_train['tax_value_pred_median'] = y_train['tax_value'].median()
    y_validate['tax_value_pred_median'] = y_validate['tax_value'].median()

    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_median)**(1/2)

    print("\nRMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))


# In[7]:


def get_ols():
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm'] = lm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm'] = lm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)


# In[8]:


def get_lassolars():
    lars = LassoLars(alpha=1.0)
    #scaler = MinMaxScaler()
    #x_train_scaled = x_train_scaled.copy()
    #x_train_scaled[['bedrooms','bathrooms','sq_feet','year_built','tax_amount','fips']] = scaler.fit_transform(x_train_scaled)
    #x_validate_scaled = x_validate_scaled.copy()
    #x_validate_scaled[['bedrooms','bathrooms','sq_feet','year_built','tax_amount','fips']] = scaler.fit_transform(x_validate_scaled)

    lars.fit(x_train_scaled, y_train.tax_value)

    y_train['tax_value_pred_lars'] = lars.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)


# In[9]:


def get_tweedie():
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_glm'] = glm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_glm'] = glm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)


# In[10]:


def get_poly():
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_scaled_degree2 = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_scaled_degree2 = pf.transform(x_validate_scaled)
    x_test_scaled_degree2 = pf.transform(x_test_scaled)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(x_train_scaled_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(x_train_scaled_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(x_validate_scaled_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    


# In[11]:


#def get_poly_test(x_train_scaled, y_test):
def get_poly_test():
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_scaled_degree2 = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_scaled_degree2 = pf.transform(x_validate_scaled)
    x_test_scaled_degree2 = pf.transform(x_test_scaled)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(x_train_scaled_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(x_train_scaled_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)

    # predict test
    y_test['tax_value_pred_lm2'] = lm2.predict(x_test_scaled_degree2)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
          "\nTest/Out-of-Sample: ", rmse_test)
    


# In[12]:


def calc_performance(y, yhat, featureN = 2):
        # A version of R Squared which may be more accurate than when done by hand
        r2 = r2_score(y, yhat)
        # Adjusted R Squared
        #adjR2= 1-(1-r2)*(len(y)-1)/(len(y)-featureN-1)
        print('R^2 Coefficient for Variance: ', r2)
      


# # R^2 related

# In[13]:


def get_baseline_1():
#def get_baseline():
    y_train['tax_value_pred_mean']= y_train['tax_value'].mean()
    y_validate['tax_value_pred_mean']= y_validate['tax_value'].mean()

    y_train['tax_value_pred_median'] = y_train['tax_value'].median()
    y_validate['tax_value_pred_median'] = y_validate['tax_value'].median()

    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_mean)**(1/2)

    
    
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_median)**(1/2)

    


# In[14]:


def get_ols_1():
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm'] = lm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm'] = lm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm)**(1/2)

   


# In[15]:


def get_lassolars_1():
    lars = LassoLars(alpha=1.0)
    #scaler = MinMaxScaler()
    #x_train_scaled = x_train_scaled.copy()
    #x_train_scaled[['bedrooms','bathrooms','sq_feet','year_built','tax_amount','fips']] = scaler.fit_transform(x_train_scaled)
    #x_validate_scaled = x_validate_scaled.copy()
    #x_validate_scaled[['bedrooms','bathrooms','sq_feet','year_built','tax_amount','fips']] = scaler.fit_transform(x_validate_scaled)

    lars.fit(x_train_scaled, y_train.tax_value)

    y_train['tax_value_pred_lars'] = lars.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars)**(1/2)

   


# In[16]:


def get_tweedie_1():
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(x_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_glm'] = glm.predict(x_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_glm'] = glm.predict(x_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_glm)**(1/2)


# In[17]:


def get_poly_1():
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_scaled_degree2 = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_scaled_degree2 = pf.transform(x_validate_scaled)
    x_test_scaled_degree2 = pf.transform(x_test_scaled)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(x_train_scaled_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(x_train_scaled_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(x_validate_scaled_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm2)**(1/2)

    


# In[18]:


#def get_poly_test(x_train_scaled, y_test):
def get_poly_test_1():
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_scaled_degree2 = pf.fit_transform(x_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    x_validate_scaled_degree2 = pf.transform(x_validate_scaled)
    x_test_scaled_degree2 = pf.transform(x_test_scaled)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(x_train_scaled_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(x_train_scaled_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)

    # predict test
    y_test['tax_value_pred_lm2'] = lm2.predict(x_test_scaled_degree2)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2)**(1/2)

    


# In[19]:


def performance():
    #use all models to get a fully updated y_train and a y_test dataframe
    get_baseline_1()
    get_ols_1()
    get_lassolars_1()
    get_tweedie_1()
    get_poly_1()
    get_poly_test_1()
    
    calc_performance(y_test.tax_value, y_test.tax_value_pred_lm2)


# In[20]:


#performance()


# In[ ]:




