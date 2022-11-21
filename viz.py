#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

import wrangle as w
import env
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")


# In[3]:


df = w.wrangle_zillow()


# In[4]:


train, validate, test = w.split_data(df)


# In[5]:


x_train = train[['bedrooms', 'bathrooms', 'sq_feet']]
y_train = train[['tax_value']]

x_validate = validate[['bedrooms', 'bathrooms', 'sq_feet']]
y_validate = validate[['tax_value']]

x_test = test[['bedrooms', 'bathrooms', 'sq_feet']]
y_test = test[['tax_value']]


# In[6]:


def get_strip(string):
    train_viz = train.sample(frac=0.04, replace=True, random_state=777)
    sns.stripplot(x=string, y='tax_value', data=train_viz, size=3)
    plt.show()


# In[14]:


def get_scatter_sq_feet():
    train_viz = train.sample(frac=0.04, replace=True, random_state=777)
    train_viz.plot.scatter('sq_feet','tax_value')
    plt.title('Square Feet vs Tax Value')


# In[15]:


#def get_fips():
#    train_viz = train.sample(frac=0.04, replace=True, random_state=777)
#    #fipsname = map(train_viz['fips'],['LA', 'Orange','Something else'])
#    sns.stripplot(x=train_viz['fips'], y='tax_value', data=train_viz, size=3)
#    plt.show()
#    print('6037 = Los Angeles County\n6059 = Orange County\n6111 = Ventura County')


# In[19]:


def get_fips():
    plt.bar(['Los Angeles','Orange','Ventura'], pd.value_counts(train['fips']), color=['#FF6663','orange','#6EB5FF'])


# In[9]:


def get_corr(string):
    corr, p = stats.pearsonr(x_train[string], y_train)
    print(f'corr = {float(corr):.4f}')
    print(f'p    = {p:.4f}')

