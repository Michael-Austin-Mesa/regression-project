#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import wrangle as w


# In[2]:


def plot_variable_pairs(df, target_variable):
    for n in range(0,(len(df.columns))):
        sns.lmplot(x=df.columns[n], y=target_variable, data=df)
        plt.show()


# In[3]:


def plot_categorical_and_continuous_vars(df, cat_x, cont_y):
       for n in range(0,len(cont_y)):
            for i in range(0,len(cat_x)):
                df.boxplot(column=cont_y[i], by=cat_x[n], figsize=(5,5))
                plt.show()
                #sns.catplot(data=df, kind='swarm',x=cat_x[n],y=cont_y[i], col=cat_x[n])
                #sns.swarmplot(data=df, x=cat_x[n], y=cont_y[i], size=1)
                sns.stripplot(data=df, x=cat_x[n],y=cont_y[i])
                plt.show()
                sns.violinplot(data=df,x=cat_x[n],y=cont_y[i])
                plt.show()


# In[ ]:




