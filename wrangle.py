#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

import env as e


#def get_db_url(database):
#    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

"""
USAGE: 
Use `from wrangle import wrangle_zillow` at the top of your notebook.
This 
"""

def get_new_zillow_data():
    """Returns a dataframe of all 2017 properties that are Single Family Residential"""

    #sql = """
    #select 
    #bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    #from properties_2017
    #join propertylandusetype using (propertylandusetypeid)
    #where propertylandusedesc = "Single Family Residential"
    #"""

    sql = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, transactiondate, propertylandusetypeid
    FROM properties_2017 AS prop
    JOIN predictions_2017 AS pred ON prop.parcelid = pred.parcelid
    WHERE transactiondate LIKE "2017%" AND propertylandusetypeid = "261" ;

    '''
    return pd.read_sql(sql, e.get_db_url("zillow"))


def handle_nulls(df):    
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    return df


def optimize_types(df):
    # Convert some columns to integers
    # fips, yearbuilt, and bedrooms can be integers
    df["fips"] = df["fips"].astype(int)
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    return df


def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    df = df[df.calculatedfinishedsquarefeet < 10000]

    return df

def rename_cols(df):
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                            'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet':'sq_feet', 
                            'taxvaluedollarcnt':'tax_value',
                            'yearbuilt':'year_built',
                            'taxamount':'tax_amount'})
    return df

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'tax_amount', 'sq_feet'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test


def clean_zillow(df):
    df = handle_nulls(df)

    df = optimize_types(df)

    df = handle_outliers(df)

    df = rename_cols(df)

    df.to_csv("zillow.csv", index=False)

    return df

def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """

    filename = "zillow.csv"

    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = get_new_zillow_data()

        df = clean_zillow(df)

    return df


# In[ ]:




