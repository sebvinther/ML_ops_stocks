#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Import necessary libraries
import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt   
import os                         
import joblib                     
import hopsworks                  
import re

# Import specific modules from scikit-learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder   # For data preprocessing
from sklearn.metrics import accuracy_score                        # For evaluating model accuracy

from dotenv import load_dotenv
import os
load_dotenv()

#Connecting to hopsworks
api_key = os.environ.get('HOPSWORKS_API')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()


# In[27]:


# Load and display the data from CSV to get an overview of the actual data - confirming the values
amd_df = pd.read_csv('AMD_stock_prices.csv')
print(amd_df.head())    


# In[28]:


amd_df


# In[29]:


# Converting the "date" column to timestamp
amd_df['date'] = pd.to_datetime(amd_df['date'])


# In[30]:


# Defining the stocks feature group
amd_fg = fs.get_or_create_feature_group(
    name="amd_stock",
    description="amd stock dataset from alpha vantage",
    version=8,
    primary_key=["ticker"],
    event_time=['date'],
    online_enabled=False,
)


# In[25]:


# version 2 for the new feature group
amd_fg = fs.create_feature_group(
    name="amd_stock",
    description="AMD stock dataset from Alpha Vantage",
    version=2,  # Incremented version number
    primary_key=["ticker"],
    event_time="date",
    online_enabled=False,
)

# Insert data into the new feature group
amd_fg.insert(amd_df, write_options={"wait_for_job": False})

