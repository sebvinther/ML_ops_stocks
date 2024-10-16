# %%
# Importing necessary libraries
import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt   
import os                         
import joblib                     
import hopsworks                  
import re
from hsfs.client.exceptions import RestAPIError
#Making the notebook able to fetch from the .env file
from dotenv import load_dotenv
load_dotenv()

#Getting connected to hopsworks 
api_key = os.environ.get('HOPSWORKS_API') 
project = hopsworks.login(api_key_value=api_key) 
fs = project.get_feature_store()





# %%
#Defining the function to create feature view

def create_stocks_feature_view(fs, version):

    # Loading in the feature groups
    amd_fg = fs.get_feature_group('amd_stock', version=28)
    

    # Defining the query
    ds_query = amd_fg.select(['date', 'f_1__open', 'f_4__close'])
    

    # Creating the feature view
    feature_view = fs.create_feature_view(
        name='amd_stock_fv',
        query=ds_query,
        labels=['f_4__close']
    )

    return feature_view, amd_fg

# %%
#Creating the feature view
try:
    feature_view = fs.get_feature_view("amd_stock_fv", version=24)
    amd_fg = fs.get_feature_group('amd_stock', version=28)
except:
    feature_view, amd_fg = create_stocks_feature_view(fs, 24)


