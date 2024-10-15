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
from Feature_pipeline import amd_fg   #Loading in the amd_fg
load_dotenv()

#Getting connected to hopsworks 
api_key = os.environ.get('hopsworks_api') 
project = hopsworks.login(api_key_value=api_key) 
fs = project.get_feature_store()





# %%
#Defining the function to create feature view

def create_stocks_feature_view(fs, version):

    # Loading in the feature groups
    amd_fg = fs.get_feature_group('amd_stock', version=8)
    

    # Defining the query
    ds_query = amd_fg.select(['date', 'open', 'close'])
    

    # Creating the feature view
    feature_view = fs.create_feature_view(
        name='amd_stocks_fv',
        query=ds_query,
        labels=['open']
    )

    return feature_view, amd_fg







