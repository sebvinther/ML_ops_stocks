# %%
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

# %%
# Load and display the data from CSV to get an overview of the actual data - confirming the values
amd_df = pd.read_csv('AMD_stock_prices.csv')
print(amd_df.head())    

# %%
amd_df

# %%
# Converting the "date" column to timestamp
amd_df['date'] = pd.to_datetime(amd_df['date'])

# %%
import re

# Function to clean column names
def clean_column_name(col):
    # Convert to lowercase
    col = col.lower()
    # Replace any invalid characters with an underscore
    col = re.sub(r'[^a-z0-9_]', '_', col)
    # If the column name doesn't start with a letter, prefix it with 'f_'
    if not re.match(r'^[a-z]', col):
        col = 'f_' + col
    return col

# Apply the function to all column names
amd_df.columns = [clean_column_name(col) for col in amd_df.columns]

print("Cleaned column names:")
print(amd_df.columns.tolist())


# %%
# version 10 for the new feature group
amd_fg = fs.create_feature_group(
    name="amd_stock",
    description="AMD stock dataset from Alpha Vantage",
    version=11,  
    primary_key=["ticker"],
    event_time="date",
    online_enabled=False,
)


# %%
amd_fg.insert(amd_df, write_options={"wait_for_job": False})


