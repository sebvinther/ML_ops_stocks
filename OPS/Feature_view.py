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

# %%
#Getting connected to hopsworks 
api_key = os.environ.get('hopsworks_api') 
project = hopsworks.login(api_key_value=api_key) 
fs = project.get_feature_store()


# Check if the feature group already exists before creating it
def create_stocks_feature_view(fs, version):
    try:
        amd_fg = fs.get_feature_group('amd_stock', version=version)
        print(f"Feature group 'amd_stock' with version {version} already exists.")
    except RestAPIError as e:
        if e.error_code == 270089:
            print(f"Creating feature group 'amd_stock' with version {version}.")
            # Loading in the feature groups
            amd_fg = fs.get_feature_group('amd_stock', version=7)
            # Defining the query
            ds_query = amd_fg.select(['date', 'open', 'close'])
            # Creating the feature view
            feature_view = fs.create_feature_view(
                name='amd_stocks_fv',
                query=ds_query,
                labels=['open']
            )
            return feature_view, amd_fg
        else:
            raise e





# %%
#Defining the function to create feature view

def create_stocks_feature_view(fs, version):

    # Loading in the feature groups
    amd_fg = fs.get_feature_group('amd_stock', version=8)
    

    # Defining the query
    ds_query = amd_fg.select(['date', 'open', 'close'])\
        

    # Creating the feature view
    feature_view = fs.create_feature_view(
        name='amd_stocks_fv',
        query=ds_query,
        labels=['open']
    )

    return feature_view, amd_fg

# %%
def create_stocks_feature_view(fs, version):
    # Load and preprocess data
    amd_df = pd.read_csv('AMD_stock_prices.csv')
    amd_df['date'] = pd.to_datetime(amd_df['date'])

    # Ensure 'ticker' column exists
    if 'ticker' not in amd_df.columns:
        amd_df['ticker'] = 'AMD'

    # Clean column names
    def clean_column_name(col):
        col = col.lower()
        col = re.sub(r'[^a-z0-9_]', '_', col)
        if not re.match(r'^[a-z]', col):
            col = 'f_' + col
        return col

    amd_df.columns = [clean_column_name(col) for col in amd_df.columns]

    # Convert data types
    numeric_columns = [col for col in amd_df.columns if col not in ['date', 'ticker']]
    for col in numeric_columns:
        amd_df[col] = pd.to_numeric(amd_df[col], errors='coerce')

    # Create or get the feature group
    try:
        amd_fg = fs.get_feature_group('amd_stock', version=version)
        print(f"Feature group 'amd_stock' with version {version} already exists.")
    except RestAPIError as e:
        print(f"Creating feature group 'amd_stock' with version {version}.")
        amd_fg = fs.create_feature_group(
            name='amd_stock',
            version=version,
            description='AMD stock dataset',
            primary_key=['ticker'],
            event_time='date',
            online_enabled=False
        )
        # Insert data into the feature group
        amd_fg.insert(amd_df, write_options={"wait_for_job": True})
        print("Data inserted into the feature group.")

    # Create or get the feature view
    try:
        feature_view = fs.get_feature_view('amd_stocks_fv', version=version)
        print(f"Feature view 'amd_stocks_fv' with version {version} already exists.")
    except RestAPIError as e:
        print(f"Creating feature view 'amd_stocks_fv' with version {version}.")
        # Define the query
        feature_query = amd_fg.select_all()
        # Create the feature view
        feature_view = fs.create_feature_view(
            name='amd_stocks_fv',
            version=version,
            description='Feature view for AMD stock data',
            labels=['f_4__close'],  # Adjust label column as needed
            query=feature_query
        )
        print("Feature view created.")

    return feature_view, amd_fg

version = 8  # Use the desired version number

feature_view, amd_fg = create_stocks_feature_view(fs, version)

# %%
# Verify the feature view
print("Features in the feature view:")
for feature in feature_view.features:
    print(f"- {feature.name} ({feature.type})")


print("Sample data from the feature view:")



# %%
#Defining a function to get fixed data from the feature view
from OPS.preprocessing_stocks import extract_business_day


def fix_data_from_feature_view(df,start_date,end_date):
    df = df.sort_values("date")
    df = df.reset_index()
    df = df.drop(columns=["index"])

    # Create a boolean mask for rows that fall within the date range
    mask = (pd.to_datetime(df['date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(df['date']) <= pd.to_datetime(end_date))
    len_df = np.shape(df)
    df = df[mask] # Use the boolean mask to filter the DataFrame
    print('From shape {} to {} after cropping to given date range: {} to {}'.format(len_df,np.shape(df),start_date,end_date))

    # Get rid off all non-business days
    isBusinessDay, is_open = extract_business_day(start_date,end_date)
    is_open = [not i for i in is_open] # Invert the mask to be able to drop all non-buisiness days

    filtered_df = df.drop(df[is_open].index) # Use the mask to filter the rows of the DataFrame
    print('From shape {} to {} after removing non-business days'.format(np.shape(df),np.shape(filtered_df)))
    
    print(filtered_df)
    


