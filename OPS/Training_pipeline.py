# %%
#Importing necessary libraries
import hopsworks
import hsfs
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler  # Import StandardScaler from scikit-learn
import joblib
from prophet import Prophet

load_dotenv()


#Another connection to hopsworks
api_key = os.getenv('HOPSWORKS_API')
 
if not api_key:
    raise ValueError("HOPSWORKS_API environment variable is not set.")

 
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()



# %%
#Getting the feature view
feature_view = fs.get_feature_view(
    name='amd_stock_fv',
    version=19
)

print(feature_view)

# %%
df = feature_view.get_batch_data(read_options={"use_hive": True})



# %%
df['date'] = pd.to_datetime(df['date'])


# %%
df = df.sort_values('date')

# %%
print("Columns in df:")
print(df.columns.tolist())

# %%
prophet_df = df[['date', 'f_1__open']].rename(columns={'date': 'ds', 'f_1__open': 'y'})

# %%
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Remove timezone information from 'ds' column
prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

# Ensure the data is sorted by date
prophet_df.sort_values('ds', inplace=True)

# Verify the data range in prophet_df
print("Data range in prophet_df:", prophet_df['ds'].min().date(), "to", prophet_df['ds'].max().date())

# Get the maximum date in the data
max_date = prophet_df['ds'].max()

# Define the testing period (latest year)
test_end_date = max_date
test_start_date = test_end_date - pd.DateOffset(years=1) + pd.DateOffset(days=1)

# Define the training period (4 years prior to testing period)
train_end_date = test_start_date - pd.DateOffset(days=1)
train_start_date = train_end_date - pd.DateOffset(years=4) + pd.DateOffset(days=1)

# Convert date ranges to strings for display
print(f"Training period: {train_start_date.date()} to {train_end_date.date()}")
print(f"Testing period: {test_start_date.date()} to {test_end_date.date()}")

# Split the data
train_df = prophet_df[
    (prophet_df['ds'] >= train_start_date) & (prophet_df['ds'] <= train_end_date)
].copy()

test_df = prophet_df[
    (prophet_df['ds'] >= test_start_date) & (prophet_df['ds'] <= test_end_date)
].copy()

# Check if test_df is not empty
if test_df.empty:
    print("Test DataFrame is empty. Please check the test date range.")
else:
    # Proceed with modeling
    model = Prophet(daily_seasonality=True)
    model.fit(train_df)
    
    # Using the dates from the test set for prediction
    future_dates = test_df[['ds']]
    
    # Generate forecasts
    forecast = model.predict(future_dates)
    
    # Merge the forecast with the actual test data
    forecast_df = forecast[['ds', 'yhat']].set_index('ds')
    actual_df = test_df.set_index('ds')
    
    comparison_df = actual_df.join(forecast_df, how='left').dropna()
    
    # Sort the data by date (ascending order)
    comparison_df.sort_index(inplace=True)
    
    # Compute evaluation metrics
    mae = mean_absolute_error(comparison_df['y'], comparison_df['yhat'])
    rmse = mean_squared_error(comparison_df['y'], comparison_df['yhat'], squared=False)
    
    print(f"\nMean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df.index, comparison_df['y'], label='Actual')
    plt.plot(comparison_df.index, comparison_df['yhat'], label='Predicted')
    plt.legend()
    plt.title('Actual vs. Predicted AMD Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


# %%
import joblib
import hopsworks
import os
import shutil
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# Save the trained Prophet model to a file
joblib.dump(model, 'prophet_model.pkl')

# Get your Hopsworks API key (ensure it's set in your environment variables)
api_key = os.environ.get('HOPSWORKS_API')

# Log in to Hopsworks
project = hopsworks.login(api_key_value=api_key)

# Get the model registry
mr = project.get_model_registry()

# Define the model name and metadata
model_name = "ProphetModel"
description = "Prophet model for time series forecasting AMD stock prices"

# Prepare the input example (pandas DataFrame)
input_example = train_df[['ds']].head(1)
input_schema = Schema(input_example)

# Prepare the output example (pandas DataFrame)
output_example = forecast[['yhat']].head(1)
output_schema = Schema(output_example)

# Create the model schema
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# Define evaluation metrics
metrics = {
    'MAE': mae,
    'RMSE': rmse
}

# Create the model in the registry
model_registry_entry = mr.python.create_model(
    name=model_name,
    description=description,
    input_example=input_example,
    model_schema=model_schema,
    metrics=metrics
)

# Ensure the model directory exists
model_dir = 'prophet_model_dir'
os.makedirs(model_dir, exist_ok=True)

# Move the model file into the model directory
shutil.move('prophet_model.pkl', os.path.join(model_dir, 'prophet_model.pkl'))

# Save the model artifacts to Hopsworks
model_registry_entry.save(model_dir)

print(f"Model '{model_name}' saved successfully to the Hopsworks Model Registry.")


# %%
#Setting up train & test split dates
train_start = "2020-03-10"
train_end = "2023-12-31"

test_start = '2024-01-01'
test_end = "2024-10-14"

# %%
#Connecting to hopsworks
api_key = os.environ.get('HOPSWORKS_API')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

#Another connection to hopsworks
api_key = os.getenv('HOPSWORKS_API')
connection = hsfs.connection()
fs = connection.get_feature_store()

# %%
import hopsworks
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch the Hopsworks API key from environment variables
api_key = os.environ.get('HOPSWORKS_API')
if not api_key:
    raise ValueError("HOPSWORKS_API environment variable is not set or is empty.")

# Login to your Hopsworks project
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# Retrieve the feature view
feature_view = fs.get_feature_view(
    name='amd_stock_fv',
    version=19
)

# Setting up train & test split dates
train_start = "2020-03-10"
train_end = "2023-12-31"

test_start = "2024-01-01"
test_end = "2024-10-14"

# Creating the train/test split on the feature view with the split dates
feature_view.create_train_test_split(
    train_start=train_start,
    train_end=train_end,
    test_start=test_start,
    test_end=test_end,
    data_format='csv',
    coalesce=True,
    statistics_config={'histogram': True, 'correlations': True}
)



