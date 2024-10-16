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

#Connecting to hopsworks
api_key = os.environ.get('HOPSWORKS_API')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

#Another connection to hopsworks
api_key = os.getenv('HOPSWORKS_API')
connection = hsfs.connection()
fs = connection.get_feature_store()

# %%
#Getting the feature view
feature_view = fs.get_feature_view(
    name='amd_stock_fv',
    version=25
)

# %%
print(feature_view)

# %%
df = feature_view.get_batch_data(read_options={"use_hive": True})


# %%
# Inspect sample 'date' values before conversion
print("Sample 'date' values before conversion:")
print(df['date'].head())

# %%
# Convert 'date' to datetime with error handling
df['date'] = pd.to_datetime(df['date'], errors='coerce')


# %%
# Drop rows with invalid 'date'
initial_count = len(df)
df = df.dropna(subset=['date'])
dropped_count = initial_count - len(df)
if dropped_count > 0:
    print(f"Dropped {dropped_count} rows due to invalid 'date'.")


# %%
df = df.sort_values('date')

# %%
print("Columns in df:")
print(df.columns.tolist())

# %%
df.head()

# %%
# Prepare prophet_df
prophet_df = df[['date', 'f_1__open']].rename(columns={'date': 'ds', 'f_1__open': 'y'})
print("Columns in Prophet_df:")
print(prophet_df.columns.tolist())

# %%
# Inspect sample 'ds' values after conversion
print("Sample 'ds' values after conversion:")
print(prophet_df['ds'].head())

# %%
# Remove timezone information from 'ds' column
prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

# %%
# Ensure the data is sorted by date
prophet_df.sort_values('ds', inplace=True)



# %%
# Define train and test split dates as specified
train_start = "2020-03-10"
train_end = "2023-12-31"

test_start = "2024-01-01"
test_end = "2024-10-14"

train_start_date = pd.to_datetime(train_start)
train_end_date = pd.to_datetime(train_end)
test_start_date = pd.to_datetime(test_start)
test_end_date = pd.to_datetime(test_end)


# %%
# Check if max_date is valid
if pd.isnull(max_date):
    print("Maximum date is NaT. Cannot proceed with splitting.")
else:
    # Check if test_start_date is after max_date
    if test_start_date > max_date:
        print("Test start date is after the last date in the data. Adjusting test_start_date.")
        test_start_date = max_date - pd.DateOffset(years=1) + pd.DateOffset(days=1)
        print(f"Adjusted test_start_date to: {test_start_date.date()}")
    
    # Check if test_end_date is after max_date
    if test_end_date > max_date:
        print("Test end date is after the last date in the data. Adjusting test_end_date.")
        test_end_date = max_date
        print(f"Adjusted test_end_date to: {test_end_date.date()}")
    
    print(f"Training period: {train_start_date.date()} to {train_end_date.date()}")
    print(f"Testing period: {test_start_date.date()} to {test_end_date.date()}")

     # Split the data
    train_df = prophet_df[
        (prophet_df['ds'] >= train_start_date) & (prophet_df['ds'] <= train_end_date)
    ].copy()
    
    test_df = prophet_df[
        (prophet_df['ds'] >= test_start_date) & (prophet_df['ds'] <= test_end_date)
    ].copy()
    
    print(f"Training DataFrame has {len(train_df)} records.")
    print(f"Testing DataFrame has {len(test_df)} records.")

# %%
import hopsworks
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import matplotlib.pyplot as plt
import joblib
import shutil
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


# Check if test_df is not empty and ds has valid dates
if test_df.empty or prophet_df['ds'].isnull().all():
    print("Test DataFrame is empty or 'ds' contains only NaT. Please check the test date range.")
else:
    # Proceed with modeling
    model = Prophet(daily_seasonality=True)
    model.fit(train_df)
    print("Prophet model trained successfully.")
    
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
    
    # Save the trained Prophet model to a file
    joblib.dump(model, 'prophet_model.pkl')
    print("Prophet model saved successfully as 'prophet_model.pkl'.")
    
    
    # Register the model with Hopsworks Model Registry
    # Get the model registry
    mr = project.get_model_registry()
    print("Accessed Hopsworks Model Registry.")
    
    # Define the model name and metadata
    model_name = "ProphetModel"
    description = "Prophet model for time series forecasting AMD stock prices"
    
    # Prepare the input example 
    input_example = train_df[['ds']].head(1)
    input_schema = Schema(input_example)
    
    # Prepare the output example
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
    print(f"Model '{model_name}' created successfully in the registry.")
    
    # Ensure the model directory exists
    model_dir = 'prophet_model_dir'
    os.makedirs(model_dir, exist_ok=True)
    
    # Move the model file into the model directory
    shutil.move('prophet_model.pkl', os.path.join(model_dir, 'prophet_model.pkl'))
    print(f"Moved model file to '{model_dir}'.")
    
    # Save the model artifacts to Hopsworks
    model_registry_entry.save(model_dir)
    print(f"Model '{model_name}' saved successfully to the Hopsworks Model Registry.")


