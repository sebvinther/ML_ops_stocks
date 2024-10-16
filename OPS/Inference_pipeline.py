# %%
import pandas as pd 
import hopsworks 
from datetime import datetime, timedelta

import numpy as np



#Making the notebook able to fetch from the .env file
from dotenv import load_dotenv
import os

load_dotenv()

# %%
api_key = os.environ.get('HOPSWORKS_API')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
mr = project.get_model_registry() 

# %%
start_date = datetime.now() - timedelta(hours=48)
print(start_date.strftime("%Y-%m-%d"))

# %%
end_date = datetime.now() - timedelta(hours=24)
print(end_date.strftime("%Y-%m-%d"))

# %%
feature_view = fs.get_feature_view('amd_stock_fv', 18)
feature_view.init_batch_scoring(training_dataset_version=1)

# %%
df = feature_view.get_batch_data(start_time=start_date, end_time=end_date)

# %%
df.head()

# %%
import joblib

the_model = mr.get_model("ProphetModel", version=8)
model_dir = the_model.download()

model = joblib.load(model_dir + "/Prophet_model.pkl")

# %%
print("Columns in df:", df.columns.tolist())

# %%
# Remove timezone information from 'ds' column
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)


# %%
print(df['date'].dtype)


# %%
df.head()

# %%
prophet_df = df[['date', 'f_1__open']].rename(columns={'date': 'ds', 'f_1__open': 'y'})

# %%
prophet_df.head()

# %%
# Predict using the model
predictions = model.predict(prophet_df)

# Display predictions
print(predictions)


# %%
predictions 

# %%
api_key = os.environ.get('HOPSWORKS_API')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# %%
results_fg = fs.get_or_create_feature_group(
    name= 'stock_prediction_results',
    version = 4,
    description = 'Predction of AMD close stock price',
    primary_key = ['y'],
    event_time = ['ds'],
    online_enabled = False,
)

# %%
results_fg.insert(prophet_df, write_options={"wait_for_job" : False})


