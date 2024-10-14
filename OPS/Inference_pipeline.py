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
api_key = os.environ.get('hopsworks_api')
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
feature_view = fs.get_feature_view('amd_stocks_fv', 8)
feature_view.init_batch_scoring(training_dataset_version=1)

# %%
df = feature_view.get_batch_data(start_time=start_date, end_time=end_date)

# %%
df.head()

# %%
import joblib

the_model = mr.get_model("ProphetModel", version=2)
model_dir = the_model.download()

model = joblib.load(model_dir + "/Prophet_model.pkl")

# %%
predictions = model.predict(df)

# %%
predictions 

# %%
df['predictions'] = predictions.tolist()

# %%
api_key = os.environ.get('hopsworks_api')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# %%
results_fg = fs.get_or_create_feature_group(
    name= 'stock_prediction_results',
    version = 1,
    description = 'Predction of AMD close stock price',
    primary_key = ['ticker'],
    event_time = ['date'],
    online_enabled = False,
)

# %%
results_fg.insert(df, write_options={"wait_for_job" : False})


