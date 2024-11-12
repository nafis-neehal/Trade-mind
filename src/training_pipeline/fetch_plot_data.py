import pprint
from dotenv import load_dotenv
import yaml
from pathlib import Path
from Trainer import Trainer  # Assuming Trainer.py is in the same directory
import requests
import os
import json
import warnings
import pandas as pd
import hsml
warnings.filterwarnings('ignore')

load_dotenv()

# Hopsworks API configuration
# Or replace with your actual API key
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Define the base directory as the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Use BASE_DIR to dynamically load the config file
CONFIG_FILE = BASE_DIR / "src" / "config.yml"
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)


# Initialize Trainer instance with Hopsworks project configurations
symbol = configs['stock_api_params']['symbol']
# Initialize Trainer with relevant project details
trainer = Trainer(
    project_name=configs['hopsworks']['project_name'],
    feature_group_name=f"{symbol.split('/')[0].lower()}_features",
    model_registry_name=f"{symbol.split('/')[0].lower()}_regressor_model",
    api_key=os.getenv("HOPSWORKS_API_KEY")
)


def return_plot_data(hours):
    # Create or retrieve feature view
    trainer.create_feature_view()

    # Get the plot data from the feature view
    input_df = trainer.get_plot_data_from_feature_view(hours)

    # get the datetime column from the input_df
    datetime_column = input_df['datetime']

    input_features, input_labels = trainer.get_features_labels(input_df)

    return input_features, input_labels, datetime_column


def return_plot_data_prediction(input_features):
    # Get the prediction
    prediction = trainer.predict_with_hopsworks_api(input_features)
    return prediction


def get_plot_data(hours):
    # Get the plot data
    input_features, input_labels, datetime_column = return_plot_data(
        hours)
    prediction = return_plot_data_prediction(input_features)
    return {"features": input_features, "labels": input_labels,
            "prediction": prediction['predictions'], "datetime": datetime_column}


# f, l, d = return_plot_data()
# print(f)
# print(l)
# print(trainer.predict_with_hopsworks_api(f))


# # Example input data (replace with your actual input structure)
# input_ls = [76480.91, 76648.94, 76390.51, 76541.99, 76330.78, 76339.94, 76312.67, 76319.28, 76246.58, 76413.26, 76206.41, 76333.14, 76396.64, 76732.32, 76151.9, 76244.62, 76279.09, 76429.21, 76222.1, 76396.63, 76122.3, 76283.43, 75758.58,
#             76272.1, 76349.99, 76366.2, 76093.0, 76117.98, 76395.53, 76456.16, 76319.87, 76348.18, 76461.01, 76481.48, 76300.38, 76395.53, 76330.91, 76517.26, 76323.53, 76461.02, 76532.39, 76583.19, 76319.32, 76330.91, 76509.82, 76570.6, 76415.72, 76534.61]
# input_columns = ["open_lag_1", "high_lag_1", "low_lag_1", "close_lag_1", "open_lag_2", "high_lag_2", "low_lag_2", "close_lag_2", "open_lag_3", "high_lag_3", "low_lag_3", "close_lag_3", "open_lag_4", "high_lag_4", "low_lag_4", "close_lag_4", "open_lag_5", "high_lag_5", "low_lag_5", "close_lag_5", "open_lag_6", "high_lag_6", "low_lag_6", "close_lag_6",
#                  "open_lag_7", "high_lag_7", "low_lag_7", "close_lag_7", "open_lag_8", "high_lag_8", "low_lag_8", "close_lag_8", "open_lag_9", "high_lag_9", "low_lag_9", "close_lag_9", "open_lag_10", "high_lag_10", "low_lag_10", "close_lag_10", "open_lag_11", "high_lag_11", "low_lag_11", "close_lag_11", "open_lag_12", "high_lag_12", "low_lag_12", "close_lag_12"]
# input_df = pd.DataFrame([input_ls], columns=input_columns)
