import yaml
from dotenv import load_dotenv
import os
import json
import pandas as pd


import warnings
warnings.filterwarnings('ignore')

load_dotenv()

with open('../config.yml', 'r') as file:
    configs = yaml.safe_load(file)


"""
This class will be used to process the features from the data fetched from the API.
Will be specific to a particular syumbol, start_date, end_date, time_interval.
This would:
    - Function 1: create filename based on symbol, start_date, end_date, time_interval 
    - Function 2: read the json file 
    - Function 3: save the features into a dataframe 
    - Function 4: perform feature engineering
        - create new features by using feature lagging technique
            - use open/close/high/low data from 2, 4, 6, 8, 10, 12 hrs ago to predict the current hour's close price
        - create the new dataframe with features and target 
            - (2hropen, 2hrclose, 2hrhigh, 2hrlow, 4hropen, 4hrclose, ... , 12hrclose) -> current hour's close price 
            - 24 feature columns, 1 target column 
        - drop NaN rows 
            - initial rows will have NaN values as we are using lagging technique
    - Function 5: save the processed features into a new file 
"""


class FeatureProcessor:
    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        if 'start_date' in kwargs:
            self.start_date = kwargs['start_date']
        if 'end_date' in kwargs:
            self.end_date = kwargs['end_date']

    def create_filename(self):
        data_dir = '../../data'
        file_name = f"stockdata_{self.symbol.split('/')[0]}.json"
        return os.path.join(data_dir, file_name)

    def read_json_file(self, file_name):
        # check if the file exists or not
        if not os.path.exists(file_name):
            print(f"File {file_name} does not exist")
            return None
        else:
            with open(file_name, 'r') as file:
                data = json.load(file)
            return data

    # gets json data as input and returns a dataframe
    def save_json_features_to_dataframe(self, data):
        print(data.keys())
        df = pd.json_normalize(data['values'])
        return df

    def feature_engineering(self, df):

        # convert datetime column to datetime type '
        df['datetime'] = pd.to_datetime(df['datetime'])

        # sort data by datetime (earliest at the top)
        df = df.sort_values(
            by='datetime', ascending=True).reset_index(drop=True)

        # Convert price columns from strings to floats
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        # Feature engineering: Creating lag features for 'open', 'high', 'low', 'close'
        # We create lagged features for the past 2, 4, 6, 8, 10, 12 hours
        for lag in range(1, 13):
            df[f'open_lag_{lag}'] = df['open'].shift(lag)
            df[f'high_lag_{lag}'] = df['high'].shift(lag)
            df[f'low_lag_{lag}'] = df['low'].shift(lag)
            df[f'close_lag_{lag}'] = df['close'].shift(lag)

        # Drop rows with NaN values (created by lagging)
        df = df.dropna().reset_index(drop=True)

        # Target variable: predicting the next hour's close price
        df['target'] = df['close'].shift(-1)

        # Drop the last row with NaN in the target column
        df = df.dropna().reset_index(drop=True)

        # sort data by datetime (latest at the top)
        df = df.sort_values(
            by='datetime', ascending=False).reset_index(drop=True)

        # drop original feature columns
        df = df.drop(['open', 'high', 'low', 'close'], axis=1)

        # if start and end date are provided, filter the data
        if hasattr(self, 'start_date') and hasattr(self, 'end_date'):
            df = df[(df['datetime'] >= self.start_date) & (
                df['datetime'] <= self.end_date)]

        return df

    def save_new_features_to_file(self, df):
        data_dir = '../../data/engineered/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # if start and end date are provided, add them to the file name
        if hasattr(self, 'start_date') and hasattr(self, 'end_date'):
            file_name = f"stockdata_{self.symbol.split('/')[0]}_{self.start_date}_{self.end_date}_engineered.csv"
        else:
            file_name = f"stockdata_{self.symbol.split('/')[0]}_engineered.csv"
        df.to_csv(os.path.join(data_dir, file_name), index=False)
        print(f"Features engineered and saved successfully in {file_name}")
