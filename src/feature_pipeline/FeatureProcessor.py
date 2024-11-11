from pathlib import Path
import yaml
from dotenv import load_dotenv
import os
import json
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# Define the base directory as the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load the configuration file using BASE_DIR
CONFIG_FILE = BASE_DIR / "src" / "config.yml"
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)


class FeatureProcessor:
    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        if 'start_date' in kwargs:
            self.start_date = kwargs['start_date']
        if 'end_date' in kwargs:
            self.end_date = kwargs['end_date']

    def create_filename(self):
        """Create the file name for the JSON data file based on the symbol."""
        data_dir = BASE_DIR / "data"
        file_name = f"stockdata_{self.symbol.split('/')[0]}.json"
        return data_dir / file_name

    def read_json_file(self, file_path):
        """Read data from a JSON file if it exists."""
        if not file_path.exists():
            print(f"File {file_path} does not exist")
            return None
        else:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data

    def save_json_features_to_dataframe(self, data):
        """Convert JSON data to a DataFrame."""
        df = pd.json_normalize(data['values'])
        return df

    def feature_engineering(self, df):
        """Perform feature engineering on the DataFrame."""
        # Convert datetime column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Sort data by datetime (earliest at the top)
        df = df.sort_values(
            by='datetime', ascending=True).reset_index(drop=True)

        # Convert price columns from strings to floats
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        # Creating lag features for 'open', 'high', 'low', 'close' for the past 1 to 12 hours
        for lag in range(0, 13):
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

        # Sort data by datetime (latest at the top)
        df = df.sort_values(
            by='datetime', ascending=False).reset_index(drop=True)

        # Drop original feature columns
        df = df.drop(['open', 'high', 'low', 'close'], axis=1)

        # Filter data if start and end dates are provided
        if hasattr(self, 'start_date') and hasattr(self, 'end_date'):
            df = df[(df['datetime'] >= self.start_date)
                    & (df['datetime'] <= self.end_date)]

        # create uid column for feature store combining strings of double lag features
        df['uid'] = df['open_lag_1'].astype(str) + '_' + df['high_lag_2'].astype(
            str) + '_' + df['low_lag_3'].astype(str) + '_' + df['close_lag_4'].astype(str)

        # df['uid'] = f"{df['open_lag_1']}_{df['high_lag_2']}_{df['low_lag_3']}_{df['close_lag_4']}"

        return df

    def save_new_features_to_file(self, df):
        """Save engineered features to a CSV file."""
        data_dir = BASE_DIR / "data" / "engineered"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Construct the filename for engineered data
        if hasattr(self, 'start_date') and hasattr(self, 'end_date'):
            file_name = f"stockdata_{self.symbol.split('/')[0]}_{self.start_date}_{self.end_date}_engineered.csv"
        else:
            file_name = f"stockdata_{self.symbol.split('/')[0]}_engineered.csv"

        # Save the DataFrame to a CSV file
        file_path = data_dir / file_name
        df.to_csv(file_path, index=False)
        print(f"Features engineered and saved successfully in {file_path}")
