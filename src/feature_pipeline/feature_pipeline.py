import yaml
from dotenv import load_dotenv
import os
import pandas as pd

# Custom imports
from FeatureProcessor import FeatureProcessor
from HopsworkFeatureStore import HopsworkFeatureStore
from StockData import StockData


import warnings
warnings.filterwarnings('ignore')

load_dotenv()

with open('../config.yml', 'r') as file:
    configs = yaml.safe_load(file)


def run_stock_profile(symbol, init=False, **kwargs):
    stock = StockData(symbol)
    if init:
        # <--- use this if we don't have any data on stock yet
        if 'days_before' in kwargs:
            stock.init_data(days_before=kwargs['days_before'])
        else:
            # <---- default to 30 days, hourly data
            stock.init_data(days_before=30)
    else:
        # <--- use this to periodically update the data upto latest hour
        stock.update_data()


def run_feature_engineering_pipeline(symbol):
    # Process the features for stock
    feature_processor = FeatureProcessor(
        symbol=symbol
    )

    file_name = feature_processor.create_filename()
    json_data = feature_processor.read_json_file(file_name)
    if json_data:
        print(
            f"after reading updated file {json_data.keys()}")
        df = feature_processor.save_json_features_to_dataframe(json_data)
        engineered_df = feature_processor.feature_engineering(df)
        feature_processor.save_new_features_to_file(engineered_df)


def run_feature_store_ingestion(symbol):
    # Define your configurations
    PROJECT_NAME = "trade_mind"  # Replace with your Hopsworks project name
    # Replace with your feature group name
    FEATURE_GROUP_NAME = f"{symbol.split('/')[0].lower()}_features"

    # Replace with your Hopsworks API key
    API_KEY = os.getenv("HOPSWORKS_API_KEY")

    data_eng_dir = '../../data/engineered/'
    file_name = f"stockdata_{symbol.split('/')[0]}_engineered.csv"

    # Replace with your CSV file path
    CSV_PATH = os.path.join(data_eng_dir, file_name)

    # Initialize and run the pipeline
    hopswork_fs = HopsworkFeatureStore(
        PROJECT_NAME,
        FEATURE_GROUP_NAME,
        API_KEY,
        CSV_PATH
    )
    hopswork_fs.run_pipeline()


if __name__ == "__main__":

    # Fetch data for BTC/USD for the last 20 days - initial data fetch
    symbol = "BTC/USD"
    print(f"Fetching data for {symbol}...")
    run_stock_profile(symbol, init=False)
    print(f"Data fetched for {symbol}")
    print("Running feature engineering pipeline...")
    run_feature_engineering_pipeline(symbol)
    print("Feature engineering pipeline completed")
    print("Running feature store ingestion pipeline...")
    run_feature_store_ingestion(symbol)
    print("Feature store ingestion pipeline completed")
