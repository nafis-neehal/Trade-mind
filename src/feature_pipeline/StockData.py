"""
Module to fetch stock data from the TwelveData Stock API and save it to a JSON file.
"""

import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv()

# Define the base directory as the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load the configuration file using BASE_DIR
CONFIG_FILE = BASE_DIR / "src" / "config.yml"
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    configs = yaml.safe_load(file)


class StockData:
    def __init__(self, symbol):
        self.symbol = symbol

    #### Used for retraining model ####

    def calculate_date_range(self, before_days):
        """Calculate the date range for data retrieval."""
        end_date = datetime.now() + timedelta(days=1)
        start_date = end_date - timedelta(days=before_days)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    def fetch_range_data_from_api(self, start_date, end_date):
        """Fetch data from the API within a specified date range."""
        url = configs["stock_api_params"]["base_url"] + \
            configs["stock_api_params"]["endpoint"]
        query_string = (
            f"?apikey={os.getenv('STOCK_API_KEY')}&symbol={self.symbol}"
            f"&interval={configs['stock_api_params']['time_interval']}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&timezone={configs['stock_api_params']['timezone']}"
        )
        print(url + query_string)
        response = requests.get(url + query_string, timeout=10)
        return response

    def save_response_to_json(self, response, file_name):
        """Save the API response to a JSON file."""
        data_dir = BASE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        file_path = data_dir / file_name
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(response.json(), json_file, indent=4)
            print(f"Data fetched successfully and saved in {file_path}")

    def init_data(self, days_before):
        """Initialize data by fetching a specified number of days before the current date."""
        start_date, end_date = self.calculate_date_range(days_before)
        response = self.fetch_range_data_from_api(start_date, end_date)

        if response.status_code == 200:
            file_name = f"stockdata_{self.symbol.split('/')[0]}.json"
            self.save_response_to_json(response, file_name)
        else:
            print(f"Failed to fetch data: {response.status_code}")

    #### Used for hourly updates ####
    def update_data(self):
        """Update data by fetching the latest available information and appending it."""
        file_name = f"stockdata_{self.symbol.split('/')[0]}.json"
        data_file_path = BASE_DIR / "data" / file_name

        # Load existing data
        with open(data_file_path, 'r', encoding='utf-8') as data_file:
            data = json.load(data_file)
            last_datetime_str = data['values'][0]['datetime']
            last_datetime = datetime.strptime(
                last_datetime_str, '%Y-%m-%d %H:%M:%S')

        # Calculate the date range for the update
        start_date = last_datetime
        end_date = datetime.now() + timedelta(days=1)

        response = self.fetch_range_data_from_api(start_date, end_date)

        if response.status_code == 200:
            new_data = response.json()

            # Check if the latest data is already up to date
            if new_data['values'][0]['datetime'] == last_datetime_str:
                print("Data already up to date")
                return -1  # Data is already up to date
            else:
                # Remove the last item in new_data to avoid duplication
                new_data['values'] = new_data['values'][:-1]
                data['values'] = new_data['values'] + data['values']

                # Save the updated data back to the JSON file
                with open(data_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"Data in {file_name} updated successfully")
                return 1  # Data updated successfully
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return 0
