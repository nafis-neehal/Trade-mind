import yaml
from dotenv import load_dotenv
import os
import requests
import json
import pandas as pd
import pprint
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

with open('../config.yml', 'r') as file:
    configs = yaml.safe_load(file)


class StockData:
    def __init__(self, symbol):
        self.symbol = symbol

    #### Used for retraining model ####

    def calculate_date_range(self, before_days):
        from datetime import datetime, timedelta
        end_date = datetime.now() + timedelta(days=1)
        start_date = end_date - timedelta(days=before_days)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    def fetch_range_data_from_api(self, start_date, end_date):
        url = configs["stock_api_params"]["base_url"] + \
            configs["stock_api_params"]["endpoint"]
        query_string = f"?apikey={os.getenv('STOCK_API_KEY')}&symbol={self.symbol}&interval={configs['stock_api_params']['time_interval']}"
        query_string += f"&start_date={start_date}&end_date={end_date}&timezone={configs['stock_api_params']['timezone']}"
        print(url + query_string)
        response = requests.get(url + query_string)
        return response

    def save_response_to_json(self, response, file_name):
        # check if dir exists o/w create it
        data_dir = '../../data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        with open(os.path.join(data_dir, file_name), 'w+') as file:
            json.dump(response.json(), file, indent=4)
            print(f"Data fetched successfully and saved in {file_name}")

    def init_data(self, days_before):
        start_date, end_date = self.calculate_date_range(days_before)
        response = self.fetch_range_data_from_api(start_date, end_date)

        if response.status_code == 200:
            # save the data in a pretty format in json file with indent - use json dump
            self.save_response_to_json(
                response,
                # f"stockdata_{self.symbol.split('/')[0]}_{start_date}_{end_date}_{configs['stock_api_params']['time_interval']}.json"
                f"stockdata_{self.symbol.split('/')[0]}.json"
            )
        else:
            print(f"Failed to fetch data: {response.status_code}")

    #### Used for hourly updates ####
    # get the latest hour data that was saved in json file and then append the new data to it
    def update_data(self):
        from datetime import datetime, timedelta

        file_name = f"stockdata_{self.symbol.split('/')[0]}.json"
        with open(f"../../data/{file_name}", 'r') as file:
            data = json.load(file)
            last_datetime_str = data['values'][0]['datetime']
            last_datetime = datetime.strptime(
                last_datetime_str, '%Y-%m-%d %H:%M:%S')

        start_date = last_datetime  # + timedelta(hours=1)
        end_date = datetime.now() + timedelta(days=1)

        response = self.fetch_range_data_from_api(start_date, end_date)

        if response.status_code == 200:
            new_data = response.json()

            # check if new_data latest is already what we have
            if new_data['values'][0]['datetime'] == last_datetime_str:
                print(f"Data already up to date")
                return -1  # data already up to date
            else:
                # take all the new data values except the last one
                new_data['values'] = new_data['values'][:-1]
                data['values'] = new_data['values'] + data['values']
                json.dump(data, open(
                    f"../../data/{file_name}", 'w+'), indent=4)
                print(f"Data in {file_name} updated successfully")
                return 1  # data updated successfully
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return 0
