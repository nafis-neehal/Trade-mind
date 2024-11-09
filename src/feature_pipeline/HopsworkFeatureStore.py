from pathlib import Path
import yaml
from dotenv import load_dotenv
import pandas as pd
import hopsworks
from hsfs.feature_group import FeatureGroup
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

# Define the base directory as the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load the configuration file using BASE_DIR
CONFIG_FILE = BASE_DIR / "src" / "config.yml"
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)


class HopsworkFeatureStore:
    def __init__(self, project_name, feature_group_name, api_key, csv_path):
        """
        Initializes the HopsworkFeatureStore instance.

        Args:
            project_name (str): Name of the Hopsworks project.
            feature_group_name (str): Name of the feature group.
            api_key (str): Hopsworks API key.
            csv_path (str or Path): Path to the CSV file.
        """
        self.project_name = project_name
        self.feature_group_name = feature_group_name
        self.csv_path = Path(csv_path)  # Ensure csv_path is a Path object
        self.api_key = api_key
        self.project = hopsworks.login(api_key_value=self.api_key)
        self.fs = self.project.get_feature_store()
        self.feature_group = None

    def load_data(self):
        """Loads data from the specified CSV file."""
        # Use Path object for CSV path
        self.df = pd.read_csv(self.csv_path, parse_dates=['datetime'])
        print(f"Loaded {len(self.df)} rows from {self.csv_path}")

    def get_or_create_feature_group(self, description="Feature group for time-series data ingestion"):
        """
        Retrieves an existing feature group or creates a new one if it doesn't exist.

        Args:
            description (str): Description for the feature group if created.
        """
        try:
            self.feature_group = self.fs.get_feature_group(
                name=self.feature_group_name, version=1)
            print(f"Using existing feature group: {self.feature_group_name}")
        except:
            self.feature_group = self.fs.create_feature_group(
                name=self.feature_group_name,
                version=1,
                description=description,
                primary_key=["datetime"],
                event_time="datetime"
            )
            print(f"Created new feature group: {self.feature_group_name}")

    def find_new_rows(self):
        """Finds rows that are not already present in the feature group."""
        if self.feature_group is None:
            raise ValueError("Feature group is not initialized.")

        try:
            # Read existing data from the feature group
            existing_data = self.feature_group.read()

            # Convert datetime columns to ensure consistent comparison
            self.df['datetime'] = pd.to_datetime(
                self.df['datetime']).dt.tz_localize(None)
            existing_data['datetime'] = pd.to_datetime(
                existing_data['datetime']).dt.tz_localize(None)

            # Identify new rows based on `datetime` primary key
            new_data = self.df[~self.df['datetime'].isin(
                existing_data['datetime'])]
            print(f"Identified {len(new_data)} new rows to insert.")
        except Exception as e:
            print(f"Could not read existing data from feature group: {e}")
            # Assume the feature group is empty or newly created
            new_data = self.df
            print("Assuming the feature group is empty. All data will be considered new.")

        return new_data

    def insert_new_data(self, new_data):
        """
        Inserts new data rows into the feature group.

        Args:
            new_data (pd.DataFrame): DataFrame containing the new rows to insert.
        """
        if new_data.empty:
            print("No new rows to insert.")
        else:
            self.feature_group.insert(new_data, write_options={
                                      "wait_for_job": False})
            print(f"Inserted {len(new_data)} new rows into the feature group.")

    def run_pipeline(self):
        """Runs the complete pipeline for loading, checking, and inserting data."""
        print("Starting data ingestion pipeline...")
        self.load_data()
        self.get_or_create_feature_group()
        new_data = self.find_new_rows()
        self.insert_new_data(new_data)
        print("Data ingestion pipeline completed.")
