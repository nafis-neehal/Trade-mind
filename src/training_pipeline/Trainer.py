import hopsworks
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import hsfs
import hsml

# Define the base directory as the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Trainer:
    def __init__(self, project_name, feature_group_name, model_registry_name, api_key):
        self.project_name = project_name
        self.feature_group_name = feature_group_name
        self.model_registry_name = model_registry_name
        self.api_key = api_key
        self.project = hopsworks.login(api_key_value=self.api_key)
        self.fs = self.project.get_feature_store()
        self.model_registry = self.project.get_model_registry()
        self.feature_view = None
        self.deployment = None

    def create_feature_view(self):
        """Select features from the feature group and create a feature view."""
        selected_features = self.fs.get_or_create_feature_group(
            name=self.feature_group_name,
            version=1
        ).select_all()

        print("Feature group selected successfully......... --->>")

        """Create or get a feature view for the last 30 days of data."""
        try:
            self.feature_view = self.fs.get_or_create_feature_view(
                name=f"{self.feature_group_name}_view",
                version=1,
                description="Feature view with last 30 days of data for model training",
                query=selected_features,
            )
            print("Feature view created or retrieved successfully.")
        except hsfs.client.exceptions.RestAPIError as e:
            print(f"Error creating feature view: {e}")

    def delete_feature_view(self):
        """Delete the feature view."""
        try:
            self.feature_view.delete()
            print("Feature view deleted successfully.")
        except hsfs.client.exceptions.RestAPIError as e:
            print(f"Error deleting feature view: {e}")

    def get_retrain_data_from_feature_view(self):
        """Pull the last 30 days of data from the feature view till today."""
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()

        # Get the data as a DataFrame from the feature view
        df = self.feature_view.get_batch_data(
            start_time=start_time, end_time=end_time)

        # sort by datetime
        df = df.sort_values(by='datetime', ascending=False)
        print("Data pulled from feature view for retraining successfully.")
        return df

    def get_plot_data_from_feature_view(self, hours):
        # get last 12 hours of data starting from current hour to plot
        start_time = datetime.now() - timedelta(hours=hours)
        end_time = datetime.now()

        # Get the data as a DataFrame from the feature view
        df = self.feature_view.get_batch_data(
            start_time=start_time, end_time=end_time)

        # sort by datetime
        df = df.sort_values(by='datetime', ascending=False)
        print("Data pulled from feature view for plotting successfully.")
        return df

    def train_test_split(self, df, test_size=0.2):
        """Split data into training and test sets."""
        # Define feature columns based on lagged features
        feature_columns = [
            f"{prefix}_lag_{i}" for i in range(0, 13) for prefix in ["open", "high", "low", "close"]
        ]

        # Separate features and target
        X = df[feature_columns]
        y = df['target']

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        print("Data split into train and test sets.")
        return X_train, X_test, y_train, y_test

    def get_features_labels(self, df):
        """Split data into features and labels."""
        # Define feature columns based on lagged features
        feature_columns = [
            f"{prefix}_lag_{i}" for i in range(0, 13) for prefix in ["open", "high", "low", "close"]
        ]

        # Separate features and target
        X = df[feature_columns]
        y = df['target']
        return X, y

    def train_model(self, model, X_train, y_train):
        """Train the model on training data."""
        model.fit(X_train, y_train)
        print("Model training completed.")
        return model

    def evaluate_model(self, model, X_test, y_test, **kwargs):
        """Evaluate the model on the hold-out test set."""
        y_pred = model.predict(X_test)
        # if show_pred in kwargs is true, print the predictions
        if "show_pred" in kwargs:
            print(f"Predictions: {y_pred}")
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation:\nMSE: {mse}\nMAE: {mae}\nR2 Score: {r2}")
        return {"mse": mse, "mae": mae, "r2": r2}

    def save_model_to_registry(self, model, metrics, model_schema, X_train):
        """Save the trained model to Hopsworks Model Registry."""
        # Use BASE_DIR to define the model directory and path
        model_dir = BASE_DIR / "models"
        # Ensure the directory exists
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{self.model_registry_name}.pkl"
        joblib.dump(model, model_path)

        new_model = self.model_registry.sklearn.create_model(
            name=self.model_registry_name,
            metrics=metrics,
            model_schema=model_schema,
            input_example=X_train.sample(),
            description="Trained model with 30-day feature view data",
        )

        # Register the model and serve as endpoint
        new_model.save(str(model_path))
        # new_model.deploy()
        print("Model saved to registry successfully.")

    def model_deploy(self):
        model = self.model_registry.get_model(
            self.model_registry_name)

        # strip all _ from self.model_registry_name and keep only alphanumeric characters
        deploy_name = self.model_registry_name.replace("_", "")

        # Get the dataset API for the project
        dataset_api = self.project.get_dataset_api()

        # Upload the file "predict_example.py" to the "Models" dataset
        # If a file with the same name already exists, overwrite it
        predictor_local_path = BASE_DIR / "src" / \
            "training_pipeline" / "kserve_predict_script.py"
        uploaded_file_path = dataset_api.upload(
            predictor_local_path, "Models", overwrite=True)

        # Construct the full path to the uploaded predictor script
        predictor_script_path = os.path.join(
            "/Projects", self.project_name, uploaded_file_path)

        self.deployment = model.deploy(
            name=deploy_name,
            script_file=predictor_script_path,)

        # start the deployment
        self.deployment.start()

    def predict_with_hopsworks_api(self, X):
        """Use the deployed model to make predictions via the Hopsworks API."""
        # Get model serving handle from the project
        model_serving = self.project.get_model_serving()

        model = self.model_registry.get_model(
            self.model_registry_name, version=1)

        # Ensure the deployment name follows the required regex pattern
        deploy_name = self.model_registry_name.replace("_", "")

        try:
            # Get the deployment
            deployment = model_serving.get_deployment(name=deploy_name)

            # Make predictions
            predictions = deployment.predict(inputs=X.values.tolist())
            print("Predictions made via Hopsworks model API.")
            return predictions
        except hsml.client.exceptions.RestAPIError as e:
            print(f"Error making predictions: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def stop_model_deployment(self):
        model = self.model_registry.get_model(
            self.model_registry_name, version=1)
        # Ensure the deployment name follows the required regex pattern
        deploy_name = self.model_registry_name.replace("_", "")

        # Get model serving handle
        model_serving = self.project.get_model_serving()

        try:
            # List deployments
            deployments = model_serving.get_deployments(model)
            for deployment in deployments:
                if deployment.name == deploy_name:
                    # deployment.stop()
                    deployment.delete(force=True)
                    print(
                        f"Deployment {deploy_name} stopped and deleted successfully.")
                    break
            else:
                print(f"No deployment found with name: {deploy_name}")
        except hsml.client.exceptions.RestAPIError as e:
            print(f"Error stopping or deleting deployment: {e}")

        return model
