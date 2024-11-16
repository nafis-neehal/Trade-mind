import warnings
from pathlib import Path
import os
import yaml
from xgboost import XGBRegressor
from Trainer import Trainer
from dotenv import load_dotenv
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pprint

load_dotenv()

warnings.filterwarnings('ignore')

# Define the base directory as the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Use BASE_DIR to dynamically load the config file
CONFIG_FILE = BASE_DIR / "src" / "config.yml"
with open(CONFIG_FILE, 'r') as file:
    configs = yaml.safe_load(file)


def main():
    symbol = configs['stock_api_params']['symbol']
    # Initialize Trainer with relevant project details
    trainer = Trainer(
        project_name=configs['hopsworks']['project_name'],
        feature_group_name=f"{symbol.split('/')[0].lower()}_features",
        model_registry_name=f"{symbol.split('/')[0].lower()}_regressor_model",
        api_key=os.getenv("HOPSWORKS_API_KEY")
    )

    # Step 0: Stop old deployment and Delete old deployed model
    model_stopped = trainer.stop_model_deployment()
    model_stopped.delete()
    print("Old model deleted successfully.")

    # Step 1: Create or retrieve feature view
    trainer.create_feature_view()

    # Step 2: Pull last 30 days of data from the feature view
    df = trainer.get_retrain_data_from_feature_view()

    # Step 3: No split, train with all data
    X_train, y_train = trainer.get_features_labels(df)

    # take the first 48 rows of the data
    X_check, y_check = X_train.head(48), y_train.head(48)

    # Step 4: Train a model (using XGBoost in this example)
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model = trainer.train_model(model, X_train, y_train)

    # Step 5: Evaluate the model
    metrics = trainer.evaluate_model(model, X_check, y_check, show_pred=False)

    # Step 6: Generate Model Schema
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Step 7: Save the trained model to Hopsworks model registry
    trainer.save_model_to_registry(model, metrics, model_schema, X_train)
    trainer.model_deploy()

    # Step 8: Predict on test data after model deployed
    # print(X_check)
    # y_check_pred = trainer.predict_with_hopsworks_api(X_check)
    # print(y_check_pred)
    # mse = mean_squared_error(y_check, y_pred)
    # mae = mean_absolute_error(y_check, y_pred)
    # r2 = r2_score(y_check, y_pred)

    # print(f"Model Evaluation:\nMSE: {mse}\nMAE: {mae}\nR2 Score: {r2}")


if __name__ == "__main__":
    main()
