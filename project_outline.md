## Project Outline

- **Feature pipeline** (map raw data to ML features)
  - **ingests** real-time price data from third party API ✅
  - **transforms** price data (1 hr Open-High-Low-Close prices) into appropriate features (i.e. past 12 hrs O/L/H/C data -> target next hour's Close) ✅
  - **pushes** it to the feature store
- Train pipeline (maps trainign data to a model artifact)
  - **ingests** training data from feature store
  - **trains** a predictive ML model
  - **pushes** the model to model registry
- Inference pipeline (maps model artifact to predictions)
  - **loads** the model from model registry and also recent features from feature store
  - **serves** fresh predictions through API
- Modules (helper functions)

### URLS used:

- Stock Public API:
  - https://www.alphavantage.co/documentation/
  - https://twelvedata.com/ ✅
- Feature store: https://docs.hopsworks.ai/3.1/ ✅

## Project Requirements:

- Based on last 12 hours' BTC/USD data (open/high/low/close for each hour - 12\*4 = 48 signals), ML model will predict next hours BTC/USD closing data
- Every hour, that hour's BTC/USD data will be fetched via API and will be appended to feature store (automated script - github action)
- Model will be retrained every 7 days using last 30 day's hourly data (automated script - github action)
- Model will be running online 24\*7 serving predictions with dedicated API endpoints
- A simple Streamlit dashboard will show last 12 hours data and projection of next hour's (x_axis = date/hour, y_axis = BTC/USD value)

## Written Github Actions

- One action will run one job hourly, which will consume the last hour's data from the API, run feature engineering pipeline on the data, and then insert it into the feature store ✅
- One action will run one job weekly, which will perform data cleanup on the CSV file (feature engineered data) and JSON file (raw data). Will keep only last 30 day's data.
- One action will retrain the model on last 30 days data after the data cleanup has been performed, and then add this updated model to model registry in hopsworks. We need to make sure the updated model always serves prediction.
