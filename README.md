# Real-Time BTC/USD Prediction System

This project implements a real-time prediction system for BTC/USD, leveraging machine learning and MLOps practices.

Live URL: https://huggingface.co/spaces/nafisneehal/trade-mind

## Table of Contents

1. [Project Setup](#project-setup)
2. [API Configuration](#api-configuration)
3. [Hopsworks Configuration](#hopsworks-configuration)
4. [Detailed Workflow](#detailed-workflow)
5. [Model Serving & Predictions](#model-serving--predictions)
6. [Usage Guide](#usage-guide)
7. [Future Development](#future-development)

## Project Structure

```tree
├── .github/workflows
│   └── update_feature_store.yml          # GitHub Actions workflow for feature store update
├── data
│   └── engineered
│       ├── stockdata_BTC_engineered.csv  # Engineered features
│       └── stockdata_BTC.json            # Raw BTC/USD data
├── models
│   └── btc_regressor_model.pkl          # Trained model file
├── src
│   ├── feature_pipeline
│   │   ├── feature_pipeline.py          # Main feature pipeline script
│   │   ├── FeatureProcessor.py          # Feature transformation utilities
│   │   ├── HopsworkFeatureStore.py      # Hopsworks feature store interactions
│   │   └── StockData.py                 # Data fetching and initial processing
│   └── training_pipeline
│       ├── fetch_plot_data.py           # Data fetching and plotting utilities
│       ├── gradio_app.py                # Gradio interface for live predictions
│       ├── kserve_predict_script.py     # Prediction script for KServe endpoint
│       ├── retrain_model.py             # Script for weekly model retraining
│       ├── streamlit_app.py             # Optional Streamlit app for visualization
│       └── Trainer.py                   # Model training script
├── .env                                 # Environment variables
├── config.yml                          # Configuration file
├── trade-mind                          # Directory for additional resources
├── .gitignore                          # Git ignore file
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
└── setup.sh                            # Setup script
```

## Project Setup

To get started with the project, clone the repository, navigate to the directory, and install required dependencies.

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

## API Configuration

### Required API Keys

The following API keys are required to fetch BTC/USD data and access Hopsworks.

- `STOCK_API_KEY`: For BTC/USD data fetching
- `HOPSWORKS_API_KEY`: For Hopsworks feature store access

**Note**: Store these keys as GitHub secrets for CI/CD integration.

## Hopsworks Configuration

1. Set up a feature group in Hopsworks for the BTC/USD dataset.
2. Configure the feature store and model registry to enable data synchronization and model management.

## Detailed Workflow

### Continuous Feature Updates

The feature pipeline updates BTC/USD data hourly to maintain fresh data through github actions. This process includes:

- Fetching new BTC/USD data
- Performing feature engineering steps
- Pushing updated features with the Hopsworks feature store

GitHub Actions Workflow: `update_feature_store.yml` manages these hourly updates.

### Weekly Model Retraining

Every week, the model is retrained on the last 30 days of data. The CI/CD pipeline handles:

- Model retraining
- Hopsworks model registration

## Model Serving & Predictions

The system provides live predictions via a Gradio interface with:

- Hourly prediction updates
- Real-time data comparison
- A 12-hour historical view

## Usage Guide

### Running Components

Execute the following commands to manually update features, train the model, or launch the Gradio interface.

```bash
# Update Features Manually
python src/feature_pipeline/feature_pipeline.py

# Train Model
python src/training_pipeline/retrain_model.py

# Launch Gradio Interface
python src/training_pipeline/gradio_app.py
```

## Future Development

### Planned Enhancements

- Model drift monitoring and alert system
- Scheduled data cleanup
- Multi-step forecasting capabilities
- Integration of additional external data sources
