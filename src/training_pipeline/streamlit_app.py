import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
# Replace with your actual data-fetching function
from fetch_plot_data import get_plot_data

# Title of the Streamlit app
st.title("Live BTC/USD Time Series - Actual vs Predicted")

# Function to fetch and prepare the latest 12 hours of data


def get_time_series_data():
    # Fetch the plot data
    plot_data = get_plot_data()
    datetime_column = pd.to_datetime(plot_data["datetime"])
    actual_values = pd.Series(plot_data["labels"], name="Actual BTC/USD")
    predicted_values = pd.Series(
        plot_data["prediction"], name="Predicted BTC/USD")

    # Create a DataFrame with datetime as index and actual/predicted values as columns
    time_series_data = pd.DataFrame({
        "Datetime": datetime_column,
        "Actual BTC/USD": actual_values,
        "Predicted BTC/USD": predicted_values
    }).sort_values(by="Datetime")

    return time_series_data


# Main Streamlit loop to update the chart every hour
while True:
    # Get the latest data
    time_series_data = get_time_series_data()

    # Determine y-axis limits based on min/max of actual and predicted values
    y_min = min(time_series_data["Actual BTC/USD"].min(),
                time_series_data["Predicted BTC/USD"].min()) * 0.95
    y_max = max(time_series_data["Actual BTC/USD"].max(),
                time_series_data["Predicted BTC/USD"].max()) * 1.05

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot actual and predicted values
    ax.plot(time_series_data["Datetime"], time_series_data["Actual BTC/USD"],
            label="Actual BTC/USD", color="blue", marker="o")
    ax.plot(time_series_data["Datetime"], time_series_data["Predicted BTC/USD"],
            label="Predicted BTC/USD", color="orange", linestyle="--", marker="x")

    # Set y-axis limits
    ax.set_ylim([y_min, y_max])

    # Customize plot labels and title
    ax.set_xlabel("Datetime (Last 12 Hours)", fontsize=12)
    ax.set_ylabel("BTC/USD Value", fontsize=12)
    ax.set_title("BTC/USD Time Series - Actual vs Predicted", fontsize=15)
    ax.legend(loc="upper left", fontsize=10)

    # Format x-axis to show date and hour precisely
    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

    # Wait for one hour before refreshing the plot
    time.sleep(3600)
