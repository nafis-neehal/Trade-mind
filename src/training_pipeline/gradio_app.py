import gradio as gr
import pandas as pd
import numpy as np
from fetch_plot_data import get_plot_data


def get_time_series_data():
    # Fetch and process data
    plot_data = get_plot_data(hours=24)
    plot_data["datetime"] = pd.to_datetime(plot_data["datetime"])
    time_series_data = pd.DataFrame({
        "Datetime": plot_data["datetime"],
        "Actual BTC/USD": plot_data["labels"],
        "Predicted BTC/USD": plot_data["prediction"]
    })
    time_series_data = time_series_data.sort_values(by="Datetime")
    time_series_data["Datetime"] = time_series_data["Datetime"].dt.strftime(
        "%Y-%m-%d %H:%M")

    all_values = np.concatenate([time_series_data["Actual BTC/USD"],
                                 time_series_data["Predicted BTC/USD"]])
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    y_range = y_max - y_min
    padding = y_range * 0.0005
    y_min = y_min - padding
    y_max = y_max + padding

    long_data = time_series_data.melt(
        id_vars="Datetime",
        var_name="Series",
        value_name="BTC/USD Value"
    )
    return (long_data, y_min, y_max)


def update_plot():
    """Function to update plot data and timestamp"""
    data, y_min, y_max = get_time_series_data()
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    footer_html = f"""
    <div class="footer-content">
        <div class="footer-left">
            Last updated: {timestamp}
            <br>
            <a href="https://nafis-neehal.github.io/" target="_blank" class="developer-info">Developed by Nafis Neehal</a>
        </div>
    </div>
    """

    # Return values directly instead of using .update()
    return data, footer_html


custom_css = """
body {
    background-color: #f8fafc !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 2rem auto !important;
    padding: 2rem !important;
    background-color: white !important;
    border-radius: 1rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}
.main-title {
    color: #1e293b !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.2 !important;
}
.subtitle {
    color: #64748b !important;
    font-size: 1.125rem !important;
    text-align: center !important;
    margin-bottom: 1.5rem !important;
    font-weight: 500 !important;
}
.chart-container {
    margin-bottom: 1rem !important;
}
.footer-content {
    margin-top: 1rem !important;
    padding-top: 1rem !important;
    border-top: 1px solid #e2e8f0 !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    color: #64748b !important;
    font-size: 0.875rem !important;
}
.footer-left {
    text-align: left !important;
}
.footer-right {
    text-align: right !important;
}
.developer-info {
    color: #3b82f6 !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    transition: color 0.2s !important;
}
.developer-info:hover {
    color: #2563eb !important;
}
"""

# Initialize the Gradio app
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    with gr.Column():
        # Title and subtitle
        gr.Markdown("""
            <div class="main-title">Live BTC/USD Time Series Info</div>
            <div class="subtitle">Predictions served via Hopsworks API</div>
        """)

        initial_data, initial_y_min, initial_y_max = get_time_series_data()

        # Chart with reduced bottom margin
        with gr.Column(elem_classes=["chart-container"]):
            line_plot = gr.LinePlot(
                value=initial_data,
                x="Datetime",
                y="BTC/USD Value",
                color="Series",
                title="",
                y_title="BTC/USD Value",
                x_title="Time",
                x_label_angle=45,
                width=1000,
                height=450,  # Slightly reduced height
                colors={
                    "Actual BTC/USD": "#3b82f6",
                    "Predicted BTC/USD": "#ef4444"
                },
                tooltip=["Datetime", "BTC/USD Value", "Series"],
                overlay_point=True,
                zoom=False,
                pan=False,
                show_label=True,
                stroke_width=2,
                y_min=initial_y_min,
                y_max=initial_y_max,
                y_lim=[initial_y_min, initial_y_max],
                show_grid=True,
            )

        # Footer with timestamp and developer info
        footer = gr.Markdown(f"""
            <div class="footer-content">
                <div class="footer-left">
                    Last updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
                    <br>
                    <a href="https://nafis-neehal.github.io/" target="_blank" class="developer-info">Developed by Nafis Neehal</a>
                </div>
            </div>
        """)

        # Set up timer for hourly updates (3600 seconds)
        timer = gr.Timer(30)
        timer.tick(update_plot, inputs=None, outputs=[line_plot, footer])

# Launch the app
app.launch()
