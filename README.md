# Nifty 50 Stock Price Prediction

This project uses Machine Learning (Linear Regression) to predict future price movements of the Nifty 50 index. It processes historical data from the National Stock Exchange (NSE) to forecast closing prices for the next 5 days.

## Overview

The script performs the following steps:
1.  **Data Loading:** Reads historical stock data from a CSV file.
2.  **Preprocessing:** Cleans column names and sorts data chronologically (Oldest to Newest).
3.  **Feature Engineering:** Shifts the data to create labels for future prediction.
4.  **Training:** Trains a Linear Regression model using `scikit-learn`.
5.  **Forecasting:** Predicts the closing price for the next 5 trading days.

## Prerequisites

* Python 3.x
* Pip package manager

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/stock-price-prediction.git]
    cd stock-price-prediction
    ```

2.  Install the required libraries:
    ```bash
    pip install numpy pandas scikit-learn
    ```

## Usage

1.  **Get the Data:**
    * Download historical data for **Nifty 50** from the [NSE Website](https://www.niftyindices.com/reports/historical-data).
    * Rename the file to `nifty50.csv`.
    * Place it in the project root folder.

2.  **Run the Script:**
    ```bash
    python stock_prediction.py
    ```

## Output

The script will output the model's accuracy score and a forecast array containing the predicted closing prices for the next 5 days.

```json
{
  "test_score": 0.7721545827,
  "forecast_set": [25797.94, 25943.60, 26142.51, 26147.09, 26113.25]
}
