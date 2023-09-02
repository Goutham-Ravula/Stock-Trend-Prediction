# Stock Trend Prediction Web Application

## Overview

The Stock Trend Prediction Web Application is a data-driven tool built using Python and Streamlit. It allows users to analyze and predict stock market trends by leveraging historical stock price data, data visualization techniques, statistical analysis, and machine learning. This project serves as a valuable resource for investors, traders, and financial analysts to make informed decisions.

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Data Visualization](#data-visualization)
- [Statistical Analysis](#statistical-analysis)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Features

- **Data Acquisition:** The application fetches historical stock price data from Yahoo Finance using the "yfinance" library. Users can input stock ticker symbols to retrieve relevant data.

- **Data Visualization:** Visualizations include line charts displaying historical stock prices, closing prices, 100-day moving averages, and stock volatility over time. These visualizations help users identify trends and assess risk.

- **Statistical Analysis:** The application performs linear regression analysis on historical data, exploring relationships between opening prices, high prices, low prices, and trading volumes. It calculates the R-squared value to gauge model accuracy.

- **Model Evaluation:** The application evaluates the linear regression model using key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and model accuracy. These metrics provide insights into the model's predictive capabilities.

## Technology Stack

- Python
- Streamlit
- pandas
- yfinance
- Matplotlib
- Plotly
- statsmodels
- scikit-learn

## Usage
Open the Streamlit app in your web browser (https://goutham-ravula-stock-prediction-project.streamlit.app/).
Enter the stock ticker symbol of your choice in the text input field.
Explore historical stock price visualizations, statistical analysis results, and model predictions.

## Data Sources
Historical stock price data is sourced from Yahoo Finance using the "yfinance" library.

## Data Visualization
- **Stock's Performance:** The application provides a line chart displaying the historical performance of the selected stock over the past three years. This visualization helps users assess long-term trends.

- **Closing Price Graph:** Another line chart illustrates closing prices over time, allowing users to identify daily fluctuations.

- **Moving Average (100 days):** A line chart overlays the stock's closing prices with a 100-day moving average, helping users spot trends and filter out short-term noise.

- **Volatility of Stocks:** A line chart displays the volatility of the selected stock by calculating the rolling standard deviation of closing prices over a 10-day window.

## Statistical Analysis
- **Linear Regression:** The project employs linear regression to model the relationship between opening prices, high prices, low prices, trading volumes, and closing prices. The coefficients and intercept of the regression model are calculated.

- **R-squared Value:** The R-squared value is computed to measure the goodness of fit of the regression model. A higher R-squared value indicates better predictive accuracy.

## Model Evaluation
- **Mean Absolute Error (MAE):** MAE measures the average absolute difference between actual and predicted stock prices. Lower values indicate better model performance.

- **Mean Squared Error (MSE):** MSE measures the average squared difference between actual and predicted stock prices. Lower values signify better model accuracy.

- **Root Mean Squared Error (RMSE):** RMSE is the square root of MSE and provides a measure of the average prediction error. Lower RMSE values indicate improved model precision.

- **Model Accuracy:** The project calculates the accuracy of the model by comparing the means of actual and predicted stock prices. Higher accuracy percentages represent more reliable predictions.

## License
This project is licensed under the [MIT License](LICENSE).

---

**Note:** For better user experience use the streamlit web app in light mode. 
