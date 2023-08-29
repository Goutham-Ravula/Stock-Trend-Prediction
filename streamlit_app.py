import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import seaborn as sns 
import plotly.graph_objects as go
import sklearn.model_selection
import sklearn


#Importing data from Yahoo Finance
start_date = datetime.now() - pd.DateOffset(months=36)
end_date = datetime.now()
st.title('Stock Trend Prediction')
user_input= st.text_input('Enter Stock Ticker', 'AAPL')
tickers = [user_input]
df_list = []

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    df_list.append(data)

df = pd.concat(df_list, keys=tickers, names=['Ticker', 'Date'])

#Describing data
st.subheader('Past 3 Years Stock trend')
st.write(df.describe())


#Visualization

st.subheader("Stock's Performance")

#History of Stock's Graph
history = yf.Ticker(tickers[0]).history(period="Max")
df_history = pd.DataFrame(history)

fig_history = px.area(df_history.reset_index(), x='Date', y='Close', title='History of Stock Performance')
fig_history.update_xaxes(title='Date')
fig_history.update_yaxes(title='Value')
st.plotly_chart(fig_history)

#Closing price graph

fig = px.line(df.reset_index(), x='Date', y='Close', color='Ticker', title='Closing Price vs Time chart')
st.plotly_chart(fig)

#Data with 100 days Moving Averages
fig = px.line(df.reset_index(), x='Date', y='Close', title='Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(window=100).mean()
fig.add_scatter(x=df.reset_index()['Date'], y=ma100, mode='lines', name='Moving Average (100 days)')
st.plotly_chart(fig)

#Graph with Volatility of Stocks
volatility = df.groupby('Ticker')['Close'].pct_change().rolling(window=10).std().reset_index()
fig = px.line(volatility, x='Date', y='Close', title='Volatility of Stock')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Volatility')
st.plotly_chart(fig)


#Stock's performance history
history = yf.Ticker(tickers[0]).history(period="Max")
df = pd.DataFrame(history)

x = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Linear regression
st.subheader('Linear Regression')

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=0)

regression = LinearRegression()
regression.fit(train_x, train_y)
st.write("Regression coefficients:", regression.coef_)
st.write("Regression intercept:", regression.intercept_)

#This code is used to calculate the R^2. The higher the value the better the prediction.
regression_confidence = regression.score(test_x, test_y)
st.write("linear regression confidence: ", regression_confidence)

predicted = regression.predict(test_x)
dfr = pd.DataFrame({'Actual_Price': test_y, 'Predicted_Price': predicted})
st.write(dfr.tail(10))
st.write(dfr.describe())

#MSE
st.subheader('Model Evaluation')

#MAE Lower is better
st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(test_y, predicted))
#MSE Lower the better, 0 means the model is perfect, so this is close to perfect.
st.write('Mean Squared Error (MSE) :', metrics.mean_squared_error(test_y, predicted))
#RMSE Closer to 1 is better. 
st.write('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(test_y, predicted)))

#Model Accuracy 
st.subheader('Model Accuracy')

x2 = dfr.Actual_Price.mean()
y2 = dfr.Predicted_Price.mean()
Accuracy1 = x2/y2*100
st.write("The accuracy of the model is:", Accuracy1)

fig = px.scatter(dfr, x='Actual_Price', y='Predicted_Price', title='Actual vs Predicted Stock Price')
fig.update_xaxes(title='Actual Price')
fig.update_yaxes(title='Predicted Price')
st.plotly_chart(fig)



fig = go.Figure()
fig.add_trace(go.Scatter(x=dfr.index, y=dfr.Actual_Price, name='Actual Price', line=dict(color='black')))
fig.add_trace(go.Scatter(x=dfr.index, y=dfr.Predicted_Price, name='Predicted Price', line=dict(color='lightblue')))
fig.update_layout(
    title='Stock Prediction Chart',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig)
