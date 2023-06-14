import pandas as pd
pip install yfinance
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as pyplot
from keras.model import load_model
import numpy as np
import streamlit as st




start_date = datetime.now() - pd.DateOffset(months=36)
end_date = datetime.now()


st.title('Stock Trend Prediction')
         
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
tickers = [user_input]

df_list = []

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    df_list.append(data)

df = pd.concat(df_list, keys=tickers, names=['Ticker', 'Date'])

#Describing data
st.subheader('Past 3 Years Stock trend')
st.write(df.describe())
