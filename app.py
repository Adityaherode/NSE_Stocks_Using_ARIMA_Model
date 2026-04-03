# 📊 Streamlit Stock Market Prediction App (Interactive & Attractive)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as d
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Market Prediction App")
st.markdown("Predict stock prices using ARIMA model with interactive features.")

# Sidebar inputs
st.sidebar.header("User Input")

stocks = {
    "TCS": "TCS.NS",
    "WIPRO": "WIPRO.NS",
    "INFOSYS": "INFY.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    "LTIM": "LTIM.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "OFSS": "OFSS.NS",
    "COFORGE": "COFORGE.NS",
    "MPHASIS": "MPHASIS.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
start_date = st.sidebar.date_input("Start Date", d.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", d.date.today())
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

# Load data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.columns = data.columns.get_level_values(0)
    return data

stock_data = load_data(stocks[selected_stock], start_date, end_date)

st.subheader(f"📊 Raw Data for {selected_stock}")
st.dataframe(stock_data.tail())

# Plot closing price
st.subheader("📉 Closing Price Chart")
fig1, ax1 = plt.subplots()
ax1.plot(stock_data['Close'])
ax1.set_title("Closing Price")
st.pyplot(fig1)

# Returns
stock_data['Returns'] = stock_data['Close'].pct_change()

# ADF Test
st.subheader("📌 Stationarity Test (ADF)")

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result

adf_result = check_stationarity(stock_data['Close'])

st.write(f"ADF Statistic: {adf_result[0]:.4f}")
st.write(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    st.success("Series is Stationary ✅")
else:
    st.warning("Series is NOT Stationary ❌")

# Differencing
stock_data['Close_Diff'] = stock_data['Close'].diff()

# ARIMA Model
st.subheader("🤖 ARIMA Model Prediction")

p = st.sidebar.slider("AR (p)", 0, 10, 5)
d_order = st.sidebar.slider("I (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 10, 0)

model = ARIMA(stock_data['Close'], order=(p, d_order, q))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=forecast_days)
future_dates = pd.date_range(start=stock_data.index[-1], periods=forecast_days+1, freq='B')[1:]

# Plot prediction
st.subheader("📈 Forecast vs Actual")
fig2, ax2 = plt.subplots()
ax2.plot(stock_data['Close'], label="Actual")
ax2.plot(future_dates, forecast, linestyle='dashed', label="Forecast")
ax2.legend()
st.pyplot(fig2)

# Metrics
st.subheader("📊 Model Summary")
st.text(model_fit.summary())

# Moving Average
st.subheader("📉 Moving Averages")
ma_days = st.slider("Select MA Window", 5, 50, 20)
stock_data['MA'] = stock_data['Close'].rolling(ma_days).mean()

fig3, ax3 = plt.subplots()
ax3.plot(stock_data['Close'], label="Close")
ax3.plot(stock_data['MA'], label=f"MA {ma_days}")
ax3.legend()
st.pyplot(fig3)

# Download option
st.subheader("⬇️ Download Data")
csv = stock_data.to_csv().encode('utf-8')
st.download_button("Download CSV", csv, "stock_data.csv", "text/csv")

st.markdown("---")
st.markdown("🚀 Built with Streamlit for Data Analytics Project")
