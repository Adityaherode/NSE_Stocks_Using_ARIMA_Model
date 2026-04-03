import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as d
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ---------------- UI ----------------
st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Market Prediction App (ARIMA)")

# Sidebar
st.sidebar.header("User Input")

stocks = {
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "Infosys": "INFY.NS",
    "HCL Tech": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",
    "Persistent": "PERSISTENT.NS",
    "Oracle Financial": "OFSS.NS",
    "Coforge": "COFORGE.NS",
    "Mphasis": "MPHASIS.NS"
}

stock_name = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[stock_name]

start_date = st.sidebar.date_input("Start Date", d.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", d.date.today())

p = st.sidebar.slider("AR (p)", 0, 10, 5)
d_val = st.sidebar.slider("I (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 10, 0)

forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

# ---------------- Data ----------------
st.subheader(f"📊 Data for {stock_name}")

data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("No data found!")
    st.stop()

data = data[['Close']].copy()
data['Returns'] = data['Close'].pct_change()

st.write(data.tail())

# ---------------- ADF Test ----------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[0], result[1]

st.subheader("📉 Stationarity Test (ADF)")

adf_stat, p_value = check_stationarity(data['Close'])

st.write(f"ADF Statistic: {adf_stat:.4f}")
st.write(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    st.success("Series is Stationary ✅")
else:
    st.warning("Series is NOT Stationary ❌")

# ---------------- Differencing ----------------
data['Close_Diff'] = data['Close'].diff()

# ---------------- Model ----------------
st.subheader("🤖 ARIMA Model")

model = ARIMA(data['Close'], order=(p, d_val, q))
model_fit = model.fit()

st.text(model_fit.summary())

# ---------------- Forecast ----------------
forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1, freq='B')[1:]

# ---------------- Plot ----------------
st.subheader("📊 Forecast Visualization")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(data['Close'], label="Actual Price")
ax.plot(future_dates, forecast, linestyle='dashed', color='red', label="Forecast")

ax.set_title(f"{stock_name} Price Prediction")
ax.legend()

st.pyplot(fig)

# ---------------- Insights ----------------
st.subheader("Insights")

st.write(f"Last Price: ₹{data['Close'].iloc[-1]:.2f}")
st.write(f"Predicted Price after {forecast_days} days: ₹{forecast.iloc[-1]:.2f}")
