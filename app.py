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

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

st.title("📈 Stock Market Prediction Dashboard")
st.markdown("Analyze & Forecast Indian IT Stocks")

# -------------------------------
# STOCK LIST
# -------------------------------
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

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("User Input")

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

start_date = st.sidebar.date_input("Start Date", d.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", d.date.today())

forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]
    return data

data = load_data(stocks[selected_stock])

st.subheader(f"📊 Raw Data - {selected_stock}")
st.write(data.tail())

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
data['Returns'] = data['Close'].pct_change()
data['MA50'] = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()

# -------------------------------
# CHART 1: PRICE + MOVING AVG
# -------------------------------
st.subheader("📉 Price & Moving Averages")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data['Close'], label="Close Price")
ax.plot(data['MA50'], label="MA50")
ax.plot(data['MA200'], label="MA200")
ax.legend()
st.pyplot(fig)

# -------------------------------
# CHART 2: RETURNS
# -------------------------------
st.subheader("📊 Daily Returns")

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(data['Returns'])
st.pyplot(fig2)

# -------------------------------
# STATIONARITY CHECK FUNCTION
# -------------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1]

# -------------------------------
# AUTO STATIONARITY FIX
# -------------------------------
st.subheader("🧪 Stationarity Check")

p_value = check_stationarity(data['Close'])

if p_value < 0.05:
    st.success("Data is Stationary ✅")
    ts_data = data['Close']
    d_value = 0
else:
    st.warning("Data is NOT Stationary ⚠️ → Applying Differencing")
    ts_data = data['Close'].diff().dropna()
    d_value = 1

# -------------------------------
# ARIMA MODEL
# -------------------------------
st.subheader("🤖 ARIMA Forecast")

try:
    model = ARIMA(data['Close'], order=(5, d_value, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_days)

    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='B')[1:]

    # -------------------------------
    # PLOT FORECAST
    # -------------------------------
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(data['Close'], label="Actual Price")
    ax3.plot(future_dates, forecast, linestyle='dashed', color='red', label="Forecast")
    ax3.legend()

    st.pyplot(fig3)

    # -------------------------------
    # METRICS
    # -------------------------------
    st.subheader("📌 Key Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Price", round(data['Close'].iloc[-1], 2))
    col2.metric("Avg Return", round(data['Returns'].mean(), 4))
    col3.metric("Volatility", round(data['Returns'].std(), 4))

except Exception as e:
    st.error(f"Error in model: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Made with ❤️ for Stock Analysis")
