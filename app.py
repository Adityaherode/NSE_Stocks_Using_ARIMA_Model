import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as d
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

st.title("📈 Stock Market Prediction Dashboard")

# -------------------------------
# Sidebar Inputs
# -------------------------------
stocks = {
    "TCS": "TCS.NS",
    "WIPRO": "WIPRO.NS",
    "INFOSYS": "INFY.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "LTIM": "LTIM.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "OFSS": "OFSS.NS",
    "COFORGE": "COFORGE.NS",
    "MPHASIS": "MPHASIS.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
start_date = st.sidebar.date_input("Start Date", d.date(2025,1,1))
end_date = st.sidebar.date_input("End Date", d.date.today())
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

# -------------------------------
# Fetch Data
# -------------------------------
data = yf.download(stocks[selected_stock], start=start_date, end=end_date)

if data.empty:
    st.error("No data found. Try different dates.")
    st.stop()

data = data[['Close']]
data.dropna(inplace=True)

# -------------------------------
# ADF Test Function
# -------------------------------
def adf_test(series):
    result = adfuller(series)
    return result[0], result[1]

# -------------------------------
# Make Stationary Automatically
# -------------------------------
differencing_order = 0
temp_series = data['Close']

p_value = adf_test(temp_series)[1]

while p_value > 0.05:
    temp_series = temp_series.diff().dropna()
    differencing_order += 1
    p_value = adf_test(temp_series)[1]

# -------------------------------
# Display ADF Result
# -------------------------------
st.subheader("📊 Stationarity Test (ADF)")
st.write(f"ADF p-value: {p_value:.5f}")
st.write(f"Differencing applied: {differencing_order}")

if p_value < 0.05:
    st.success("Series is Stationary ✅")
else:
    st.warning("Series is NOT Stationary ❌")

# -------------------------------
# Plot Original Data
# -------------------------------
st.subheader("📉 Stock Price Trend")

fig1, ax1 = plt.subplots()
ax1.plot(data['Close'])
ax1.set_title(f"{selected_stock} Closing Price")
st.pyplot(fig1)

# -------------------------------
# ARIMA Model
# -------------------------------
st.subheader("🤖 ARIMA Forecast")

try:
    model = ARIMA(data['Close'], order=(5, differencing_order, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_days)

    forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='B')[1:]

    fig2, ax2 = plt.subplots()
    ax2.plot(data['Close'], label="Actual")
    ax2.plot(forecast_dates, forecast, linestyle="dashed", label="Forecast")
    ax2.legend()

    st.pyplot(fig2)

    st.subheader("📅 Forecasted Values")
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})
    st.dataframe(forecast_df)

except Exception as e:
    st.error(f"Model Error: {e}")
