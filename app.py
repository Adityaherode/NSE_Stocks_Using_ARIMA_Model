import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Forecast App", layout="wide")

# Title
st.title("📈 Stock Market Analysis & Forecasting")

# Sidebar Inputs
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
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar"])
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

start = st.sidebar.date_input("Start Date", dt.date(2025, 1, 1))
end = st.sidebar.date_input("End Date", dt.date.today())

# Load Data
data = yf.download(stocks[selected_stock], start=start, end=end)
data.dropna(inplace=True)

# ----------- CHARTS -------------
st.subheader(f"{selected_stock} Stock Chart")

if chart_type == "Candlestick":
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Line":
    st.line_chart(data['Close'])

elif chart_type == "Bar":
    st.bar_chart(data['Close'])


# ----------- STATIONARITY CHECK -------------
st.subheader("📊 Stationarity Check (ADF Test)")

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[0], result[1]

adf_stat, p_value = check_stationarity(data['Close'])

st.write(f"ADF Statistic: {adf_stat}")
st.write(f"p-value: {p_value}")

# Make Stationary
if p_value > 0.05:
    st.warning("Series is NOT stationary. Applying differencing...")
    data['Close'] = data['Close'].diff()
    data.dropna(inplace=True)

    adf_stat, p_value = check_stationarity(data['Close'])

    st.write("After Differencing:")
    st.write(f"ADF Statistic: {adf_stat}")
    st.write(f"p-value: {p_value}")
else:
    st.success("Series is already stationary ✅")


# ----------- ARIMA MODEL -------------
st.subheader("📉 Forecasting")

try:
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_days)

    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='B')[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast
    })

    # Plot Forecast
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name="Actual"
    ))

    fig2.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        name="Forecast",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(forecast_df)

except Exception as e:
    st.error(f"Model Error: {e}")
