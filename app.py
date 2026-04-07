import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as d
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Title
# -------------------------------
st.title("📈 IT Stock Prediction Dashboard")

# -------------------------------
# Stock List
# -------------------------------
stocks = {
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "Infosys": "INFY.NS",
    "HCLTech": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",
    "Persistent": "PERSISTENT.NS",
    "Oracle Financial": "OFSS.NS",
    "Coforge": "COFORGE.NS",
    "Mphasis": "MPHASIS.NS"
}

# -------------------------------
# Sidebar Inputs
# -------------------------------
stock_name = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line", "Candlestick", "Bar"])

start = st.sidebar.date_input("Start Date", d.date(2020,1,1))
end = st.sidebar.date_input("End Date", d.date.today())

# -------------------------------
# Load Data
# -------------------------------
df = yf.download(stocks[stock_name], start=start, end=end)

# -------------------------------
# Show Data
# -------------------------------
st.subheader(f"{stock_name} Data")
st.write(df.tail())

# -------------------------------
# Returns + Stationarity
# -------------------------------
df['Returns'] = df['Close'].pct_change()

# ADF Test
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1]

p_value = check_stationarity(df['Close'])

st.subheader("📊 Stationarity Check")
st.write(f"p-value: {p_value}")

# Convert Non-stationary → Stationary
if p_value > 0.05:
    st.write("Series is NOT stationary → Applying Differencing")
    df['Close'] = df['Close'].diff()
    df.dropna(inplace=True)
else:
    st.write("Series is already stationary")

# -------------------------------
# ARIMA Model
# -------------------------------
model = ARIMA(df['Close'], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=10)
future_dates = pd.date_range(start=df.index[-1], periods=11, freq='B')[1:]

# -------------------------------
# Plot Graph (Single Output)
# -------------------------------
st.subheader("📈 Stock Chart + Prediction")

fig = go.Figure()

# ---- Chart Types ----
if chart_type == "Line":
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual"))

elif chart_type == "Bar":
    fig.add_trace(go.Bar(x=df.index, y=df['Close'], name="Actual"))

elif chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ))

# ---- Forecast ----
fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    name="Prediction",
    line=dict(dash="dash")
))

fig.update_layout(
    title=f"{stock_name} Stock Forecast",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark"
)

# SINGLE GRAPH OUTPUT
st.plotly_chart(fig, use_container_width=True)
