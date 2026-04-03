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

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

st.title("📈 Stock Market Prediction Dashboard")
st.markdown("Interactive Stock Analysis & Forecasting App")

# ------------------------------
# SIDEBAR
# ------------------------------
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

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

start_date = st.sidebar.date_input("Start Date", dt.date(2023,1,1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

forecast_days = st.sidebar.slider("Forecast Days", 5, 60, 10)

# ------------------------------
# DATA FETCHING
# ------------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data = load_data(stocks[selected_stock])

st.subheader(f"📊 Data for {selected_stock}")
st.write(data.tail())

# ------------------------------
# CANDLESTICK CHART
# ------------------------------
st.subheader("🕯️ Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# RETURNS
# ------------------------------
data['Returns'] = data['Close'].pct_change()

st.subheader("📉 Daily Returns")
st.line_chart(data['Returns'])

# ------------------------------
# STATIONARITY FUNCTION
# ------------------------------
def make_stationary(series):
    result = adfuller(series.dropna())
    
    if result[1] < 0.05:
        return series, 0  # already stationary
    else:
        diff_series = series.diff().dropna()
        return diff_series, 1

# ------------------------------
# ARIMA MODEL
# ------------------------------
st.subheader("🔮 Stock Price Prediction (ARIMA)")

close_data = data.set_index('Date')['Close']

stationary_data, d = make_stationary(close_data)

st.write(f"📌 Differencing Applied: {d}")

# Simple ARIMA auto config
model = ARIMA(close_data, order=(5, d, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(
    start=close_data.index[-1],
    periods=forecast_days+1,
    freq='B'
)[1:]

# ------------------------------
# PLOT FORECAST
# ------------------------------
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=close_data.index,
    y=close_data,
    name="Actual Price"
))

fig2.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    name="Forecast",
    line=dict(color='red', dash='dash')
))

fig2.update_layout(title="Stock Price Prediction", height=500)

st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# MOVING AVERAGE
# ------------------------------
st.subheader("📊 Moving Averages")

ma1 = st.slider("Short MA", 5, 50, 20)
ma2 = st.slider("Long MA", 50, 200, 100)

data['MA1'] = data['Close'].rolling(ma1).mean()
data['MA2'] = data['Close'].rolling(ma2).mean()

fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
fig3.add_trace(go.Scatter(x=data['Date'], y=data['MA1'], name=f'MA{ma1}'))
fig3.add_trace(go.Scatter(x=data['Date'], y=data['MA2'], name=f'MA{ma2}'))

st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
