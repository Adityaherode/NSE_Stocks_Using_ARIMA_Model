import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Stock Market Analysis & Forecasting App")

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
# SIDEBAR INPUT
# -------------------------------
selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line", "Candlestick", "Bar"])

start = st.sidebar.date_input("Start Date", dt.date(2025, 1, 1))
end = st.sidebar.date_input("End Date", dt.date.today())

# -------------------------------
# DATA LOAD
# -------------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start=start, end=end)
    df.columns = df.columns.get_level_values(0)
    return df

df = load_data(stocks[selected_stock])

# -------------------------------
# SHOW DATA
# -------------------------------
st.subheader(f"📌 Raw Data - {selected_stock}")
st.dataframe(df.tail())

# -------------------------------
# STATIONARITY FUNCTION
# -------------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1]  # return p-value

# -------------------------------
# MAKE STATIONARY
# -------------------------------
p_value = check_stationarity(df['Close'])

if p_value > 0.05:
    df['Close'] = df['Close'].diff()
    df.dropna(inplace=True)
    st.warning("⚠️ Data was non-stationary → Converted using differencing")
else:
    st.success("✅ Data is already stationary")

# -------------------------------
# CHARTS
# -------------------------------
st.subheader("📈 Stock Chart")

if chart_type == "Line":
    st.line_chart(df['Close'])

elif chart_type == "Bar":
    st.bar_chart(df['Close'])

elif chart_type == "Candlestick":
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# RETURNS
# -------------------------------
df['Returns'] = df['Close'].pct_change()

st.subheader("📉 Returns Distribution")
st.line_chart(df['Returns'])

# -------------------------------
# ARIMA MODEL
# -------------------------------
st.subheader("🔮 Forecasting (ARIMA)")

try:
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)
    future_dates = pd.date_range(start=df.index[-1], periods=11, freq='B')[1:]

    # Plot
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual"))

    fig2.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        name="Forecast",
        line=dict(dash='dash')
    ))

    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"Error in ARIMA: {e}")
