import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

st.title("📊 Stock Market Analysis & Forecasting")

# -------------------------
# STOCK LIST
# -------------------------
stocks = {
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "Infosys": "INFY.NS",
    "HCLTech": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",
    "Persistent": "PERSISTENT.NS",
    "OFSS": "OFSS.NS",
    "Coforge": "COFORGE.NS",
    "Mphasis": "MPHASIS.NS"
}

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Settings")

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

start_date = st.sidebar.date_input("Start Date", dt.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

chart_type = st.sidebar.radio("Select Chart Type", ["Candlestick", "Line", "Bar"])

forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

# -------------------------
# DATA LOADING
# -------------------------
df = yf.download(stocks[selected_stock], start=start_date, end=end_date)

df.dropna(inplace=True)

st.subheader(f"📌 {selected_stock} Data")

# -------------------------
# CHARTS
# -------------------------
if chart_type == "Candlestick":
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])

elif chart_type == "Line":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

elif chart_type == "Bar":
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Close'], name='Close'))

fig.update_layout(title=f"{selected_stock} Price Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# RETURNS
# -------------------------
df['Returns'] = df['Close'].pct_change()

st.subheader("📉 Daily Returns")
st.line_chart(df['Returns'])

# -------------------------
# ADF TEST
# -------------------------
st.subheader("🧪 Stationarity Test (ADF)")

result = adfuller(df['Close'].dropna())

st.write(f"ADF Statistic: {result[0]}")
st.write(f"p-value: {result[1]}")

if result[1] < 0.05:
    st.success("Series is Stationary")
else:
    st.warning("Series is NOT Stationary")

# -------------------------
# ARIMA MODEL
# -------------------------
st.subheader("🔮 ARIMA Forecast")

model = ARIMA(df['Close'], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_days)

future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='B')[1:]

# Plot forecast
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index, y=df['Close'],
    mode='lines', name='Actual'
))

fig2.add_trace(go.Scatter(
    x=future_dates, y=forecast,
    mode='lines', name='Forecast'
))

fig2.update_layout(title="Forecast vs Actual")

st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# RAW DATA
# -------------------------
if st.checkbox("Show Raw Data"):
    st.write(df.tail())
