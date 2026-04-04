# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as d
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

st.title("📈 Stock Analysis & Forecasting App")

# Sidebar
stocks = {
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "Infosys": "INFY.NS",
    "HCLTECH": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIM": "LTIM.NS",
    "Persistent": "PERSISTENT.NS",
    "OFSS": "OFSS.NS",
    "Coforge": "COFORGE.NS",
    "Mphasis": "MPHASIS.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar"])
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10)

start = st.sidebar.date_input("Start Date", d.date(2025,1,1))
end = st.sidebar.date_input("End Date", d.date.today())

# Load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start, end=end)
    data.columns = data.columns.get_level_values(0)
    return data


data = load_data(stocks[selected_stock])

st.subheader(f"📊 {selected_stock} Stock Data")

# Chart Visualization
fig = go.Figure()

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    ))

elif chart_type == "Line":
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))

elif chart_type == "Bar":
    fig.add_trace(go.Bar(x=data.index, y=data['Close'], name='Close'))

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Stationarity Check
st.subheader("📉 Stationarity Check (ADF Test)")

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1]

p_value = check_stationarity(data['Close'])

st.write(f"p-value: {p_value:.5f}")

if p_value < 0.05:
    st.success("Series is Stationary ✅")
    ts = data['Close']
    d_order = 0
else:
    st.warning("Series is NOT Stationary ⚠️ → Applying Differencing")
    ts = data['Close'].diff().dropna()
    d_order = 1

# ARIMA Model
st.subheader("🤖 ARIMA Forecast")

try:
    model = ARIMA(data['Close'], order=(5, d_order, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_days)
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='B')[1:]

    # Plot forecast
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Actual"))
    fig2.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(dash='dash')))

    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(pd.DataFrame({"Date": future_dates, "Forecast": forecast}).set_index("Date"))

except Exception as e:
    st.error(f"Error in ARIMA model: {e}")

st.markdown("---")
st.markdown("### 🚀 Features:")
st.markdown("- Multiple stock selection")
st.markdown("- Candlestick, Line, Bar charts")
st.markdown("- Automatic stationarity check")
st.markdown("- ARIMA forecasting")

# Run command
st.sidebar.markdown("---")
st.sidebar.markdown("### ▶️ Run App:")
st.sidebar.code("streamlit run streamlit_app.py")
