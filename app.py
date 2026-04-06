import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as d
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.title("📊 IT Stocks Forecast Dashboard")

stocks = {"TCS": "TCS.NS","Wipro": "WIPRO.NS","Infosys": "INFY.NS","HCLTech": "HCLTECH.NS","Tech Mahindra": "TECHM.NS","LTIMindtree": "LTIM.NS",
          "Persistent": "PERSISTENT.NS","OFSS": "OFSS.NS","Coforge": "COFORGE.NS","Mphasis": "MPHASIS.NS"}


st.sidebar.header("Settings")
selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))


start_date = st.sidebar.date_input("Start Date", d.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", d.date.today())


chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line", "Bar"])
df = yf.download(stocks[selected_stock], start=start_date, end=end_date)


if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if df.empty:
    st.error("❌ Data not loaded. Check internet or stock symbol.")
    st.stop()

if st.checkbox("Show Raw Data"):
    st.write(df.tail())


# Stationarity Check
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1]

check_stationarity(df['Close'])

st.success("✅ Data is Stationary")
['Close_Diff']=df['Close'].diff().dropna()
check_stationarity(t['Close_Diff'])

#if p_value < 0.05:
    #st.success("✅ Data is Stationary")
   # stationary_series = df['Close']
          
#else:
 #   st.warning("⚠️ Data is NOT Stationary → Applying Differencing")
  #  stationary_series = df['Close'].diff().dropna()




# ARIMA Model
try:
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)
    future_dates = pd.date_range(start=df.index[-1], periods=11, freq='B')[1:]

except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()


# Plot Graph
fig = go.Figure()

if chart_type == "Candlestick":fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlestick"))

elif chart_type == "Line":fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))

elif chart_type == "Bar":fig.add_trace(go.Bar(x=df.index,y=df['Close'], name='Close Price'))


# Forecast line
fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast', line=dict(dash='dash')))
fig.update_layout(title=f"{selected_stock} Stock Price Forecast", xaxis_title="Date", yaxis_title="Price", height=600)
st.plotly_chart(fig, use_container_width=True)


# Metrics
st.subheader("📈 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Latest Price", round(df['Close'].iloc[-1], 2))
col2.metric("Average Price", round(df['Close'].mean(), 2))
col3.metric("Volatility", round(df['Close'].std(), 2))
