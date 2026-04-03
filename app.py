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

st.set_page_config(page_title="Stock Predictor", layout="wide")

# 🎯 Title
st.title("📈 Stock Market Prediction App (ARIMA Model)")

# 📊 Stock list
stocks = {
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "Infosys": "INFY.NS",
    "HCLTech": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",
    "Persistent": "PERSISTENT.NS",
    "Oracle Financial Services": "OFSS.NS",
    "Coforge": "COFORGE.NS",
    "Mphasis": "MPHASIS.NS"
}

# 🎛️ Sidebar controls
st.sidebar.header("⚙️ Controls")

selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

start_date = st.sidebar.date_input("Start Date", d.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", d.date.today())

forecast_days = st.sidebar.slider("Forecast Days", 5, 60, 10)

# 📥 Load data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

df = load_data(stocks[selected_stock], start_date, end_date)

# 📉 Plot raw data
st.subheader(f"📊 {selected_stock} Closing Price")
st.line_chart(df['Close'])

# 🔍 ADF Test
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1]

p_value = check_stationarity(df['Close'])

st.write(f"📌 ADF Test p-value: {p_value:.5f}")

# ⚠️ Make stationary if needed
if p_value > 0.05:
    st.warning("Data is NOT stationary → Applying Differencing")
    df['Close'] = df['Close'].diff()
    df.dropna(inplace=True)
else:
    st.success("Data is Stationary ✅")

# 🤖 Model Training
st.subheader("⚙️ Training ARIMA Model...")

model = ARIMA(df['Close'], order=(5,1,0))
model_fit = model.fit()

# 🔮 Forecast
forecast = model_fit.forecast(steps=forecast_days)

# 📅 Forecast Dates
future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='B')[1:]

# 📈 Plot
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(df.index, df['Close'], label="Actual")
ax.plot(future_dates, forecast, label="Forecast", linestyle='dashed')

ax.set_title(f"{selected_stock} Price Prediction")
ax.legend()

st.pyplot(fig)

# 📊 Show forecast table
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": forecast
})

st.subheader("📅 Forecast Data")
st.dataframe(forecast_df)

# 📉 Returns Visualization
st.subheader("📉 Daily Returns")
df['Returns'] = df['Close'].pct_change()

st.line_chart(df['Returns'])

# 📊 Moving Average
st.subheader("📊 Moving Averages")

df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(df['Close'], label="Close")
ax2.plot(df['MA50'], label="MA50")
ax2.plot(df['MA200'], label="MA200")
ax2.legend()

st.pyplot(fig2)

# 📌 Footer
st.write("Built with ❤️ using Streamlit + ARIMA")
