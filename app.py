import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Stock Pro Dashboard", layout="wide")

# ---------- Sidebar ----------
st.sidebar.title("📊 Stock Controls")

stock = st.sidebar.text_input("Enter NSE Stock Symbol", "TCS.NS")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))

show_ma20 = st.sidebar.checkbox("Show MA20", True)
show_ma50 = st.sidebar.checkbox("Show MA50", True)
show_ma200 = st.sidebar.checkbox("Show MA200", False)

# ---------- Fetch Data ----------
@st.cache_data
def load_data(stock, start, end):
    return yf.download(stock, start=start, end=end)

data = load_data(stock, start, end)

st.title("📈 NSE Stock Analytics Dashboard")

if not data.empty:

    # ---------- KPIs ----------
    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"₹{round(data['Close'].iloc[-1],2)}")
    col2.metric("Day High", f"₹{round(data['High'].iloc[-1],2)}")
    col3.metric("Day Low", f"₹{round(data['Low'].iloc[-1],2)}")
    col4.metric("Volume", f"{data['Volume'].iloc[-1]:,}")

    # ---------- Moving Averages ----------
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    # ---------- RSI ----------
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # ---------- Candlestick + Volume ----------
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick"
    ), row=1, col=1)

    # Moving Averages
    if show_ma20:
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"],
                                 line=dict(width=1),
                                 name="MA20"), row=1, col=1)

    if show_ma50:
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"],
                                 line=dict(width=1),
                                 name="MA50"), row=1, col=1)

    if show_ma200:
        fig.add_trace(go.Scatter(x=data.index, y=data["MA200"],
                                 line=dict(width=1),
                                 name="MA200"), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"],
                         name="Volume"), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"],
                             name="RSI"), row=3, col=1)

    fig.update_layout(height=800, title=f"{stock} Stock Analysis",
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # ---------- Multi Stock Comparison ----------
    st.subheader("📊 Compare Multiple Stocks")

    multi_stocks = st.multiselect(
        "Select Stocks for Comparison",
        ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"],
        default=["TCS.NS"]
    )

    if multi_stocks:
        compare_df = pd.DataFrame()
        for s in multi_stocks:
            temp = yf.download(s, start=start, end=end)
            compare_df[s] = temp["Close"]

        st.line_chart(compare_df)

    # ---------- Raw Data ----------
    with st.expander("🔍 View Raw Data"):
        st.dataframe(data)

else:
    st.error("No data found. Please check stock symbol.")
