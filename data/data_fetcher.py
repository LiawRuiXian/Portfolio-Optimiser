import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def fetch_price_data(tickers, start, end, price_field="Close"):
    tickers = [t.strip().upper() for t in tickers]
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df[price_field].copy() if price_field in df.columns.levels[0] else df["Close"].copy()
    else:
        prices = df.copy()
    prices = prices.ffill().dropna(how='all')
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.dropna(axis=1, how='all')
    return prices
