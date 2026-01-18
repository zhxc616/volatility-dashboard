import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta

DB_NAME = "financial_data.db"


def fetch_and_save_data(ticker):

    # Fetches 1y historical data from Yahoo Finance and caches it in SQLite.
    # Raises ValueError if no data is found (e.g., invalid ticker).

    stock = yf.Ticker(ticker)
    df = stock.history(period="1y", auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for '{ticker}'. This might be an API issue or invalid symbol."
        )

    df.reset_index(inplace=True)

    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(
            f"Data format error for '{ticker}'. Missing Date or Close columns."
        )

    df = df[["Date", "Close"]]

    with sqlite3.connect(DB_NAME) as conn:
        df.to_sql(name=f"{ticker}_data", con=conn, if_exists="replace", index=False)


def calculate_volatility(ticker):

    # annualised volatility calculation

    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql(f"SELECT Date, Close FROM {ticker}_data", conn)

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date")

    df["Returns"] = df["Close"].pct_change()
    daily_volatility = df["Returns"].std()

    return daily_volatility * np.sqrt(252) * 100


def get_company_info(ticker):

    # fetches stats data

    stock = yf.Ticker(ticker)
    info = stock.info

    def format_market_cap(value):
        if not value or value == "N/A":
            return "N/A"
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        if value >= 1e9:
            return f"${value / 1e9:.2f}B"
        if value >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value}"

    return {
        "sector": info.get("sector", "N/A"),
        "market_cap": format_market_cap(info.get("marketCap")),
        "pe_ratio": (
            f"{info.get('trailingPE', 0):.2f}" if info.get("trailingPE") else "N/A"
        ),
        "high_52": (
            f"${info.get('fiftyTwoWeekHigh', 0):.2f}"
            if info.get("fiftyTwoWeekHigh")
            else "N/A"
        ),
    }


def get_chart_data_json(ticker):
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql(f"SELECT Date, Close FROM {ticker}_data", conn)

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date")

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["Lower_Band"] = df["SMA_20"] - (df["Std_Dev"] * 2)

    df["Date_Ordinal"] = df["Date"].apply(lambda x: x.toordinal())
    model = LinearRegression()

    # only train on valid data
    train_df = df.dropna()
    if not train_df.empty:
        model.fit(train_df[["Date_Ordinal"]], train_df["Close"])

        last_date = df["Date"].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        future_ordinals = [[d.toordinal()] for d in future_dates]
        predicted_prices = model.predict(future_ordinals)
    else:
        # fallback if not enough data
        future_dates = []
        predicted_prices = np.array([])

    # helper function to clean nan
    def clean_nan(data_list):
        return [None if np.isnan(x) else x for x in data_list]

    return {
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": clean_nan(df["Close"].tolist()),
        "sma": clean_nan(df["SMA_20"].tolist()),
        "upper": clean_nan(df["Upper_Band"].tolist()),
        "lower": clean_nan(df["Lower_Band"].tolist()),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_prices": predicted_prices.tolist(),
    }
