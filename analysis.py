import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend for non-GUI server rendering
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from datetime import timedelta

DB_NAME = "financial_data.db"


def fetch_and_save_data(ticker):
    """
    Fetches 1y historical data from Yahoo Finance and caches it in SQLite.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")

    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]

    with sqlite3.connect(DB_NAME) as conn:
        df.to_sql(name=f'{ticker}_data', con=conn, if_exists='replace', index=False)


def calculate_volatility(ticker):
    """
    Calculates annualised volatility based on daily returns.
    """
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql(f"SELECT Date, Close FROM {ticker}_data", conn)

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')

    # Calculate standard deviation of daily percentage change
    df['Returns'] = df['Close'].pct_change()
    daily_volatility = df['Returns'].std()

    # Annualise (approx 252 trading days)
    return daily_volatility * np.sqrt(252) * 100


def visualise_data(ticker):
    """
    Generates a static plot with SMA, Bollinger Bands, and Linear Regression forecast.
    Returns: Base64 encoded PNG string.
    """
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql(f"SELECT Date, Close FROM {ticker}_data", conn)

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')

    # --- Technical Indicators ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)

    # --- Trend Forecasting (Linear Regression) ---
    df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())

    model = LinearRegression()
    model.fit(df[['Date_Ordinal']], df['Close'])

    # Predict next 7 days
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    future_ordinals = [[d.toordinal()] for d in future_dates]
    predicted_prices = model.predict(future_ordinals)

    # --- Plotting ---
    # Define Dark Mode palette
    bg_color = '#121212'
    text_color = '#FFFFFF'
    accent_color = '#00B4F0'
    sma_color = '#FF9500'
    band_color = '#333333'
    grid_color = '#333333'
    prediction_color = '#FFD700'

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Plot layers
    ax.fill_between(df['Date'], df['Upper_Band'], df['Lower_Band'], color=band_color, alpha=0.3,
                    label='Bollinger Bands (2σ)')
    ax.plot(df['Date'], df['SMA_20'], label='20-Day SMA', color=sma_color, linestyle='-', linewidth=1.5, alpha=0.8)
    ax.plot(df['Date'], df['Close'], label=f'{ticker} Price', color=accent_color, linewidth=2)
    ax.plot(future_dates, predicted_prices, label='AI Forecast (7d)', color=prediction_color, linestyle='--',
            linewidth=2)

    # Styling
    ax.set_title(f'{ticker} Technical Analysis', color=text_color, weight='bold', pad=20)
    ax.set_xlabel('Date', color=text_color, labelpad=10)
    ax.set_ylabel('Price (USD)', color=text_color, labelpad=10)

    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.7)

    legend = ax.legend(loc='upper left', fontsize='small')
    legend.get_frame().set_facecolor(bg_color)
    legend.get_frame().set_edgecolor(grid_color)
    for text in legend.get_texts():
        text.set_color(text_color)

    # Save to buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()