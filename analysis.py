import matplotlib
matplotlib.use('Agg') # Fixes "main thread is not in main loop" error on web servers
import matplotlib.pyplot as plt
import io
import base64
import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# database file name constant
DB_NAME = "financial_data.db"


def fetch_and_save_data(ticker):
    """
    EXTRACT & LOAD:
    1. Fetches 1 year of data from Yahoo Finance.
    2. Saves (caches) it into a local SQLite database.
    """
    print(f"--- Fetching data for {ticker} ---")

    # 1. Extract: Get the data object
    stock = yf.Ticker(ticker)
    # Download 1 year of history
    df = stock.history(period="1y")

    # Reset index so 'Date' becomes a standard column, not the index
    df.reset_index(inplace=True)

    # Select only the columns we need to save space
    df = df[['Date', 'Close']]

    # 2. Load: Connect to database and save
    # 'with' context manager ensures the connection closes automatically
    with sqlite3.connect(DB_NAME) as conn:
        # if_exists='replace' refreshes the cache completely each time
        df.to_sql(name=f'{ticker}_data', con=conn, if_exists='replace', index=False)
        print(f"Success: Data for {ticker} saved to {DB_NAME}")


def calculate_volatility(ticker):
    """
    QUERY & ANALYSE:
    1. Reads the 'Close' prices from the SQL database.
    2. Calculates annualised volatility.
    """
    # 1. Query: Read from SQL
    with sqlite3.connect(DB_NAME) as conn:
        query = f"SELECT Date, Close FROM {ticker}_data"
        df = pd.read_sql(query, conn)

    # Ensure data is sorted by date for accurate math
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')

    # 2. Analyse: Calculate Daily Returns (Percentage Change)
    df['Returns'] = df['Close'].pct_change()

    # Calculate Standard Deviation of returns (Volatility)
    daily_volatility = df['Returns'].std()

    # Annualise: Multiply by square root of trading days (approx 252)
    annual_volatility = daily_volatility * np.sqrt(252)

    # Convert to percentage for display
    return annual_volatility * 100


def visualise_data(ticker):
    """
    VISUALISE (Full FinTech Edition):
    1. Query data.
    2. Calculate Technical Indicators (SMA + Bollinger Bands).
    3. Train AI Model (Linear Regression).
    4. Plot EVERYTHING.
    """
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql(f"SELECT Date, Close FROM {ticker}_data", conn)

    # 1. Prepare Data
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')

    # --- TECHNICAL INDICATORS SECTION ---
    # 20-Day Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Bollinger Bands (SMA +/- 2 Standard Deviations)
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)

    # --- AI/ML SECTION ---
    df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())
    X = df[['Date_Ordinal']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)

    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    future_ordinals = [[d.toordinal()] for d in future_dates]
    predicted_prices = model.predict(future_ordinals)

    # --- PLOTTING SECTION ---
    bg_color = '#121212'
    text_color = '#FFFFFF'
    accent_color = '#00B4F0'  # Cyan for Price
    sma_color = '#FF9500'  # Orange for SMA
    band_color = '#333333'  # Dark Gray for Bollinger Band Fill
    grid_color = '#333333'
    prediction_color = '#FFD700'  # Gold for AI

    fig, ax = plt.subplots(figsize=(10, 6))  # Made slightly taller
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 1. Bollinger Bands (Shaded Area) - Plot this first so it's in the background
    ax.fill_between(df['Date'], df['Upper_Band'], df['Lower_Band'], color=band_color, alpha=0.3,
                    label='Bollinger Bands (2σ)')

    # 2. SMA (Simple Moving Average)
    ax.plot(df['Date'], df['SMA_20'], label='20-Day SMA', color=sma_color, linestyle='-', linewidth=1.5, alpha=0.8)

    # 3. Historical Price
    ax.plot(df['Date'], df['Close'], label=f'{ticker} Price', color=accent_color, linewidth=2)

    # 4. AI Forecast
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

    # Legend setup (Loc 2 = Top Left)
    legend = ax.legend(loc='upper left', fontsize='small')
    legend.get_frame().set_facecolor(bg_color)
    legend.get_frame().set_edgecolor(grid_color)
    for text in legend.get_texts():
        text.set_color(text_color)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()

# This block allows us to test the script independently
if __name__ == "__main__":
    ticker_symbol = "AAPL"  # Apple Inc.

    # Run the ETL pipeline
    fetch_and_save_data(ticker_symbol)

    # Run the Analysis
    vol = calculate_volatility(ticker_symbol)

    print(f"--------------------------------------------------")
    print(f"The Annualised Volatility for {ticker_symbol} is: {vol:.2f}%")
    print(f"--------------------------------------------------")