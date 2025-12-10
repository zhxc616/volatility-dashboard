import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta

DB_NAME = "financial_data.db"


def fetch_and_save_data(ticker):
    """
    Fetches 1y historical data from Yahoo Finance and caches it in SQLite.
    Raises ValueError if no data is found (e.g., invalid ticker).
    """
    stock = yf.Ticker(ticker)

    # auto_adjust=True fixes issues with some stock splits/dividends
    df = stock.history(period="1y", auto_adjust=True)

    # SAFETY CHECK 1: was data received
    if df.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for '{ticker}'. This might be an API issue or invalid symbol.")

    df.reset_index(inplace=True)

    # SAFETY CHECK 2: does data contain what is needed
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError(f"Data format error for '{ticker}'. Missing Date or Close columns.")

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


def get_company_info(ticker):
    """
    Fetches fundamental data for the Key Stats panel.
    Returns a dictionary of formatted strings.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    # Helper to format large numbers
    def format_market_cap(value):
        if not value or value == "N/A": return "N/A"
        if value >= 1e12: return f"${value / 1e12:.2f}T"
        if value >= 1e9: return f"${value / 1e9:.2f}B"
        if value >= 1e6: return f"${value / 1e6:.2f}M"
        return f"${value}"

    return {
        "sector": info.get("sector", "N/A"),
        "market_cap": format_market_cap(info.get("marketCap")),
        "pe_ratio": f"{info.get('trailingPE', 0):.2f}" if info.get("trailingPE") else "N/A",
        "high_52": f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get("fiftyTwoWeekHigh") else "N/A"
    }

def visualise_data(ticker):
    """
    Generates an interactive Plotly chart with SMA, Bollinger Bands, and AI Forecast.
    Returns: HTML string containing the interactive plot div.
    """
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql(f"SELECT Date, Close FROM {ticker}_data", conn)

    # 1. Prepare Data
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')

    # --- Technical Indicators ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)

    # --- AI Forecasting (Linear Regression) ---
    df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())

    model = LinearRegression()
    model.fit(df[['Date_Ordinal']], df['Close'])

    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    future_ordinals = [[d.toordinal()] for d in future_dates]
    predicted_prices = model.predict(future_ordinals)

    # --- Plotly Interactive Chart ---
    fig = go.Figure()

    # 1. Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Upper_Band'],
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Lower_Band'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)',
        name='Bollinger Bands', hoverinfo='skip'
    ))

    # 2. SMA (Orange)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['SMA_20'],
        mode='lines', name='20-Day SMA',
        line=dict(color='#FF9500', width=1.5)
    ))

    # 3. Actual Price (Blue)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name=f'{ticker} Price',
        line=dict(color='#00B4F0', width=2)
    ))

    # 4. AI Forecast (Gold Dashed)
    fig.add_trace(go.Scatter(
        x=future_dates, y=predicted_prices,
        mode='lines', name='7-Day AI Forecast',
        line=dict(color='#FFD700', width=2, dash='dash')
    ))

    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),

        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="#121212",
            font_color="#FFFFFF",
            bordercolor="#333333"
        ),

        xaxis=dict(showgrid=True, gridcolor='#333333'),
        yaxis=dict(showgrid=True, gridcolor='#333333', title='Price (USD)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Return HTML string
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})