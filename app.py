from flask import Flask, render_template, request
from analysis import fetch_and_save_data, calculate_volatility, visualise_data, get_company_info

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def dashboard():
    # Default to AAPL on initial load
    ticker = "AAPL"

    # Handle user search submission
    if request.method == 'POST':
        user_input = request.form.get('ticker')
        if user_input:
            ticker = user_input.upper()

    volatility = None
    chart_data = None
    company_info = None
    error_msg = None

    try:
        # 1. Run ETL pipeline to update local database
        fetch_and_save_data(ticker)

        # 2. Get specific analytics and chart generation
        volatility = calculate_volatility(ticker)
        chart_data = visualise_data(ticker)

        # 3. Get Fundamentals (Key Stats)
        company_info = get_company_info(ticker)

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        error_msg = f"Could not find data for '{ticker}'. Please check the symbol."

    return render_template('index.html',
                           ticker=ticker,
                           vol=volatility,
                           chart=chart_data,
                           info=company_info,
                           error=error_msg)


if __name__ == "__main__":
    app.run(debug=True)