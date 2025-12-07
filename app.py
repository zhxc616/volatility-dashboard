from flask import Flask, render_template, request
from analysis import fetch_and_save_data, calculate_volatility, visualise_data

app = Flask(__name__)


# Update route to accept both GET (viewing the page) and POST (submitting the form)
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    # Default to Apple if the user just visits the page
    ticker = "AAPL"

    # If the user submitted the form (POST), get their input
    if request.method == 'POST':
        user_input = request.form.get('ticker')
        # specific check: did they actually type something?
        if user_input:
            ticker = user_input.upper()  # Convert "goog" to "GOOG"

    # Define variables to hold our data
    volatility = None
    chart_data = None
    error_msg = None

    try:
        # 1. Run the Pipeline
        fetch_and_save_data(ticker)

        # 2. Get Analytics
        volatility = calculate_volatility(ticker)
        chart_data = visualise_data(ticker)

    except Exception as e:
        # If anything goes wrong (e.g., invalid ticker, no internet), catch it here
        print(f"Error processing {ticker}: {e}")
        error_msg = f"Could not find data for '{ticker}'. Please check the symbol."

    # 3. Render the Template
    # We pass the 'error' variable so index.html knows whether to show the red banner
    return render_template('index.html',
                           ticker=ticker,
                           vol=volatility,
                           chart=chart_data,
                           error=error_msg)


if __name__ == "__main__":
    app.run(debug=True)