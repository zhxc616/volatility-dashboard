from flask import Flask, render_template, request, jsonify
from analysis import (
    fetch_and_save_data,
    calculate_volatility,
    visualise_data,
    get_company_info,
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def dashboard():
    # Default to AAPL on initial load
    ticker = "AAPL"

    # Handle user search submission
    if request.method == "POST":
        user_input = request.form.get("ticker")
        if user_input:
            ticker = user_input.upper()

    volatility = None
    chart_data = None
    company_info = None
    error_msg = None

    try:
        # 1. Run ETL pipeline to update local database
        fetch_and_save_data(ticker)

        # 2. Get Chart & Info (Still server-side for now)
        chart_data = visualise_data(ticker)
        company_info = get_company_info(ticker)

        # Note: We are NO LONGER calculating volatility here.
        # The JavaScript will ask for it separately!

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        error_msg = f"Could not find data for '{ticker}'. Please check the symbol."

    return render_template(
        "index.html",
        ticker=ticker,
        chart=chart_data,
        info=company_info,
        error=error_msg,
    )


# --- NEW API ROUTE FOR JAVASCRIPT ---
@app.route("/api/get-volatility")
def get_volatility_api():
    # Get the ticker from the URL query (e.g., ?ticker=MSFT)
    ticker = request.args.get('ticker', 'AAPL')

    try:
        # Calculate it on the fly
        vol = calculate_volatility(ticker)

        # Return pure JSON data
        return jsonify({
            "ticker": ticker,
            "volatility": round(vol, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
