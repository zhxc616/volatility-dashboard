from flask import Flask, render_template, request, jsonify
from analysis import (
    fetch_and_save_data,
    calculate_volatility,
    get_company_info,
    get_chart_data_json
)

app = Flask(__name__)


@app.route("/")
def dashboard():
    # initial ticker shown
    ticker = "AAPL"

    try:
        fetch_and_save_data(ticker)
    except:
        pass  # error handled by js

    return render_template("index.html", ticker=ticker)


# api endpoints

@app.route("/api/get-volatility")
def get_volatility_api():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        fetch_and_save_data(ticker)

        vol = calculate_volatility(ticker)
        return jsonify({"volatility": round(vol, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get-stats")
def get_stats_api():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        fetch_and_save_data(ticker)

        info = get_company_info(ticker)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get-chart-data")
def get_chart_data_api():
    ticker = request.args.get('ticker', 'AAPL')
    try:
        fetch_and_save_data(ticker)

        data = get_chart_data_json(ticker)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)