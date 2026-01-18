document.addEventListener('DOMContentLoaded', function() {

    // initial apple ticker
    let currentTicker = document.getElementById('volatility-box').getAttribute('data-ticker');

    // fetch volatility
    async function fetchVolatility(ticker) {
        const displayElement = document.getElementById('volatility-display');
        displayElement.innerText = "Loading...";
        displayElement.style.color = "#A0A0A0";

        try {
            const response = await fetch(`/api/get-volatility?ticker=${ticker}`);
            const data = await response.json();

            if (data.error) {
                displayElement.innerText = "N/A";
            } else {
                displayElement.innerText = data.volatility + "%";
                if (data.volatility > 40) displayElement.style.color = "#FF3B30";
                else displayElement.style.color = "#00B4F0";
            }
        } catch (error) {
            console.error(error);
            displayElement.innerText = "Error";
        }
    }

    // fetch stats for panel
    async function fetchStats(ticker) {
        // reset
        ['stat-sector', 'stat-market-cap', 'stat-pe', 'stat-high'].forEach(id => {
            document.getElementById(id).innerText = "...";
        });

        try {
            const response = await fetch(`/api/get-stats?ticker=${ticker}`);
            const data = await response.json();

            if (!data.error) {
                document.getElementById('stat-sector').innerText = data.sector;
                document.getElementById('stat-market-cap').innerText = data.market_cap;
                document.getElementById('stat-pe').innerText = data.pe_ratio;
                document.getElementById('stat-high').innerText = data.high_52;
            }
        } catch (error) {
            console.error(error);
        }
    }

    // fetch chart on client side
    async function fetchChart(ticker) {
        const chartBox = document.getElementById('chart-box');
        chartBox.style.opacity = "0.5";
        chartBox.innerHTML = ""; // Clear old chart

        try {
            const response = await fetch(`/api/get-chart-data?ticker=${ticker}`);
            const data = await response.json();

            if (!data.error) {
                // plotly config
                const traceUpper = {
                    x: data.dates, y: data.upper, type: 'scatter', mode: 'lines',
                    line: { width: 0 }, showlegend: false, hoverinfo: 'skip'
                };
                const traceLower = {
                    x: data.dates, y: data.lower, type: 'scatter', mode: 'lines',
                    line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(255, 255, 255, 0.05)',
                    name: 'Bollinger Bands', hoverinfo: 'skip'
                };
                const tracePrice = {
                    x: data.dates, y: data.close, type: 'scatter', mode: 'lines',
                    name: ticker, line: { color: '#00B4F0', width: 2 }
                };
                const traceSMA = {
                    x: data.dates, y: data.sma, type: 'scatter', mode: 'lines',
                    name: '20-Day SMA', line: { color: '#FF9500', width: 1.5 }
                };
                const traceForecast = {
                    x: data.forecast_dates, y: data.forecast_prices, type: 'scatter', mode: 'lines',
                    name: '7-Day Forecast', line: { color: '#FFD700', width: 2, dash: 'dash' }
                };

                const layout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    height: 500,
                    margin: { l: 40, r: 20, t: 40, b: 30 },
                    xaxis: { showgrid: true, gridcolor: '#333333' },
                    yaxis: { showgrid: true, gridcolor: '#333333', title: 'Price (USD)' },
                    legend: { orientation: 'h', y: 1.02, x: 1, xanchor: 'right', font: {color: '#FFF'} },

                    // popup styling
                    hovermode: 'x unified',
                    hoverlabel: {
                        bgcolor: "#1E1E1E",
                        bordercolor: "#333333",
                        font: { color: "#FFFFFF" }
                    },

                    font: { color: '#A0A0A0' }
                };
                const config = { responsive: true, displayModeBar: false };

                Plotly.newPlot('chart-box',
                    [traceUpper, traceLower, traceSMA, tracePrice, traceForecast],
                    layout, config
                );
            } else {
                chartBox.innerHTML = "<div style='padding:20px; text-align:center;'>Chart Unavailable</div>";
            }
        } catch (error) {
            console.error(error);
        } finally {
            chartBox.style.opacity = "1";
        }
    }

    // form submit handlerr
    const form = document.getElementById('search-form');
    form.addEventListener('submit', function(event) {
        event.preventDefault(); // STOP RELOAD

        const input = document.getElementById('ticker-input');
        const newTicker = input.value.toUpperCase();

        // Update Title
        document.querySelector('.highlight-blue').innerText = newTicker;

        // Run updates
        fetchVolatility(newTicker);
        fetchStats(newTicker);
        fetchChart(newTicker);
    });

    // run
    fetchVolatility(currentTicker);
    fetchStats(currentTicker);
    fetchChart(currentTicker);
});