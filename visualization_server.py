from flask import Flask, render_template
import webbrowser
import threading
from datetime import datetime

app = Flask(__name__)

# Store data globally for the Flask app
chart_data = {}


@app.route('/')
def index():
    if not chart_data:
        return "<h1>Loading data...</h1><p>Please wait while the model trains.</p>", 503
    return render_template('visualization.html', **chart_data)


def start_server(port=5000):
    """Start the Flask server in a separate thread."""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, threaded=True)


def open_browser(port=5000):
    """Open the browser after a short delay."""
    import time
    time.sleep(2)
    webbrowser.open(f'http://127.0.0.1:{port}')


def visualize_predictions(item_id, historical_dates, historical_prices,
                         test_dates, test_actual, test_predicted,
                         future_dates, future_predictions,
                         split_date, today, start_date,
                         metrics, forecast_summary, outliers=None):
    """
    Create and display interactive visualization in browser.
    
    Args:
        item_id: Item identifier
        historical_dates: List of datetime objects for all historical data
        historical_prices: List of prices for all historical data
        test_dates: List of datetime objects for test data
        test_actual: List of actual prices in test set
        test_predicted: List of predicted prices in test set
        future_dates: List of datetime objects for future predictions
        future_predictions: List of predicted future prices
        split_date: Datetime object for train/test split
        today: Datetime object for today
        start_date: Datetime object for data start
        metrics: Dict with keys: mae, rmse, r2, mape
        forecast_summary: Dict with keys: current_price, avg_forecast, trend, min_pred, max_pred
        outliers: Optional dict with outlier information from detect_outliers_for_viz()
    """
    global chart_data
    
    # Prepare outlier data for visualization
    outliers_data = None
    if outliers:
        outliers_data = {
            'count': outliers['count'],
            'percentage': outliers['percentage'],
            'timestamps': outliers['outlier_timestamps'],
            'prices': outliers['outlier_prices']
        }
    
    # Prepare data for chart
    chart_data = {
        'title': f'{item_id} - Price Prediction Analysis',
        'subtitle': f'Training Period: {start_date.strftime("%Y-%m-%d")} to {today.strftime("%Y-%m-%d")} | 30-Day Forecast',
        'metrics': metrics,
        'forecast': {
            'trend': forecast_summary['trend']
        },
        'outliers': outliers_data,
        'chart_data': {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in historical_dates],
                'prices': [float(p) for p in historical_prices]
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in test_dates],
                'actual': [float(p) for p in test_actual],
                'predicted': [float(p) for p in test_predicted]
            },
            'future': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'predictions': [float(p) for p in future_predictions],
                'days': list(range(1, len(future_predictions) + 1)),
                'upper_bound': [float(p * 1.05) for p in future_predictions],
                'lower_bound': [float(p * 0.95) for p in future_predictions]
            },
            'outliers': outliers_data,
            'split_date': split_date.strftime('%Y-%m-%d'),
            'today': today.strftime('%Y-%m-%d')
        }
    }
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open browser
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print(f"\n{'='*60}")
    print(f"Visualization ready at: http://127.0.0.1:5000")
    print(f"Press Ctrl+C to stop the server")
    print(f"{'='*60}\n")
    
    # Keep the main thread alive
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\nServer stopped.")
