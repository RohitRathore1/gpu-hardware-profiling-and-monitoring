from flask import Flask, render_template, jsonify
from threading import Thread
import time

from .monitor import GPUMonitor

app = Flask(__name__, template_folder='../templates')

monitor = GPUMonitor(sampling_interval=2.0)

@app.route('/')
def dashboard():
    """
    Renders the main dashboard page.
    The frontend JavaScript will then call the /api/metrics endpoint to populate the data.
    """
    return render_template('dashboard.html')

@app.route('/api/metrics')
def api_metrics():
    """
    Provides the latest collected metrics as a JSON response.
    This is the endpoint that the dashboard's JavaScript will poll.
    """
    latest_metrics = monitor.get_latest_metrics()
    return jsonify(latest_metrics)

@app.route('/api/status')
def api_status():
    """
    Returns the current status of the monitor (running or not).
    """
    return jsonify({
        'monitoring': monitor.monitoring,
        'data_buffer_size': len(monitor.data_buffer)
    })

def start_monitoring_thread():
    """
    Starts the GPU monitoring in a background thread.
    This ensures the Flask app remains responsive.
    """
    if not monitor.monitoring:
        monitor.start_monitoring()

def run_app():
    """
    Main function to run the Flask application.
    """
    monitor_thread = Thread(target=start_monitoring_thread, daemon=True)
    monitor_thread.start()
    
    app.run(host='0.0.0.0', port=8000, debug=False)

if __name__ == '__main__':
    run_app()
