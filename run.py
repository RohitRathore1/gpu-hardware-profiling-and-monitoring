import argparse
import json
import sys
import os
import platform

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_profile():
    """
    Imports the GPUProfiler and prints a JSON report of the system's hardware and software profile.
    """
    from src.profiler import GPUProfiler
    
    print("Detected Operating System:", platform.system())
    print("Gathering system profile...")
    try:
        profiler = GPUProfiler()
        profile_data = profiler.profile_system()
        print(json.dumps(profile_data, indent=2))
        print("\nProfile report complete.")
    except Exception as e:
        print(f"An error occurred during profiling: {e}")

def run_monitor():
    """
    Imports the dashboard app and starts the Flask server.
    """
    from src.dashboard import app, monitor

    print("Starting monitoring dashboard...")
    try:
        app.run(host='0.0.0.0', port=8000, debug=True)
    except Exception as e:
        print(f"An error occurred while starting the dashboard: {e}")

def main():
    """
    Parses command-line arguments to determine which action to perform.
    """
    parser = argparse.ArgumentParser(
        description="GPU Hardware Profiling and Monitoring Tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'action',
        choices=['profile', 'monitor'],
        help=(
            "Choose the action to perform:\n"
            "  profile   - Run a one-time scan and print a JSON report of the system's GPU and software stack.\n"
            "  monitor   - Launch a web-based dashboard for real-time monitoring of GPU and system metrics."
        )
    )

    args = parser.parse_args()

    # Check for OS, as some features are platform-specific.
    # This is informational, as the modules themselves handle OS-specific logic.
    print(f"Detected Operating System: {platform.system()}")

    if args.action == 'profile':
        run_profile()
    elif args.action == 'monitor':
        run_monitor()

if __name__ == '__main__':
    main() 
    