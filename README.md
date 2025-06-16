# GPU Hardware Profiling and Monitoring Tool

It provides a comprehensive, cross-platform toolkit to capture GPU hardware details, profile the software stack, and implement active, real-time monitoring with a web-based dashboard.

It is designed to work on both **Linux**, **MacOS** and **Windows** and can detect GPUs from major vendors like **NVIDIA, AMD, Intel, and Google TPUs (on Linux)**.

## Features

- **Platform Profiling**: Gathers detailed information about:
  - **GPU Hardware**: Vendor, model, memory (supports NVIDIA, AMD, Intel, and Google TPUs).
  - **System**: OS, architecture, hostname.
  - **Software Stack**: GPU drivers, CUDA version, OpenCL availability.
- **Workload Monitoring**: Provides real-time metrics for:
  - GPU utilization, memory usage, temperature, and power draw.
  - System-level stats like CPU and RAM usage.
  - A list of processes currently using the GPU (NVIDIA only).
- **Web Dashboard**: A clean and modern web interface to visualize all monitoring data in real time.
- **Alerting**: Basic threshold-based alerting for key metrics.
- **Cross-Platform**: Designed to run on Linux and Windows.

### Prerequisites

- **Python 3.10+**
- **Git** (for cloning the repository)
- **GPU Drivers**: Ensure the appropriate drivers for your GPU(s) are installed.
  - **NVIDIA**: NVIDIA drivers, which include `nvidia-smi`.
  - **AMD (Linux)**: AMDGPU drivers and the `rocm-smi` tool.
  - **Google TPUs (Linux)**: TPU drivers (gasket, apex modules).
  - **Windows**: Standard WDDM drivers for all vendors.

### Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/RohitRathore1/gpu-hardware-profiling-and-monitoring.git
cd gpu-hardware-profiling-and-monitoring
```

Next, install the required Python packages using `pip`. It is highly recommended to do this within a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Usage

The tool is operated via the `run.py` script, which provides two main commands: `profile` and `monitor`.

#### To Run a System Profile:

This command performs a one-time scan of the system and prints a detailed JSON report of the hardware and software configuration.

```bash
python3 run.py profile
```

This is useful for quickly inventorying a machine's capabilities.

#### To Launch the Monitoring Dashboard:

This command starts a continuous monitoring service and launches a web server to display the data.

```bash
python3 run.py monitor
```

The dashboard will be available at `http://127.0.0.1:8000`.
