import time
import threading
import json
import logging
import subprocess
import psutil
import platform
from datetime import datetime
from typing import Dict, List

class GPUAlerting:
    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or {
            'gpu_utilization': 90,  # %
            'memory_utilization': 85,  # %
            'temperature': 85,  # °C
            'power_draw_percent': 90,  # % of limit
        }
    
    def check_alerts(self, metrics: Dict) -> List[str]:
        alerts = []
        
        for gpu in metrics.get('nvidia_gpus', []):
            if gpu.get('temp_gpu') and gpu['temp_gpu'] > self.thresholds['temperature']:
                alerts.append(f"ALERT: GPU {gpu['gpu_index']} temperature high: {gpu['temp_gpu']}°C")
            
            if gpu.get('util_gpu') and gpu['util_gpu'] > self.thresholds['gpu_utilization']:
                alerts.append(f"ALERT: GPU {gpu['gpu_index']} utilization high: {gpu['util_gpu']}%")
            
            if gpu.get('power_draw') and gpu.get('power_limit') and gpu['power_limit'] > 0:
                power_percent = (gpu['power_draw'] / gpu['power_limit']) * 100
                if power_percent > self.thresholds['power_draw_percent']:
                    alerts.append(f"ALERT: GPU {gpu['gpu_index']} power draw high: {power_percent:.1f}%")

        
        # Check TPU alerts
        for tpu in metrics.get('tpu_gpus', []):
            if tpu.get('temp_tpu') and tpu['temp_tpu'] > self.thresholds['temperature']:
                alerts.append(f"ALERT: TPU {tpu['device']} temperature high: {tpu['temp_tpu']}°C")
            
            if tpu.get('power_draw') and tpu['power_draw'] > 100:  # Example threshold for TPU power
                alerts.append(f"ALERT: TPU {tpu['device']} power draw high: {tpu['power_draw']}W")
        
        return alerts

class GPUMonitor:
    def __init__(self, sampling_interval=1.0, alert_handler: GPUAlerting = None):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.data_buffer = []
        self.max_buffer_size = 1000
        self.os_type = platform.system().lower()
        self.alerter = alert_handler or GPUAlerting()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _safe_cast(self, value: str, type_func):
        """Safely convert string to a given type."""
        try:
            # Handle cases where nvidia-smi returns "[Not Supported]" or "N/A"
            if isinstance(value, str) and any(s in value for s in ['Not Supported', 'N/A']):
                return None
            return type_func(value)
        except (ValueError, TypeError):
            return None

    def get_nvidia_metrics(self) -> Dict:
        """Get comprehensive NVIDIA GPU metrics."""
        if self.os_type not in ['linux', 'windows']:
            return {'nvidia_gpus': []}
            
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
            
            metrics = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    metrics.append({
                        'gpu_index': self._safe_cast(parts[0], int),
                        'timestamp': parts[1],
                        'name': parts[2],
                        'temp_gpu': self._safe_cast(parts[3], int),
                        'util_gpu': self._safe_cast(parts[5], int),
                        'util_memory': self._safe_cast(parts[6], int),
                        'memory_total': self._safe_cast(parts[7], int),
                        'memory_free': self._safe_cast(parts[8], int),
                        'memory_used': self._safe_cast(parts[9], int),
                        'power_draw': self._safe_cast(parts[10], float),
                        'power_limit': self._safe_cast(parts[11], float),
                        'clock_graphics': self._safe_cast(parts[12], int),
                        'clock_memory': self._safe_cast(parts[13], int),
                        'fan_speed': self._safe_cast(parts[14], int)
                    })
            return {'nvidia_gpus': metrics}
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            return {'nvidia_gpus': []} # nvidia-smi not found or failed
        except Exception as e:
            self.logger.error(f"Error getting NVIDIA metrics: {e}")
            return {'nvidia_gpus': []}
    
    def get_amd_metrics(self) -> Dict:
        """Get AMD GPU metrics using rocm-smi (Linux). Placeholder for Windows."""
        if self.os_type != 'linux':
            return {'amd_gpus': []}
        
        try:
            # This is a simplified example. `rocm-smi` can output JSON directly.
            # rocm-smi --showalljson
            result = subprocess.run(['rocm-smi', '-a', '--json'], capture_output=True, text=True, timeout=10, check=True)
            smi_output = json.loads(result.stdout)
            
            amd_metrics = []
            for key, device in smi_output.items():
                if key.startswith('card'):
                    amd_metrics.append({
                        'gpu_index': int(device['Card series']),
                        'name': device['Card model'],
                        'temp_gpu': self._safe_cast(device.get('Temperature (C)'), float),
                        'util_gpu': self._safe_cast(device.get('GPU use (%)'), int),
                        'util_memory': self._safe_cast(device.get('GPU memory use (%)'), int),
                        'memory_total': self._safe_cast(device.get('GPU memory total (MB)'), int),
                        'memory_used': self._safe_cast(device.get('GPU memory used (MB)'), int),
                        'power_draw': self._safe_cast(device.get('Average Graphics Package Power (W)'), float),
                    })
            return {'amd_gpus': amd_metrics}
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            return {'amd_gpus': []} # rocm-smi not found or failed
        except Exception as e:
            self.logger.error(f"Error getting AMD metrics: {e}")
            return {'amd_gpus': []}

    def get_tpu_metrics(self) -> Dict:
        """Get Google TPU metrics on Linux."""
        if self.os_type != 'linux':
            return {'tpu_gpus': []}
        
        tpu_metrics = []
        
        try:
            import os
            import glob
            
            # Check for TPU devices in /sys/class/accel
            accel_path = '/sys/class/accel'
            if os.path.exists(accel_path):
                for device in sorted(os.listdir(accel_path)):
                    device_path = os.path.join(accel_path, device)
                    if os.path.islink(device_path):
                        try:
                            real_path = os.path.realpath(device_path)
                            metrics_dict = {
                                'device': device,
                                'name': f'TPU {device}',
                                'vendor': 'Google'
                            }
                            
                            # Read TPU status
                            status_file = os.path.join(real_path, 'status')
                            if os.path.exists(status_file):
                                try:
                                    with open(status_file, 'r') as f:
                                        metrics_dict['status'] = f.read().strip()
                                except:
                                    metrics_dict['status'] = 'Unknown'
                            
                            # Read memory ranges
                            mem_ranges_file = os.path.join(real_path, 'user_mem_ranges')
                            if os.path.exists(mem_ranges_file):
                                try:
                                    with open(mem_ranges_file, 'r') as f:
                                        mem_ranges = f.read().strip().split('\n')
                                        # Parse memory ranges to calculate total memory
                                        total_memory_mb = 0
                                        for mem_range in mem_ranges:
                                            if '-' in mem_range:
                                                start, end = mem_range.split('-')
                                                start_addr = int(start, 16)
                                                end_addr = int(end, 16)
                                                total_memory_mb += (end_addr - start_addr) / (1024 * 1024)
                                        metrics_dict['memory_total'] = int(total_memory_mb)
                                except:
                                    pass
                            
                            # Get PCI device info
                            device_dir = os.path.join(real_path, 'device')
                            if os.path.exists(device_dir):
                                # Read vendor and device IDs
                                vendor_file = os.path.join(device_dir, 'vendor')
                                device_file = os.path.join(device_dir, 'device')
                                if os.path.exists(vendor_file) and os.path.exists(device_file):
                                    try:
                                        with open(vendor_file, 'r') as f:
                                            vendor_id = f.read().strip()
                                        with open(device_file, 'r') as f:
                                            device_id = f.read().strip()
                                        metrics_dict['pci_id'] = f"{vendor_id}:{device_id}"
                                    except:
                                        pass
                            
                            # Note: Temperature, power, and detailed memory usage are not available
                            # for Google Cloud TPUs through sysfs
                            metrics_dict['temp_tpu'] = None  # Not available
                            metrics_dict['power_draw'] = None  # Not available
                            metrics_dict['memory_used'] = None  # Not available
                            
                            tpu_metrics.append(metrics_dict)
                        except Exception:
                            pass
        except Exception as e:
            self.logger.debug(f"Error reading TPU metrics from sysfs: {e}")
        
        return {'tpu_gpus': tpu_metrics}

    def get_system_metrics(self) -> Dict:
        """Get system-level metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=None), # non-blocking
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            'network_io': psutil.net_io_counters()._asdict(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [],
        }

    def get_gpu_processes(self) -> List[Dict]:
        """Get processes using NVIDIA GPUs."""
        if self.os_type not in ['linux', 'windows']:
            return []
            
        try:
            cmd = ['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
            
            processes = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        processes.append({
                            'pid': self._safe_cast(parts[0], int),
                            'process_name': parts[1],
                            'gpu_uuid': parts[2],
                            'used_memory': self._safe_cast(parts[3], int)
                        })
            return processes
        except (FileNotFoundError, subprocess.CalledProcessError):
            return []
        except Exception as e:
            self.logger.error(f"Error getting GPU processes: {e}")
            return []

    def collect_metrics(self) -> Dict:
        """Collect all metrics in one call."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_metrics(),
        }
        
        gpu_metrics = {}
        gpu_metrics.update(self.get_nvidia_metrics())
        gpu_metrics.update(self.get_amd_metrics())
        gpu_metrics.update(self.get_tpu_metrics())
        # Placeholder for Intel monitoring
        gpu_metrics['intel_gpus'] = []

        metrics['gpus'] = gpu_metrics
        metrics['gpu_processes'] = self.get_gpu_processes()

        # Check for alerts
        alerts = self.alerter.check_alerts(gpu_metrics)
        for alert in alerts:
            self.logger.warning(alert)
        metrics['alerts'] = alerts
        
        return metrics

    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring:
            self.logger.info("Monitoring is already running.")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("GPU monitoring started.")

    def stop_monitoring(self):
        """Stop monitoring."""
        if not self.monitoring:
            self.logger.info("Monitoring is not running.")
            return

        self.monitoring = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        self.logger.info("GPU monitoring stopped.")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.collect_metrics()
                if len(self.data_buffer) >= self.max_buffer_size:
                    self.data_buffer.pop(0) # Keep buffer size fixed
                self.data_buffer.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)

    def get_latest_metrics(self) -> Dict:
        """Get the most recent metrics."""
        if self.data_buffer:
            return self.data_buffer[-1]
        return self.collect_metrics() # Collect fresh data if buffer is empty

    def export_data(self, filename: str = None):
        """Export collected data to JSON file."""
        if not filename:
            filename = f"gpu_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.data_buffer, f, indent=2)
        
        self.logger.info(f"Data exported to {filename}")

    def get_gpu_metrics(self) -> Dict:
        """Dispatches to the correct metrics function based on OS."""
        latest_metrics = {}
        if self.os_type == 'linux':
            latest_metrics = self.get_gpu_metrics_linux()
        elif self.os_type == 'windows':
            latest_metrics = self.get_gpu_metrics_windows()
        elif self.os_type == 'darwin':
            latest_metrics = self.get_gpu_metrics_darwin()

        return latest_metrics

    def get_gpu_metrics_darwin(self) -> Dict:
        """
        Gathers real-time GPU metrics on macOS using asitop for Apple Silicon.
        """
        if platform.machine() != 'arm64':
            if not hasattr(self, '_darwin_intel_warning_logged'):
                self.logger.warning("Real-time GPU monitoring on macOS is only supported on Apple Silicon (arm64).")
                self._darwin_intel_warning_logged = True
            return {}

        try:
            from asitop.api import ASITopAPI
            api = ASITopAPI()
            
            gpu_stats = api.gpu
            
            # Format the data to match the dashboard's expectations
            apple_gpus = [{
                'index': 0,
                'name': 'Apple Silicon GPU',
                'util_gpu': f"{gpu_stats.get('utilization', 0):.1f}", # E.g., 25.5
                'temp_gpu': f"{gpu_stats.get('temp', 0):.1f}", # E.g., 55.0
                'power_draw_w': f"{gpu_stats.get('power', 0):.2f}", # E.g., 5.20
            }]
            return {"apple_gpus": apple_gpus}

        except ImportError:
            if not hasattr(self, '_darwin_asitop_warning_logged'):
                self.logger.warning("`asitop` is not installed. Please run `pip install asitop` for real-time GPU monitoring.")
                self._darwin_asitop_warning_logged = True
            return {}
        except Exception as e:
            if not hasattr(self, '_darwin_error_logged'):
                self.logger.error(f"An error occurred while fetching GPU metrics via asitop: {e}")
                self._darwin_error_logged = True
            return {}

    def get_gpu_metrics_windows(self) -> Dict:
        """
        Gathers basic GPU information on Windows.
        Currently, real-time metrics are not implemented for Windows in this script
        """
        # This is a placeholder. WMI or other libraries would be needed for real-time data.
        return {"nvidia_gpus": [], "amd_gpus": [], "intel_gpus": []}

    def get_gpu_metrics_linux(self) -> Dict:
        """Gathers real-time GPU metrics on Linux."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_metrics(),
            'gpus': self.get_nvidia_metrics(),
            'amd_gpus': self.get_amd_metrics(),
            'tpu_gpus': self.get_tpu_metrics(),
            'gpu_processes': self.get_gpu_processes()
        }

if __name__ == "__main__":
    monitor = GPUMonitor(sampling_interval=2.0)
    monitor.start_monitoring()
    
    try:
        # Run for a short period, then print latest metrics
        print("Monitoring for 10 seconds...")
        time.sleep(10)
        
        print("\nLatest metrics:")
        latest = monitor.get_latest_metrics()
        print(json.dumps(latest, indent=2))
        
    except KeyboardInterrupt:
        print("\nStopping monitoring.")
    finally:
        monitor.stop_monitoring()
        monitor.export_data()
 