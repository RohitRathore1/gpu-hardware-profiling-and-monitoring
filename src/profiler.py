import subprocess
import platform
import json
import re
import logging
import pyopencl as cl
from typing import Dict, List, Optional
from datetime import datetime

class GPUProfiler:
    def __init__(self):
        """Initializes the profiler and sets up logging."""
        self.os_type = platform.system().lower()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_gpus(self) -> List[Dict]:
        """Detects and returns a list of GPUs on the system."""
        if self.os_type == 'windows':
            return self.get_gpus_windows()
        elif self.os_type == 'linux':
            return self.get_gpus_linux()
        elif self.os_type == 'darwin':
            return self.get_gpus_darwin()
        else:
            # Fallback for other OSes or if specific detection fails
            return self.get_opencl_gpus()

    def get_gpus_darwin(self) -> List[Dict]:
        """Get GPU information using system_profiler on macOS."""
        gpus = []
        try:
            cmd = ['system_profiler', 'SPDisplaysDataType', '-json']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            if 'SPDisplaysDataType' in data:
                for i, gpu_info in enumerate(data['SPDisplaysDataType']):
                    vendor = gpu_info.get('spdisplays_vendor', 'Unknown')
                    # Clean up vendor name (e.g., "NVIDIA (0x10de)")
                    vendor_match = re.match(r'(.+?)(?:\s*\(0x.*\))?$', vendor)
                    if vendor_match:
                        vendor = vendor_match.group(1).strip()

                    gpus.append({
                        'index': i,
                        'vendor': vendor,
                        'name': gpu_info.get('sppci_model', 'Unknown GPU'),
                        'driver_version': gpu_info.get('spdisplays_driver-version', 'N/A'),
                        'memory_total_mb': gpu_info.get('spdisplays_vram_shared', 'N/A'),
                    })
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not query macOS system_profiler: {e}")
            print("Falling back to OpenCL detection for macOS.")
            return self.get_opencl_gpus()
            
        # If system_profiler returns nothing, try OpenCL as a backup
        if not gpus:
            gpus = self.get_opencl_gpus()
            
        return gpus

    def get_gpus_linux(self) -> List[Dict]:
        """Get GPU information on Linux by combining results from all vendors."""
        gpus = []
        
        nvidia_gpus = self.get_nvidia_gpus_linux()
        gpus.extend(nvidia_gpus)
        
        amd_gpus = self.get_amd_gpus_linux()
        for i, gpu in enumerate(amd_gpus):
            gpu['index'] = len(nvidia_gpus) + i
        gpus.extend(amd_gpus)
        
        intel_gpus = self.get_intel_gpus_linux()
        for i, gpu in enumerate(intel_gpus):
            gpu['index'] = len(nvidia_gpus) + len(amd_gpus) + i
        gpus.extend(intel_gpus)
        
        # Try to get Google TPUs
        tpu_gpus = self.get_tpu_gpus_linux()
        for i, tpu in enumerate(tpu_gpus):
            tpu['index'] = len(nvidia_gpus) + len(amd_gpus) + len(intel_gpus) + i
        gpus.extend(tpu_gpus)
        
        if not gpus:
            gpus = self.get_opencl_gpus()
            
        return gpus

    def get_gpus_windows(self) -> List[Dict]:
        """Get GPU information using WMI on Windows."""
        import wmi
        gpus = []
        try:
            c = wmi.WMI()
            video_controllers = c.Win32_VideoController()
            for i, gpu in enumerate(video_controllers):
                vendor = "Unknown"
                if "nvidia" in gpu.Name.lower():
                    vendor = "NVIDIA"
                elif "amd" in gpu.Name.lower() or "radeon" in gpu.Name.lower():
                    vendor = "AMD"
                elif "intel" in gpu.Name.lower():
                    vendor = "Intel"
                
                gpus.append({
                    'vendor': vendor,
                    'index': i,
                    'name': gpu.Name,
                    'driver_version': gpu.DriverVersion,
                    'memory_total_mb': int(gpu.AdapterRAM) / (1024**2) if gpu.AdapterRAM else None,
                })
        except ImportError:
            self.logger.warning("WMI library not found. Skipping WMI-based GPU detection.")
            return []
        except Exception as e:
            self.logger.error(f"An error occurred during WMI query: {e}")
            return []
        return gpus

    def get_opencl_gpus(self) -> List[Dict]:
        """Get GPU information using OpenCL as a fallback method."""
        gpus = []
        try:
            platforms = cl.get_platforms()
            gpu_index = 0
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    vendor = device.vendor.strip()
                    name = device.name.strip()
                    
                    try:
                        memory_mb = device.global_mem_size / (1024 * 1024)
                    except:
                        memory_mb = None
                        
                    try:
                        driver_version = device.driver_version
                    except:
                        driver_version = "N/A"
                    
                    gpus.append({
                        'index': gpu_index,
                        'vendor': vendor,
                        'name': name,
                        'driver_version': driver_version,
                        'memory_total_mb': memory_mb,
                        'opencl_device': True  # Mark as OpenCL detected
                    })
                    gpu_index += 1
        except Exception as e:
            self.logger.warning(f"OpenCL detection failed: {e}")
            
        return gpus

    def get_nvidia_gpus_linux(self) -> List[Dict]:
        """Get NVIDIA GPU information using nvidia-smi on Linux."""
        try:
            cmd = [
                'nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    gpus.append({
                        'vendor': 'NVIDIA',
                        'index': int(parts[0]),
                        'name': parts[1],
                        'driver_version': parts[2],
                        'memory_total_mb': int(parts[3]),
                    })
            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # nvidia-smi not found or returned an error
            return []
        except Exception as e:
            print(f"Error getting NVIDIA GPUs on Linux: {e}")
            return []

    def get_amd_gpus_linux(self) -> List[Dict]:
        """Get AMD GPU information using rocm-smi or lspci on Linux."""
        gpus = []
        try:
            # Try rocm-smi first
            result = subprocess.run(['rocm-smi', '--showid', '--showproductname'], 
                                    capture_output=True, text=True, check=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Card' in line and 'AMD' in line:
                    gpus.append({
                        'vendor': 'AMD',
                        'name': line.split(':')[-1].strip(),
                        'driver_version': self._get_amd_driver_version_linux()
                    })
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to lspci
            try:
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if 'VGA' in line and ('AMD' in line or 'ATI' in line):
                        gpus.append({
                            'vendor': 'AMD',
                            'name': line.split(':')[-1].strip(),
                            'driver_version': self._get_amd_driver_version_linux()
                        })
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass # Neither tool is available
        return gpus

    def get_intel_gpus_linux(self) -> List[Dict]:
        """Get Intel GPU information using lspci on Linux."""
        gpus = []
        try:
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if 'VGA' in line and 'Intel' in line:
                    gpus.append({
                        'vendor': 'Intel',
                        'name': line.split(':')[-1].strip(),
                        'driver_version': self._get_intel_driver_version_linux()
                    })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass # lspci not available
        return gpus
    
    def get_tpu_gpus_linux(self) -> List[Dict]:
        """Get Google TPU information on Linux."""
        gpus = []
        
        try:
            import glob
            tpu_devices = glob.glob('/dev/accel*') + glob.glob('/dev/apex_*')
            if tpu_devices:
                for i, device in enumerate(tpu_devices):
                    gpus.append({
                        'vendor': 'Google',
                        'name': f'TPU Device {i}',
                        'device_path': device,
                        'driver_version': self._get_tpu_driver_version_linux()
                    })
        except Exception:
            pass
        
        try:
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):

                if '1ae0' in line or 'Google' in line.lower() and ('tpu' in line.lower() or 'tensor' in line.lower()):
                    gpus.append({
                        'vendor': 'Google',
                        'name': line.split(':')[-1].strip(),
                        'driver_version': self._get_tpu_driver_version_linux()
                    })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            result = subprocess.run(['which', 'lspci'], capture_output=True, text=True)
            if result.returncode == 0:
                import os
                accel_path = '/sys/class/accel'
                if os.path.exists(accel_path):
                    for device in os.listdir(accel_path):
                        device_path = os.path.join(accel_path, device)
                        if os.path.islink(device_path):
                            try:
                                name_file = os.path.join(device_path, 'device', 'name')
                                if os.path.exists(name_file):
                                    with open(name_file, 'r') as f:
                                        name = f.read().strip()
                                else:
                                    name = f'TPU Accelerator {device}'
                                
                                gpus.append({
                                    'vendor': 'Google',
                                    'name': name,
                                    'device': device,
                                    'driver_version': self._get_tpu_driver_version_linux()
                                })
                            except Exception:
                                pass
        except Exception:
            pass
        
        seen = set()
        unique_gpus = []
        for gpu in gpus:
            if gpu['name'] not in seen:
                seen.add(gpu['name'])
                unique_gpus.append(gpu)
        
        return unique_gpus
    
    def _get_amd_driver_version_linux(self) -> Optional[str]:
        """Get AMD driver version on Linux."""
        try:
            result = subprocess.run(['modinfo', 'amdgpu'], capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if 'version:' in line:
                    return line.split(':')[-1].strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return None

    def _get_intel_driver_version_linux(self) -> Optional[str]:
        """Get Intel driver version on Linux."""
        try:
            result = subprocess.run(['modinfo', 'i915'], capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if 'version:' in line:
                    return line.split(':')[-1].strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return None

    def _get_tpu_driver_version_linux(self) -> Optional[str]:
        """Get TPU driver version on Linux."""
        try:
            for module_name in ['tpu', 'apex', 'gasket', 'google_tpu']:
                try:
                    result = subprocess.run(['modinfo', module_name], capture_output=True, text=True, check=True)
                    for line in result.stdout.split('\n'):
                        if 'version:' in line:
                            return line.split(':')[-1].strip()
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        except Exception:
            pass
        return None

    def get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.architecture()[0],
            'hostname': platform.node(),
            'python_version': platform.python_version()
        }

    def get_gpu_libraries(self) -> Dict:
        """Check for GPU computing libraries."""
        libraries = {}
        
        # Check CUDA
        try:
            cmd = 'nvcc' if self.os_type == 'linux' else 'nvcc.exe'
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
                if version_line:
                    libraries['cuda'] = version_line[0].split('release')[-1].split(',')[0].strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            libraries['cuda'] = None
        
        # Check ROCm (Linux-only for now)
        if self.os_type == 'linux':
            try:
                result = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    libraries['rocm'] = result.stdout.strip().split('\n')[0]
            except (FileNotFoundError, subprocess.CalledProcessError):
                libraries['rocm'] = None
        
        # Check OpenCL
        try:
            platforms = cl.get_platforms()
            libraries['opencl'] = [str(p) for p in platforms]
        except (ImportError, Exception):
            libraries['opencl'] = None
        
        return libraries

    def profile_system(self) -> Dict:
        """Gathers a complete system profile."""
        self.logger.info("Starting system profile...")
        
        gpus = self.get_gpus()

        profile = {
            'system_info': self.get_system_info(),
            'gpus': gpus,
            'gpu_count': len(gpus),
            'gpu_libraries': self.get_gpu_libraries(),
            'profiling_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Found {profile['gpu_count']} GPUs.")
        
        return profile

if __name__ == "__main__":
    profiler = GPUProfiler()
    profile_data = profiler.profile_system()
    print(json.dumps(profile_data, indent=2))
 