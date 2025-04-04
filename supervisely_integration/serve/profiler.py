import logging
import os
import time
import psutil
import torch
import json
from typing import Dict, List, Tuple, Union
from datetime import datetime

from supervisely import logger
from supervisely import Api


class MemoryProfiler:
    """
    A profiler class to monitor memory usage of the main process, child processes,
    and GPU memory for PyTorch applications.
    """
    
    def __init__(self):
        """
        Initialize the memory profiler.
        """
        self.enabled = logger.getEffectiveLevel() <= logging.DEBUG
        if not self.enabled:
            return
        self.main_pid = os.getpid()
        self.main_process = psutil.Process(self.main_pid)

        self.last_cpu_memory: Dict[int, int] = {}  # pid -> memory in bytes
        self.last_gpu_memory: Dict[int, Tuple[int, int]] = {}  # device -> (allocated, reserved) in bytes
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.checkpoint_count = 0

        self.measurements = []
        self.checkpoint_names = []

        self.cuda_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0

        self.measure("Initial measurement")

    def _get_process_tree(self) -> List[psutil.Process]:
        """Get the main process and all its children recursively."""
        try:
            children = self.main_process.children(recursive=True)
            return [self.main_process] + children
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    def _format_memory(self, bytes_value: Union[int, float]) -> str:
        """Format memory size from bytes to a human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(bytes_value) < 1024.0 or unit == 'TB':
                if unit == 'B':
                    return f"{bytes_value:.0f} {unit}"
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0

    def _get_cpu_memory(self) -> Dict[int, Dict[str, Union[int, float, str]]]:
        """Get CPU memory usage for all processes in the tree."""
        result = {}
        for process in self._get_process_tree():
            try:
                process_info = {
                    'pid': process.pid,
                    'name': process.name(),
                    'memory_bytes': process.memory_info().rss,
                    'memory_formatted': self._format_memory(process.memory_info().rss),
                    'parent_pid': process.ppid() if process.pid != self.main_pid else None,
                }

                if process.pid in self.last_cpu_memory:
                    diff = process_info['memory_bytes'] - self.last_cpu_memory[process.pid]
                else:
                    diff = process_info['memory_bytes']
                process_info['diff_bytes'] = diff
                process_info['diff_formatted'] = ('+' if diff >= 0 else '') + self._format_memory(diff)


                self.last_cpu_memory[process.pid] = process_info['memory_bytes']

                result[process.pid] = process_info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        total_memory = sum(info['memory_bytes'] for info in result.values())
        result["total"] = {
            'pid': None,
            'name': None,
            'memory_bytes': total_memory,
            'memory_formatted': self._format_memory(total_memory),
            'parent_pid': None,
        }

        return result

    def _get_gpu_memory(self) -> Dict[int, Dict[str, Union[int, float, str]]]:
        """Get GPU memory usage for all available GPUs."""
        result = {}

        if not self.cuda_available:
            return result

        for device in range(self.gpu_count):
            try:
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)

                gpu_info = {
                    'device': device,
                    'name': torch.cuda.get_device_name(device),
                    'allocated_bytes': allocated,
                    'allocated_formatted': self._format_memory(allocated),
                    'reserved_bytes': reserved,
                    'reserved_formatted': self._format_memory(reserved),
                    'utilization': torch.cuda.utilization(device) if hasattr(torch.cuda, 'utilization') else None
                }

                if device in self.last_gpu_memory:
                    last_allocated, last_reserved = self.last_gpu_memory[device]
                    allocated_diff = allocated - last_allocated
                    reserved_diff = reserved - last_reserved
                    
                    gpu_info['allocated_diff_bytes'] = allocated_diff
                    gpu_info['allocated_diff_formatted'] = ('+' if allocated_diff >= 0 else '') + self._format_memory(allocated_diff)
                    gpu_info['reserved_diff_bytes'] = reserved_diff
                    gpu_info['reserved_diff_formatted'] = ('+' if reserved_diff >= 0 else '') + self._format_memory(reserved_diff)

                self.last_gpu_memory[device] = (allocated, reserved)
                result[device] = gpu_info
            except Exception as e:
                logger.debug(f"Error getting GPU {device} memory: {e}")

        return result
    
    def profile(self, func):
        """
        Decorator to profile a function and measure memory usage.
        
        Args:
            func: Function to be profiled
        
        Returns:
            Wrapped function with memory profiling
        """
        def wrapper(*args, **kwargs):
            self.measure(f"Before {func.__qualname__} call")
            result = func(*args, **kwargs)
            self.measure(f"After {func.__qualname__} call")
            return result
        return wrapper

    def measure(self, checkpoint_name: str = None) -> Dict:
        """
        Measure and report memory usage at the current checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint for identification
        
        Returns:
            Dictionary with all memory measurements
        """
        if not self.enabled:
            return
        self.checkpoint_count += 1
        if checkpoint_name is None:
            checkpoint_name = f"Checkpoint {self.checkpoint_count}"

        current_time = time.time()
        elapsed_since_start = current_time - self.start_time
        elapsed_since_last = current_time - self.last_checkpoint_time
        self.last_checkpoint_time = current_time

        cpu_memory = self._get_cpu_memory()
        gpu_memory = self._get_gpu_memory()

        total_cpu_diff = 0
        total_gpu_allocated_diff = 0
        total_gpu_reserved_diff = 0

        for _, info in cpu_memory.items():
            if 'diff_bytes' in info:
                total_cpu_diff += info['diff_bytes']

        for _, info in gpu_memory.items():
            if 'allocated_diff_bytes' in info:
                total_gpu_allocated_diff += info['allocated_diff_bytes']
            if 'reserved_diff_bytes' in info:
                total_gpu_reserved_diff += info['reserved_diff_bytes']

        if checkpoint_name in self.checkpoint_names:
            i = 1
            while f"{checkpoint_name} #{i}" in self.checkpoint_names:
                i += 1
            checkpoint_name = f"{checkpoint_name} #{i}"

        result = {
            'checkpoint': checkpoint_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_total': elapsed_since_start,
            'elapsed_since_last': elapsed_since_last,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'memory_diff': {
                'total_cpu_diff_bytes': total_cpu_diff,
                'total_cpu_diff_formatted': ('+' if total_cpu_diff >= 0 else '') + self._format_memory(total_cpu_diff),
                'total_gpu_allocated_diff_bytes': total_gpu_allocated_diff,
                'total_gpu_allocated_diff_formatted': ('+' if total_gpu_allocated_diff >= 0 else '') + self._format_memory(total_gpu_allocated_diff),
                'total_gpu_reserved_diff_bytes': total_gpu_reserved_diff,
                'total_gpu_reserved_diff_formatted': ('+' if total_gpu_reserved_diff >= 0 else '') + self._format_memory(total_gpu_reserved_diff)
            }
        }
        self._print_report(result)
        self.measurements.append(result)
        self.checkpoint_names.append(checkpoint_name)

        return result

    def _print_report(self, result: Dict) -> None:
        """Print a formatted memory usage report."""

        profile_str = f"MEMORY PROFILE: {result['checkpoint']}:\n"
        if 'memory_diff' in result and self.checkpoint_count > 1:
            diff = result['memory_diff']
            profile_str += "MEMORY DIFF:\n"
            profile_str += f"  CPU Total:     {diff['total_cpu_diff_formatted']}\n"
            profile_str += f"  GPU Allocated: {diff['total_gpu_allocated_diff_formatted']}\n"
            profile_str += f"  GPU Reserved:  {diff['total_gpu_reserved_diff_formatted']}\n"
            profile_str += "\n"

        profile_str += "CPU MEMORY USAGE:"
        for pid, info in result['cpu_memory'].items():
            diff_str = f" ({info['diff_formatted']})" if 'diff_formatted' in info else ""
            parent_str = f" (child of {info['parent_pid']})" if info['parent_pid'] else " (main)"
            profile_str += f"  Process {pid}{parent_str}: {info['name']} - {info['memory_formatted']}{diff_str}"
        profile_str += "\n"

        if result['gpu_memory']:
            profile_str += "GPU MEMORY USAGE:"
            for device, info in result['gpu_memory'].items():
                allocated_diff = f" ({info['allocated_diff_formatted']})" if 'allocated_diff_formatted' in info else ""
                reserved_diff = f" ({info['reserved_diff_formatted']})" if 'reserved_diff_formatted' in info else ""

                profile_str += f"  GPU {device} ({info['name']}):"
                profile_str += f"    Allocated: {info['allocated_formatted']}{allocated_diff}"
                profile_str += f"    Reserved:  {info['reserved_formatted']}{reserved_diff}"
                if info['utilization'] is not None:
                    profile_str += f"    Utilization: {info['utilization']}%"

        logger.debug(profile_str)

    def save(self, filename: str = "memory_profile_results.json"):
        """
        Save all collected measurements to a JSON file.
        
        Args:
            filename: Path to the output JSON file
        """
        if not self.enabled:
            return
        with open(filename, 'w') as f:
            json.dump(self.measurements, f, indent=2)

        logger.debug("Memory profile measurements saved to %s", filename)


    def upload(self, api: Api, team_id: int, dst_dir):
        if not self.enabled:
            return
        try:
            tmp_path = "/tmp/memory_profile_results.json"
            self.save(tmp_path)
            dst = os.path.join(dst_dir, os.path.basename(tmp_path))
            api.file.upload(team_id, src=tmp_path, dst=dst)
            logger.debug("Memory profile measurements uploaded to %s", dst)
            os.remove(tmp_path)
        except Exception as e:
            logger.debug(f"Failed to upload memory profile measurements: {e}", exc_info=True)
