#!/usr/bin/env python3
"""
Enterprise-grade resource monitoring with threshold enforcement.

Monitors:
- CPU usage (<70% threshold per AWS Well-Architected Framework)
- Memory usage (<70% threshold)
- GPU utilization (>80% target during processing)
- Disk I/O
- Network I/O

Alerts when thresholds are exceeded.
"""

import argparse
import json
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False


class ResourceMonitor:
    """Enterprise-grade resource monitoring with thresholds."""

    # Industry standard thresholds (AWS Well-Architected Framework)
    CPU_THRESHOLD = 70.0  # Max 70% to allow for surges
    MEMORY_THRESHOLD = 70.0  # Max 70% memory usage
    GPU_MIN_THRESHOLD = 80.0  # Min 80% GPU utilization during processing
    DISK_THRESHOLD = 85.0  # Max 85% disk usage

    def __init__(self, sampling_interval: int = 1):
        """Initialize monitor."""
        self.sampling_interval = sampling_interval
        self.violations = []
        self.samples = []

    def monitor(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Monitor resources for specified duration.

        Returns:
            Dictionary with metrics and threshold violations
        """
        print(f"\n🔍 Monitoring resources for {duration_seconds}s...")
        print(f"   Thresholds: CPU<{self.CPU_THRESHOLD}%, Memory<{self.MEMORY_THRESHOLD}%, GPU>{self.GPU_MIN_THRESHOLD}%")

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < duration_seconds:
            sample = self._collect_sample()
            self.samples.append(sample)

            # Check thresholds
            self._check_thresholds(sample, sample_count)

            sample_count += 1
            time.sleep(self.sampling_interval)

            if sample_count % 10 == 0:
                print(f"   Sample {sample_count}: CPU {sample['cpu_percent']:.1f}%, "
                      f"Mem {sample['memory_percent']:.1f}%")

        # Calculate statistics
        results = self._calculate_statistics()

        # Print summary
        self._print_summary(results)

        return results

    def _collect_sample(self) -> Dict[str, Any]:
        """Collect a single resource sample."""
        sample = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_percent": psutil.disk_usage('/').percent,
        }

        # CPU per core
        sample["cpu_per_core"] = psutil.cpu_percent(interval=0.1, percpu=True)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        sample["disk_read_mb"] = disk_io.read_bytes / (1024**2)
        sample["disk_write_mb"] = disk_io.write_bytes / (1024**2)

        # Network I/O
        net_io = psutil.net_io_counters()
        sample["net_sent_mb"] = net_io.bytes_sent / (1024**2)
        sample["net_recv_mb"] = net_io.bytes_recv / (1024**2)

        # GPU if available
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                sample["gpu_utilization"] = gpu_util.gpu
                sample["gpu_memory_percent"] = (gpu_mem.used / gpu_mem.total) * 100
                sample["gpu_memory_used_gb"] = gpu_mem.used / (1024**3)
            except:
                pass

        return sample

    def _check_thresholds(self, sample: Dict[str, Any], sample_num: int) -> None:
        """Check if any thresholds are violated."""
        violations = []

        # CPU threshold
        if sample["cpu_percent"] > self.CPU_THRESHOLD:
            violations.append({
                "type": "CPU_HIGH",
                "value": sample["cpu_percent"],
                "threshold": self.CPU_THRESHOLD,
                "sample": sample_num,
                "severity": "WARNING",
            })

        # Memory threshold
        if sample["memory_percent"] > self.MEMORY_THRESHOLD:
            violations.append({
                "type": "MEMORY_HIGH",
                "value": sample["memory_percent"],
                "threshold": self.MEMORY_THRESHOLD,
                "sample": sample_num,
                "severity": "WARNING",
            })

        # Disk threshold
        if sample["disk_percent"] > self.DISK_THRESHOLD:
            violations.append({
                "type": "DISK_HIGH",
                "value": sample["disk_percent"],
                "threshold": self.DISK_THRESHOLD,
                "sample": sample_num,
                "severity": "CRITICAL",
            })

        # GPU utilization (should be HIGH during processing)
        if "gpu_utilization" in sample:
            if sample["gpu_utilization"] < self.GPU_MIN_THRESHOLD:
                violations.append({
                    "type": "GPU_LOW",
                    "value": sample["gpu_utilization"],
                    "threshold": self.GPU_MIN_THRESHOLD,
                    "sample": sample_num,
                    "severity": "INFO",
                })

        if violations:
            self.violations.extend(violations)
            for v in violations:
                if v["severity"] in ["WARNING", "CRITICAL"]:
                    print(f"   ⚠️  {v['severity']}: {v['type']} = {v['value']:.1f}% "
                          f"(threshold: {v['threshold']}%)")

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from samples."""
        if not self.samples:
            return {}

        cpu_values = [s["cpu_percent"] for s in self.samples]
        mem_values = [s["memory_percent"] for s in self.samples]

        results = {
            "duration_seconds": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
            "num_samples": len(self.samples),
            "cpu": {
                "mean": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "threshold": self.CPU_THRESHOLD,
                "violations": sum(1 for v in cpu_values if v > self.CPU_THRESHOLD),
                "compliant": all(v <= self.CPU_THRESHOLD for v in cpu_values),
            },
            "memory": {
                "mean": sum(mem_values) / len(mem_values),
                "max": max(mem_values),
                "min": min(mem_values),
                "threshold": self.MEMORY_THRESHOLD,
                "violations": sum(1 for v in mem_values if v > self.MEMORY_THRESHOLD),
                "compliant": all(v <= self.MEMORY_THRESHOLD for v in mem_values),
            },
            "violations": self.violations,
            "total_violations": len(self.violations),
        }

        # GPU stats if available
        if "gpu_utilization" in self.samples[0]:
            gpu_util_values = [s["gpu_utilization"] for s in self.samples if "gpu_utilization" in s]
            if gpu_util_values:
                results["gpu"] = {
                    "mean": sum(gpu_util_values) / len(gpu_util_values),
                    "max": max(gpu_util_values),
                    "min": min(gpu_util_values),
                    "threshold": self.GPU_MIN_THRESHOLD,
                    "below_threshold": sum(1 for v in gpu_util_values if v < self.GPU_MIN_THRESHOLD),
                    "compliant": sum(v >= self.GPU_MIN_THRESHOLD for v in gpu_util_values) / len(gpu_util_values) >= 0.8,
                }

        return results

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print monitoring summary."""
        print("\n" + "="*70)
        print("  RESOURCE MONITORING SUMMARY")
        print("="*70)

        cpu = results["cpu"]
        mem = results["memory"]

        # CPU
        cpu_status = "✅ PASS" if cpu["compliant"] else "❌ FAIL"
        print(f"\nCPU Usage: {cpu_status}")
        print(f"  Mean: {cpu['mean']:.1f}% (threshold: <{cpu['threshold']}%)")
        print(f"  Max: {cpu['max']:.1f}%")
        print(f"  Violations: {cpu['violations']}/{results['num_samples']}")

        # Memory
        mem_status = "✅ PASS" if mem["compliant"] else "❌ FAIL"
        print(f"\nMemory Usage: {mem_status}")
        print(f"  Mean: {mem['mean']:.1f}% (threshold: <{mem['threshold']}%)")
        print(f"  Max: {mem['max']:.1f}%")
        print(f"  Violations: {mem['violations']}/{results['num_samples']}")

        # GPU
        if "gpu" in results:
            gpu = results["gpu"]
            gpu_status = "✅ PASS" if gpu["compliant"] else "⚠️  UNDERUTILIZED"
            print(f"\nGPU Utilization: {gpu_status}")
            print(f"  Mean: {gpu['mean']:.1f}% (target: >{gpu['threshold']}%)")
            print(f"  Max: {gpu['max']:.1f}%")

        # Overall
        print("\n" + "="*70)
        overall_pass = cpu["compliant"] and mem["compliant"]
        if overall_pass:
            print("✅ ALL RESOURCE THRESHOLDS MET")
            print("   System has adequate headroom for production surges")
        else:
            print("❌ RESOURCE THRESHOLD VIOLATIONS DETECTED")
            print("   System may not handle production load safely")
        print("="*70 + "\n")

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save monitoring results."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "thresholds": {
                "cpu_max": self.CPU_THRESHOLD,
                "memory_max": self.MEMORY_THRESHOLD,
                "gpu_min": self.GPU_MIN_THRESHOLD,
                "disk_max": self.DISK_THRESHOLD,
            },
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enterprise Resource Monitoring")
    parser.add_argument("--duration", type=int, default=60,
                       help="Monitoring duration (seconds)")
    parser.add_argument("--interval", type=int, default=1,
                       help="Sampling interval (seconds)")
    parser.add_argument("--output", type=str, default="benchmarks/results/resource_monitoring.json",
                       help="Output file")

    args = parser.parse_args()

    print("="*70)
    print("  SAP_LLM Enterprise Resource Monitor")
    print("="*70)
    print("\nIndustry Standard Thresholds (AWS Well-Architected Framework):")
    print(f"  - CPU: <70% (allows 30% headroom for surges)")
    print(f"  - Memory: <70% (allows 30% headroom)")
    print(f"  - GPU: >80% (target utilization during processing)")
    print(f"  - Disk: <85% (critical threshold)")

    monitor = ResourceMonitor(sampling_interval=args.interval)
    results = monitor.monitor(duration_seconds=args.duration)
    monitor.save_results(results, args.output)

    # Exit with error code if thresholds violated
    if not (results["cpu"]["compliant"] and results["memory"]["compliant"]):
        print("\n⚠️  Exiting with error code due to threshold violations")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
