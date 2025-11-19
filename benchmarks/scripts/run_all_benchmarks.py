#!/usr/bin/env python3
"""
Main orchestrator for running all SAP_LLM benchmarks.

Runs:
1. Test data generation
2. Latency benchmarks
3. Throughput benchmarks
4. Accuracy benchmarks
5. Resource monitoring
6. Report generation
"""

import argparse
import asyncio
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import time


class BenchmarkOrchestrator:
    """Orchestrate all benchmarks."""

    def __init__(self, output_dir: str = "benchmarks/results", quick_mode: bool = False):
        """Initialize orchestrator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.quick_mode = quick_mode
        self.results = {}

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"benchmark_run_{self.timestamp}"

    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("="*70)
        print("  SAP_LLM COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
        print("="*70)
        print(f"\nRun ID: {self.run_id}")
        print(f"Output: {self.output_dir}")
        print(f"Quick mode: {self.quick_mode}")
        print()

        overall_start = time.time()

        # Step 1: Generate test data
        if not self._check_test_data_exists():
            print("\n" + "="*70)
            print("STEP 1: Generating Test Data")
            print("="*70)
            self._generate_test_data()
        else:
            print("\n✓ Test data already exists, skipping generation")

        # Step 2: Latency benchmark
        print("\n" + "="*70)
        print("STEP 2: Running Latency Benchmarks")
        print("="*70)
        self.results["latency"] = self._run_latency_benchmark()

        # Step 3: Throughput benchmark
        print("\n" + "="*70)
        print("STEP 3: Running Throughput Benchmarks")
        print("="*70)
        self.results["throughput"] = self._run_throughput_benchmark()

        # Step 4: Accuracy benchmark
        print("\n" + "="*70)
        print("STEP 4: Running Accuracy Benchmarks")
        print("="*70)
        self.results["accuracy"] = self._run_accuracy_benchmark()

        # Step 5: Resource monitoring summary
        print("\n" + "="*70)
        print("STEP 5: Resource Usage Summary")
        print("="*70)
        self.results["resources"] = self._collect_resource_stats()

        overall_elapsed = time.time() - overall_start

        # Save combined results
        self._save_combined_results(overall_elapsed)

        # Print final summary
        self._print_final_summary()

        return self.results

    def _check_test_data_exists(self) -> bool:
        """Check if test data already exists."""
        test_data_dir = Path("benchmarks/data/sample_documents")
        if not test_data_dir.exists():
            return False

        # Check if we have enough documents
        docs = list(test_data_dir.glob("*.png")) + list(test_data_dir.glob("*.jpg"))
        return len(docs) >= 100  # Minimum threshold

    def _generate_test_data(self) -> None:
        """Generate test data."""
        num_docs = 100 if self.quick_mode else 1000

        cmd = [
            sys.executable,
            "benchmarks/scripts/generate_test_data.py",
            "--num-docs", str(num_docs),
            "--output", "benchmarks/data",
        ]

        self._run_command(cmd, "Test data generation")

    def _run_latency_benchmark(self) -> Dict[str, Any]:
        """Run latency benchmarks."""
        num_iterations = 50 if self.quick_mode else 100

        output_file = self.output_dir / f"latency_{self.timestamp}.json"

        cmd = [
            sys.executable,
            "benchmarks/scripts/run_latency_benchmark.py",
            "--test-data", "benchmarks/data/sample_documents",
            "--num-iterations", str(num_iterations),
            "--output", str(output_file),
            "--simulation",  # Use simulation for now
        ]

        self._run_command(cmd, "Latency benchmark")

        return self._load_results(output_file)

    def _run_throughput_benchmark(self) -> Dict[str, Any]:
        """Run throughput benchmarks."""
        duration = 60 if self.quick_mode else 300

        output_file = self.output_dir / f"throughput_{self.timestamp}.json"

        cmd = [
            sys.executable,
            "benchmarks/scripts/run_throughput_benchmark.py",
            "--test-data", "benchmarks/data/sample_documents",
            "--duration", str(duration),
            "--workers", "1", "2", "4",
            "--output", str(output_file),
            "--simulation",
        ]

        if self.quick_mode:
            cmd.append("--quick")

        self._run_command(cmd, "Throughput benchmark")

        return self._load_results(output_file)

    def _run_accuracy_benchmark(self) -> Dict[str, Any]:
        """Run accuracy benchmarks."""
        output_file = self.output_dir / f"accuracy_{self.timestamp}.json"

        cmd = [
            sys.executable,
            "benchmarks/scripts/run_accuracy_benchmark.py",
            "--test-dataset", "benchmarks/data/ground_truth",
            "--output", str(output_file),
            "--simulation",
        ]

        self._run_command(cmd, "Accuracy benchmark")

        return self._load_results(output_file)

    def _collect_resource_stats(self) -> Dict[str, Any]:
        """Collect resource usage statistics."""
        import psutil

        stats = {
            "cpu": {
                "count": psutil.cpu_count(),
                "percent": psutil.cpu_percent(interval=1),
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "percent": psutil.disk_usage('/').percent,
            },
        }

        # GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu"] = {
                    "count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(0),
                    "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                }
        except:
            stats["gpu"] = {"available": False}

        print(f"\nSystem Resources:")
        print(f"  CPU: {stats['cpu']['count']} cores ({stats['cpu']['percent']}% used)")
        print(f"  Memory: {stats['memory']['available_gb']:.1f}/{stats['memory']['total_gb']:.1f} GB available")
        if stats['gpu'].get('available') != False:
            print(f"  GPU: {stats['gpu'].get('name', 'N/A')}")

        return stats

    def _run_command(self, cmd: list, description: str) -> None:
        """Run a command and handle errors."""
        print(f"\nRunning: {description}")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n⚠️  {description} failed with error code {e.returncode}")
            print(f"   Continuing with remaining benchmarks...")
        except FileNotFoundError:
            print(f"\n⚠️  Script not found: {cmd[1]}")
            print(f"   Skipping {description}")

    def _load_results(self, file_path: Path) -> Dict[str, Any]:
        """Load results from JSON file."""
        if file_path.exists():
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            return {"error": "Results file not found"}

    def _save_combined_results(self, elapsed_time: float) -> None:
        """Save all results to a combined file."""
        combined = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_time,
            "quick_mode": self.quick_mode,
            "results": self.results,
        }

        output_file = self.output_dir / f"combined_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"\n💾 Combined results saved to: {output_file}")

    def _print_final_summary(self) -> None:
        """Print final summary of all benchmarks."""
        print("\n\n" + "="*70)
        print("  FINAL BENCHMARK SUMMARY")
        print("="*70)

        # Latency
        latency = self.results.get("latency", {}).get("e2e_latency", {})
        if latency:
            p95 = latency.get("p95", 0)
            target_met = latency.get("target_met", False)
            status = "✅ PASS" if target_met else "❌ FAIL"
            print(f"\nLatency (P95): {p95:.1f}ms (target: <600ms) {status}")

        # Throughput
        throughput = self.results.get("throughput", {}).get("sustained", {})
        if throughput:
            avg_tput = throughput.get("avg_throughput_per_min", 0)
            target_met = throughput.get("target_met", False)
            status = "✅ PASS" if target_met else "❌ FAIL"
            print(f"Throughput: {avg_tput:,.0f} docs/min (target: ≥100k) {status}")

        # Accuracy
        accuracy = self.results.get("accuracy", {})
        if accuracy:
            classification_acc = accuracy.get("classification", {}).get("accuracy", 0)
            extraction_f1 = accuracy.get("extraction", {}).get("avg_f1", 0)
            routing_acc = accuracy.get("routing", {}).get("accuracy", 0)

            all_met = all([
                accuracy.get("classification", {}).get("target_met", False),
                accuracy.get("extraction", {}).get("target_met", False),
                accuracy.get("routing", {}).get("target_met", False),
            ])

            status = "✅ PASS" if all_met else "❌ FAIL"

            print(f"\nAccuracy Metrics: {status}")
            print(f"  - Classification: {classification_acc*100:.2f}% (target: ≥99%)")
            print(f"  - Extraction F1: {extraction_f1*100:.2f}% (target: ≥97%)")
            print(f"  - Routing: {routing_acc*100:.2f}% (target: ≥99.5%)")

        # Overall
        all_targets = [
            latency.get("target_met", False) if latency else False,
            throughput.get("target_met", False) if throughput else False,
            all([
                accuracy.get("classification", {}).get("target_met", False),
                accuracy.get("extraction", {}).get("target_met", False),
                accuracy.get("routing", {}).get("target_met", False),
            ]) if accuracy else False,
        ]

        print("\n" + "="*70)
        if all(all_targets):
            print("✅ ALL PERFORMANCE TARGETS MET")
        else:
            print("❌ SOME PERFORMANCE TARGETS NOT MET")

            print("\nTargets not met:")
            if not (latency.get("target_met", False) if latency else False):
                print("  - Latency P95 <600ms")
            if not (throughput.get("target_met", False) if throughput else False):
                print("  - Throughput ≥100k docs/min")
            if accuracy:
                if not accuracy.get("classification", {}).get("target_met", False):
                    print("  - Classification accuracy ≥99%")
                if not accuracy.get("extraction", {}).get("target_met", False):
                    print("  - Extraction F1 ≥97%")
                if not accuracy.get("routing", {}).get("target_met", False):
                    print("  - Routing accuracy ≥99.5%")

        print("="*70)

        print(f"\n📊 Detailed results: {self.output_dir}")
        print(f"   Run ID: {self.run_id}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run all SAP_LLM benchmarks")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode (reduced iterations)")

    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator(
        output_dir=args.output_dir,
        quick_mode=args.quick,
    )

    results = orchestrator.run_all()

    # Exit with error code if targets not met
    all_targets_met = all([
        results.get("latency", {}).get("e2e_latency", {}).get("target_met", False),
        results.get("throughput", {}).get("sustained", {}).get("target_met", False),
        all([
            results.get("accuracy", {}).get("classification", {}).get("target_met", False),
            results.get("accuracy", {}).get("extraction", {}).get("target_met", False),
            results.get("accuracy", {}).get("routing", {}).get("target_met", False),
        ]),
    ])

    sys.exit(0 if all_targets_met else 1)


if __name__ == "__main__":
    main()
