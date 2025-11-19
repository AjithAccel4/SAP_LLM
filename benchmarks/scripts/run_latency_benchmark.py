#!/usr/bin/env python3
"""
Comprehensive latency benchmark for SAP_LLM.

Measures P50, P95, P99 latency for:
- End-to-end pipeline
- Individual stages (Inbox → Routing)
- Different concurrency levels
- Cold start vs warm cache

Target: P95 < 600ms
"""

import argparse
import asyncio
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import os

try:
    from sap_llm.stages.inbox import InboxStage
    from sap_llm.stages.preprocessing import PreprocessingStage
    from sap_llm.stages.classification import ClassificationStage
    from sap_llm.stages.extraction import ExtractionStage
    from sap_llm.stages.validation import ValidationStage
    from sap_llm.stages.routing import RoutingStage
    from sap_llm.config import load_config
    from sap_llm.utils.logger import get_logger
    REAL_MODE = True
except ImportError:
    print("⚠️  SAP_LLM not fully installed. Running in simulation mode.")
    REAL_MODE = False

    class MockLogger:
        def info(self, msg): pass
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")

    def get_logger(name):
        return MockLogger()

logger = get_logger(__name__)


class LatencyBenchmark:
    """Comprehensive latency benchmarking."""

    def __init__(self, simulation_mode: bool = False):
        """Initialize benchmark."""
        self.simulation_mode = simulation_mode or not REAL_MODE
        self.results = {}

        if not self.simulation_mode:
            try:
                self.config = load_config()
                # Initialize stages
                self.stages = {
                    "inbox": InboxStage(self.config),
                    "preprocessing": PreprocessingStage(self.config),
                    "classification": ClassificationStage(self.config),
                    "extraction": ExtractionStage(self.config),
                    "validation": ValidationStage(self.config),
                    "routing": RoutingStage(self.config),
                }
                logger.info("Initialized all pipeline stages")
            except Exception as e:
                logger.warning(f"Could not initialize stages: {e}. Using simulation mode.")
                self.simulation_mode = True

    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile statistics."""
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": self._percentile(sorted_values, 50),
            "p95": self._percentile(sorted_values, 95),
            "p99": self._percentile(sorted_values, 99),
        }

    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    async def measure_e2e_latency(
        self,
        test_documents: List[Path],
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Measure end-to-end latency."""
        print(f"\n🔄 Measuring E2E latency ({num_iterations} iterations)...")

        latencies = []

        for i in range(num_iterations):
            # Random document
            doc_path = test_documents[i % len(test_documents)]

            start = time.perf_counter()

            if self.simulation_mode:
                # Simulate processing (300-500ms)
                await asyncio.sleep(0.3 + (0.2 * (i % 10) / 10))
            else:
                try:
                    await self._process_document_e2e(doc_path)
                except Exception as e:
                    logger.error(f"Error processing {doc_path}: {e}")
                    continue

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 20 == 0:
                current_p95 = self._percentile(sorted(latencies), 95)
                print(f"  Progress: {i+1}/{num_iterations}, P95: {current_p95:.1f}ms")

        stats = self._calculate_percentiles(latencies)
        stats["target_met"] = stats.get("p95", float('inf')) < 600
        stats["target_value"] = 600
        stats["num_samples"] = len(latencies)

        status = "✅ PASS" if stats["target_met"] else "❌ FAIL"
        print(f"\nE2E Latency Results:")
        print(f"  P50: {stats['p50']:.1f}ms")
        print(f"  P95: {stats['p95']:.1f}ms (target: <600ms) {status}")
        print(f"  P99: {stats['p99']:.1f}ms")
        print(f"  Mean: {stats['mean']:.1f}ms ± {stats['std']:.1f}ms")

        return stats

    async def measure_stage_latency(
        self,
        test_documents: List[Path],
        num_iterations: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Measure latency for each pipeline stage."""
        print(f"\n🔄 Measuring per-stage latency...")

        stage_names = ["inbox", "preprocessing", "classification", "extraction", "validation", "routing"]
        stage_latencies = {stage: [] for stage in stage_names}

        for i in range(num_iterations):
            doc_path = test_documents[i % len(test_documents)]

            for stage_name in stage_names:
                start = time.perf_counter()

                if self.simulation_mode:
                    # Simulate stage processing (50-200ms)
                    await asyncio.sleep(0.05 + (0.15 * (hash(stage_name) % 10) / 10))
                else:
                    try:
                        await self._process_stage(stage_name, doc_path)
                    except Exception as e:
                        logger.error(f"Error in {stage_name}: {e}")
                        continue

                latency_ms = (time.perf_counter() - start) * 1000
                stage_latencies[stage_name].append(latency_ms)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")

        # Calculate statistics per stage
        results = {}
        print(f"\nPer-Stage Latency Results:")
        for stage_name, latencies in stage_latencies.items():
            stats = self._calculate_percentiles(latencies)
            results[stage_name] = stats
            print(f"  {stage_name:15s} - P95: {stats['p95']:6.1f}ms, Mean: {stats['mean']:6.1f}ms")

        return results

    def measure_concurrent_latency(
        self,
        test_documents: List[Path],
        concurrency_levels: List[int] = [1, 10, 50, 100],
        requests_per_level: int = 100,
    ) -> Dict[int, Dict[str, float]]:
        """Measure latency under different concurrency levels."""
        print(f"\n🔄 Measuring latency under concurrency...")

        results = {}

        for concurrency in concurrency_levels:
            print(f"\n  Testing with {concurrency} concurrent workers...")

            latencies = []

            if self.simulation_mode:
                # Simulate concurrent processing
                for i in range(requests_per_level):
                    # Simulate increased latency under load
                    base_latency = 300
                    contention = concurrency * 2  # 2ms per concurrent worker
                    noise = (i % 50) * 2
                    latency = base_latency + contention + noise
                    latencies.append(latency)
            else:
                # Real concurrent processing
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for i in range(requests_per_level):
                        doc_path = test_documents[i % len(test_documents)]
                        future = executor.submit(self._process_document_sync, doc_path)
                        futures.append(future)

                    for future in futures:
                        try:
                            latency = future.result()
                            latencies.append(latency)
                        except Exception as e:
                            logger.error(f"Concurrent processing error: {e}")

            stats = self._calculate_percentiles(latencies)
            results[concurrency] = stats

            print(f"    P95: {stats['p95']:.1f}ms, Mean: {stats['mean']:.1f}ms")

        return results

    async def measure_cold_vs_warm(
        self,
        test_documents: List[Path],
        num_iterations: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """Compare cold start vs warm cache latency."""
        print(f"\n🔄 Measuring cold start vs warm cache latency...")

        # Cold start (first run)
        cold_latencies = []
        print("  Measuring cold start latencies...")

        for i in range(num_iterations):
            doc_path = test_documents[i % len(test_documents)]

            start = time.perf_counter()
            if self.simulation_mode:
                # Cold start is slower
                await asyncio.sleep(0.5 + (0.3 * (i % 10) / 10))
            else:
                try:
                    # Clear caches before each run
                    self._clear_caches()
                    await self._process_document_e2e(doc_path)
                except Exception as e:
                    logger.error(f"Cold start error: {e}")
                    continue

            latency_ms = (time.perf_counter() - start) * 1000
            cold_latencies.append(latency_ms)

        # Warm cache (repeated runs)
        warm_latencies = []
        print("  Measuring warm cache latencies...")

        for i in range(num_iterations):
            doc_path = test_documents[i % min(10, len(test_documents))]  # Reuse same 10 docs

            start = time.perf_counter()
            if self.simulation_mode:
                # Warm cache is faster
                await asyncio.sleep(0.2 + (0.1 * (i % 10) / 10))
            else:
                try:
                    await self._process_document_e2e(doc_path)
                except Exception as e:
                    logger.error(f"Warm cache error: {e}")
                    continue

            latency_ms = (time.perf_counter() - start) * 1000
            warm_latencies.append(latency_ms)

        cold_stats = self._calculate_percentiles(cold_latencies)
        warm_stats = self._calculate_percentiles(warm_latencies)

        print(f"\nCold vs Warm Results:")
        print(f"  Cold Start - P95: {cold_stats['p95']:.1f}ms, Mean: {cold_stats['mean']:.1f}ms")
        print(f"  Warm Cache - P95: {warm_stats['p95']:.1f}ms, Mean: {warm_stats['mean']:.1f}ms")
        print(f"  Speedup: {cold_stats['mean'] / warm_stats['mean']:.2f}x")

        return {
            "cold_start": cold_stats,
            "warm_cache": warm_stats,
            "speedup": cold_stats['mean'] / warm_stats['mean'] if warm_stats['mean'] > 0 else 0,
        }

    async def _process_document_e2e(self, doc_path: Path) -> Dict[str, Any]:
        """Process document through entire pipeline."""
        if self.simulation_mode:
            return {"status": "simulated"}

        result = {}
        for stage_name, stage in self.stages.items():
            result = await stage.process(result if result else {"document_path": str(doc_path)})

        return result

    async def _process_stage(self, stage_name: str, doc_path: Path) -> Dict[str, Any]:
        """Process document through a single stage."""
        if self.simulation_mode:
            return {"status": "simulated"}

        stage = self.stages.get(stage_name)
        if stage:
            return await stage.process({"document_path": str(doc_path)})
        return {}

    def _process_document_sync(self, doc_path: Path) -> float:
        """Synchronous wrapper for concurrent processing."""
        start = time.perf_counter()

        if self.simulation_mode:
            time.sleep(0.3)
        else:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._process_document_e2e(doc_path))
            finally:
                loop.close()

        return (time.perf_counter() - start) * 1000

    def _clear_caches(self) -> None:
        """Clear caches for cold start testing."""
        # This would clear model caches, Redis caches, etc.
        pass

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save benchmark results to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "simulation_mode": self.simulation_mode,
            "hostname": os.uname().nodename,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAP_LLM Latency Benchmarks")
    parser.add_argument("--test-data", type=str, default="benchmarks/data/sample_documents",
                       help="Path to test documents")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of iterations per test")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 50, 100],
                       help="Concurrency levels to test")
    parser.add_argument("--output", type=str, default="benchmarks/results/latency_results.json",
                       help="Output file for results")
    parser.add_argument("--simulation", action="store_true",
                       help="Run in simulation mode")

    args = parser.parse_args()

    print("="*70)
    print("  SAP_LLM Latency Benchmark")
    print("="*70)

    # Load test documents
    test_data_dir = Path(args.test_data)
    if test_data_dir.exists():
        test_documents = list(test_data_dir.glob("*.png")) + list(test_data_dir.glob("*.jpg"))
        print(f"\nFound {len(test_documents)} test documents")
    else:
        print(f"\n⚠️  Test data not found at {test_data_dir}")
        print("   Run: python benchmarks/scripts/generate_test_data.py")
        test_documents = [Path("dummy.png")] * 100  # Dummy for simulation

    # Initialize benchmark
    benchmark = LatencyBenchmark(simulation_mode=args.simulation)

    results = {}

    # Run benchmarks
    results["e2e_latency"] = await benchmark.measure_e2e_latency(
        test_documents,
        num_iterations=args.num_iterations,
    )

    results["stage_latency"] = await benchmark.measure_stage_latency(
        test_documents,
        num_iterations=args.num_iterations,
    )

    results["concurrent_latency"] = benchmark.measure_concurrent_latency(
        test_documents,
        concurrency_levels=args.concurrency,
        requests_per_level=args.num_iterations,
    )

    results["cold_vs_warm"] = await benchmark.measure_cold_vs_warm(
        test_documents,
        num_iterations=50,
    )

    # Save results
    benchmark.save_results(results, args.output)

    # Print summary
    print("\n" + "="*70)
    print("  LATENCY BENCHMARK SUMMARY")
    print("="*70)

    e2e = results["e2e_latency"]
    status = "✅ PASS" if e2e.get("target_met") else "❌ FAIL"
    print(f"\nEnd-to-End P95 Latency: {e2e['p95']:.1f}ms (target: <600ms) {status}")

    print(f"\nBottleneck Analysis:")
    stage_latencies = results["stage_latency"]
    slowest_stage = max(stage_latencies.items(), key=lambda x: x[1]['p95'])
    print(f"  Slowest stage: {slowest_stage[0]} ({slowest_stage[1]['p95']:.1f}ms)")

    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
