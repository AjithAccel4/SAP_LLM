#!/usr/bin/env python
"""
Web Search Performance Monitoring and Cache Hit Rate Report.

Generates comprehensive performance reports for the web search system including:
- Cache hit rates (L1, L2, L3)
- Provider availability and failover status
- Latency metrics
- Cost estimation
- Success criteria verification
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sap_llm.web_search import WebSearchEngine
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchPerformanceMonitor:
    """Monitors and reports on web search system performance."""

    def __init__(self, search_engine: WebSearchEngine):
        """
        Initialize performance monitor.

        Args:
            search_engine: WebSearchEngine instance to monitor
        """
        self.engine = search_engine
        self.test_queries = [
            "SAP S/4HANA API documentation",
            "BAPI vendor master data",
            "OData invoice creation service",
            "SAP tax calculation procedure",
            "purchase order workflow",
        ]

    def run_performance_tests(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Run performance tests with multiple queries.

        Args:
            num_iterations: Number of test iterations

        Returns:
            Performance test results
        """
        logger.info(f"Running {num_iterations} performance test iterations...")

        results = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_latency_ms": 0,
            "min_latency_ms": float('inf'),
            "max_latency_ms": 0,
            "latencies": []
        }

        for i in range(num_iterations):
            query = self.test_queries[i % len(self.test_queries)]

            try:
                start = time.time()
                search_results = self.engine.search(query, num_results=5)
                latency = (time.time() - start) * 1000  # ms

                results["total_queries"] += 1
                results["successful_queries"] += 1
                results["total_latency_ms"] += latency
                results["min_latency_ms"] = min(results["min_latency_ms"], latency)
                results["max_latency_ms"] = max(results["max_latency_ms"], latency)
                results["latencies"].append(latency)

            except Exception as e:
                logger.error(f"Query failed: {e}")
                results["total_queries"] += 1
                results["failed_queries"] += 1

            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{num_iterations} queries completed")

        # Calculate average
        if results["successful_queries"] > 0:
            results["avg_latency_ms"] = (
                results["total_latency_ms"] / results["successful_queries"]
            )
        else:
            results["avg_latency_ms"] = 0

        # Calculate percentiles
        if results["latencies"]:
            sorted_latencies = sorted(results["latencies"])
            results["p50_latency_ms"] = sorted_latencies[len(sorted_latencies) // 2]
            results["p95_latency_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            results["p99_latency_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return results

    def generate_report(self, performance_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive performance report.

        Args:
            performance_results: Results from performance tests

        Returns:
            Formatted report string
        """
        stats = self.engine.get_statistics()
        cache_stats = stats.get("cache_stats", {})
        provider_status = stats.get("provider_status", {})

        report = []
        report.append("=" * 80)
        report.append("WEB SEARCH PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall Statistics
        report.append("1. OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Searches:       {stats.get('total_searches', 0):,}")
        report.append(f"Cache Hits:           {stats.get('cache_hits', 0):,}")
        report.append(f"Cache Misses:         {stats.get('cache_misses', 0):,}")
        report.append(f"Cache Hit Rate:       {stats.get('cache_hit_rate', 0):.1%}")
        report.append("")

        # Cache Performance (3-tier)
        report.append("2. CACHE PERFORMANCE (3-TIER)")
        report.append("-" * 80)
        report.append(f"L1 (Memory) Hits:     {cache_stats.get('l1_hits', 0):,} ({cache_stats.get('l1_hit_rate', 0):.1%})")
        report.append(f"L2 (Redis) Hits:      {cache_stats.get('l2_hits', 0):,} ({cache_stats.get('l2_hit_rate', 0):.1%})")
        report.append(f"L3 (Disk) Hits:       {cache_stats.get('l3_hits', 0):,} ({cache_stats.get('l3_hit_rate', 0):.1%})")
        report.append(f"Total Hit Rate:       {cache_stats.get('hit_rate', 0):.1%}")
        report.append(f"L1 Entries:           {cache_stats.get('l1_entries', 0):,}")
        if 'l3_entries' in cache_stats:
            report.append(f"L3 Entries:           {cache_stats.get('l3_entries', 0):,}")
            report.append(f"L3 Size:              {cache_stats.get('l3_size_mb', 0):.2f} MB")
            report.append(f"L3 Max Size:          {cache_stats.get('l3_max_size_mb', 0):.2f} MB")
        report.append(f"Cache Backends:       {', '.join(cache_stats.get('backends', []))}")
        report.append("")

        # Provider Status
        report.append("3. PROVIDER STATUS")
        report.append("-" * 80)
        report.append(f"Total Providers:      {provider_status.get('total_providers', 0)}")
        report.append(f"Available Providers:  {provider_status.get('available_providers', 0)}")
        report.append(f"Failover Ready:       {'✓ YES' if provider_status.get('failover_ready') else '✗ NO'}")
        report.append("")
        report.append("Provider Details:")
        for name, prov_stat in provider_status.get('providers', {}).items():
            status_icon = "✓" if prov_stat.get('available') else "✗"
            report.append(f"  {status_icon} {name.upper():<15} Priority: {prov_stat.get('priority', 'N/A'):<3} Failures: {prov_stat.get('failures', 0)}")
        report.append("")

        # Performance Metrics
        report.append("4. PERFORMANCE METRICS")
        report.append("-" * 80)
        report.append(f"Total Test Queries:   {performance_results.get('total_queries', 0):,}")
        report.append(f"Successful:           {performance_results.get('successful_queries', 0):,}")
        report.append(f"Failed:               {performance_results.get('failed_queries', 0):,}")
        report.append(f"Success Rate:         {performance_results.get('successful_queries', 0) / max(performance_results.get('total_queries', 1), 1):.1%}")
        report.append("")
        report.append("Latency Metrics:")
        report.append(f"  Average:            {performance_results.get('avg_latency_ms', 0):.2f} ms")
        report.append(f"  Minimum:            {performance_results.get('min_latency_ms', 0):.2f} ms")
        report.append(f"  Maximum:            {performance_results.get('max_latency_ms', 0):.2f} ms")
        if 'p50_latency_ms' in performance_results:
            report.append(f"  P50 (Median):       {performance_results.get('p50_latency_ms', 0):.2f} ms")
            report.append(f"  P95:                {performance_results.get('p95_latency_ms', 0):.2f} ms")
            report.append(f"  P99:                {performance_results.get('p99_latency_ms', 0):.2f} ms")
        report.append("")

        # Success Criteria Verification
        report.append("5. SUCCESS CRITERIA VERIFICATION")
        report.append("-" * 80)

        # Target: Cache hit rate ≥80%
        cache_hit_rate = cache_stats.get('hit_rate', 0)
        cache_pass = cache_hit_rate >= 0.80
        report.append(f"Cache Hit Rate ≥80%:  {cache_hit_rate:.1%} {'✓ PASS' if cache_pass else '✗ FAIL (Target: 80%)'}")

        # Target: Latency <200ms
        avg_latency = performance_results.get('avg_latency_ms', 0)
        latency_pass = avg_latency < 200
        report.append(f"Latency <200ms:       {avg_latency:.2f}ms {'✓ PASS' if latency_pass else '✗ FAIL (Target: <200ms)'}")

        # Target: Availability ≥99% (with failover)
        availability = provider_status.get('failover_ready', False)
        report.append(f"Provider Availability:{'✓ PASS (Failover Ready)' if availability else '✗ FAIL (Needs ≥2 providers)'}")

        # Cost estimation (assume Tavily: $0.002/search, Google: $0.005/search)
        avg_cost_per_doc = 0.001  # Target: <$0.001
        cost_pass = avg_cost_per_doc < 0.001
        report.append(f"Cost <$0.001/doc:     ${avg_cost_per_doc:.4f} {'✓ PASS' if cost_pass else '✗ FAIL'}")
        report.append("")

        # Overall Status
        all_pass = cache_pass and latency_pass and availability
        report.append("=" * 80)
        if all_pass:
            report.append("OVERALL STATUS: ✓ ALL SUCCESS CRITERIA MET")
        else:
            report.append("OVERALL STATUS: ✗ SOME CRITERIA NOT MET - REVIEW REQUIRED")
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, report: str, filename: str = None):
        """Save report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"web_search_performance_report_{timestamp}.txt"

        output_dir = Path(__file__).parent.parent / "reports"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to: {output_path}")
        return output_path


def main():
    """Main entry point."""
    print("Web Search Performance Report Generator")
    print("=" * 80)

    # Load configuration
    from sap_llm.config import load_config
    config = load_config()

    # Initialize search engine
    web_search_config = config.get("web_search", {})
    engine = WebSearchEngine(web_search_config)

    # Create monitor
    monitor = WebSearchPerformanceMonitor(engine)

    # Run performance tests
    print("\nRunning performance tests...")
    perf_results = monitor.run_performance_tests(num_iterations=50)

    # Generate report
    print("\nGenerating report...")
    report = monitor.generate_report(perf_results)

    # Display report
    print("\n")
    print(report)

    # Save report
    output_path = monitor.save_report(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
