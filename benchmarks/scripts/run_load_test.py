#!/usr/bin/env python3
"""
Locust load testing for SAP_LLM API.

Tests:
- Concurrent user load
- Stress testing
- Spike testing
- Soak testing (sustained load)
"""

from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import time
import json
import random
from pathlib import Path
from typing import Optional
import io


class DocumentProcessorUser(HttpUser):
    """Simulated user processing documents."""

    wait_time = between(0.1, 0.5)  # Wait 100-500ms between requests

    def on_start(self):
        """Initialize user (load test documents)."""
        self.document_index = 0
        self.test_documents = self._load_test_documents()

    def _load_test_documents(self):
        """Load test documents from disk or generate synthetic."""
        test_data_dir = Path("benchmarks/data/sample_documents")

        if test_data_dir.exists():
            docs = list(test_data_dir.glob("*.png")) + list(test_data_dir.glob("*.jpg"))
            return docs if docs else [None] * 100
        else:
            # Return dummy paths for simulation
            return [None] * 100

    @task(10)
    def process_document(self):
        """Process a single document (most common task)."""
        doc_path = self.test_documents[self.document_index % len(self.test_documents)]
        self.document_index += 1

        if doc_path and doc_path.exists():
            with open(doc_path, "rb") as f:
                files = {"file": (doc_path.name, f, "image/png")}
                with self.client.post(
                    "/api/v1/process",
                    files=files,
                    catch_response=True,
                    name="POST /api/v1/process"
                ) as response:
                    if response.status_code == 200:
                        response.success()
                    else:
                        response.failure(f"Got status code {response.status_code}")
        else:
            # Simulate with JSON payload
            payload = {
                "document_id": f"doc_{self.document_index}",
                "document_type": random.choice(["PURCHASE_ORDER", "INVOICE", "DELIVERY_NOTE"]),
            }
            with self.client.post(
                "/api/v1/process",
                json=payload,
                catch_response=True,
                name="POST /api/v1/process (simulated)"
            ) as response:
                if response.status_code in [200, 404]:  # 404 expected for simulated endpoint
                    response.success()
                else:
                    response.failure(f"Got status code {response.status_code}")

    @task(3)
    def get_document_status(self):
        """Check document processing status."""
        doc_id = f"doc_{random.randint(1, 10000)}"

        with self.client.get(
            f"/api/v1/status/{doc_id}",
            catch_response=True,
            name="GET /api/v1/status/:id"
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def get_statistics(self):
        """Get system statistics."""
        with self.client.get(
            "/api/v1/stats",
            catch_response=True,
            name="GET /api/v1/stats"
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class HighLoadUser(HttpUser):
    """High-load stress testing user."""

    wait_time = between(0.01, 0.05)  # Very aggressive - 10-50ms between requests

    @task
    def rapid_fire_requests(self):
        """Fire requests as fast as possible."""
        payload = {
            "document_id": f"stress_{time.time_ns()}",
            "urgent": True,
        }

        self.client.post(
            "/api/v1/process",
            json=payload,
            catch_response=True,
            name="POST /api/v1/process (stress)"
        )


# Programmatic execution functions
def run_load_test(
    host: str = "http://localhost:8000",
    users: int = 100,
    spawn_rate: int = 10,
    run_time: str = "5m",
) -> dict:
    """
    Run load test programmatically.

    Args:
        host: API endpoint
        users: Number of concurrent users
        spawn_rate: Users to spawn per second
        run_time: How long to run (e.g., "5m", "1h")

    Returns:
        Test statistics
    """
    print(f"\n🔥 Starting load test:")
    print(f"   Host: {host}")
    print(f"   Users: {users}")
    print(f"   Spawn rate: {spawn_rate}/sec")
    print(f"   Duration: {run_time}")

    # Setup Locust environment
    setup_logging("INFO", None)

    env = Environment(user_classes=[DocumentProcessorUser])
    env.create_local_runner()

    # Start test
    env.runner.start(user_count=users, spawn_rate=spawn_rate)

    # Run for specified time
    duration_seconds = _parse_duration(run_time)
    time.sleep(duration_seconds)

    # Stop test
    env.runner.quit()

    # Collect stats
    stats = {
        "total_requests": env.stats.total.num_requests,
        "total_failures": env.stats.total.num_failures,
        "average_response_time": env.stats.total.avg_response_time,
        "min_response_time": env.stats.total.min_response_time,
        "max_response_time": env.stats.total.max_response_time,
        "requests_per_second": env.stats.total.total_rps,
        "failure_rate": env.stats.total.fail_ratio,
    }

    print(f"\n✅ Load test complete:")
    print(f"   Total requests: {stats['total_requests']:,}")
    print(f"   Failures: {stats['total_failures']:,}")
    print(f"   Avg response: {stats['average_response_time']:.0f}ms")
    print(f"   RPS: {stats['requests_per_second']:.1f}")

    return stats


def _parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds."""
    if duration_str.endswith('s'):
        return int(duration_str[:-1])
    elif duration_str.endswith('m'):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * 3600
    else:
        return int(duration_str)


if __name__ == "__main__":
    """
    Run with Locust CLI:

    locust -f run_load_test.py \\
        --host http://localhost:8000 \\
        --users 100 \\
        --spawn-rate 10 \\
        --run-time 10m \\
        --headless \\
        --csv benchmarks/results/load_test

    Or for web UI:

    locust -f run_load_test.py --host http://localhost:8000
    # Then open http://localhost:8089
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run load test programmatically")
    parser.add_argument("--host", default="http://localhost:8000", help="API host")
    parser.add_argument("--users", type=int, default=100, help="Number of users")
    parser.add_argument("--spawn-rate", type=int, default=10, help="Spawn rate")
    parser.add_argument("--run-time", default="5m", help="Duration (e.g., 5m, 1h)")
    parser.add_argument("--output", default="benchmarks/results/load_test_results.json",
                       help="Output file")

    args = parser.parse_args()

    results = run_load_test(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
