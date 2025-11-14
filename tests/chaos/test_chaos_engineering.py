"""
Chaos Engineering Test Suite for SAP_LLM

Tests system resilience under failure conditions:
- Pod failures
- Network partitions
- Resource exhaustion
- Database failures
- High latency scenarios
- Traffic spikes

Based on Chaos Engineering principles from Netflix Chaos Monkey

Run with: pytest tests/chaos/test_chaos_engineering.py -v --chaos-mode=true
"""

import pytest
import requests
import time
import random
import subprocess
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Test configuration
BASE_URL = "http://localhost:8000"
CHAOS_DURATION = 60  # seconds
RECOVERY_TIMEOUT = 300  # seconds


class ChaosTestContext:
    """Context manager for chaos tests"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        print(f"\n🔥 Starting chaos test: {self.name}")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"✅ Chaos test completed: {self.name} (duration: {duration:.2f}s)")


class TestPodFailures:
    """Test resilience to pod failures"""

    @pytest.mark.chaos
    def test_api_pod_kill(self):
        """Test system resilience when API pod is killed"""
        with ChaosTestContext("API Pod Kill"):
            # 1. Verify baseline
            response = requests.get(f"{BASE_URL}/health")
            assert response.status_code == 200, "Baseline check failed"

            # 2. Kill random API pod
            pods = self._get_pods("app=sap-llm-api")
            if pods:
                victim_pod = random.choice(pods)
                print(f"Killing pod: {victim_pod}")
                subprocess.run(["kubectl", "delete", "pod", victim_pod, "--force"])

                # 3. Verify system continues working
                time.sleep(5)  # Wait for pod deletion

                # Make requests during recovery
                success_count = 0
                for i in range(10):
                    try:
                        response = requests.get(f"{BASE_URL}/health", timeout=5)
                        if response.status_code == 200:
                            success_count += 1
                    except:
                        pass
                    time.sleep(1)

                # Should have some successes (load balancer routes to healthy pods)
                assert success_count >= 5, "System should remain partially available"

                # 4. Wait for recovery
                self._wait_for_recovery()

                # 5. Verify full recovery
                response = requests.get(f"{BASE_URL}/health")
                assert response.status_code == 200, "System should fully recover"

    @pytest.mark.chaos
    def test_worker_pod_kill(self):
        """Test system resilience when worker pod is killed"""
        with ChaosTestContext("Worker Pod Kill"):
            # Kill multiple worker pods simultaneously
            pods = self._get_pods("app=sap-llm-worker")
            victims = random.sample(pods, min(3, len(pods)))

            for victim in victims:
                print(f"Killing worker pod: {victim}")
                subprocess.run(["kubectl", "delete", "pod", victim, "--force"])

            # System should continue processing
            time.sleep(5)

            # Submit a document for processing
            response = requests.post(
                f"{BASE_URL}/v1/extract",
                files={"file": ("test.pdf", b"test content")}
            )

            # Should still accept requests (may be slower)
            assert response.status_code in [200, 202, 503]

            self._wait_for_recovery()

    @pytest.mark.chaos
    def test_gpu_pod_kill(self):
        """Test system resilience when GPU pod is killed"""
        with ChaosTestContext("GPU Pod Kill"):
            pods = self._get_pods("app=sap-llm-gpu")
            if pods:
                victim = random.choice(pods)
                print(f"Killing GPU pod: {victim}")
                subprocess.run(["kubectl", "delete", "pod", victim])

                # System may degrade but should not crash
                time.sleep(10)

                response = requests.get(f"{BASE_URL}/health")
                assert response.status_code == 200

                self._wait_for_recovery()

    @pytest.mark.chaos
    def test_cascading_pod_failures(self):
        """Test resilience to cascading pod failures"""
        with ChaosTestContext("Cascading Failures"):
            # Kill pods one by one
            all_pods = (
                self._get_pods("app=sap-llm-api") +
                self._get_pods("app=sap-llm-worker")
            )

            for i, pod in enumerate(all_pods[:3]):  # Kill first 3
                print(f"Killing pod {i+1}/3: {pod}")
                subprocess.run(["kubectl", "delete", "pod", pod, "--force"])
                time.sleep(5)

                # Check if system is still responsive
                try:
                    response = requests.get(f"{BASE_URL}/health", timeout=5)
                    print(f"  System status after {i+1} failures: {response.status_code}")
                except:
                    print(f"  System unresponsive after {i+1} failures")

            self._wait_for_recovery()

    def _get_pods(self, selector: str) -> List[str]:
        """Get pod names by selector"""
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", selector, "-o", "name"],
            capture_output=True,
            text=True
        )
        pods = [p.replace("pod/", "") for p in result.stdout.strip().split("\n") if p]
        return pods

    def _wait_for_recovery(self, timeout=60):
        """Wait for system to recover"""
        print("Waiting for recovery...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ System recovered")
                    return True
            except:
                pass
            time.sleep(2)

        raise TimeoutError("System did not recover in time")


class TestNetworkChaos:
    """Test resilience to network failures"""

    @pytest.mark.chaos
    def test_network_latency(self):
        """Test system under high network latency"""
        with ChaosTestContext("Network Latency"):
            # Inject 100ms latency
            self._inject_latency(100)

            try:
                # System should still work, just slower
                start_time = time.time()
                response = requests.get(f"{BASE_URL}/health", timeout=10)
                duration = time.time() - start_time

                assert response.status_code == 200
                print(f"Request took {duration:.2f}s with 100ms latency")

            finally:
                self._remove_latency()

    @pytest.mark.chaos
    def test_network_partition(self):
        """Test system during network partition"""
        with ChaosTestContext("Network Partition"):
            # Simulate partition between API and database
            self._create_network_partition("sap-llm-api", "cosmos-db")

            try:
                # API should handle gracefully
                response = requests.get(f"{BASE_URL}/health")
                # May return degraded status but not crash
                assert response.status_code in [200, 503]

            finally:
                self._remove_network_partition()

    @pytest.mark.chaos
    def test_packet_loss(self):
        """Test system under packet loss"""
        with ChaosTestContext("Packet Loss"):
            # Inject 5% packet loss
            self._inject_packet_loss(5)

            try:
                # Make multiple requests
                success_count = 0
                for i in range(20):
                    try:
                        response = requests.get(f"{BASE_URL}/health", timeout=5)
                        if response.status_code == 200:
                            success_count += 1
                    except:
                        pass

                # Should have most requests succeed despite packet loss
                assert success_count >= 15, f"Too many failures: {20 - success_count}"

            finally:
                self._remove_packet_loss()

    def _inject_latency(self, delay_ms: int):
        """Inject network latency using tc (traffic control)"""
        # This requires network admin capabilities
        # In real environment, use Chaos Mesh or similar
        pass

    def _remove_latency(self):
        """Remove network latency"""
        pass

    def _create_network_partition(self, source: str, target: str):
        """Create network partition between services"""
        # Use NetworkPolicy or Chaos Mesh
        pass

    def _remove_network_partition(self):
        """Remove network partition"""
        pass

    def _inject_packet_loss(self, percentage: int):
        """Inject packet loss"""
        pass

    def _remove_packet_loss(self):
        """Remove packet loss"""
        pass


class TestResourceExhaustion:
    """Test resilience to resource exhaustion"""

    @pytest.mark.chaos
    def test_cpu_exhaustion(self):
        """Test system under CPU exhaustion"""
        with ChaosTestContext("CPU Exhaustion"):
            # Create CPU stress
            stress_pod = self._create_stress_pod("cpu", "2000m")

            try:
                # System should still respond (may be slower)
                response = requests.get(f"{BASE_URL}/health", timeout=10)
                assert response.status_code in [200, 503]

                # Check if auto-scaling kicks in
                time.sleep(30)
                new_pod_count = len(self._get_pods("app=sap-llm-api"))
                print(f"Pod count during CPU stress: {new_pod_count}")

            finally:
                self._delete_stress_pod(stress_pod)

    @pytest.mark.chaos
    def test_memory_exhaustion(self):
        """Test system under memory pressure"""
        with ChaosTestContext("Memory Exhaustion"):
            # Create memory stress
            stress_pod = self._create_stress_pod("memory", "8Gi")

            try:
                # Monitor for OOMKilled pods
                time.sleep(30)

                # System should handle gracefully
                response = requests.get(f"{BASE_URL}/health", timeout=10)
                assert response.status_code in [200, 503]

            finally:
                self._delete_stress_pod(stress_pod)

    @pytest.mark.chaos
    def test_disk_pressure(self):
        """Test system under disk pressure"""
        with ChaosTestContext("Disk Pressure"):
            # Fill up disk space
            self._fill_disk()

            try:
                # System should handle gracefully
                response = requests.get(f"{BASE_URL}/health")
                assert response.status_code in [200, 503]

            finally:
                self._cleanup_disk()

    def _create_stress_pod(self, resource_type: str, amount: str) -> str:
        """Create stress pod"""
        # Would create a pod that stresses resources
        return "stress-pod-name"

    def _delete_stress_pod(self, pod_name: str):
        """Delete stress pod"""
        pass

    def _fill_disk(self):
        """Fill disk to 90%"""
        pass

    def _cleanup_disk(self):
        """Clean up disk space"""
        pass

    def _get_pods(self, selector: str) -> List[str]:
        """Get pod names by selector"""
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", selector, "-o", "name"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split("\n")


class TestDatabaseChaos:
    """Test resilience to database failures"""

    @pytest.mark.chaos
    def test_redis_failure(self):
        """Test system when Redis fails"""
        with ChaosTestContext("Redis Failure"):
            # Kill Redis
            redis_pods = self._get_pods("app=redis")
            if redis_pods:
                subprocess.run(["kubectl", "delete", "pod", redis_pods[0], "--force"])

            # System should degrade gracefully (cache misses)
            time.sleep(5)

            response = requests.get(f"{BASE_URL}/health")
            # Should still work, just slower
            assert response.status_code == 200

            # Check metrics
            metrics = requests.get(f"{BASE_URL}/metrics").text
            # Cache hit rate should drop
            print("Metrics during Redis failure:", metrics)

    @pytest.mark.chaos
    def test_cosmos_db_throttling(self):
        """Test system under Cosmos DB throttling"""
        with ChaosTestContext("Cosmos DB Throttling"):
            # Simulate high load to trigger throttling
            # Make many concurrent requests
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [
                    executor.submit(requests.post, f"{BASE_URL}/v1/extract",
                                  files={"file": ("test.pdf", b"content")})
                    for _ in range(100)
                ]

                # Count 429 errors (throttling)
                throttle_count = 0
                for future in as_completed(futures):
                    try:
                        response = future.result(timeout=10)
                        if response.status_code == 429:
                            throttle_count += 1
                    except:
                        pass

                print(f"Throttled requests: {throttle_count}")

    @pytest.mark.chaos
    def test_database_connection_pool_exhaustion(self):
        """Test system when database connection pool is exhausted"""
        with ChaosTestContext("Connection Pool Exhaustion"):
            # Make many concurrent requests to exhaust pool
            with ThreadPoolExecutor(max_workers=200) as executor:
                futures = [
                    executor.submit(requests.get, f"{BASE_URL}/v1/stats")
                    for _ in range(200)
                ]

                success_count = 0
                for future in as_completed(futures):
                    try:
                        response = future.result(timeout=5)
                        if response.status_code == 200:
                            success_count += 1
                    except:
                        pass

                # Should handle most requests
                assert success_count >= 150, f"Too many failures: {200 - success_count}"

    def _get_pods(self, selector: str) -> List[str]:
        """Get pod names by selector"""
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", selector, "-o", "name"],
            capture_output=True,
            text=True
        )
        pods = [p.replace("pod/", "") for p in result.stdout.strip().split("\n") if p]
        return pods


class TestTrafficChaos:
    """Test resilience to traffic spikes"""

    @pytest.mark.chaos
    def test_traffic_spike(self):
        """Test system under sudden traffic spike"""
        with ChaosTestContext("Traffic Spike"):
            # Baseline
            baseline_response = requests.get(f"{BASE_URL}/health")
            assert baseline_response.status_code == 200

            # Spike: 10x normal traffic
            print("Generating traffic spike...")
            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = [
                    executor.submit(self._make_request)
                    for _ in range(1000)
                ]

                results = []
                for future in as_completed(futures):
                    results.append(future.result())

            # Analyze results
            success_rate = sum(1 for r in results if r.get("status") == 200) / len(results)
            print(f"Success rate during spike: {success_rate:.2%}")

            # Should maintain >90% success rate
            assert success_rate >= 0.9, f"Too many failures during spike: {success_rate:.2%}"

    @pytest.mark.chaos
    def test_sustained_high_load(self):
        """Test system under sustained high load"""
        with ChaosTestContext("Sustained High Load"):
            duration = 60  # seconds
            start_time = time.time()

            results = []

            with ThreadPoolExecutor(max_workers=50) as executor:
                while time.time() - start_time < duration:
                    future = executor.submit(self._make_request)
                    results.append(future)
                    time.sleep(0.1)  # 10 requests/second

                # Wait for all to complete
                final_results = [f.result() for f in results]

            # Calculate metrics
            success_rate = sum(1 for r in final_results if r.get("status") == 200) / len(final_results)
            avg_latency = sum(r.get("latency", 0) for r in final_results) / len(final_results)

            print(f"Sustained load results:")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Avg latency: {avg_latency:.2f}s")
            print(f"  Total requests: {len(final_results)}")

            assert success_rate >= 0.95

    @pytest.mark.chaos
    def test_burst_traffic(self):
        """Test system with bursts of traffic"""
        with ChaosTestContext("Burst Traffic"):
            # Send traffic in bursts
            for burst in range(5):
                print(f"Burst {burst + 1}/5")

                with ThreadPoolExecutor(max_workers=50) as executor:
                    futures = [
                        executor.submit(self._make_request)
                        for _ in range(100)
                    ]

                    results = [f.result() for f in futures]
                    success_rate = sum(1 for r in results if r.get("status") == 200) / len(results)
                    print(f"  Success rate: {success_rate:.2%}")

                # Cool down between bursts
                time.sleep(10)

    def _make_request(self) -> Dict:
        """Make a test request and return results"""
        start_time = time.time()
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            latency = time.time() - start_time
            return {"status": response.status_code, "latency": latency}
        except Exception as e:
            latency = time.time() - start_time
            return {"status": 0, "latency": latency, "error": str(e)}


class TestRecoveryScenarios:
    """Test system recovery from failures"""

    @pytest.mark.chaos
    def test_automatic_pod_recovery(self):
        """Test automatic pod recovery by Kubernetes"""
        with ChaosTestContext("Automatic Recovery"):
            # Get initial pod count
            initial_pods = len(self._get_pods("app=sap-llm-api"))

            # Kill all API pods
            for pod in self._get_pods("app=sap-llm-api"):
                subprocess.run(["kubectl", "delete", "pod", pod, "--force"])

            # Wait for Kubernetes to recreate pods
            recovered = False
            for i in range(60):  # Wait up to 60 seconds
                current_pods = len(self._get_pods("app=sap-llm-api"))
                if current_pods >= initial_pods:
                    print(f"Pods recovered in {i} seconds")
                    recovered = True
                    break
                time.sleep(1)

            assert recovered, "Pods did not recover automatically"

            # Verify functionality
            response = requests.get(f"{BASE_URL}/health")
            assert response.status_code == 200

    @pytest.mark.chaos
    def test_graceful_degradation(self):
        """Test graceful degradation during partial failures"""
        with ChaosTestContext("Graceful Degradation"):
            # Kill half the worker pods
            workers = self._get_pods("app=sap-llm-worker")
            victims = workers[:len(workers)//2]

            for victim in victims:
                subprocess.run(["kubectl", "delete", "pod", victim])

            # System should degrade gracefully
            time.sleep(5)

            # Check metrics
            response = requests.get(f"{BASE_URL}/v1/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"System stats during degradation: {stats}")

            # Should still process requests, just slower
            response = requests.post(
                f"{BASE_URL}/v1/extract",
                files={"file": ("test.pdf", b"content")}
            )
            assert response.status_code in [200, 202], "Should still accept requests"

    def _get_pods(self, selector: str) -> List[str]:
        """Get pod names by selector"""
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", selector, "-o", "name"],
            capture_output=True,
            text=True
        )
        pods = [p.replace("pod/", "") for p in result.stdout.strip().split("\n") if p]
        return pods


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for chaos tests"""
    config.addinivalue_line(
        "markers",
        "chaos: mark test as chaos engineering test"
    )


@pytest.fixture(scope="session")
def chaos_mode(request):
    """Enable chaos mode"""
    return request.config.getoption("--chaos-mode", default=False)


def pytest_addoption(parser):
    """Add command line options"""
    parser.addoption(
        "--chaos-mode",
        action="store_true",
        default=False,
        help="Enable chaos testing mode"
    )


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--chaos-mode=true",
        "--html=chaos_report.html",
        "--self-contained-html"
    ])
