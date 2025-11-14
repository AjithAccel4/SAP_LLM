"""
Load Testing for SAP_LLM API

Tests system performance under load using Locust.

Run with:
    locust -f tests/load/test_api.py --host http://localhost:8000
"""

from locust import HttpUser, task, between, events
import random
import os

class SAPLLMUser(HttpUser):
    """Simulated user for load testing"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Called when a user starts"""
        # Could authenticate here if needed
        pass

    @task(5)
    def extract_document_async(self):
        """
        Test asynchronous document extraction (50% of requests)

        Most common operation - upload document and poll for results
        """
        # Simulate document upload
        files = {
            "file": ("test_invoice.pdf", b"fake pdf content", "application/pdf")
        }

        with self.client.post(
            "/v1/extract",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 202:
                job_id = response.json().get("job_id")

                if job_id:
                    # Poll for result
                    self.client.get(f"/v1/jobs/{job_id}")
                    response.success()
                else:
                    response.failure("No job_id in response")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(2)
    def extract_document_sync(self):
        """
        Test synchronous document extraction (20% of requests)

        Wait for processing to complete
        """
        files = {
            "file": ("test_invoice.pdf", b"fake pdf content", "application/pdf")
        }

        self.client.post("/v1/extract/sync", files=files)

    @task(2)
    def get_metrics(self):
        """
        Test metrics endpoint (20% of requests)

        Monitoring queries
        """
        self.client.get("/metrics")

    @task(1)
    def health_check(self):
        """
        Test health check (10% of requests)

        Health monitoring
        """
        self.client.get("/health")

    @task(1)
    def detect_language(self):
        """Test language detection"""
        languages = [
            "Invoice Number: 12345",
            "Rechnung Nummer: 12345",
            "Factura Número: 12345",
            "发票号码: 12345",
            "رقم الفاتورة: 12345"
        ]

        text = random.choice(languages)

        self.client.post("/v1/detect-language", json={"text": text})


class StressTestUser(HttpUser):
    """
    Stress test user - aggressive testing

    Run with: locust -f tests/load/test_api.py StressTestUser
    """

    wait_time = between(0.1, 0.5)  # Minimal wait time

    @task
    def rapid_fire_extraction(self):
        """Rapid document extraction"""
        files = {"file": ("test.pdf", b"content", "application/pdf")}
        self.client.post("/v1/extract", files=files)


# Event handlers for reporting

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    print("🚀 Load test starting...")
    print(f"   Target: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    print("✅ Load test completed")

    # Print summary statistics
    stats = environment.stats.total
    print(f"\n📊 Summary:")
    print(f"   Total requests: {stats.num_requests}")
    print(f"   Failures: {stats.num_failures}")
    print(f"   Average response time: {stats.avg_response_time:.2f}ms")
    print(f"   Max response time: {stats.max_response_time:.2f}ms")
    print(f"   RPS: {stats.total_rps:.2f}")

    if stats.num_failures > 0:
        print(f"\n⚠️  {stats.num_failures} failures detected!")
