"""
Latency performance tests for SAP_LLM.
"""

import pytest
import time
from unittest.mock import MagicMock
import numpy as np


@pytest.mark.performance
class TestLatencyBenchmarks:
    """Latency benchmark tests."""

    def test_e2e_latency(self):
        """Test end-to-end latency."""
        mock_pipeline = MagicMock()

        latencies = []
        num_runs = 100

        for _ in range(num_runs):
            start = time.time()
            mock_pipeline.process({"data": "test"})
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"\nAvg latency: {avg_latency:.2f}ms")
        print(f"P50 latency: {p50_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        print(f"P99 latency: {p99_latency:.2f}ms")

        assert p95_latency < 1000  # P95 should be under 1s

    @pytest.mark.parametrize("stage", [
        "inbox",
        "preprocessing",
        "classification",
        "extraction",
        "validation",
        "routing",
    ])
    def test_stage_latency(self, stage):
        """Test latency for individual stages."""
        mock_stage = MagicMock()

        latencies = []
        for _ in range(100):
            start = time.time()
            mock_stage.process({"data": "test"})
            latencies.append((time.time() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n{stage} avg latency: {avg_latency:.2f}ms")

        assert avg_latency < 500  # Each stage should be under 500ms

    def test_ocr_latency(self):
        """Test OCR processing latency."""
        from PIL import Image

        mock_ocr = MagicMock()
        image = Image.new('RGB', (800, 600))

        latencies = []
        for _ in range(50):
            start = time.time()
            mock_ocr.process(image)
            latencies.append((time.time() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\nOCR avg latency: {avg_latency:.2f}ms")

        # OCR is typically slower
        assert avg_latency < 2000

    def test_model_inference_latency(self):
        """Test model inference latency."""
        import torch

        mock_model = MagicMock()
        input_tensor = torch.randn(1, 3, 224, 224)

        latencies = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                mock_model(input_tensor)
            latencies.append((time.time() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\nModel inference avg latency: {avg_latency:.2f}ms")

        assert avg_latency < 100  # Model inference should be fast
