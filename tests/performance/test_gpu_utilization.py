"""
GPU utilization performance tests for SAP_LLM.
"""

import pytest
import torch
import time


@pytest.mark.performance
@pytest.mark.requires_gpu
class TestGPUUtilizationBenchmarks:
    """GPU utilization benchmark tests."""

    def test_gpu_availability(self):
        """Test GPU availability."""
        assert torch.cuda.is_available(), "GPU not available"
        gpu_count = torch.cuda.device_count()
        print(f"\nAvailable GPUs: {gpu_count}")
        assert gpu_count > 0

    def test_gpu_memory_utilization(self):
        """Test GPU memory utilization."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Allocate tensors
        tensors = []
        for _ in range(10):
            tensor = torch.randn(1000, 1000).cuda()
            tensors.append(tensor)

        peak_memory = torch.cuda.memory_allocated() / 1024 / 1024
        memory_used = peak_memory - initial_memory

        print(f"\nGPU memory used: {memory_used:.2f} MB")

        # Cleanup
        del tensors
        torch.cuda.empty_cache()

    def test_gpu_inference_speed(self):
        """Test GPU inference speed."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        model = torch.nn.Linear(1000, 100).cuda()
        input_tensor = torch.randn(32, 1000).cuda()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)

        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(input_tensor)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        throughput = 100 / elapsed

        print(f"\nGPU inference throughput: {throughput:.2f} batches/sec")
        assert throughput > 10
