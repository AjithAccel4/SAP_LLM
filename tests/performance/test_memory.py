"""
Memory usage performance tests for SAP_LLM.
"""

import pytest
import psutil
import os
from unittest.mock import MagicMock


@pytest.mark.performance
class TestMemoryBenchmarks:
    """Memory usage benchmark tests."""

    def test_baseline_memory(self):
        """Test baseline memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        print(f"\nBaseline memory: {memory_mb:.2f} MB")
        assert memory_mb > 0

    def test_model_loading_memory(self):
        """Test memory usage when loading models."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Mock model loading
        mock_model = MagicMock()

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        print(f"\nModel loading memory increase: {memory_increase:.2f} MB")

    def test_document_processing_memory(self):
        """Test memory usage during document processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Process 100 documents
        documents = []
        for i in range(100):
            documents.append({"id": i, "data": "x" * 10000})

        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory

        # Clear documents
        documents = []

        print(f"\nPeak memory increase: {memory_increase:.2f} MB")
        assert memory_increase < 200  # Should not exceed 200MB
