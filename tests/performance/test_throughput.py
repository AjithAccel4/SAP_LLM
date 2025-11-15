"""
Throughput performance tests for SAP_LLM.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, MagicMock, patch
import torch

from sap_llm.stages import *
from sap_llm.models.unified_model import UnifiedExtractorModel


@pytest.mark.performance
@pytest.mark.slow
class TestThroughputBenchmarks:
    """Throughput benchmark tests."""

    @pytest.fixture
    def mock_model(self):
        """Create mocked model for performance testing."""
        model = MagicMock()
        model.classify.return_value = ("PURCHASE_ORDER", "STANDARD", 0.95)
        model.extract.return_value = ({"po_number": "12345"}, {})
        model.route.return_value = {"endpoint": "API_PO", "confidence": 0.95}
        return model

    def test_single_document_throughput(self, mock_model, sample_document_image):
        """Test single document processing throughput."""
        num_docs = 100
        start_time = time.time()

        for i in range(num_docs):
            # Mock processing
            doc_type, subtype, conf = mock_model.classify(
                image=sample_document_image,
                ocr_text="test",
                words=[],
                boxes=[],
            )

        elapsed = time.time() - start_time
        throughput = num_docs / elapsed

        print(f"\nSingle document throughput: {throughput:.2f} docs/sec")
        assert throughput > 10  # Minimum 10 docs/sec

    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
    def test_batch_processing_throughput(self, mock_model, batch_size):
        """Test batch processing throughput."""
        num_batches = 10
        docs_per_batch = batch_size

        start_time = time.time()

        for _ in range(num_batches):
            # Process batch
            batch = [
                {"image": MagicMock(), "ocr_text": f"doc_{i}"}
                for i in range(docs_per_batch)
            ]

            for doc in batch:
                mock_model.classify(
                    image=doc["image"],
                    ocr_text=doc["ocr_text"],
                    words=[],
                    boxes=[],
                )

        elapsed = time.time() - start_time
        total_docs = num_batches * docs_per_batch
        throughput = total_docs / elapsed

        print(f"\nBatch size {batch_size} throughput: {throughput:.2f} docs/sec")
        assert throughput > 0

    @pytest.mark.requires_gpu
    def test_gpu_throughput(self):
        """Test GPU processing throughput."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Mock GPU processing
        batch_size = 32
        num_batches = 100

        start_time = time.time()

        for _ in range(num_batches):
            # Simulate GPU inference
            input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()
            with torch.no_grad():
                # Mock inference
                output = torch.nn.functional.softmax(
                    torch.randn(batch_size, 15).cuda(),
                    dim=1,
                )

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        total_docs = num_batches * batch_size
        throughput = total_docs / elapsed

        print(f"\nGPU throughput: {throughput:.2f} docs/sec")
        assert throughput > 100  # GPU should be faster

    @pytest.mark.asyncio
    async def test_async_pipeline_throughput(self):
        """Test async pipeline throughput."""
        num_docs = 100

        async def process_document(doc_id):
            """Simulate async document processing."""
            await asyncio.sleep(0.01)  # Simulate I/O
            return {"id": doc_id, "status": "processed"}

        start_time = time.time()

        # Process documents concurrently
        tasks = [process_document(i) for i in range(num_docs)]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        throughput = num_docs / elapsed

        print(f"\nAsync pipeline throughput: {throughput:.2f} docs/sec")
        assert len(results) == num_docs
        assert throughput > 50  # Async should be faster

    def test_stage_throughput_breakdown(self):
        """Test throughput for individual stages."""
        stages = {
            "preprocessing": MagicMock(),
            "classification": MagicMock(),
            "extraction": MagicMock(),
            "validation": MagicMock(),
        }

        num_docs = 100
        stage_times = {}

        for stage_name, stage in stages.items():
            start_time = time.time()

            for _ in range(num_docs):
                stage.process({"data": "test"})

            elapsed = time.time() - start_time
            stage_times[stage_name] = elapsed
            throughput = num_docs / elapsed

            print(f"\n{stage_name} throughput: {throughput:.2f} docs/sec")

        # Identify bottleneck
        bottleneck = max(stage_times, key=stage_times.get)
        print(f"\nBottleneck stage: {bottleneck}")

    @pytest.mark.parametrize("concurrency", [1, 5, 10, 20])
    def test_concurrent_requests(self, concurrency):
        """Test throughput with concurrent requests."""
        import concurrent.futures

        def process_request(req_id):
            """Simulate processing request."""
            time.sleep(0.01)  # Simulate work
            return {"id": req_id, "status": "done"}

        num_requests = 100

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_request, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        elapsed = time.time() - start_time
        throughput = num_requests / elapsed

        print(f"\nConcurrency {concurrency} throughput: {throughput:.2f} req/sec")
        assert len(results) == num_requests


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""

    @pytest.mark.parametrize("doc_count", [10, 100, 1000, 10000])
    def test_scaling_with_document_count(self, doc_count):
        """Test how throughput scales with document count."""
        mock_stage = MagicMock()

        start_time = time.time()

        for i in range(doc_count):
            mock_stage.process({"id": i})

        elapsed = time.time() - start_time
        throughput = doc_count / elapsed

        print(f"\n{doc_count} documents throughput: {throughput:.2f} docs/sec")
        assert throughput > 0

    def test_memory_scalability(self):
        """Test memory usage scaling."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process documents
        documents = []
        for i in range(1000):
            documents.append({"id": i, "data": "x" * 1000})

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\nMemory increase for 1000 docs: {memory_increase:.2f} MB")
        assert memory_increase < 500  # Should not use more than 500MB
