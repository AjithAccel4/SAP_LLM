"""
Real End-to-End Integration Tests with Actual Models

Tests the complete SAP_LLM pipeline with real model inference (not mocks).
Tests all 8 pipeline stages with actual LayoutLMv3, LLaMA-2, and Mixtral models.

IMPORTANT: These tests require:
- GPU access (CUDA-capable GPU with 24GB+ VRAM)
- HuggingFace model access tokens
- Actual model weights downloaded

Run with: pytest tests/integration/test_real_models_e2e.py -v -s --gpu
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import pytest
import torch
from PIL import Image

# Import pipeline components
from sap_llm.config import Config, load_config
from sap_llm.models.unified_model import UnifiedExtractorModel
from sap_llm.stages.inbox import InboxStage
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.classification import ClassificationStage
from sap_llm.stages.type_identifier import TypeIdentifierStage
from sap_llm.stages.extraction import ExtractionStage
from sap_llm.stages.quality_check import QualityCheckStage
from sap_llm.stages.validation import ValidationStage
from sap_llm.stages.routing import RoutingStage


# Skip these tests if not running with GPU flag
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_GPU_TESTS", "false").lower() == "true",
    reason="GPU tests require RUN_GPU_TESTS=true environment variable"
)


@pytest.fixture(scope="module")
def gpu_check():
    """Verify GPU availability before running tests."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available - skipping real model tests")

    # Check VRAM
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gpu_memory < 20:  # 20GB minimum
        pytest.skip(f"Insufficient GPU memory: {gpu_memory:.1f}GB (need 20GB+)")

    return True


@pytest.fixture(scope="module")
def config() -> Config:
    """Load configuration for testing."""
    return load_config()


@pytest.fixture(scope="module")
def real_unified_model(config: Config, gpu_check) -> UnifiedExtractorModel:
    """
    Load actual unified model with real weights.

    This fixture loads:
    - Vision: microsoft/layoutlmv3-base (~500MB)
    - Language: meta-llama/Llama-2-7b-hf (~13GB)
    - Reasoning: mistralai/Mixtral-8x7B-v0.1 (~46GB)

    Total VRAM: ~24GB+ required
    """
    print("\n[LOADING REAL MODELS - This will take 2-5 minutes...]")

    model = UnifiedExtractorModel(
        vision_model="microsoft/layoutlmv3-base",
        language_model="meta-llama/Llama-2-7b-hf",
        reasoning_model="mistralai/Mixtral-8x7B-v0.1",
        device="cuda",
        config=config
    )

    # Warm up models
    print("[Warming up models with dummy inference...]")
    dummy_image = torch.randn(1, 3, 224, 224).cuda()
    _ = model.vision_encoder(dummy_image)

    print("[Models loaded and warmed up successfully]")

    yield model

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("[GPU memory cleaned up]")


@pytest.fixture
def sample_invoice_pdf(tmp_path) -> Path:
    """Create a sample invoice PDF for testing."""
    # In real implementation, this would load an actual test invoice PDF
    # For now, create a dummy PDF path
    pdf_path = tmp_path / "test_invoice.pdf"

    # TODO: Add actual test PDF creation or load from test fixtures
    # For now, return path (tests will need actual PDFs to run)
    return pdf_path


@pytest.fixture
def sample_po_pdf(tmp_path) -> Path:
    """Create a sample purchase order PDF for testing."""
    pdf_path = tmp_path / "test_po.pdf"
    return pdf_path


class TestRealModelsEndToEnd:
    """
    End-to-end integration tests with real model inference.

    These tests verify that the complete pipeline works with actual
    model weights and real inference, not mocks.
    """

    @pytest.mark.gpu
    @pytest.mark.slow
    async def test_supplier_invoice_real_inference(
        self,
        real_unified_model: UnifiedExtractorModel,
        sample_invoice_pdf: Path,
        config: Config
    ):
        """
        Test supplier invoice processing with real model inference.

        This test:
        1. Loads a real invoice PDF
        2. Processes through all 8 stages with REAL models
        3. Validates extraction accuracy
        4. Checks quality scores
        5. Verifies routing decisions
        6. Measures real performance metrics
        """
        print("\n[Testing Supplier Invoice with Real Models]")

        # Stage 1: Inbox (load document)
        inbox_stage = InboxStage(config=config.stages.inbox)

        if not sample_invoice_pdf.exists():
            pytest.skip("Sample invoice PDF not available")

        inbox_output = inbox_stage.process({
            "file_path": str(sample_invoice_pdf)
        })

        assert inbox_output["success"] is True
        print(f"✓ Inbox stage completed: {inbox_output['file_type']}")

        # Stage 2: Preprocessing (OCR + image enhancement)
        preprocessing_stage = PreprocessingStage(config=config.stages.preprocessing)
        preprocessing_output = preprocessing_stage.process(inbox_output)

        assert "images" in preprocessing_output
        assert "ocr_text" in preprocessing_output
        print(f"✓ Preprocessing completed: {len(preprocessing_output['images'])} pages")

        # Stage 3: Classification (REAL LayoutLMv3 inference)
        start_time = time.time()
        classification_stage = ClassificationStage(
            model=real_unified_model.vision_encoder,
            config=config.stages.classification
        )
        classification_output = classification_stage.process(preprocessing_output)
        classification_latency = (time.time() - start_time) * 1000

        assert classification_output["document_type"] is not None
        assert classification_output["confidence"] > 0.5
        print(f"✓ Classification: {classification_output['document_type']} "
              f"(confidence: {classification_output['confidence']:.2%}, "
              f"latency: {classification_latency:.1f}ms)")

        # Stage 4: Type Identifier (identify subtype)
        type_stage = TypeIdentifierStage(
            model=real_unified_model.vision_encoder,
            config=config.stages.type_identifier
        )
        type_output = type_stage.process(classification_output)

        assert "document_subtype" in type_output
        print(f"✓ Type Identified: {type_output['document_subtype']}")

        # Stage 5: Extraction (REAL LLaMA-2 inference)
        start_time = time.time()
        extraction_stage = ExtractionStage(
            model=real_unified_model.language_decoder,
            config=config.stages.extraction
        )
        extraction_output = extraction_stage.process(type_output)
        extraction_latency = (time.time() - start_time) * 1000

        assert "extracted_data" in extraction_output
        assert extraction_output["extracted_data"] is not None

        # Verify key invoice fields were extracted
        extracted_data = extraction_output["extracted_data"]
        expected_fields = [
            "invoice_number", "invoice_date", "vendor_name",
            "total_amount", "currency"
        ]

        extracted_fields_count = sum(
            1 for field in expected_fields if field in extracted_data
        )
        extraction_coverage = extracted_fields_count / len(expected_fields)

        print(f"✓ Extraction: {extracted_fields_count}/{len(expected_fields)} "
              f"fields extracted ({extraction_coverage:.1%} coverage, "
              f"latency: {extraction_latency:.1f}ms)")

        # Validate extraction coverage is high
        assert extraction_coverage >= 0.80, \
            f"Extraction coverage {extraction_coverage:.1%} below 80% threshold"

        # Stage 6: Quality Check (assess extraction quality)
        quality_stage = QualityCheckStage(
            model=real_unified_model.language_decoder,
            pmg=None,  # PMG optional for this test
            config=config.stages.quality_check
        )
        quality_output = quality_stage.process(extraction_output)

        assert "quality_score" in quality_output
        assert quality_output["quality_score"] >= 0.70
        print(f"✓ Quality Check: {quality_output['quality_score']:.2%} quality score")

        # Stage 7: Validation (business rules)
        validation_stage = ValidationStage(config=config.stages.validation)
        validation_output = validation_stage.process(quality_output)

        assert "validation_result" in validation_output
        print(f"✓ Validation: {len(validation_output['validation_result'].get('errors', []))} errors")

        # Stage 8: Routing (REAL Mixtral reasoning)
        start_time = time.time()
        routing_stage = RoutingStage(
            reasoning_engine=real_unified_model.reasoning_engine,
            pmg=None,
            config=config.stages.routing
        )
        routing_output = routing_stage.process(validation_output)
        routing_latency = (time.time() - start_time) * 1000

        assert "routing_decision" in routing_output
        assert routing_output["routing_decision"]["sap_endpoint"] is not None
        print(f"✓ Routing: {routing_output['routing_decision']['sap_endpoint']} "
              f"(latency: {routing_latency:.1f}ms)")

        # FINAL ASSERTIONS - Production-Ready Criteria
        total_latency = classification_latency + extraction_latency + routing_latency

        # Latency assertions
        assert classification_latency < 200, \
            f"Classification latency {classification_latency:.1f}ms exceeds 200ms threshold"
        assert extraction_latency < 1500, \
            f"Extraction latency {extraction_latency:.1f}ms exceeds 1500ms threshold"
        assert routing_latency < 300, \
            f"Routing latency {routing_latency:.1f}ms exceeds 300ms threshold"
        assert total_latency < 2000, \
            f"Total latency {total_latency:.1f}ms exceeds 2000ms threshold"

        # Accuracy assertions
        assert classification_output["confidence"] >= 0.90, \
            "Classification confidence below 90%"
        assert quality_output["quality_score"] >= 0.85, \
            "Quality score below 85%"

        print(f"\n[SUCCESS] Invoice processing with REAL models completed")
        print(f"  - Total latency: {total_latency:.1f}ms")
        print(f"  - Classification: {classification_output['confidence']:.2%}")
        print(f"  - Quality: {quality_output['quality_score']:.2%}")
        print(f"  - Fields extracted: {extraction_coverage:.1%}")

    @pytest.mark.gpu
    @pytest.mark.slow
    async def test_purchase_order_real_inference(
        self,
        real_unified_model: UnifiedExtractorModel,
        sample_po_pdf: Path,
        config: Config
    ):
        """
        Test purchase order processing with real models.

        Validates that the pipeline works for different document types
        with the same real model infrastructure.
        """
        print("\n[Testing Purchase Order with Real Models]")

        # Similar structure to invoice test, but for purchase orders
        inbox_stage = InboxStage(config=config.stages.inbox)

        if not sample_po_pdf.exists():
            pytest.skip("Sample PO PDF not available")

        # Run through all stages (abbreviated for brevity)
        inbox_output = inbox_stage.process({"file_path": str(sample_po_pdf)})
        assert inbox_output["success"] is True

        preprocessing_stage = PreprocessingStage(config=config.stages.preprocessing)
        preprocessing_output = preprocessing_stage.process(inbox_output)

        classification_stage = ClassificationStage(
            model=real_unified_model.vision_encoder,
            config=config.stages.classification
        )
        classification_output = classification_stage.process(preprocessing_output)

        # Verify PO was classified correctly
        assert classification_output["document_type"] == "PURCHASE_ORDER" or \
               "purchase" in classification_output["document_type"].lower()

        print(f"✓ PO Classification: {classification_output['document_type']}")

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_model_performance_benchmarking(
        self,
        real_unified_model: UnifiedExtractorModel
    ):
        """
        Benchmark real model performance with multiple runs.

        Tests:
        - Average inference latency
        - P95 latency
        - Throughput
        - GPU utilization
        - Memory consumption
        """
        print("\n[Benchmarking Real Model Performance]")

        latencies = []
        gpu_memory_used = []

        # Run 100 inference iterations
        num_iterations = 100

        for i in range(num_iterations):
            start = time.time()

            # Simulate real inference with dummy inputs
            dummy_image = torch.randn(1, 3, 224, 224).cuda()
            with torch.no_grad():
                _ = real_unified_model.vision_encoder(dummy_image)

            latency = (time.time() - start) * 1000
            latencies.append(latency)

            # Measure GPU memory
            if i % 10 == 0:  # Sample every 10 iterations
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_used.append(memory_allocated)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations} iterations")

        # Calculate statistics
        mean_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[int(0.50 * len(latencies))]
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

        max_gpu_memory = max(gpu_memory_used) if gpu_memory_used else 0
        avg_gpu_memory = sum(gpu_memory_used) / len(gpu_memory_used) if gpu_memory_used else 0

        # Performance assertions
        assert p95_latency < 100, f"P95 latency {p95_latency:.1f}ms exceeds 100ms threshold"
        assert mean_latency < 50, f"Mean latency {mean_latency:.1f}ms exceeds 50ms threshold"
        assert max_gpu_memory < 20, f"GPU memory {max_gpu_memory:.1f}GB exceeds 20GB limit"

        print(f"\n[Performance Benchmark Results]")
        print(f"  - Mean latency: {mean_latency:.2f}ms")
        print(f"  - P50 latency: {p50_latency:.2f}ms")
        print(f"  - P95 latency: {p95_latency:.2f}ms")
        print(f"  - P99 latency: {p99_latency:.2f}ms")
        print(f"  - Throughput: {1000/mean_latency:.1f} inferences/sec")
        print(f"  - Avg GPU memory: {avg_gpu_memory:.2f}GB")
        print(f"  - Max GPU memory: {max_gpu_memory:.2f}GB")

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_batch_processing_real_models(
        self,
        real_unified_model: UnifiedExtractorModel
    ):
        """
        Test batch processing with real models.

        Validates that the system can handle batches efficiently
        and maintains performance under load.
        """
        print("\n[Testing Batch Processing with Real Models]")

        batch_sizes = [1, 4, 8, 16]
        results = {}

        for batch_size in batch_sizes:
            # Create dummy batch
            dummy_batch = torch.randn(batch_size, 3, 224, 224).cuda()

            # Measure batch processing time
            start = time.time()
            with torch.no_grad():
                _ = real_unified_model.vision_encoder(dummy_batch)
            batch_time = (time.time() - start) * 1000

            # Calculate per-item latency
            per_item_latency = batch_time / batch_size
            throughput = (batch_size / batch_time) * 1000  # items/sec

            results[batch_size] = {
                "batch_time": batch_time,
                "per_item_latency": per_item_latency,
                "throughput": throughput
            }

            print(f"  Batch size {batch_size:2d}: "
                  f"total={batch_time:.1f}ms, "
                  f"per-item={per_item_latency:.1f}ms, "
                  f"throughput={throughput:.1f} items/sec")

        # Verify batch processing is more efficient than sequential
        batch_16_per_item = results[16]["per_item_latency"]
        batch_1_latency = results[1]["batch_time"]

        efficiency_gain = (batch_1_latency - batch_16_per_item) / batch_1_latency

        print(f"\n  Batch efficiency gain: {efficiency_gain:.1%}")

        # Assert that batch processing provides at least 50% efficiency gain
        assert efficiency_gain >= 0.50, \
            f"Batch processing efficiency gain {efficiency_gain:.1%} below 50%"


class TestRealModelsAccuracy:
    """
    Accuracy tests with real models and ground truth data.

    These tests verify that extraction accuracy meets production requirements
    when using actual model weights.
    """

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_classification_accuracy_real_models(
        self,
        real_unified_model: UnifiedExtractorModel
    ):
        """
        Test classification accuracy with real models on labeled dataset.

        Target: ≥99% accuracy on document type classification
        """
        print("\n[Testing Classification Accuracy with Real Models]")

        # TODO: Load labeled test dataset with ground truth labels
        # For now, this is a placeholder for the test structure

        # Expected structure:
        # test_samples = load_labeled_test_set("classification")
        # correct_predictions = 0
        # total_predictions = 0
        #
        # for sample in test_samples:
        #     prediction = real_unified_model.classify(sample["image"])
        #     if prediction == sample["ground_truth_label"]:
        #         correct_predictions += 1
        #     total_predictions += 1
        #
        # accuracy = correct_predictions / total_predictions
        # assert accuracy >= 0.99, f"Classification accuracy {accuracy:.2%} below 99%"

        pytest.skip("Labeled test dataset not available - implement when dataset ready")

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_extraction_f1_score_real_models(
        self,
        real_unified_model: UnifiedExtractorModel
    ):
        """
        Test extraction F1 score with real models.

        Target: ≥97% F1 score on field extraction
        """
        print("\n[Testing Extraction F1 Score with Real Models]")

        # TODO: Implement F1 score calculation with ground truth
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # f1_score = 2 * (precision * recall) / (precision + recall)

        pytest.skip("Ground truth dataset not available - implement when dataset ready")


# Additional test utilities

def calculate_f1_score(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> float:
    """
    Calculate F1 score for field extraction.

    Args:
        predictions: List of predicted field extractions
        ground_truth: List of ground truth field values

    Returns:
        F1 score (0.0 to 1.0)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, truth in zip(predictions, ground_truth):
        for field_name, true_value in truth.items():
            pred_value = pred.get(field_name)

            if pred_value is not None and pred_value == true_value:
                true_positives += 1
            elif pred_value is not None and pred_value != true_value:
                false_positives += 1
            elif pred_value is None:
                false_negatives += 1

    if true_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


if __name__ == "__main__":
    """Run tests with GPU requirements."""
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--gpu",
        "-m", "gpu"
    ])
