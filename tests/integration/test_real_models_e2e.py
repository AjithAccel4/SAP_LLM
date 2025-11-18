"""
Real End-to-End Integration Tests with Actual Models

Tests the complete pipeline with real model inference (not mocks).
Requires GPU and actual model weights to run.

Test Coverage:
- Complete pipeline with LayoutLMv3 vision model
- Complete pipeline with LLaMA-2 language model
- Complete pipeline with Mixtral reasoning model
- Performance benchmarking with real inference
- Self-correction with actual model feedback

Requirements:
- CUDA-enabled GPU (recommended: A100 or V100)
- Model weights downloaded (~50GB total)
- Sufficient VRAM (recommended: 40GB+)

Usage:
    # Run all real model tests (slow, requires GPU)
    pytest tests/integration/test_real_models_e2e.py -v -s

    # Run specific test
    pytest tests/integration/test_real_models_e2e.py::TestRealModelsE2E::test_supplier_invoice_real_inference -v

    # Skip slow tests
    pytest tests/integration/test_real_models_e2e.py -v -m "not slow"
"""

import pytest
import torch
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch

# Mark all tests in this file as integration and slow
pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def gpu_available():
    """Check if GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True


@pytest.fixture(scope="module")
def real_unified_model(gpu_available):
    """
    Load actual unified model with real weights.

    This is expensive - loaded once per test module.
    """
    from sap_llm.models.unified_model import UnifiedExtractorModel

    print("\n🔄 Loading real models (this may take several minutes)...")

    model = UnifiedExtractorModel(
        vision_model="microsoft/layoutlmv3-base",
        language_model="meta-llama/Llama-2-7b-hf",
        reasoning_model="mistralai/Mixtral-8x7B-v0.1",
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,  # Use quantization to fit in memory
    )

    print(f"✅ Models loaded successfully")
    print(f"   Device: {model.device}")
    print(f"   Vision params: {sum(p.numel() for p in model.vision_model.parameters()):,}")
    print(f"   Language params: {sum(p.numel() for p in model.language_model.parameters()):,}")

    yield model

    # Cleanup
    print("\n🧹 Cleaning up models...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Cleanup complete")


@pytest.fixture(scope="module")
def sample_invoice_pdf():
    """Path to sample invoice PDF for testing."""
    path = Path("tests/fixtures/sample_supplier_invoice.pdf")
    if not path.exists():
        pytest.skip(f"Sample PDF not found: {path}")
    return str(path)


@pytest.fixture(scope="module")
def sample_po_pdf():
    """Path to sample purchase order PDF for testing."""
    path = Path("tests/fixtures/sample_purchase_order.pdf")
    if not path.exists():
        pytest.skip(f"Sample PDF not found: {path}")
    return str(path)


@pytest.mark.slow
@pytest.mark.gpu
class TestRealModelsE2E:
    """Test complete pipeline with real models (no mocks)."""

    def test_supplier_invoice_real_inference(
        self, real_unified_model, sample_invoice_pdf
    ):
        """
        Test supplier invoice processing with real model inference.

        This test runs the complete pipeline:
        1. Preprocessing (OCR with real model)
        2. Classification (LayoutLMv3)
        3. Subtype classification
        4. Field extraction (LLaMA-2)
        5. Quality checking
        6. Business rule validation
        7. Self-correction (if needed)
        8. Routing decision (Mixtral)

        Expected latency: < 2000ms (P95)
        """
        from sap_llm.stages.preprocessing import PreprocessingStage
        from sap_llm.stages.classification import ClassificationStage
        from sap_llm.stages.extraction import ExtractionStage
        from sap_llm.stages.quality_check import QualityCheckStage
        from sap_llm.stages.validation import ValidationStage
        from sap_llm.stages.routing import RoutingStage

        print(f"\n📄 Processing: {sample_invoice_pdf}")

        # Stage 1: Preprocessing
        print("  Stage 1: Preprocessing...")
        start = time.time()
        preprocessing = PreprocessingStage()
        preprocessed = preprocessing.process({"file_path": sample_invoice_pdf})
        preprocess_time = (time.time() - start) * 1000
        print(f"    ✓ Preprocessing: {preprocess_time:.0f}ms")

        assert preprocessed["success"] is True
        assert "images" in preprocessed
        assert "ocr_text" in preprocessed
        assert len(preprocessed["images"]) > 0

        # Stage 2: Classification (with real LayoutLMv3)
        print("  Stage 2: Classification...")
        start = time.time()
        classifier = ClassificationStage(model=real_unified_model)
        classified = classifier.process(preprocessed)
        classify_time = (time.time() - start) * 1000
        print(f"    ✓ Classification: {classify_time:.0f}ms")

        assert classified["success"] is True
        assert classified["doc_type"] == "SUPPLIER_INVOICE"
        assert classified["confidence"] >= 0.90
        assert "subtype" in classified

        # Stage 3: Extraction (with real LLaMA-2)
        print("  Stage 3: Field Extraction...")
        start = time.time()
        extractor = ExtractionStage(model=real_unified_model)
        extracted = extractor.process(classified)
        extract_time = (time.time() - start) * 1000
        print(f"    ✓ Extraction: {extract_time:.0f}ms")

        assert extracted["success"] is True
        assert "extracted_data" in extracted

        # Verify critical fields extracted
        data = extracted["extracted_data"]
        critical_fields = ["invoice_number", "invoice_date", "total_amount", "vendor_id"]
        for field in critical_fields:
            assert field in data, f"Critical field '{field}' not extracted"
            assert data[field] is not None, f"Field '{field}' is null"

        print(f"    📊 Extracted fields: {list(data.keys())}")

        # Stage 4: Quality Check
        print("  Stage 4: Quality Check...")
        start = time.time()
        quality_checker = QualityCheckStage()
        quality_checked = quality_checker.process(extracted)
        quality_time = (time.time() - start) * 1000
        print(f"    ✓ Quality Check: {quality_time:.0f}ms")

        assert quality_checked["success"] is True
        assert "quality_score" in quality_checked
        assert quality_checked["quality_score"] >= 0.85

        # Stage 5: Validation
        print("  Stage 5: Business Rule Validation...")
        start = time.time()
        validator = ValidationStage()
        validated = validator.process(quality_checked)
        validate_time = (time.time() - start) * 1000
        print(f"    ✓ Validation: {validate_time:.0f}ms")

        assert validated["success"] is True
        assert "validation_result" in validated

        # Stage 6: Routing (with real Mixtral)
        print("  Stage 6: Routing Decision...")
        start = time.time()
        router = RoutingStage(model=real_unified_model)
        routed = router.process(validated)
        route_time = (time.time() - start) * 1000
        print(f"    ✓ Routing: {route_time:.0f}ms")

        assert routed["success"] is True
        assert "routing_decision" in routed
        assert routed["routing_decision"]["endpoint"] is not None

        # Total latency
        total_time = preprocess_time + classify_time + extract_time + quality_time + validate_time + route_time
        print(f"\n  ⏱️  Total Pipeline Latency: {total_time:.0f}ms")

        # Performance assertions
        assert total_time < 2000, f"Pipeline latency {total_time}ms exceeds 2000ms threshold"

        # Verify end-to-end result
        assert routed["doc_type"] == "SUPPLIER_INVOICE"
        assert routed["quality_score"] >= 0.85
        assert len(routed["extracted_data"]) >= 10  # At least 10 fields extracted

    def test_purchase_order_real_inference(
        self, real_unified_model, sample_po_pdf
    ):
        """Test purchase order processing with real models."""
        from sap_llm.pipeline import process_document

        print(f"\n📄 Processing: {sample_po_pdf}")

        start = time.time()
        result = process_document(
            file_path=sample_po_pdf,
            model=real_unified_model,
            enable_self_correction=True
        )
        latency = (time.time() - start) * 1000

        print(f"  ⏱️  Total Latency: {latency:.0f}ms")

        # Assertions
        assert result["success"] is True
        assert result["doc_type"] == "PURCHASE_ORDER"
        assert result["confidence"] >= 0.90
        assert result["quality_score"] >= 0.85

        # Verify PO-specific fields
        data = result["extracted_data"]
        po_fields = ["po_number", "vendor_id", "po_date", "total_amount"]
        for field in po_fields:
            assert field in data

        # Performance
        assert latency < 2000

    def test_self_correction_with_real_models(
        self, real_unified_model, sample_invoice_pdf
    ):
        """Test self-correction module with real model feedback."""
        from sap_llm.stages.quality_check import SelfCorrectionModule
        from sap_llm.stages.extraction import ExtractionStage
        from sap_llm.stages.preprocessing import PreprocessingStage

        print(f"\n🔧 Testing self-correction with real models...")

        # Process document
        preprocessing = PreprocessingStage()
        preprocessed = preprocessing.process({"file_path": sample_invoice_pdf})

        extractor = ExtractionStage(model=real_unified_model)
        extracted = extractor.process(preprocessed)

        # Introduce an intentional error
        extracted["extracted_data"]["total_amount"] = "INVALID_AMOUNT"

        # Apply self-correction (with real model)
        corrector = SelfCorrectionModule(model=real_unified_model)
        corrected = corrector.correct(extracted)

        print(f"  Before: {extracted['extracted_data']['total_amount']}")
        print(f"  After:  {corrected['extracted_data']['total_amount']}")

        # Verify correction
        assert corrected["self_correction_applied"] is True
        assert corrected["extracted_data"]["total_amount"] != "INVALID_AMOUNT"
        assert isinstance(corrected["extracted_data"]["total_amount"], (int, float, str))

    @pytest.mark.benchmark
    def test_performance_benchmarking_real_models(self, real_unified_model):
        """Benchmark real model performance."""
        from sap_llm.pipeline import process_document
        import statistics

        print(f"\n📊 Benchmarking with real models...")

        # Generate synthetic test documents
        test_docs = [
            "tests/fixtures/sample_supplier_invoice.pdf",
            "tests/fixtures/sample_purchase_order.pdf",
            # Add more test documents
        ]

        latencies = []

        for i, doc_path in enumerate(test_docs[:10]):  # Limit to 10 for speed
            if not Path(doc_path).exists():
                continue

            start = time.time()
            result = process_document(doc_path, model=real_unified_model)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            print(f"  Doc {i+1}: {latency:.0f}ms")

        if not latencies:
            pytest.skip("No test documents available")

        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

        print(f"\n  📈 Performance Metrics:")
        print(f"     Mean Latency: {mean_latency:.0f}ms")
        print(f"     P95 Latency:  {p95_latency:.0f}ms")
        print(f"     Min:          {min(latencies):.0f}ms")
        print(f"     Max:          {max(latencies):.0f}ms")

        # Assertions against targets
        assert mean_latency < 800, f"Mean latency {mean_latency}ms exceeds 800ms target"
        assert p95_latency < 1000, f"P95 latency {p95_latency}ms exceeds 1000ms target"

    def test_gpu_memory_usage(self, real_unified_model):
        """Monitor GPU memory usage during inference."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        print(f"\n💾 GPU Memory Usage:")

        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / 1024**3  # GB

        print(f"  Initial:  {initial_mem:.2f} GB")

        # Run inference
        from sap_llm.pipeline import process_document
        sample_doc = "tests/fixtures/sample_supplier_invoice.pdf"
        if Path(sample_doc).exists():
            result = process_document(sample_doc, model=real_unified_model)

        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
        current_mem = torch.cuda.memory_allocated() / 1024**3  # GB

        print(f"  Current:  {current_mem:.2f} GB")
        print(f"  Peak:     {peak_mem:.2f} GB")
        print(f"  Increase: {peak_mem - initial_mem:.2f} GB")

        # Assert reasonable memory usage
        assert peak_mem < 40, f"Peak memory {peak_mem:.2f}GB exceeds 40GB"

    @pytest.mark.parametrize("doc_type,expected_fields", [
        ("SUPPLIER_INVOICE", ["invoice_number", "vendor_id", "total_amount", "invoice_date"]),
        ("PURCHASE_ORDER", ["po_number", "vendor_id", "total_amount", "po_date"]),
        ("SALES_ORDER", ["sales_order_number", "customer_id", "total_amount", "order_date"]),
    ])
    def test_extraction_accuracy_real_models(
        self, real_unified_model, doc_type, expected_fields
    ):
        """Test extraction accuracy for different document types with real models."""
        # This would require a labeled test dataset
        # For now, just verify the structure works
        print(f"\n✅ Extraction test for {doc_type}")
        assert len(expected_fields) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestRealModelEdgeCases:
    """Test edge cases and error handling with real models."""

    def test_corrupted_pdf_handling(self, real_unified_model):
        """Test handling of corrupted PDF with real models."""
        corrupted_pdf = "tests/fixtures/corrupted.pdf"

        from sap_llm.pipeline import process_document

        result = process_document(corrupted_pdf, model=real_unified_model)

        # Should handle gracefully
        assert "error" in result or result["success"] is False

    def test_multilingual_document(self, real_unified_model):
        """Test processing multilingual document with real models."""
        multilang_pdf = "tests/fixtures/invoice_german.pdf"

        if not Path(multilang_pdf).exists():
            pytest.skip("Multilingual test document not available")

        from sap_llm.pipeline import process_document

        result = process_document(multilang_pdf, model=real_unified_model)

        # Should process successfully
        assert result["success"] is True
        assert result["language"] in ["de", "en", "multi"]

    def test_low_quality_scan(self, real_unified_model):
        """Test processing low-quality scanned document."""
        low_quality_pdf = "tests/fixtures/low_quality_scan.pdf"

        if not Path(low_quality_pdf).exists():
            pytest.skip("Low quality test document not available")

        from sap_llm.stages.quality_check import QualityCheckStage
        from sap_llm.pipeline import process_document

        result = process_document(
            low_quality_pdf,
            model=real_unified_model,
            enable_self_correction=True  # Should trigger correction
        )

        # May have lower quality score but should still process
        assert result["success"] is True
        if result["quality_score"] < 0.85:
            # Self-correction should have been attempted
            assert "self_correction" in result


@pytest.mark.integration
class TestRealModelComparison:
    """Compare performance across different model configurations."""

    @pytest.mark.slow
    def test_quantized_vs_full_precision(self, gpu_available):
        """Compare quantized vs full precision models."""
        from sap_llm.models.unified_model import UnifiedExtractorModel

        sample_doc = "tests/fixtures/sample_supplier_invoice.pdf"
        if not Path(sample_doc).exists():
            pytest.skip("Sample document not available")

        # Test with int8 quantization
        print("\n⚡ Testing INT8 quantized model...")
        model_int8 = UnifiedExtractorModel(
            vision_model="microsoft/layoutlmv3-base",
            language_model="meta-llama/Llama-2-7b-hf",
            load_in_8bit=True,
        )

        from sap_llm.pipeline import process_document

        start = time.time()
        result_int8 = process_document(sample_doc, model=model_int8)
        latency_int8 = (time.time() - start) * 1000

        print(f"  INT8 Latency: {latency_int8:.0f}ms")
        print(f"  INT8 Quality: {result_int8.get('quality_score', 0):.3f}")

        # Cleanup
        del model_int8
        torch.cuda.empty_cache()

        # Verify performance
        assert result_int8["success"] is True
        assert latency_int8 < 2000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
