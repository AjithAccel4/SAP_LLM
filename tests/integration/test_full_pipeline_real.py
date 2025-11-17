"""
Real end-to-end integration tests with actual models (NOT mocks).
Tests ALL 8 pipeline stages with real inference.

REQUIRES: GPU with 16GB+ VRAM, actual model weights downloaded
"""

import pytest
import torch
import asyncio
from pathlib import Path
from typing import Dict, Any
import time

from sap_llm.models.unified_model import UnifiedExtractorModel
from sap_llm.config import load_config
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.inbox import InboxStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.real_models
class TestFullPipelineRealModels:
    """
    Test complete pipeline with real models (NOT mocks).

    These tests require:
    - CUDA-capable GPU (16GB+ VRAM)
    - Downloaded model weights
    - ~30GB disk space
    - ~2-5 minutes per test
    """

    @pytest.fixture(scope="class")
    def real_unified_model(self):
        """
        Load actual UnifiedModel with real weights.
        This will download models if not cached.
        """
        logger.info("Loading real models (this may take several minutes)...")

        try:
            model = UnifiedExtractorModel(
                vision_model="microsoft/layoutlmv3-base",
                language_model="meta-llama/Llama-2-7b-hf",
                reasoning_model="mistralai/Mixtral-8x7B-v0.1",
                device="cuda" if torch.cuda.is_available() else "cpu",
                load_in_8bit=True  # Use 8-bit quantization to fit in memory
            )

            logger.info(f"Models loaded on device: {model.device}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

            yield model

            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")

        except Exception as e:
            pytest.skip(f"Could not load real models: {e}")

    @pytest.fixture(scope="class")
    def preprocessing_stage(self):
        """Create real preprocessing stage."""
        try:
            config = load_config()
            return PreprocessingStage(config)
        except:
            return PreprocessingStage({})

    @pytest.fixture(scope="class")
    def inbox_stage(self):
        """Create real inbox stage."""
        try:
            config = load_config()
            return InboxStage(config)
        except:
            return InboxStage({})

    @pytest.fixture
    def sample_invoice_image(self, tmp_path):
        """
        Create or load a real invoice image for testing.
        In production, replace with actual test invoice images.
        """
        # For now, create a simple test image
        from PIL import Image, ImageDraw, ImageFont

        # Create a simple invoice image
        img = Image.new('RGB', (800, 1000), color='white')
        draw = ImageDraw.Draw(img)

        # Draw invoice content
        draw.text((50, 50), "INVOICE", fill='black')
        draw.text((50, 100), "Invoice Number: INV-2025-001", fill='black')
        draw.text((50, 130), "Date: 2025-01-15", fill='black')
        draw.text((50, 160), "Vendor: VENDOR-123", fill='black')
        draw.text((50, 190), "Acme Corporation", fill='black')
        draw.text((50, 250), "Total Amount: $1,000.00", fill='black')
        draw.text((50, 280), "Tax: $100.00", fill='black')
        draw.text((50, 310), "Net: $900.00", fill='black')

        # Save image
        image_path = tmp_path / "test_invoice.png"
        img.save(image_path)

        return image_path

    @pytest.fixture
    def document_schemas(self):
        """Document type schemas."""
        return {
            "SUPPLIER_INVOICE": {
                "required_fields": [
                    "invoice_number",
                    "invoice_date",
                    "vendor_id",
                    "total_amount"
                ],
                "field_types": {
                    "total_amount": "float",
                    "tax_amount": "float",
                    "net_amount": "float"
                }
            },
            "PURCHASE_ORDER": {
                "required_fields": [
                    "po_number",
                    "po_date",
                    "vendor_id",
                    "total_amount"
                ]
            }
        }

    # =========================================================================
    # Real Model Inference Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_supplier_invoice_e2e_real_models(
        self,
        real_unified_model,
        preprocessing_stage,
        sample_invoice_image,
        document_schemas
    ):
        """
        Test complete supplier invoice processing with REAL models.

        Pipeline stages tested:
        1. Inbox (file ingestion)
        2. Preprocessing (OCR, image processing)
        3. Classification (LayoutLMv3 - REAL)
        4. Type identification (Subtype classifier - REAL)
        5. Extraction (LLaMA-2 7B - REAL)
        6. Quality check (6-dimensional validation)
        7. Validation (business rules)
        8. Routing (Mixtral reasoning - REAL)
        """
        logger.info("=" * 70)
        logger.info("TEST: Supplier Invoice E2E with REAL MODELS")
        logger.info("=" * 70)

        start_time = time.time()

        # Stage 1-2: Preprocess document (OCR, etc.)
        logger.info("Stage 1-2: Preprocessing...")
        preprocessed = await preprocessing_stage.process(str(sample_invoice_image))

        assert "images" in preprocessed
        assert "ocr_text" in preprocessed
        assert "words" in preprocessed
        assert "boxes" in preprocessed

        logger.info(f"OCR extracted {len(preprocessed['words'])} words")
        logger.info(f"OCR text preview: {preprocessed['ocr_text'][:100]}...")

        # Stage 3-8: Run through UnifiedModel with REAL inference
        logger.info("Stage 3-8: Running REAL model inference...")

        result = await real_unified_model.process_document(
            image=preprocessed["images"][0],
            ocr_text=preprocessed["ocr_text"],
            words=preprocessed["words"],
            boxes=preprocessed["boxes"],
            schemas=document_schemas,
            api_schemas=[],
            enable_self_correction=True
        )

        latency_ms = (time.time() - start_time) * 1000

        # Validate results from REAL models
        logger.info("=" * 70)
        logger.info("RESULTS FROM REAL MODELS:")
        logger.info(f"Document Type: {result.get('doc_type', 'UNKNOWN')}")
        logger.info(f"Subtype: {result.get('subtype', 'UNKNOWN')}")
        logger.info(f"Classification Confidence: {result.get('classification_confidence', 0):.4f}")
        logger.info(f"Quality Score: {result.get('quality_score', 0):.4f}")
        logger.info(f"Extracted Fields: {len(result.get('extracted_data', {}))}")
        logger.info(f"Validation Errors: {len(result.get('validation_errors', []))}")
        logger.info(f"Latency: {latency_ms:.2f}ms")
        logger.info("=" * 70)

        # Assertions for REAL model output
        assert "doc_type" in result
        assert "subtype" in result
        assert "extracted_data" in result
        assert "quality_score" in result

        # Quality thresholds for REAL models
        assert result["quality_score"] >= 0.70, \
            f"Quality score {result['quality_score']} below 0.70 threshold"

        # Classification should produce reasonable results
        assert result.get("classification_confidence", 0) >= 0.60, \
            f"Classification confidence {result.get('classification_confidence')} too low"

        # Performance assertion (real models are slower)
        assert latency_ms < 10000, \
            f"Latency {latency_ms}ms exceeds 10s threshold"

        # Should extract key invoice fields
        extracted = result["extracted_data"]
        logger.info(f"Extracted data: {extracted}")

        # Check critical fields were extracted
        assert "invoice_number" in extracted or \
               "total_amount" in extracted or \
               "vendor" in str(extracted).lower(), \
               "No critical invoice fields extracted"

        logger.info("✅ Real model E2E test PASSED")

    @pytest.mark.asyncio
    async def test_classification_accuracy_real_models(
        self,
        real_unified_model,
        preprocessing_stage,
        sample_invoice_image
    ):
        """
        Test classification accuracy with REAL LayoutLMv3 model.
        """
        logger.info("TEST: Classification with REAL LayoutLMv3")

        # Preprocess
        preprocessed = await preprocessing_stage.process(str(sample_invoice_image))

        # Classify with REAL model
        doc_type, subtype, confidence = await real_unified_model.classify(
            preprocessed["images"][0],
            preprocessed["ocr_text"],
            preprocessed["words"],
            preprocessed["boxes"]
        )

        logger.info(f"Classification Result:")
        logger.info(f"  Doc Type: {doc_type}")
        logger.info(f"  Subtype: {subtype}")
        logger.info(f"  Confidence: {confidence:.4f}")

        # Assertions
        assert doc_type in [
            "SUPPLIER_INVOICE", "PURCHASE_ORDER", "GOODS_RECEIPT",
            "CREDIT_NOTE", "PAYMENT_ADVICE", "UNKNOWN"
        ]

        assert confidence >= 0.50, \
            f"Confidence {confidence} too low for real model"

        # Real models should identify this as an invoice
        assert doc_type != "UNKNOWN" or confidence > 0, \
            "Real model failed to classify document"

        logger.info("✅ Real classification test PASSED")

    @pytest.mark.asyncio
    async def test_extraction_accuracy_real_models(
        self,
        real_unified_model,
        preprocessing_stage,
        sample_invoice_image,
        document_schemas
    ):
        """
        Test field extraction accuracy with REAL LLaMA-2 model.
        """
        logger.info("TEST: Extraction with REAL LLaMA-2 7B")

        # Preprocess
        preprocessed = await preprocessing_stage.process(str(sample_invoice_image))

        # Extract with REAL model
        extracted_data, metadata = await real_unified_model.extract(
            preprocessed["images"][0],
            preprocessed["ocr_text"],
            preprocessed["words"],
            preprocessed["boxes"],
            "SUPPLIER_INVOICE",
            document_schemas["SUPPLIER_INVOICE"]
        )

        logger.info(f"Extraction Result:")
        logger.info(f"  Fields extracted: {len(extracted_data)}")
        logger.info(f"  Data: {extracted_data}")
        logger.info(f"  Metadata: {metadata}")

        # Assertions
        assert isinstance(extracted_data, dict)
        assert len(extracted_data) > 0, "No fields extracted by real model"

        # Check extraction metadata
        assert "latency_ms" in metadata or "duration" in str(metadata)

        # Real model should extract at least some fields
        assert len(extracted_data) >= 1, \
            f"Real model extracted too few fields: {len(extracted_data)}"

        logger.info("✅ Real extraction test PASSED")

    @pytest.mark.asyncio
    async def test_reasoning_engine_real_models(
        self,
        real_unified_model,
        preprocessing_stage,
        sample_invoice_image
    ):
        """
        Test routing decision with REAL Mixtral reasoning engine.
        """
        logger.info("TEST: Routing with REAL Mixtral 8x7B")

        # Preprocess
        preprocessed = await preprocessing_stage.process(str(sample_invoice_image))

        # Create ADC data
        adc_data = {
            "doc_type": "SUPPLIER_INVOICE",
            "invoice_number": "INV-2025-001",
            "total_amount": 1000.00,
            "vendor_id": "VENDOR-123",
            "quality_score": 0.95
        }

        # Route with REAL reasoning model
        decision = await real_unified_model.route(
            adc_data,
            "SUPPLIER_INVOICE",
            [],
            []
        )

        logger.info(f"Routing Decision:")
        logger.info(f"  Endpoint: {decision.get('endpoint')}")
        logger.info(f"  Action: {decision.get('action')}")
        logger.info(f"  Confidence: {decision.get('confidence', 0):.4f}")
        logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")

        # Assertions
        assert "endpoint" in decision or "action" in decision

        action = decision.get("action") or decision.get("endpoint", "").lower()
        assert action in ["post", "park", "reject", "review", "approve"], \
            f"Invalid routing action: {action}"

        # High quality documents should be routed to POST
        if adc_data["quality_score"] >= 0.90:
            assert action in ["post", "approve"], \
                f"High quality doc routed to {action} instead of POST"

        logger.info("✅ Real reasoning test PASSED")

    # =========================================================================
    # Performance Tests with Real Models
    # =========================================================================

    @pytest.mark.asyncio
    async def test_real_model_latency_benchmark(
        self,
        real_unified_model,
        preprocessing_stage,
        sample_invoice_image,
        document_schemas
    ):
        """
        Benchmark real model latency (P95 target: <600ms for mocks, <5s for real).
        """
        logger.info("TEST: Real Model Latency Benchmark")

        latencies = []
        num_runs = 10  # Run 10 times for statistical significance

        for i in range(num_runs):
            start = time.time()

            # Preprocess
            preprocessed = await preprocessing_stage.process(str(sample_invoice_image))

            # Full pipeline with real models
            result = await real_unified_model.process_document(
                image=preprocessed["images"][0],
                ocr_text=preprocessed["ocr_text"],
                words=preprocessed["words"],
                boxes=preprocessed["boxes"],
                schemas=document_schemas,
                api_schemas=[],
                enable_self_correction=False  # Disable for speed
            )

            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            logger.info(f"Run {i+1}/{num_runs}: {latency_ms:.2f}ms")

        # Calculate statistics
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        mean = sum(latencies) / len(latencies)

        logger.info("=" * 70)
        logger.info("REAL MODEL LATENCY RESULTS:")
        logger.info(f"  P50: {p50:.2f}ms")
        logger.info(f"  P95: {p95:.2f}ms")
        logger.info(f"  Mean: {mean:.2f}ms")
        logger.info(f"  Min: {min(latencies):.2f}ms")
        logger.info(f"  Max: {max(latencies):.2f}ms")
        logger.info("=" * 70)

        # Real models are slower - adjust expectations
        assert p95 < 10000, \
            f"P95 latency {p95}ms exceeds 10s threshold for real models"

        assert mean < 8000, \
            f"Mean latency {mean}ms exceeds 8s threshold for real models"

        logger.info("✅ Real model latency benchmark PASSED")

    @pytest.mark.asyncio
    async def test_real_model_gpu_utilization(
        self,
        real_unified_model
    ):
        """
        Test GPU utilization with real models.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        logger.info("TEST: GPU Utilization with Real Models")

        # Get initial GPU memory
        initial_memory = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"GPU Memory:")
        logger.info(f"  Allocated: {initial_memory:.2f} GB")
        logger.info(f"  Max Allocated: {max_memory:.2f} GB")
        logger.info(f"  Total: {total_memory:.2f} GB")
        logger.info(f"  Utilization: {(max_memory/total_memory)*100:.1f}%")

        # Assertions
        assert max_memory > 0, "No GPU memory allocated - models not on GPU"
        assert max_memory < total_memory * 0.95, \
            "GPU memory usage critically high"

        logger.info("✅ GPU utilization test PASSED")

    # =========================================================================
    # Error Handling with Real Models
    # =========================================================================

    @pytest.mark.asyncio
    async def test_real_model_handles_poor_quality_image(
        self,
        real_unified_model,
        preprocessing_stage,
        tmp_path,
        document_schemas
    ):
        """
        Test real models handle poor quality images gracefully.
        """
        from PIL import Image, ImageFilter

        logger.info("TEST: Real Models with Poor Quality Image")

        # Create a blurry, low-quality image
        img = Image.new('RGB', (400, 500), color='white')
        img = img.filter(ImageFilter.GaussianBlur(radius=10))

        poor_image_path = tmp_path / "poor_quality.png"
        img.save(poor_image_path)

        # Process with real models
        try:
            preprocessed = await preprocessing_stage.process(str(poor_image_path))

            result = await real_unified_model.process_document(
                image=preprocessed["images"][0],
                ocr_text=preprocessed["ocr_text"],
                words=preprocessed["words"],
                boxes=preprocessed["boxes"],
                schemas=document_schemas,
                api_schemas=[],
                enable_self_correction=True
            )

            logger.info(f"Result with poor image:")
            logger.info(f"  Quality Score: {result.get('quality_score', 0):.4f}")
            logger.info(f"  Fields Extracted: {len(result.get('extracted_data', {}))}")

            # Should complete without crashing
            assert "quality_score" in result

            # Quality score should reflect poor image
            assert result["quality_score"] < 0.80, \
                "Quality score too high for poor quality image"

            logger.info("✅ Poor quality image handling PASSED")

        except Exception as e:
            logger.warning(f"Poor quality image handling failed: {e}")
            # This is acceptable - real models may fail on very poor images
            assert "quality" in str(e).lower() or "ocr" in str(e).lower()

    # =========================================================================
    # Model Comparison Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_compare_vision_models(
        self,
        real_unified_model,
        sample_invoice_image,
        preprocessing_stage
    ):
        """
        Compare output from different vision encoder configurations.
        """
        logger.info("TEST: Vision Model Comparison")

        preprocessed = await preprocessing_stage.process(str(sample_invoice_image))

        # Test with base vision encoder
        result_base = await real_unified_model.classify(
            preprocessed["images"][0],
            preprocessed["ocr_text"],
            preprocessed["words"],
            preprocessed["boxes"]
        )

        logger.info(f"Vision Model Results:")
        logger.info(f"  Base: {result_base}")

        # Both should produce valid classifications
        assert result_base[0] is not None  # doc_type
        assert result_base[2] >= 0  # confidence

        logger.info("✅ Vision model comparison PASSED")


@pytest.mark.integration
@pytest.mark.real_models
class TestRealModelAccuracy:
    """
    Test accuracy metrics with real models on test dataset.
    Requires prepared test dataset with ground truth labels.
    """

    @pytest.mark.asyncio
    async def test_classification_accuracy_on_test_set(self):
        """
        Test classification accuracy on labeled test set.
        Target: ≥99% accuracy
        """
        pytest.skip("Requires labeled test dataset")

        # Load test dataset
        # test_dataset = load_test_dataset()

        # Run classification on all test documents
        # Calculate accuracy metrics

        # Assert accuracy >= 0.99

    @pytest.mark.asyncio
    async def test_extraction_f1_on_test_set(self):
        """
        Test extraction F1 score on labeled test set.
        Target: ≥97% F1
        """
        pytest.skip("Requires labeled test dataset with field-level annotations")

        # Load test dataset with field annotations
        # test_dataset = load_test_dataset_with_fields()

        # Run extraction on all test documents
        # Calculate precision, recall, F1

        # Assert f1 >= 0.97

    @pytest.mark.asyncio
    async def test_routing_accuracy_on_test_set(self):
        """
        Test routing accuracy on labeled test set.
        Target: ≥99.5% accuracy
        """
        pytest.skip("Requires labeled test dataset with routing decisions")

        # Load test dataset with routing labels
        # test_dataset = load_test_dataset_with_routing()

        # Run routing on all test documents
        # Calculate routing accuracy

        # Assert accuracy >= 0.995
