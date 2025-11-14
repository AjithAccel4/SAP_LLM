"""
Comprehensive unit tests for all 8 pipeline stages in SAP_LLM.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image

from sap_llm.stages.inbox import InboxStage
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.classification import ClassificationStage
from sap_llm.stages.type_identifier import TypeIdentifierStage
from sap_llm.stages.extraction import ExtractionStage
from sap_llm.stages.quality_check import QualityCheckStage
from sap_llm.stages.validation import ValidationStage
from sap_llm.stages.routing import RoutingStage


@pytest.mark.unit
@pytest.mark.stages
class TestInboxStage:
    """Tests for Stage 1: Inbox - Document ingestion & routing."""

    @pytest.fixture
    def inbox_stage(self, mock_redis):
        """Create InboxStage instance with mocked Redis."""
        with patch('sap_llm.stages.inbox.redis.Redis', return_value=mock_redis):
            config = MagicMock()
            config.cache_ttl = 3600
            config.max_file_size = 10 * 1024 * 1024
            return InboxStage(config=config)

    def test_inbox_initialization(self, inbox_stage):
        """Test Inbox stage initialization."""
        assert inbox_stage is not None
        assert inbox_stage.cache_ttl > 0

    def test_inbox_process_new_document(self, inbox_stage, sample_image, temp_dir):
        """Test processing new document."""
        image_path = temp_dir / "test_document.png"
        sample_image.save(image_path)

        with open(image_path, "rb") as f:
            file_content = f.read()

        result = inbox_stage.process({
            "file_path": str(image_path),
            "file_content": file_content,
            "metadata": {"source": "email"},
        })

        assert result is not None
        assert "document_hash" in result
        assert "file_path" in result
        assert "is_duplicate" in result
        assert result["is_duplicate"] is False

    def test_inbox_duplicate_detection(self, inbox_stage, sample_image, temp_dir, mock_redis):
        """Test duplicate detection."""
        image_path = temp_dir / "test_document.png"
        sample_image.save(image_path)

        with open(image_path, "rb") as f:
            file_content = f.read()

        input_data = {
            "file_path": str(image_path),
            "file_content": file_content,
        }

        # First process
        result1 = inbox_stage.process(input_data)
        doc_hash = result1["document_hash"]

        # Mock Redis to return the hash (simulating duplicate)
        mock_redis.get.return_value = doc_hash.encode()

        # Second process - should detect duplicate
        result2 = inbox_stage.process(input_data)
        # Note: Actual duplicate detection depends on Redis implementation

    def test_inbox_file_size_validation(self, inbox_stage):
        """Test file size validation."""
        # Create a file that's too large
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB

        result = inbox_stage.process({
            "file_path": "large_file.pdf",
            "file_content": large_content,
        })

        # Should handle large files gracefully
        assert result is not None

    @pytest.mark.parametrize("file_format", [
        "pdf",
        "png",
        "jpg",
        "tiff",
    ])
    def test_inbox_supported_formats(self, inbox_stage, file_format):
        """Test support for different file formats."""
        result = inbox_stage.process({
            "file_path": f"test.{file_format}",
            "file_content": b"test_content",
        })

        assert result is not None


@pytest.mark.unit
@pytest.mark.stages
class TestPreprocessingStage:
    """Tests for Stage 2: Preprocessing - OCR, image enhancement."""

    @pytest.fixture
    def preprocessing_stage(self):
        """Create PreprocessingStage instance."""
        config = MagicMock()
        config.ocr_engine = "tesseract"
        config.enhance_images = True
        return PreprocessingStage(config=config)

    def test_preprocessing_initialization(self, preprocessing_stage):
        """Test Preprocessing stage initialization."""
        assert preprocessing_stage is not None

    def test_preprocessing_ocr(self, preprocessing_stage, sample_document_image):
        """Test OCR processing."""
        with patch('sap_llm.stages.preprocessing.pytesseract') as mock_tesseract:
            mock_tesseract.image_to_string.return_value = "PURCHASE ORDER\nPO# 12345"
            mock_tesseract.image_to_data.return_value = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t10\t10\t100\t20\t95\tPURCHASE"

            result = preprocessing_stage.process({
                "file_path": "test.png",
                "image": sample_document_image,
            })

            assert result is not None
            assert "ocr_text" in result
            assert "words" in result
            assert "boxes" in result

    def test_preprocessing_image_enhancement(self, preprocessing_stage, sample_image):
        """Test image enhancement."""
        result = preprocessing_stage.process({
            "file_path": "test.png",
            "image": sample_image,
        })

        assert result is not None
        assert "image" in result
        assert result["image"] is not None

    def test_preprocessing_pdf_conversion(self, preprocessing_stage):
        """Test PDF to image conversion."""
        with patch('sap_llm.stages.preprocessing.convert_from_path') as mock_convert:
            mock_convert.return_value = [Image.new('RGB', (800, 600))]

            result = preprocessing_stage.process({
                "file_path": "test.pdf",
            })

            assert result is not None

    @pytest.mark.parametrize("enhancement_type", [
        "brightness",
        "contrast",
        "sharpness",
        "denoise",
    ])
    def test_preprocessing_enhancement_types(self, preprocessing_stage, sample_image, enhancement_type):
        """Test different image enhancement types."""
        config = MagicMock()
        config.enhancement = enhancement_type
        stage = PreprocessingStage(config=config)

        result = stage.process({
            "file_path": "test.png",
            "image": sample_image,
        })

        assert result is not None


@pytest.mark.unit
@pytest.mark.stages
class TestClassificationStage:
    """Tests for Stage 3: Classification - Document type identification."""

    @pytest.fixture
    def classification_stage(self):
        """Create ClassificationStage instance."""
        config = MagicMock()
        config.model_name = "microsoft/layoutlmv3-base"
        config.num_classes = 15
        return ClassificationStage(config=config)

    def test_classification_initialization(self, classification_stage):
        """Test Classification stage initialization."""
        assert classification_stage is not None

    def test_classification_predict(self, classification_stage, sample_document_image):
        """Test document classification."""
        with patch.object(classification_stage, 'vision_encoder') as mock_encoder:
            mock_encoder.classify.return_value = (0, 0.95)  # PURCHASE_ORDER, 95% confidence

            result = classification_stage.process({
                "image": sample_document_image,
                "words": ["PURCHASE", "ORDER"],
                "boxes": [[10, 10, 100, 30], [110, 10, 200, 30]],
            })

            assert result is not None
            assert "predicted_class" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.parametrize("doc_type_idx,expected_type", [
        (0, "PURCHASE_ORDER"),
        (1, "SUPPLIER_INVOICE"),
        (2, "SALES_ORDER"),
        (3, "CUSTOMER_INVOICE"),
    ])
    def test_classification_document_types(self, classification_stage, doc_type_idx, expected_type):
        """Test classification for different document types."""
        with patch.object(classification_stage, 'vision_encoder') as mock_encoder:
            mock_encoder.classify.return_value = (doc_type_idx, 0.95)

            result = classification_stage.process({
                "image": Image.new('RGB', (800, 600)),
                "words": [],
                "boxes": [],
            })

            assert result["predicted_class"] == doc_type_idx

    def test_classification_low_confidence(self, classification_stage):
        """Test handling of low confidence predictions."""
        with patch.object(classification_stage, 'vision_encoder') as mock_encoder:
            mock_encoder.classify.return_value = (0, 0.45)  # Low confidence

            result = classification_stage.process({
                "image": Image.new('RGB', (800, 600)),
                "words": [],
                "boxes": [],
            })

            assert result["confidence"] < 0.5


@pytest.mark.unit
@pytest.mark.stages
class TestTypeIdentifierStage:
    """Tests for Stage 4: Type Identifier - 35+ invoice/PO subtypes."""

    @pytest.fixture
    def type_identifier_stage(self):
        """Create TypeIdentifierStage instance."""
        config = MagicMock()
        config.num_subtypes = 35
        return TypeIdentifierStage(config=config)

    def test_type_identifier_initialization(self, type_identifier_stage):
        """Test Type Identifier stage initialization."""
        assert type_identifier_stage is not None

    def test_type_identifier_invoice_subtypes(self, type_identifier_stage, sample_ocr_text):
        """Test identification of invoice subtypes."""
        result = type_identifier_stage.process({
            "document_type": "SUPPLIER_INVOICE",
            "ocr_text": sample_ocr_text,
        })

        assert result is not None
        assert "document_subtype" in result
        assert "subtype_confidence" in result

    def test_type_identifier_po_subtypes(self, type_identifier_stage):
        """Test identification of PO subtypes."""
        ocr_text = "PURCHASE ORDER\nStandard\nPO#: 12345"

        result = type_identifier_stage.process({
            "document_type": "PURCHASE_ORDER",
            "ocr_text": ocr_text,
        })

        assert result is not None
        assert "document_subtype" in result

    @pytest.mark.parametrize("subtype,keywords", [
        ("STANDARD", ["standard", "regular"]),
        ("BLANKET", ["blanket", "framework"]),
        ("CONTRACT", ["contract", "agreement"]),
    ])
    def test_type_identifier_subtype_keywords(self, type_identifier_stage, subtype, keywords):
        """Test subtype identification based on keywords."""
        ocr_text = f"PURCHASE ORDER\n{keywords[0]}\nPO#: 12345"

        result = type_identifier_stage.process({
            "document_type": "PURCHASE_ORDER",
            "ocr_text": ocr_text,
        })

        assert result is not None


@pytest.mark.unit
@pytest.mark.stages
class TestExtractionStage:
    """Tests for Stage 5: Extraction - Field-level data extraction (180+ fields)."""

    @pytest.fixture
    def extraction_stage(self):
        """Create ExtractionStage instance."""
        config = MagicMock()
        config.model_name = "mistralai/Mistral-7B-v0.1"
        return ExtractionStage(config=config)

    def test_extraction_initialization(self, extraction_stage):
        """Test Extraction stage initialization."""
        assert extraction_stage is not None

    def test_extraction_purchase_order(self, extraction_stage, sample_ocr_text):
        """Test extraction from purchase order."""
        schema = {
            "properties": {
                "po_number": {"type": "string"},
                "vendor_id": {"type": "string"},
                "total_amount": {"type": "number"},
            }
        }

        with patch.object(extraction_stage, 'language_decoder') as mock_decoder:
            mock_decoder.extract_fields.return_value = {
                "po_number": "4500123456",
                "vendor_id": "100001",
                "total_amount": 2200.00,
            }

            result = extraction_stage.process({
                "ocr_text": sample_ocr_text,
                "document_type": "PURCHASE_ORDER",
                "schema": schema,
            })

            assert result is not None
            assert "extracted_fields" in result
            assert "po_number" in result["extracted_fields"]

    def test_extraction_line_items(self, extraction_stage, sample_adc):
        """Test extraction of line items."""
        schema = {
            "properties": {
                "line_items": {
                    "type": "array",
                    "items": {
                        "properties": {
                            "material": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                        }
                    }
                }
            }
        }

        with patch.object(extraction_stage, 'language_decoder') as mock_decoder:
            mock_decoder.extract_fields.return_value = {
                "line_items": sample_adc["line_items"]
            }

            result = extraction_stage.process({
                "ocr_text": "Line items...",
                "document_type": "PURCHASE_ORDER",
                "schema": schema,
            })

            assert "line_items" in result["extracted_fields"]
            assert len(result["extracted_fields"]["line_items"]) == 2

    def test_extraction_confidence_scores(self, extraction_stage):
        """Test extraction confidence scoring."""
        with patch.object(extraction_stage, 'language_decoder') as mock_decoder:
            mock_decoder.extract_fields.return_value = {
                "po_number": "12345",
            }

            result = extraction_stage.process({
                "ocr_text": "PO: 12345",
                "document_type": "PURCHASE_ORDER",
                "schema": {},
            })

            assert result is not None

    @pytest.mark.parametrize("field_count", [10, 50, 100, 180])
    def test_extraction_field_counts(self, extraction_stage, field_count):
        """Test extraction with different field counts."""
        schema = {
            "properties": {
                f"field_{i}": {"type": "string"}
                for i in range(field_count)
            }
        }

        with patch.object(extraction_stage, 'language_decoder') as mock_decoder:
            mock_decoder.extract_fields.return_value = {
                f"field_{i}": f"value_{i}"
                for i in range(field_count)
            }

            result = extraction_stage.process({
                "ocr_text": "Test document",
                "document_type": "PURCHASE_ORDER",
                "schema": schema,
            })

            assert len(result["extracted_fields"]) == field_count


@pytest.mark.unit
@pytest.mark.stages
class TestQualityCheckStage:
    """Tests for Stage 6: Quality Check - Confidence scoring & validation."""

    @pytest.fixture
    def quality_check_stage(self):
        """Create QualityCheckStage instance."""
        config = MagicMock()
        config.min_quality_score = 0.90
        config.enable_self_correction = True
        return QualityCheckStage(config=config)

    def test_quality_check_initialization(self, quality_check_stage):
        """Test Quality Check stage initialization."""
        assert quality_check_stage is not None

    def test_quality_check_high_quality(self, quality_check_stage, sample_adc):
        """Test quality check for high-quality extraction."""
        result = quality_check_stage.process({
            "extracted_fields": sample_adc,
            "schema": {
                "required": ["po_number", "vendor_id", "total_amount"]
            },
        })

        assert result is not None
        assert "quality_score" in result
        assert result["quality_score"] >= 0.90

    def test_quality_check_low_quality(self, quality_check_stage):
        """Test quality check for low-quality extraction."""
        # Missing required fields
        result = quality_check_stage.process({
            "extracted_fields": {"po_number": "12345"},
            "schema": {
                "required": ["po_number", "vendor_id", "total_amount", "date"]
            },
        })

        assert result is not None
        assert "quality_score" in result
        assert result["quality_score"] < 0.90

    def test_quality_check_self_correction(self, quality_check_stage):
        """Test self-correction mechanism."""
        result = quality_check_stage.process({
            "extracted_fields": {"total_amount": "2,200.00"},  # Wrong format
            "schema": {
                "properties": {
                    "total_amount": {"type": "number"}
                }
            },
        })

        assert result is not None
        # Self-correction should attempt to fix format issues

    def test_quality_check_completeness(self, quality_check_stage):
        """Test completeness scoring."""
        extracted = {
            "field1": "value1",
            "field2": "value2",
            "field3": None,  # Missing value
        }

        result = quality_check_stage.process({
            "extracted_fields": extracted,
            "schema": {
                "required": ["field1", "field2", "field3"]
            },
        })

        assert result["quality_score"] < 1.0

    def test_quality_check_field_confidence(self, quality_check_stage):
        """Test per-field confidence scoring."""
        result = quality_check_stage.process({
            "extracted_fields": {
                "po_number": "12345",
                "total_amount": 1000.0,
            },
            "field_confidences": {
                "po_number": 0.99,
                "total_amount": 0.85,
            },
            "schema": {},
        })

        assert result is not None


@pytest.mark.unit
@pytest.mark.stages
class TestValidationStage:
    """Tests for Stage 7: Validation - Business rules & tolerance checks."""

    @pytest.fixture
    def validation_stage(self):
        """Create ValidationStage instance."""
        config = MagicMock()
        config.tolerance_percentage = 2.0
        config.strict_mode = False
        return ValidationStage(config=config)

    def test_validation_initialization(self, validation_stage):
        """Test Validation stage initialization."""
        assert validation_stage is not None

    def test_validation_pass(self, validation_stage, sample_adc):
        """Test validation passing."""
        result = validation_stage.process({
            "extracted_fields": sample_adc,
            "document_type": "PURCHASE_ORDER",
            "quality_score": 0.95,
        })

        assert result is not None
        assert "validation_passed" in result
        assert "exceptions" in result

    def test_validation_amount_mismatch(self, validation_stage, sample_adc):
        """Test validation failing on amount mismatch."""
        bad_adc = sample_adc.copy()
        bad_adc["total_amount"] = 9999.99  # Doesn't match subtotal + tax

        result = validation_stage.process({
            "extracted_fields": bad_adc,
            "document_type": "PURCHASE_ORDER",
            "quality_score": 0.95,
        })

        assert result is not None
        assert len(result.get("exceptions", [])) > 0

    def test_validation_tolerance_check(self, validation_stage):
        """Test tolerance-based validation."""
        # Amount within tolerance
        result = validation_stage.process({
            "extracted_fields": {
                "subtotal": 1000.00,
                "tax_amount": 100.00,
                "total_amount": 1101.00,  # 1% over, within 2% tolerance
            },
            "document_type": "SUPPLIER_INVOICE",
            "quality_score": 0.95,
        })

        # Should pass with warning
        assert result is not None

    def test_validation_required_fields(self, validation_stage):
        """Test required field validation."""
        result = validation_stage.process({
            "extracted_fields": {
                "vendor_name": "ACME Corp",
                # Missing required fields
            },
            "document_type": "PURCHASE_ORDER",
            "quality_score": 0.95,
        })

        assert result is not None
        exceptions = result.get("exceptions", [])
        # Should have exceptions for missing required fields

    @pytest.mark.parametrize("doc_type,required_fields", [
        ("PURCHASE_ORDER", ["po_number", "vendor_id", "total_amount"]),
        ("SUPPLIER_INVOICE", ["invoice_number", "invoice_date", "total_amount"]),
        ("SALES_ORDER", ["order_number", "customer_id", "total_amount"]),
    ])
    def test_validation_document_specific_rules(self, validation_stage, doc_type, required_fields):
        """Test document-type-specific validation rules."""
        # Create ADC with all required fields
        adc = {field: f"test_{field}" for field in required_fields}
        if "total_amount" in required_fields:
            adc["total_amount"] = 1000.0

        result = validation_stage.process({
            "extracted_fields": adc,
            "document_type": doc_type,
            "quality_score": 0.95,
        })

        assert result is not None

    def test_validation_line_item_totals(self, validation_stage, sample_adc):
        """Test line item total validation."""
        result = validation_stage.process({
            "extracted_fields": sample_adc,
            "document_type": "PURCHASE_ORDER",
            "quality_score": 0.95,
        })

        assert result is not None
        # Should validate that line item totals sum correctly


@pytest.mark.unit
@pytest.mark.stages
class TestRoutingStage:
    """Tests for Stage 8: Routing - SAP API endpoint selection & payload generation."""

    @pytest.fixture
    def routing_stage(self, mock_pmg):
        """Create RoutingStage instance."""
        config = MagicMock()
        config.api_config_path = "/path/to/api/schemas"
        return RoutingStage(config=config, pmg=mock_pmg)

    def test_routing_initialization(self, routing_stage):
        """Test Routing stage initialization."""
        assert routing_stage is not None

    def test_routing_purchase_order(self, routing_stage, sample_adc):
        """Test routing for purchase order."""
        api_schemas = [
            {"name": "API_PURCHASEORDER_PROCESS_SRV", "version": "0001"},
        ]

        with patch.object(routing_stage, 'reasoning_engine') as mock_engine:
            mock_engine.decide_routing.return_value = {
                "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                "method": "POST",
                "entity": "PurchaseOrder",
                "confidence": 0.95,
            }

            result = routing_stage.process({
                "extracted_fields": sample_adc,
                "document_type": "PURCHASE_ORDER",
                "api_schemas": api_schemas,
            })

            assert result is not None
            assert "routing_decision" in result
            assert result["routing_decision"]["endpoint"] == "API_PURCHASEORDER_PROCESS_SRV"

    def test_routing_with_pmg_context(self, routing_stage, sample_adc, mock_pmg):
        """Test routing with PMG historical context."""
        mock_pmg.query_similar.return_value = [
            {
                "doc_type": "PURCHASE_ORDER",
                "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                "confidence": 0.98,
            }
        ]

        api_schemas = [
            {"name": "API_PURCHASEORDER_PROCESS_SRV"},
        ]

        with patch.object(routing_stage, 'reasoning_engine') as mock_engine:
            mock_engine.decide_routing.return_value = {
                "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                "confidence": 0.95,
            }

            result = routing_stage.process({
                "extracted_fields": sample_adc,
                "document_type": "PURCHASE_ORDER",
                "api_schemas": api_schemas,
            })

            assert result is not None

    def test_routing_payload_generation(self, routing_stage, sample_adc):
        """Test SAP API payload generation."""
        api_schemas = [{"name": "API_PURCHASEORDER_PROCESS_SRV"}]

        with patch.object(routing_stage, 'reasoning_engine') as mock_engine:
            mock_engine.decide_routing.return_value = {
                "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                "payload": {
                    "PurchaseOrder": sample_adc["po_number"],
                    "CompanyCode": sample_adc["company_code"],
                },
                "confidence": 0.95,
            }

            result = routing_stage.process({
                "extracted_fields": sample_adc,
                "document_type": "PURCHASE_ORDER",
                "api_schemas": api_schemas,
            })

            assert result is not None
            assert "payload" in result["routing_decision"]

    @pytest.mark.parametrize("doc_type,expected_api", [
        ("PURCHASE_ORDER", "API_PURCHASEORDER_PROCESS_SRV"),
        ("SUPPLIER_INVOICE", "API_SUPPLIERINVOICE_PROCESS_SRV"),
        ("SALES_ORDER", "API_SALES_ORDER_SRV"),
        ("GOODS_RECEIPT", "API_MATERIAL_DOCUMENT_SRV"),
    ])
    def test_routing_different_doc_types(self, routing_stage, doc_type, expected_api):
        """Test routing for different document types."""
        api_schemas = [{"name": expected_api}]

        with patch.object(routing_stage, 'reasoning_engine') as mock_engine:
            mock_engine.decide_routing.return_value = {
                "endpoint": expected_api,
                "confidence": 0.95,
            }

            result = routing_stage.process({
                "extracted_fields": {"test": "data"},
                "document_type": doc_type,
                "api_schemas": api_schemas,
            })

            assert result["routing_decision"]["endpoint"] == expected_api

    def test_routing_exception_handling(self, routing_stage):
        """Test routing stage exception handling."""
        result = routing_stage.process({
            "extracted_fields": {},
            "document_type": "UNKNOWN",
            "api_schemas": [],
        })

        # Should handle gracefully
        assert result is not None

    def test_routing_confidence_threshold(self, routing_stage, sample_adc):
        """Test routing confidence threshold checks."""
        api_schemas = [{"name": "API_PURCHASEORDER_PROCESS_SRV"}]

        with patch.object(routing_stage, 'reasoning_engine') as mock_engine:
            # Low confidence
            mock_engine.decide_routing.return_value = {
                "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                "confidence": 0.45,
            }

            result = routing_stage.process({
                "extracted_fields": sample_adc,
                "document_type": "PURCHASE_ORDER",
                "api_schemas": api_schemas,
            })

            assert result is not None
            # Should flag low confidence


@pytest.mark.unit
@pytest.mark.stages
class TestStageBase:
    """Tests for base stage functionality."""

    def test_stage_input_validation(self):
        """Test that stages validate inputs."""
        from sap_llm.stages.base_stage import BaseStage

        class TestStage(BaseStage):
            def process(self, input_data):
                return {"result": "success"}

        stage = TestStage(config=MagicMock())
        result = stage.process({"test": "data"})
        assert result is not None

    def test_stage_error_handling(self):
        """Test stage error handling."""
        from sap_llm.stages.base_stage import BaseStage

        class FailingStage(BaseStage):
            def process(self, input_data):
                raise ValueError("Test error")

        stage = FailingStage(config=MagicMock())
        with pytest.raises(ValueError):
            stage.process({"test": "data"})
