"""
Comprehensive unit tests for SAP_LLM model components.
"""

import json
from unittest.mock import Mock, MagicMock, patch
import pytest
import torch
import numpy as np
from PIL import Image

from sap_llm.models.vision_encoder import VisionEncoder
from sap_llm.models.language_decoder import LanguageDecoder
from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.models.unified_model import UnifiedExtractorModel


@pytest.mark.unit
@pytest.mark.models
class TestVisionEncoder:
    """Tests for VisionEncoder component."""

    @pytest.fixture
    def mock_vision_encoder(self, mocker):
        """Create mocked VisionEncoder."""
        with patch('sap_llm.models.vision_encoder.AutoModel') as mock_model, \
             patch('sap_llm.models.vision_encoder.AutoProcessor') as mock_processor:

            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model_instance.to.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            # Mock processor
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance

            encoder = VisionEncoder(
                model_name="microsoft/layoutlmv3-base",
                device="cpu",
            )

            # Mock outputs
            mock_model_instance.return_value = MagicMock(
                logits=torch.randn(1, 15),  # 15 document types
                last_hidden_state=torch.randn(1, 512, 768),
            )

            return encoder

    def test_vision_encoder_initialization(self, mock_vision_encoder):
        """Test VisionEncoder initialization."""
        assert mock_vision_encoder is not None
        assert mock_vision_encoder.device == "cpu"
        assert hasattr(mock_vision_encoder, 'model')
        assert hasattr(mock_vision_encoder, 'processor')

    def test_vision_encoder_encode(self, mock_vision_encoder):
        """Test encoding visual features."""
        # Create test inputs
        image = Image.new('RGB', (800, 600), color='white')
        words = ['Purchase', 'Order', 'PO#', '12345']
        boxes = [[10, 10, 100, 30], [110, 10, 150, 30],
                 [10, 50, 50, 70], [60, 50, 120, 70]]

        # Mock processor output
        mock_vision_encoder.processor.return_value = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'bbox': torch.randint(0, 1000, (1, 512, 4)),
        }

        # Encode
        features = mock_vision_encoder.encode(image, words, boxes)

        assert features is not None
        assert isinstance(features, torch.Tensor)

    def test_vision_encoder_classify(self, mock_vision_encoder):
        """Test document classification."""
        # Create test inputs
        image = Image.new('RGB', (800, 600), color='white')
        words = ['Purchase', 'Order']
        boxes = [[10, 10, 100, 30], [110, 10, 150, 30]]

        # Mock processor and model output
        mock_vision_encoder.processor.return_value = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'bbox': torch.randint(0, 1000, (1, 512, 4)),
        }

        mock_vision_encoder.model.return_value = MagicMock(
            logits=torch.tensor([[2.0, 5.0, 1.0, 0.5] + [0.0] * 11])
        )

        # Classify
        predicted_class, confidence = mock_vision_encoder.classify(image, words, boxes)

        assert isinstance(predicted_class, int)
        assert 0 <= predicted_class < 15
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.parametrize("input_size", [
        (224, 224),
        (800, 600),
        (1000, 1500),
    ])
    def test_vision_encoder_different_sizes(self, mock_vision_encoder, input_size):
        """Test handling different image sizes."""
        image = Image.new('RGB', input_size, color='white')
        words = ['Test']
        boxes = [[10, 10, 100, 30]]

        mock_vision_encoder.processor.return_value = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'bbox': torch.randint(0, 1000, (1, 512, 4)),
        }

        features = mock_vision_encoder.encode(image, words, boxes)
        assert features is not None


@pytest.mark.unit
@pytest.mark.models
class TestLanguageDecoder:
    """Tests for LanguageDecoder component."""

    @pytest.fixture
    def mock_language_decoder(self, mocker):
        """Create mocked LanguageDecoder."""
        with patch('sap_llm.models.language_decoder.AutoModelForCausalLM') as mock_model, \
             patch('sap_llm.models.language_decoder.AutoTokenizer') as mock_tokenizer:

            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model_instance.to.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            # Mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.pad_token = "[PAD]"
            mock_tokenizer_instance.eos_token = "[EOS]"
            mock_tokenizer_instance.pad_token_id = 0
            mock_tokenizer_instance.eos_token_id = 1
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            decoder = LanguageDecoder(
                model_name="mistralai/Mistral-7B-v0.1",
                device="cpu",
                precision="fp32",
            )

            # Mock generation output
            mock_tokenizer_instance.decode.return_value = json.dumps({
                "po_number": "4500123456",
                "vendor_id": "100001",
                "total_amount": 2200.00,
            })

            return decoder

    def test_language_decoder_initialization(self, mock_language_decoder):
        """Test LanguageDecoder initialization."""
        assert mock_language_decoder is not None
        assert mock_language_decoder.device == "cpu"
        assert hasattr(mock_language_decoder, 'model')
        assert hasattr(mock_language_decoder, 'tokenizer')

    def test_create_extraction_prompt(self, mock_language_decoder):
        """Test extraction prompt creation."""
        ocr_text = "PURCHASE ORDER\nPO Number: 4500123456\nVendor: ACME Corp"
        doc_type = "PURCHASE_ORDER"
        schema = {
            "type": "object",
            "properties": {
                "po_number": {"type": "string"},
                "vendor_name": {"type": "string"},
            }
        }

        prompt = mock_language_decoder.create_extraction_prompt(
            ocr_text, doc_type, schema
        )

        assert "PURCHASE ORDER" in prompt
        assert "po_number" in prompt
        assert "vendor_name" in prompt
        assert isinstance(prompt, str)

    def test_extract_fields(self, mock_language_decoder, sample_ocr_text):
        """Test field extraction."""
        schema = {
            "type": "object",
            "properties": {
                "po_number": {"type": "string"},
                "vendor_id": {"type": "string"},
                "total_amount": {"type": "number"},
            }
        }

        # Mock tokenizer and generation
        mock_language_decoder.tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 100)),
            'attention_mask': torch.ones(1, 100),
        }

        mock_language_decoder.model.generate.return_value = torch.randint(0, 1000, (1, 150))

        extracted = mock_language_decoder.extract_fields(
            sample_ocr_text,
            "PURCHASE_ORDER",
            schema,
        )

        assert isinstance(extracted, dict)
        assert "po_number" in extracted
        assert "vendor_id" in extracted
        assert "total_amount" in extracted

    @pytest.mark.parametrize("doc_type,schema_fields", [
        ("PURCHASE_ORDER", ["po_number", "vendor_id", "total_amount"]),
        ("SUPPLIER_INVOICE", ["invoice_number", "invoice_date", "total_amount"]),
        ("SALES_ORDER", ["order_number", "customer_id", "total_amount"]),
    ])
    def test_extract_different_doc_types(self, mock_language_decoder, doc_type, schema_fields):
        """Test extraction for different document types."""
        ocr_text = f"{doc_type}\nTest Document"
        schema = {
            "type": "object",
            "properties": {field: {"type": "string"} for field in schema_fields}
        }

        mock_language_decoder.tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 100)),
            'attention_mask': torch.ones(1, 100),
        }

        mock_language_decoder.model.generate.return_value = torch.randint(0, 1000, (1, 150))

        extracted = mock_language_decoder.extract_fields(ocr_text, doc_type, schema)
        assert isinstance(extracted, dict)

    def test_parse_json_response(self, mock_language_decoder):
        """Test JSON parsing from generated text."""
        # Valid JSON
        text = 'Some prefix text {"field": "value", "number": 123} some suffix'
        result = mock_language_decoder._parse_json_response(text)
        assert result == {"field": "value", "number": 123}

        # Invalid JSON
        text = "This is not JSON"
        result = mock_language_decoder._parse_json_response(text)
        assert result == {}


@pytest.mark.unit
@pytest.mark.models
class TestReasoningEngine:
    """Tests for ReasoningEngine component."""

    @pytest.fixture
    def mock_reasoning_engine(self, mocker):
        """Create mocked ReasoningEngine."""
        with patch('sap_llm.models.reasoning_engine.AutoModelForCausalLM') as mock_model, \
             patch('sap_llm.models.reasoning_engine.AutoTokenizer') as mock_tokenizer:

            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model_instance.to.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            # Mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.pad_token = "[PAD]"
            mock_tokenizer_instance.eos_token = "[EOS]"
            mock_tokenizer_instance.pad_token_id = 0
            mock_tokenizer_instance.eos_token_id = 1
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            engine = ReasoningEngine(
                model_name="mistralai/Mixtral-8x7B-v0.1",
                device="cpu",
                precision="fp32",
            )

            return engine

    def test_reasoning_engine_initialization(self, mock_reasoning_engine):
        """Test ReasoningEngine initialization."""
        assert mock_reasoning_engine is not None
        assert mock_reasoning_engine.device == "cpu"
        assert hasattr(mock_reasoning_engine, 'model')
        assert hasattr(mock_reasoning_engine, 'tokenizer')

    def test_create_routing_prompt(self, mock_reasoning_engine, sample_adc):
        """Test routing prompt creation."""
        api_schemas = [
            {"name": "API_PURCHASEORDER_PROCESS_SRV"},
            {"name": "API_SUPPLIERINVOICE_PROCESS_SRV"},
        ]
        similar_cases = [
            {"doc_type": "PURCHASE_ORDER", "endpoint": "API_PURCHASEORDER_PROCESS_SRV"}
        ]

        prompt = mock_reasoning_engine.create_routing_prompt(
            sample_adc,
            "PURCHASE_ORDER",
            api_schemas,
            similar_cases,
        )

        assert "PURCHASE_ORDER" in prompt
        assert "API_PURCHASEORDER_PROCESS_SRV" in prompt
        assert isinstance(prompt, str)

    def test_create_exception_handling_prompt(self, mock_reasoning_engine):
        """Test exception handling prompt creation."""
        exception = {
            "category": "VALIDATION_ERROR",
            "severity": "HIGH",
            "field": "total_amount",
            "expected": "2200.00",
            "value": "2000.00",
            "message": "Amount mismatch",
        }
        similar_exceptions = []

        prompt = mock_reasoning_engine.create_exception_handling_prompt(
            exception,
            similar_exceptions,
        )

        assert "VALIDATION_ERROR" in prompt
        assert "total_amount" in prompt
        assert isinstance(prompt, str)

    def test_decide_routing(self, mock_reasoning_engine, sample_adc):
        """Test routing decision."""
        api_schemas = [
            {"name": "API_PURCHASEORDER_PROCESS_SRV"},
        ]

        # Mock generation to return valid JSON
        mock_reasoning_engine.tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 100)),
            'attention_mask': torch.ones(1, 100),
        }

        mock_reasoning_engine.model.generate.return_value = torch.randint(0, 1000, (1, 150))

        mock_reasoning_engine.tokenizer.decode.return_value = json.dumps({
            "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
            "method": "POST",
            "confidence": 0.95,
            "reasoning": "This is a purchase order document",
        })

        decision = mock_reasoning_engine.decide_routing(
            sample_adc,
            "PURCHASE_ORDER",
            api_schemas,
        )

        assert "endpoint" in decision
        assert decision["endpoint"] == "API_PURCHASEORDER_PROCESS_SRV"
        assert "confidence" in decision

    def test_handle_exception(self, mock_reasoning_engine):
        """Test exception handling decision."""
        exception = {
            "category": "VALIDATION_ERROR",
            "severity": "HIGH",
            "field": "total_amount",
            "message": "Amount mismatch",
        }

        # Mock generation
        mock_reasoning_engine.tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 100)),
            'attention_mask': torch.ones(1, 100),
        }

        mock_reasoning_engine.model.generate.return_value = torch.randint(0, 1000, (1, 150))

        mock_reasoning_engine.tokenizer.decode.return_value = json.dumps({
            "action": "ESCALATE",
            "reasoning": "High severity validation error",
            "confidence": 0.90,
        })

        decision = mock_reasoning_engine.handle_exception(exception)

        assert "action" in decision
        assert decision["action"] in ["AUTO_CORRECT", "ESCALATE", "REJECT", "APPLY_RULE"]

    def test_fallback_routing(self, mock_reasoning_engine):
        """Test fallback routing logic."""
        api_schemas = []

        decision = mock_reasoning_engine._fallback_routing("PURCHASE_ORDER", api_schemas)

        assert "endpoint" in decision
        assert decision["endpoint"] == "API_PURCHASEORDER_PROCESS_SRV"
        assert decision["confidence"] == 0.5  # Low confidence for fallback

    @pytest.mark.parametrize("doc_type,expected_endpoint", [
        ("PURCHASE_ORDER", "API_PURCHASEORDER_PROCESS_SRV"),
        ("SUPPLIER_INVOICE", "API_SUPPLIERINVOICE_PROCESS_SRV"),
        ("SALES_ORDER", "API_SALES_ORDER_SRV"),
    ])
    def test_fallback_routing_types(self, mock_reasoning_engine, doc_type, expected_endpoint):
        """Test fallback routing for different document types."""
        decision = mock_reasoning_engine._fallback_routing(doc_type, [])
        assert decision["endpoint"] == expected_endpoint


@pytest.mark.unit
@pytest.mark.models
class TestUnifiedModel:
    """Tests for UnifiedExtractorModel."""

    @pytest.fixture
    def mock_unified_model(self, mocker):
        """Create mocked UnifiedExtractorModel."""
        model = UnifiedExtractorModel(device="cpu")

        # Mock components
        model.vision_encoder = MagicMock()
        model.language_decoder = MagicMock()
        model.reasoning_engine = MagicMock()

        return model

    def test_unified_model_initialization(self, mock_unified_model):
        """Test UnifiedExtractorModel initialization."""
        assert mock_unified_model is not None
        assert mock_unified_model.device == "cpu"
        assert mock_unified_model.vision_encoder is not None
        assert mock_unified_model.language_decoder is not None
        assert mock_unified_model.reasoning_engine is not None

    def test_set_components(self):
        """Test setting model components."""
        model = UnifiedExtractorModel(device="cpu")

        ve = MagicMock()
        ld = MagicMock()
        re = MagicMock()

        model.set_vision_encoder(ve)
        model.set_language_decoder(ld)
        model.set_reasoning_engine(re)

        assert model.vision_encoder == ve
        assert model.language_decoder == ld
        assert model.reasoning_engine == re

    def test_classify(self, mock_unified_model):
        """Test document classification."""
        image = Image.new('RGB', (800, 600))
        ocr_text = "PURCHASE ORDER"
        words = ["PURCHASE", "ORDER"]
        boxes = [[10, 10, 100, 30], [110, 10, 200, 30]]

        # Mock vision encoder
        mock_unified_model.vision_encoder.classify.return_value = (0, 0.95)

        doc_type, subtype, confidence = mock_unified_model.classify(
            image, ocr_text, words, boxes
        )

        assert doc_type == "PURCHASE_ORDER"
        assert isinstance(subtype, str)
        assert 0.0 <= confidence <= 1.0

    def test_extract(self, mock_unified_model):
        """Test field extraction."""
        image = Image.new('RGB', (800, 600))
        ocr_text = "PO: 12345"
        words = ["PO:", "12345"]
        boxes = [[10, 10, 50, 30], [60, 10, 120, 30]]
        schema = {"properties": {"po_number": {"type": "string"}}}

        # Mock components
        mock_unified_model.vision_encoder.encode.return_value = torch.randn(1, 768)
        mock_unified_model.language_decoder.extract_fields.return_value = {
            "po_number": "12345"
        }

        extracted_data, metadata = mock_unified_model.extract(
            image, ocr_text, words, boxes, "PURCHASE_ORDER", schema
        )

        assert isinstance(extracted_data, dict)
        assert isinstance(metadata, dict)
        assert "po_number" in extracted_data

    def test_route(self, mock_unified_model, sample_adc):
        """Test routing decision."""
        api_schemas = [{"name": "API_PURCHASEORDER_PROCESS_SRV"}]

        # Mock reasoning engine
        mock_unified_model.reasoning_engine.decide_routing.return_value = {
            "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
            "confidence": 0.95,
        }

        decision = mock_unified_model.route(
            sample_adc,
            "PURCHASE_ORDER",
            api_schemas,
        )

        assert "endpoint" in decision
        assert "confidence" in decision

    def test_process_document_success(self, mock_unified_model):
        """Test successful end-to-end document processing."""
        image = Image.new('RGB', (800, 600))
        ocr_text = "PURCHASE ORDER"
        words = ["PURCHASE", "ORDER"]
        boxes = [[10, 10, 100, 30], [110, 10, 200, 30]]
        schemas = {
            "PURCHASE_ORDER": {
                "properties": {"po_number": {"type": "string"}},
                "required": ["po_number"]
            }
        }
        api_schemas = [{"name": "API_PURCHASEORDER_PROCESS_SRV"}]

        # Mock all components
        mock_unified_model.vision_encoder.classify.return_value = (0, 0.95)
        mock_unified_model.vision_encoder.encode.return_value = torch.randn(1, 768)
        mock_unified_model.language_decoder.extract_fields.return_value = {
            "po_number": "12345",
            "total_amount": 1000.0,
        }
        mock_unified_model.reasoning_engine.decide_routing.return_value = {
            "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
            "confidence": 0.95,
        }

        result = mock_unified_model.process_document(
            image, ocr_text, words, boxes, schemas, api_schemas
        )

        assert result["success"] is True
        assert "doc_type" in result
        assert "extracted_data" in result
        assert "routing" in result

    def test_process_document_failure(self, mock_unified_model):
        """Test document processing with failures."""
        image = Image.new('RGB', (800, 600))
        ocr_text = "UNKNOWN"
        words = []
        boxes = []
        schemas = {}
        api_schemas = []

        # Make classification fail
        mock_unified_model.vision_encoder.classify.side_effect = Exception("Classification failed")

        result = mock_unified_model.process_document(
            image, ocr_text, words, boxes, schemas, api_schemas
        )

        assert result["success"] is False
        assert len(result["errors"]) > 0

    def test_map_class_to_type(self, mock_unified_model):
        """Test mapping classification index to document type."""
        assert mock_unified_model._map_class_to_type(0) == "PURCHASE_ORDER"
        assert mock_unified_model._map_class_to_type(1) == "SUPPLIER_INVOICE"
        assert mock_unified_model._map_class_to_type(999) == "OTHER"

    def test_check_quality(self, mock_unified_model):
        """Test quality checking."""
        data = {"po_number": "12345", "total_amount": 1000.0}
        schema = {"required": ["po_number", "total_amount"]}

        quality = mock_unified_model._check_quality(data, schema)
        assert quality == 1.0  # All required fields present

        # Missing field
        data = {"po_number": "12345"}
        quality = mock_unified_model._check_quality(data, schema)
        assert quality == 0.5  # Only 1 of 2 fields present

    def test_validate_business_rules(self, mock_unified_model):
        """Test business rule validation."""
        data = {"total_amount": -100}
        violations = mock_unified_model._validate_business_rules(data, "SUPPLIER_INVOICE")

        assert len(violations) > 0
        assert any("total_amount" in str(v).lower() for v in violations)
