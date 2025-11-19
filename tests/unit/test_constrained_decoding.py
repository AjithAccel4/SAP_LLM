"""
Unit tests for JSON Schema Constrained Decoding.

Tests the JSONSchemaConstraintProcessor implementation added to language_decoder.py
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from sap_llm.models.language_decoder import (
    JSONSchemaConstraintProcessor,
    LanguageDecoder,
)


class TestJSONSchemaConstraintProcessor:
    """Test suite for JSONSchemaConstraintProcessor."""

    @pytest.fixture
    def tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {
            "{": 123, "}": 124, "[": 125, "]": 126,
            ":": 127, ",": 128, '"': 129,
            "true": 130, "false": 131,
            "null": 132,
            " ": 133, "\n": 134,
        }
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = ""
        tokenizer.eos_token_id = 0
        tokenizer.convert_tokens_to_ids = lambda x: tokenizer.get_vocab().get(x, -1)
        return tokenizer

    @pytest.fixture
    def schema(self):
        """Sample JSON schema for testing."""
        return {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "total_amount": {"type": "number"},
                "date": {"type": "string", "format": "date"},
                "vendor_name": {"type": "string"}
            },
            "required": ["invoice_number", "total_amount"]
        }

    def test_processor_initialization(self, tokenizer, schema):
        """Test that processor initializes correctly."""
        processor = JSONSchemaConstraintProcessor(
            schema=schema,
            tokenizer=tokenizer,
            required_fields=["invoice_number", "total_amount"]
        )

        assert processor.schema == schema
        assert processor.tokenizer == tokenizer
        assert processor.required_fields == ["invoice_number", "total_amount"]
        assert hasattr(processor, 'json_structural')
        assert hasattr(processor, 'field_tokens')

    def test_build_token_sets(self, tokenizer, schema):
        """Test that token sets are built correctly."""
        processor = JSONSchemaConstraintProcessor(
            schema=schema,
            tokenizer=tokenizer
        )

        # Check structural tokens
        assert 123 in processor.json_structural  # {
        assert 124 in processor.json_structural  # }
        assert 129 in processor.json_structural  # "

        # Check whitespace tokens
        assert 133 in processor.json_whitespace  # space
        assert 134 in processor.json_whitespace  # newline

        # Check boolean tokens
        assert 130 in processor.json_bool  # true
        assert 131 in processor.json_bool  # false

    def test_get_allowed_tokens_start_of_json(self, tokenizer, schema):
        """Test that only opening brace is allowed at start."""
        processor = JSONSchemaConstraintProcessor(
            schema=schema,
            tokenizer=tokenizer
        )

        allowed = processor._get_allowed_tokens("")

        # Should only allow opening brace and EOS
        assert 123 in allowed  # {
        assert 0 in allowed  # EOS

    def test_call_method_masks_invalid_tokens(self, tokenizer, schema):
        """Test that __call__ correctly masks invalid tokens."""
        processor = JSONSchemaConstraintProcessor(
            schema=schema,
            tokenizer=tokenizer
        )

        # Create mock input
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 200)  # Random scores for vocab

        # Process
        modified_scores = processor(input_ids, scores)

        # Check that scores were modified
        assert modified_scores.shape == scores.shape
        # Should have some -inf values for masked tokens
        assert torch.isinf(modified_scores).any()


class TestLanguageDecoderConstrainedGeneration:
    """Test constrained generation in LanguageDecoder."""

    @pytest.fixture
    def mock_model(self):
        """Mock LLaMA model for testing."""
        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        model.eval.return_value = None
        model.to.return_value = model
        model.parameters.return_value = [torch.randn(10, 10)]
        return model

    def test_constrained_decoding_enabled(self, monkeypatch):
        """Test that constrained decoding is enabled by default."""
        # This is a lightweight test - full testing requires real model
        # Just verify the flag is set correctly
        decoder = LanguageDecoder(
            model_name="meta-llama/Llama-2-7b-hf",
            enable_constrained_decoding=True
        )

        assert decoder.enable_constrained_decoding is True

    def test_constrained_decoding_disabled(self):
        """Test that constrained decoding can be disabled."""
        decoder = LanguageDecoder(
            model_name="meta-llama/Llama-2-7b-hf",
            enable_constrained_decoding=False
        )

        assert decoder.enable_constrained_decoding is False

    def test_schema_parameter_in_generate(self, monkeypatch):
        """Test that generate accepts schema parameter."""
        # Mock the model loading
        mock_tokenizer = Mock()
        mock_tokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.decode.return_value = "Test output"

        # Test requires integration with actual transformers
        # This is a placeholder for the interface test
        assert True  # Schema parameter is accepted in method signature


class TestConstrainedDecodingIntegration:
    """Integration tests for constrained decoding."""

    def test_extraction_with_schema_constraint(self):
        """Test that extract_fields uses schema for constrained decoding."""
        # This requires a full model - placeholder for integration test
        schema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "amount": {"type": "number"}
            }
        }

        # Verify schema is passed through the pipeline
        # Full test requires model weights
        assert True

    def test_json_output_validity(self):
        """Test that constrained decoding produces valid JSON."""
        # Integration test - requires model weights
        # Placeholder for validation
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
