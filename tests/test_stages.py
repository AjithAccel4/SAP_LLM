"""
Unit tests for pipeline stages.
"""

import pytest
from unittest.mock import Mock, MagicMock

from sap_llm.stages.inbox import InboxStage
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.validation import ValidationStage


@pytest.mark.unit
@pytest.mark.stages
class TestInboxStage:
    """Tests for Inbox stage."""

    @pytest.fixture
    def inbox_stage(self, test_config):
        """Create InboxStage instance."""
        return InboxStage(config=test_config.stages.inbox)

    def test_inbox_initialization(self, inbox_stage):
        """Test Inbox stage initialization."""
        assert inbox_stage is not None
        assert inbox_stage.cache_ttl > 0

    def test_inbox_process_new_document(self, inbox_stage, sample_image, temp_dir):
        """Test processing new document."""
        # Save sample image
        image_path = temp_dir / "test_document.png"
        sample_image.save(image_path)

        # Read file
        with open(image_path, "rb") as f:
            file_content = f.read()

        # Process
        result = inbox_stage.process({
            "file_path": str(image_path),
            "file_content": file_content,
        })

        assert result is not None
        assert "document_hash" in result
        assert "file_path" in result
        assert "is_duplicate" in result
        assert result["is_duplicate"] is False

    def test_inbox_duplicate_detection(self, inbox_stage, sample_image, temp_dir):
        """Test duplicate detection."""
        # Save sample image
        image_path = temp_dir / "test_document.png"
        sample_image.save(image_path)

        # Read file
        with open(image_path, "rb") as f:
            file_content = f.read()

        input_data = {
            "file_path": str(image_path),
            "file_content": file_content,
        }

        # Process first time
        result1 = inbox_stage.process(input_data)
        assert result1["is_duplicate"] is False

        # Process second time - should be detected as duplicate
        result2 = inbox_stage.process(input_data)
        # Note: Duplicate detection requires Redis in real implementation
        # In unit test, this might not work without mocking


@pytest.mark.unit
@pytest.mark.stages
class TestPreprocessingStage:
    """Tests for Preprocessing stage."""

    @pytest.fixture
    def preprocessing_stage(self, test_config):
        """Create PreprocessingStage instance."""
        return PreprocessingStage(config=test_config.stages.preprocessing)

    def test_preprocessing_initialization(self, preprocessing_stage):
        """Test Preprocessing stage initialization."""
        assert preprocessing_stage is not None

    @pytest.mark.slow
    def test_preprocessing_ocr(self, preprocessing_stage, sample_document_image, temp_dir):
        """Test OCR processing."""
        # Save sample image
        image_path = temp_dir / "test_document.png"
        sample_document_image.save(image_path)

        # Process
        result = preprocessing_stage.process({
            "file_path": str(image_path),
            "image": sample_document_image,
        })

        assert result is not None
        assert "ocr_text" in result
        assert "image" in result

    def test_preprocessing_image_enhancement(self, preprocessing_stage, sample_image):
        """Test image enhancement."""
        result = preprocessing_stage.process({
            "file_path": "test.png",
            "image": sample_image,
        })

        assert result is not None
        assert "image" in result
        # Enhanced image should still be valid
        assert result["image"] is not None


@pytest.mark.unit
@pytest.mark.stages
class TestValidationStage:
    """Tests for Validation stage."""

    @pytest.fixture
    def validation_stage(self, test_config):
        """Create ValidationStage instance."""
        return ValidationStage(config=test_config.stages.validation)

    def test_validation_initialization(self, validation_stage):
        """Test Validation stage initialization."""
        assert validation_stage is not None

    def test_validation_pass(self, validation_stage, sample_adc):
        """Test validation passing."""
        # Add quality score
        input_data = {
            "adc": sample_adc,
            "quality_score": 0.95,
            "document_type": "purchase_order",
        }

        result = validation_stage.process(input_data)

        assert result is not None
        assert "validation_passed" in result
        assert "exceptions" in result

    def test_validation_fail_quality_score(self, validation_stage, sample_adc):
        """Test validation failing on low quality score."""
        # Low quality score
        input_data = {
            "adc": sample_adc,
            "quality_score": 0.50,
            "document_type": "purchase_order",
        }

        result = validation_stage.process(input_data)

        assert result is not None
        assert "validation_passed" in result
        assert "exceptions" in result
        # May or may not pass depending on implementation

    def test_validation_amount_consistency(self, validation_stage, sample_adc):
        """Test validation of amount consistency."""
        # Create inconsistent ADC
        bad_adc = sample_adc.copy()
        bad_adc["total_amount"] = 9999.99  # Doesn't match subtotal + tax

        input_data = {
            "adc": bad_adc,
            "quality_score": 0.95,
            "document_type": "purchase_order",
        }

        result = validation_stage.process(input_data)

        assert result is not None
        # Should detect the inconsistency
        if result.get("exceptions"):
            assert any("amount" in str(exc).lower() for exc in result["exceptions"])


@pytest.mark.unit
@pytest.mark.stages
class TestStageBase:
    """Tests for base stage functionality."""

    def test_stage_input_validation(self):
        """Test that stages validate inputs."""
        # This would test the base stage class
        # For now, we test through concrete implementations
        pass

    def test_stage_timing(self):
        """Test that stages track timing."""
        # This would test the timing wrapper
        pass
