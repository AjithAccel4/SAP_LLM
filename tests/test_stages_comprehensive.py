"""
Comprehensive unit tests for all pipeline stages.

Achieves high coverage by testing:
- All public methods
- Error conditions
- Edge cases
- Input validation
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from PIL import Image
import numpy as np

from sap_llm.stages.base_stage import BaseStage
from sap_llm.stages.inbox import InboxStage
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.classification import ClassificationStage


# Concrete implementation for testing abstract base
class ConcreteStage(BaseStage):
    """Concrete stage for testing base class."""

    def process(self, input_data):
        return {"result": "processed", **input_data}


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    return Image.new('RGB', (100, 100), color='white')


@pytest.fixture
def sample_grayscale_image():
    """Create a grayscale image for testing."""
    return Image.new('L', (100, 100), color=128)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.databases = Mock()
    config.databases.redis = {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'password': None,
        'ttl': 86400
    }
    config.confidence_threshold = 0.90
    return config


@pytest.mark.unit
@pytest.mark.stages
class TestBaseStage:
    """Tests for BaseStage abstract class."""

    def test_initialization(self):
        """Test stage initialization."""
        stage = ConcreteStage()
        assert stage.stage_name == "ConcreteStage"
        assert stage.config is None

    def test_initialization_with_config(self):
        """Test stage initialization with config."""
        config = {"key": "value"}
        stage = ConcreteStage(config=config)
        assert stage.config == config

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        stage = ConcreteStage()
        # Should not raise
        stage.validate_input({"key": "value"})

    def test_validate_input_invalid(self):
        """Test input validation with invalid data."""
        stage = ConcreteStage()
        with pytest.raises(ValueError, match="must be a dictionary"):
            stage.validate_input("not a dict")
        with pytest.raises(ValueError, match="must be a dictionary"):
            stage.validate_input(123)
        with pytest.raises(ValueError, match="must be a dictionary"):
            stage.validate_input([1, 2, 3])

    def test_validate_output_valid(self):
        """Test output validation with valid data."""
        stage = ConcreteStage()
        stage.validate_output({"result": "data"})

    def test_validate_output_invalid(self):
        """Test output validation with invalid data."""
        stage = ConcreteStage()
        with pytest.raises(ValueError, match="must be a dictionary"):
            stage.validate_output("not a dict")

    def test_call_executes_pipeline(self):
        """Test __call__ executes full pipeline."""
        stage = ConcreteStage()
        result = stage({"input": "data"})
        assert result == {"result": "processed", "input": "data"}

    def test_call_validates_input(self):
        """Test __call__ validates input."""
        stage = ConcreteStage()
        with pytest.raises(ValueError):
            stage("invalid input")

    def test_process_method_called(self):
        """Test that process method is called."""
        stage = ConcreteStage()
        with patch.object(stage, 'process', return_value={"data": "test"}) as mock_process:
            result = stage({"input": "test"})
            mock_process.assert_called_once_with({"input": "test"})


@pytest.mark.unit
@pytest.mark.stages
class TestInboxStage:
    """Comprehensive tests for InboxStage."""

    @patch('sap_llm.stages.inbox.redis.Redis')
    def test_initialization_with_redis(self, mock_redis):
        """Test initialization with Redis cache."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance

        config = Mock()
        config.databases = Mock()
        config.databases.redis = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'password': None
        }

        stage = InboxStage(config=config)
        assert stage.cache is not None

    def test_initialization_without_config(self):
        """Test initialization without config."""
        stage = InboxStage()
        assert stage.cache is None
        assert stage.visual_model is None
        assert stage.text_model is None

    @patch('sap_llm.stages.inbox.redis.Redis')
    def test_initialization_redis_failure(self, mock_redis):
        """Test handling Redis connection failure."""
        mock_redis.side_effect = Exception("Connection refused")

        config = Mock()
        config.databases = Mock()
        config.databases.redis = {'host': 'localhost', 'port': 6379, 'db': 0}

        stage = InboxStage(config=config)
        assert stage.cache is None

    def test_process_missing_document_path(self):
        """Test process raises error without document_path."""
        stage = InboxStage()
        with pytest.raises(ValueError, match="document_path is required"):
            stage.process({})

    @patch('sap_llm.stages.inbox.compute_file_hash')
    @patch.object(InboxStage, '_check_cache')
    @patch.object(InboxStage, '_create_thumbnail')
    @patch.object(InboxStage, '_extract_first_page_text')
    @patch.object(InboxStage, '_classify_fast')
    def test_process_new_document(
        self, mock_classify, mock_text, mock_thumb, mock_cache, mock_hash
    ):
        """Test processing a new document."""
        mock_hash.return_value = "abc123"
        mock_cache.return_value = None
        mock_thumb.return_value = Image.new('RGB', (256, 256))
        mock_text.return_value = "Invoice for payment"
        mock_classify.return_value = ("INVOICE", 0.95)

        stage = InboxStage()
        result = stage.process({"document_path": "/path/to/doc.pdf"})

        assert result["document_id"] == "abc123"
        assert result["document_hash"] == "abc123"
        assert result["category"] == "INVOICE"
        assert result["should_process"] is True
        assert result["cached"] is False
        assert result["confidence"] == 0.95

    @patch('sap_llm.stages.inbox.compute_file_hash')
    @patch.object(InboxStage, '_check_cache')
    def test_process_cached_document(self, mock_cache, mock_hash):
        """Test processing a cached document."""
        mock_hash.return_value = "abc123"
        mock_cache.return_value = {
            "category": "INVOICE",
            "result": "cached_data"
        }

        stage = InboxStage()
        result = stage.process({"document_path": "/path/to/doc.pdf"})

        assert result["cached"] is True
        assert result["should_process"] is False
        assert result["confidence"] == 1.0

    def test_check_cache_no_redis(self):
        """Test cache check when Redis not available."""
        stage = InboxStage()
        result = stage._check_cache("doc_hash")
        assert result is None

    @patch('sap_llm.stages.inbox.redis.Redis')
    def test_check_cache_hit(self, mock_redis_class):
        """Test cache hit."""
        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps({"category": "INVOICE"})
        mock_redis_class.return_value = mock_redis

        config = Mock()
        config.databases = Mock()
        config.databases.redis = {'host': 'localhost', 'port': 6379, 'db': 0}

        stage = InboxStage(config=config)
        stage.cache = mock_redis

        result = stage._check_cache("doc_hash")
        assert result == {"category": "INVOICE"}

    @patch('sap_llm.stages.inbox.redis.Redis')
    def test_check_cache_miss(self, mock_redis_class):
        """Test cache miss."""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        stage = InboxStage()
        stage.cache = mock_redis

        result = stage._check_cache("doc_hash")
        assert result is None

    def test_store_in_cache_no_redis(self):
        """Test cache store when Redis not available."""
        stage = InboxStage()
        # Should not raise
        stage._store_in_cache("hash", {"data": "test"})

    @patch('sap_llm.stages.inbox.convert_from_path')
    def test_create_thumbnail_pdf(self, mock_convert):
        """Test creating thumbnail from PDF."""
        mock_image = Image.new('RGB', (1000, 1000))
        mock_convert.return_value = [mock_image]

        stage = InboxStage()
        thumbnail = stage._create_thumbnail("/path/to/doc.pdf", size=256)

        assert thumbnail.size[0] <= 256
        assert thumbnail.size[1] <= 256

    @patch('PIL.Image.open')
    def test_create_thumbnail_image(self, mock_open):
        """Test creating thumbnail from image."""
        mock_image = Image.new('RGB', (1000, 1000))
        mock_open.return_value = mock_image

        stage = InboxStage()
        thumbnail = stage._create_thumbnail("/path/to/doc.png", size=256)

        assert thumbnail is not None

    def test_classify_fast_invoice_keywords(self):
        """Test fast classification with invoice keywords."""
        stage = InboxStage()
        category, confidence = stage._classify_fast(
            Image.new('RGB', (100, 100)),
            "This is an invoice for payment amount due"
        )
        assert category == "INVOICE"
        assert confidence >= 0.80

    def test_classify_fast_po_keywords(self):
        """Test fast classification with PO keywords."""
        stage = InboxStage()
        category, confidence = stage._classify_fast(
            Image.new('RGB', (100, 100)),
            "Purchase order PO number 12345"
        )
        assert category == "PURCHASE_ORDER"
        assert confidence >= 0.80

    def test_classify_fast_so_keywords(self):
        """Test fast classification with sales order keywords."""
        stage = InboxStage()
        category, confidence = stage._classify_fast(
            Image.new('RGB', (100, 100)),
            "Sales order confirmation"
        )
        assert category == "SALES_ORDER"

    def test_classify_fast_no_keywords(self):
        """Test fast classification with no keywords."""
        stage = InboxStage()
        category, confidence = stage._classify_fast(
            Image.new('RGB', (100, 100)),
            "Random text without keywords"
        )
        assert category == "INVOICE"  # Default
        assert confidence == 0.70

    def test_classify_fast_empty_text(self):
        """Test fast classification with empty text."""
        stage = InboxStage()
        category, confidence = stage._classify_fast(
            Image.new('RGB', (100, 100)),
            ""
        )
        assert confidence == 0.70

    def test_should_process_high_confidence(self):
        """Test should_process with high confidence."""
        stage = InboxStage()
        assert stage._should_process("INVOICE", 0.95) is True

    def test_should_process_low_confidence(self):
        """Test should_process with low confidence."""
        stage = InboxStage()
        assert stage._should_process("INVOICE", 0.50) is False

    def test_should_process_other_category(self):
        """Test should_process with OTHER category."""
        stage = InboxStage()
        assert stage._should_process("OTHER", 0.99) is False

    def test_should_process_with_config_threshold(self, mock_config):
        """Test should_process respects config threshold."""
        stage = InboxStage(config=mock_config)
        assert stage._should_process("INVOICE", 0.91) is True
        assert stage._should_process("INVOICE", 0.89) is False


@pytest.mark.unit
@pytest.mark.stages
class TestPreprocessingStage:
    """Comprehensive tests for PreprocessingStage."""

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_initialization_easyocr(self, mock_reader):
        """Test initialization with EasyOCR."""
        stage = PreprocessingStage()
        assert stage.ocr_engine == "easyocr"
        assert stage.target_dpi == 300

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Mock()
        config.ocr_engine = "tesseract"
        config.target_dpi = 150
        config.languages = ["en", "de"]
        config.trocr_model_name = "custom/model"
        config.enable_video_processing = False
        config.max_keyframes = 10
        config.scene_threshold = 30.0

        with patch('sap_llm.stages.preprocessing.easyocr.Reader'):
            stage = PreprocessingStage(config=config)
            assert stage.target_dpi == 150
            assert stage.languages == ["en", "de"]

    def test_init_ocr_engine_unknown(self):
        """Test initialization with unknown OCR engine."""
        config = Mock()
        config.ocr_engine = "unknown_engine"

        with pytest.raises(ValueError, match="Unknown OCR engine"):
            PreprocessingStage(config=config)

    def test_process_missing_document_path(self):
        """Test process raises error without document_path."""
        with patch('sap_llm.stages.preprocessing.easyocr.Reader'):
            stage = PreprocessingStage()
            with pytest.raises(ValueError, match="document_path is required"):
                stage.process({})

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    @patch.object(PreprocessingStage, '_pdf_to_images')
    @patch.object(PreprocessingStage, '_enhance_image')
    @patch.object(PreprocessingStage, '_run_ocr')
    def test_process_document(
        self, mock_ocr, mock_enhance, mock_pdf, mock_reader
    ):
        """Test processing a document."""
        mock_pdf.return_value = [Image.new('RGB', (100, 100))]
        mock_enhance.return_value = Image.new('L', (100, 100))
        mock_ocr.return_value = {
            "text": "Sample text",
            "words": ["Sample", "text"],
            "boxes": [[0, 0, 50, 50], [50, 0, 100, 50]],
            "confidences": [0.9, 0.95]
        }

        stage = PreprocessingStage()
        result = stage.process({"document_path": "/path/to/doc.pdf"})

        assert "pages" in result
        assert "enhanced_images" in result
        assert "ocr_results" in result
        assert result["num_pages"] == 1
        assert result["document_type"] == "pdf"

    @patch('sap_llm.stages.preprocessing.convert_from_path')
    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_pdf_to_images_pdf(self, mock_reader, mock_convert):
        """Test PDF to images conversion."""
        mock_images = [Image.new('RGB', (100, 100)), Image.new('RGB', (100, 100))]
        mock_convert.return_value = mock_images

        stage = PreprocessingStage()
        images = stage._pdf_to_images("/path/to/doc.pdf")

        assert len(images) == 2
        mock_convert.assert_called_once()

    @patch('PIL.Image.open')
    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_pdf_to_images_image(self, mock_reader, mock_open):
        """Test loading image file directly."""
        mock_image = Image.new('RGB', (100, 100))
        mock_open.return_value = mock_image

        stage = PreprocessingStage()
        images = stage._pdf_to_images("/path/to/doc.png")

        assert len(images) == 1

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_enhance_image_rgb(self, mock_reader):
        """Test image enhancement with RGB image."""
        stage = PreprocessingStage()
        image = Image.new('RGB', (100, 100), color=(128, 128, 128))

        enhanced = stage._enhance_image(image)

        assert enhanced is not None
        assert isinstance(enhanced, Image.Image)

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_enhance_image_grayscale(self, mock_reader):
        """Test image enhancement with grayscale image."""
        stage = PreprocessingStage()
        image = Image.new('L', (100, 100), color=128)

        enhanced = stage._enhance_image(image)

        assert enhanced is not None

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_deskew_empty_image(self, mock_reader):
        """Test deskew with empty image."""
        stage = PreprocessingStage()
        # Black image with no text
        image = np.zeros((100, 100), dtype=np.uint8)

        result = stage._deskew(image)

        assert result.shape == image.shape

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_deskew_with_content(self, mock_reader):
        """Test deskew with image content."""
        stage = PreprocessingStage()
        # Image with some white pixels
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 20:80] = 255  # Horizontal line

        result = stage._deskew(image)

        assert result.shape == image.shape

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_remove_borders(self, mock_reader):
        """Test border removal."""
        stage = PreprocessingStage()
        # Create image with border
        image = np.ones((100, 100), dtype=np.uint8) * 255

        result = stage._remove_borders(image)

        assert result.shape == image.shape

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    @patch('sap_llm.stages.preprocessing.pytesseract')
    def test_run_tesseract(self, mock_tesseract, mock_reader):
        """Test Tesseract OCR."""
        mock_tesseract.image_to_data.return_value = {
            "text": ["", "Hello", "World", ""],
            "left": [0, 10, 60, 0],
            "top": [0, 10, 10, 0],
            "width": [0, 40, 50, 0],
            "height": [0, 20, 20, 0],
            "conf": [0, 95, 90, 0]
        }
        mock_tesseract.Output = Mock()
        mock_tesseract.Output.DICT = "dict"

        config = Mock()
        config.ocr_engine = "tesseract"
        config.target_dpi = 300
        config.languages = ["en"]
        config.trocr_model_name = "microsoft/trocr-base-handwritten"
        config.enable_video_processing = True
        config.max_keyframes = 30
        config.scene_threshold = 27.0

        stage = PreprocessingStage(config=config)
        image = Image.new('L', (100, 100))

        result = stage._run_tesseract(image)

        assert "text" in result
        assert "words" in result
        assert "boxes" in result
        assert len(result["words"]) == 2

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_run_easyocr(self, mock_reader):
        """Test EasyOCR."""
        mock_ocr = Mock()
        mock_ocr.readtext.return_value = [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "Hello", 0.95),
            ([[60, 0], [100, 0], [100, 20], [60, 20]], "World", 0.90),
        ]
        mock_reader.return_value = mock_ocr

        stage = PreprocessingStage()
        image = Image.new('L', (100, 100))

        result = stage._run_easyocr(image)

        assert len(result["words"]) == 2
        assert result["words"][0] == "Hello"
        assert result["confidences"][0] == 0.95

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_validate_temporal_consistency_single_frame(self, mock_reader):
        """Test temporal consistency with single frame."""
        stage = PreprocessingStage()
        result = stage._validate_temporal_consistency([{"text": "test"}])
        assert result == 1.0

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_validate_temporal_consistency_multiple_frames(self, mock_reader):
        """Test temporal consistency with multiple frames."""
        stage = PreprocessingStage()
        ocr_results = [
            {"text": "Invoice 12345"},
            {"text": "Invoice 12345"},
            {"text": "Total: $100"}
        ]
        result = stage._validate_temporal_consistency(ocr_results)
        assert 0.0 <= result <= 1.0


@pytest.mark.unit
@pytest.mark.stages
class TestClassificationStage:
    """Comprehensive tests for ClassificationStage."""

    def test_initialization(self):
        """Test stage initialization."""
        stage = ClassificationStage()
        assert stage.model is None  # Lazy loading
        assert stage.confidence_threshold == 0.90

    def test_initialization_with_config(self):
        """Test initialization with config."""
        config = Mock()
        config.confidence_threshold = 0.85

        stage = ClassificationStage(config=config)
        assert stage.confidence_threshold == 0.85

    def test_document_types_defined(self):
        """Test that document types are properly defined."""
        assert len(ClassificationStage.DOCUMENT_TYPES) >= 15
        assert "PURCHASE_ORDER" in ClassificationStage.DOCUMENT_TYPES
        assert "SUPPLIER_INVOICE" in ClassificationStage.DOCUMENT_TYPES

    @patch.object(ClassificationStage, '_load_model')
    def test_process(self, mock_load):
        """Test document classification."""
        stage = ClassificationStage()

        # Mock model
        mock_model = Mock()
        mock_model.classify.return_value = (1, 0.95)  # SUPPLIER_INVOICE
        stage.model = mock_model

        input_data = {
            "enhanced_images": [Image.new('RGB', (100, 100))],
            "ocr_results": [{
                "words": ["Invoice", "Number"],
                "boxes": [[0, 0, 50, 50], [50, 0, 100, 50]]
            }]
        }

        result = stage.process(input_data)

        assert result["doc_type"] == "SUPPLIER_INVOICE"
        assert result["confidence"] == 0.95
        assert result["class_index"] == 1

    @patch('sap_llm.stages.classification.VisionEncoder')
    def test_load_model(self, mock_encoder):
        """Test model loading."""
        stage = ClassificationStage()
        stage._load_model()

        mock_encoder.assert_called_once()
        assert stage.model is not None

    def test_load_model_lazy(self):
        """Test that model is lazy loaded."""
        stage = ClassificationStage()
        assert stage.model is None

        # Model should not be loaded until needed
        with patch.object(stage, '_load_model') as mock_load:
            # Just creating stage doesn't load model
            assert mock_load.call_count == 0


@pytest.mark.unit
@pytest.mark.stages
class TestStageIntegration:
    """Integration tests for stage interactions."""

    def test_base_stage_concrete_implementation(self):
        """Test that concrete stages work with base class."""
        stage = ConcreteStage()
        result = stage({"key": "value"})
        assert "result" in result

    @patch('sap_llm.stages.preprocessing.easyocr.Reader')
    def test_preprocessing_output_format(self, mock_reader):
        """Test preprocessing output is suitable for classification."""
        mock_reader.return_value.readtext.return_value = []

        with patch.object(PreprocessingStage, '_pdf_to_images') as mock_pdf:
            mock_pdf.return_value = [Image.new('RGB', (100, 100))]

            stage = PreprocessingStage()
            result = stage.process({"document_path": "/path/doc.pdf"})

            # Check output has required keys for classification
            assert "enhanced_images" in result
            assert "ocr_results" in result
            assert len(result["enhanced_images"]) > 0
