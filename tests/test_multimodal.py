"""
Unit tests for multi-modal processing capabilities.

Tests:
- Video processing with keyframe extraction
- Audio processing with Whisper
- Complex table extraction
- Multi-modal fusion
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

import torch
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.models.multimodal_fusion import (
    AdvancedMultiModalFusion,
    AudioFeatureEncoder,
    VideoFeatureEncoder,
    TemporalEncoder,
    CrossModalConsistencyChecker,
)


class TestVideoProcessing(unittest.TestCase):
    """Test video processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_video_format_detection(self):
        """Test video format detection."""
        preprocessor = PreprocessingStage()

        video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        for fmt in video_formats:
            self.assertIn(
                fmt,
                preprocessor.VIDEO_FORMATS,
                f"Format {fmt} should be supported"
            )

    @patch('sap_llm.stages.preprocessing.SCENEDETECT_AVAILABLE', False)
    def test_video_processing_without_scenedetect(self):
        """Test error handling when PySceneDetect not available."""
        preprocessor = PreprocessingStage()

        # Create mock video path
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            with self.assertRaises(ImportError):
                preprocessor.process({
                    "document_path": video_path,
                    "document_id": "test_video"
                })
        finally:
            Path(video_path).unlink()

    def test_temporal_consistency_calculation(self):
        """Test temporal consistency calculation."""
        preprocessor = PreprocessingStage()

        # Mock OCR results
        ocr_results = [
            {"text": "Invoice #12345 Total: $100.00"},
            {"text": "Invoice #12345 Total: $100.00"},  # Same text (high consistency)
        ]

        consistency = preprocessor._validate_temporal_consistency(ocr_results)

        # High similarity should result in high consistency
        self.assertGreaterEqual(consistency, 0.5)

    def test_temporal_consistency_with_variation(self):
        """Test temporal consistency with varying text."""
        preprocessor = PreprocessingStage()

        ocr_results = [
            {"text": "Invoice #12345"},
            {"text": "Totally different text"},  # Low similarity
        ]

        consistency = preprocessor._validate_temporal_consistency(ocr_results)

        # Low similarity should result in lower consistency
        self.assertLess(consistency, 0.8)


class TestAudioProcessing(unittest.TestCase):
    """Test audio processing functionality."""

    @patch('sap_llm.models.audio_processor.WHISPER_AVAILABLE', False)
    def test_audio_processor_without_whisper(self):
        """Test error when Whisper not available."""
        from sap_llm.models.audio_processor import AudioProcessor

        with self.assertRaises(ImportError):
            AudioProcessor()

    def test_supported_audio_formats(self):
        """Test supported audio formats."""
        from sap_llm.models.audio_processor import AudioProcessor

        expected_formats = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

        # Check class attribute
        self.assertEqual(
            AudioProcessor.SUPPORTED_FORMATS,
            expected_formats
        )

    def test_supported_languages(self):
        """Test supported languages list."""
        from sap_llm.models.audio_processor import AudioProcessor

        # Should support at least 10 languages
        self.assertGreaterEqual(
            len(AudioProcessor.SUPPORTED_LANGUAGES),
            10
        )

        # Check some common languages
        common_langs = ["en", "es", "fr", "de", "zh"]
        for lang in common_langs:
            self.assertIn(
                lang,
                AudioProcessor.SUPPORTED_LANGUAGES,
                f"Language '{lang}' should be supported"
            )


class TestTableExtraction(unittest.TestCase):
    """Test table extraction functionality."""

    @patch('sap_llm.models.table_extractor.TRANSFORMERS_AVAILABLE', False)
    def test_table_extractor_without_transformers(self):
        """Test error when Transformers not available."""
        from sap_llm.models.table_extractor import TableExtractor

        with self.assertRaises(ImportError):
            TableExtractor()

    def test_cell_type_enum(self):
        """Test CellType enum."""
        from sap_llm.models.table_extractor import CellType

        expected_types = [
            "HEADER", "DATA", "MERGED_HORIZONTAL",
            "MERGED_VERTICAL", "SUBTOTAL", "TOTAL"
        ]

        for cell_type in expected_types:
            self.assertTrue(
                hasattr(CellType, cell_type),
                f"CellType should have {cell_type}"
            )

    def test_table_cell_dataclass(self):
        """Test TableCell dataclass."""
        from sap_llm.models.table_extractor import TableCell, CellType

        cell = TableCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            text="Test",
            bbox=[0, 0, 100, 100],
            confidence=0.9,
            cell_type=CellType.DATA,
        )

        self.assertEqual(cell.row, 0)
        self.assertEqual(cell.col, 0)
        self.assertEqual(cell.text, "Test")
        self.assertEqual(cell.confidence, 0.9)


class TestMultiModalFusion(unittest.TestCase):
    """Test multi-modal fusion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.dim = 768

    def test_temporal_encoder(self):
        """Test temporal encoder."""
        encoder = TemporalEncoder(d_model=self.dim, max_frames=100)

        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        frame_indices = torch.randint(0, 100, (self.batch_size, self.seq_len))

        output = encoder(x, frame_indices)

        # Output shape should match input
        self.assertEqual(output.shape, x.shape)

    def test_audio_feature_encoder(self):
        """Test audio feature encoder."""
        encoder = AudioFeatureEncoder(audio_dim=self.dim, output_dim=self.dim)

        audio_features = torch.randn(self.batch_size, self.seq_len, self.dim)
        confidence_scores = torch.rand(self.batch_size, self.seq_len)

        output = encoder(audio_features, confidence_scores)

        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.dim)
        )

    def test_audio_feature_encoder_without_confidence(self):
        """Test audio encoder without confidence scores."""
        encoder = AudioFeatureEncoder(audio_dim=self.dim, output_dim=self.dim)

        audio_features = torch.randn(self.batch_size, self.seq_len, self.dim)

        output = encoder(audio_features, confidence_scores=None)

        # Should work without confidence
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.dim)
        )

    def test_video_feature_encoder(self):
        """Test video feature encoder."""
        encoder = VideoFeatureEncoder(
            video_dim=self.dim,
            output_dim=self.dim,
            max_frames=100
        )

        video_features = torch.randn(self.batch_size, self.seq_len, self.dim)
        frame_indices = torch.randint(0, 100, (self.batch_size, self.seq_len))

        output = encoder(video_features, frame_indices)

        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.dim)
        )

    def test_cross_modal_consistency_checker(self):
        """Test cross-modal consistency checker."""
        checker = CrossModalConsistencyChecker(dim=self.dim)

        features1 = torch.randn(self.batch_size, self.seq_len, self.dim)
        features2 = torch.randn(self.batch_size, self.seq_len, self.dim)

        consistency = checker(features1, features2)

        # Output should be consistency scores (0-1)
        self.assertEqual(
            consistency.shape,
            (self.batch_size, self.seq_len, 1)
        )
        self.assertTrue(torch.all(consistency >= 0))
        self.assertTrue(torch.all(consistency <= 1))

    def test_advanced_multimodal_fusion_all_modalities(self):
        """Test fusion with all modalities."""
        fusion = AdvancedMultiModalFusion(
            fusion_dim=self.dim,
            num_heads=8,
            num_layers=1,
        )

        vision_features = torch.randn(self.batch_size, self.seq_len, self.dim)
        text_features = torch.randn(self.batch_size, self.seq_len, self.dim)
        audio_features = torch.randn(self.batch_size, self.seq_len, self.dim)
        video_features = torch.randn(self.batch_size, self.seq_len, self.dim)

        result = fusion(
            vision_features=vision_features,
            text_features=text_features,
            audio_features=audio_features,
            video_features=video_features,
        )

        # Check result structure
        self.assertIn("fused_features", result)
        self.assertIn("modality_weights", result)
        self.assertIn("consistency_scores", result)
        self.assertIn("attention_maps", result)
        self.assertIn("modality_names", result)

        # Check modality names
        self.assertEqual(
            set(result["modality_names"]),
            {"vision", "text", "audio", "video"}
        )

        # Check consistency scores
        self.assertGreater(len(result["consistency_scores"]), 0)

    def test_advanced_multimodal_fusion_partial_modalities(self):
        """Test fusion with only some modalities."""
        fusion = AdvancedMultiModalFusion(
            fusion_dim=self.dim,
            num_heads=8,
            num_layers=1,
        )

        vision_features = torch.randn(self.batch_size, self.seq_len, self.dim)
        text_features = torch.randn(self.batch_size, self.seq_len, self.dim)

        result = fusion(
            vision_features=vision_features,
            text_features=text_features,
            audio_features=None,
            video_features=None,
        )

        # Should work with partial modalities
        self.assertIn("fused_features", result)

        # Check modality names
        self.assertEqual(
            set(result["modality_names"]),
            {"vision", "text"}
        )

    def test_advanced_multimodal_fusion_no_modalities(self):
        """Test error when no modalities provided."""
        fusion = AdvancedMultiModalFusion(
            fusion_dim=self.dim,
            num_heads=8,
            num_layers=1,
        )

        with self.assertRaises(ValueError):
            fusion(
                vision_features=None,
                text_features=None,
                audio_features=None,
                video_features=None,
            )

    def test_modality_weights_sum_to_one(self):
        """Test that modality weights sum to 1."""
        fusion = AdvancedMultiModalFusion(
            fusion_dim=self.dim,
            num_heads=8,
            num_layers=1,
        )

        vision_features = torch.randn(self.batch_size, self.seq_len, self.dim)
        text_features = torch.randn(self.batch_size, self.seq_len, self.dim)

        result = fusion(
            vision_features=vision_features,
            text_features=text_features,
        )

        # Modality weights should sum to 1 (softmax output)
        weights_sum = result["modality_weights"].sum(dim=-1)

        torch.testing.assert_close(
            weights_sum,
            torch.ones_like(weights_sum),
            rtol=1e-5,
            atol=1e-5
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for multi-modal processing."""

    def test_end_to_end_multimodal_pipeline(self):
        """Test complete multi-modal pipeline."""
        # This is a high-level integration test
        # In production, you'd use actual files

        batch_size = 1
        seq_len = 20
        dim = 768

        # Initialize fusion
        fusion = AdvancedMultiModalFusion(
            fusion_dim=dim,
            num_heads=8,
            num_layers=2,
        )

        # Simulate features from different modalities
        vision_features = torch.randn(batch_size, seq_len, dim)
        text_features = torch.randn(batch_size, seq_len, dim)
        audio_features = torch.randn(batch_size, seq_len, dim)

        # Fuse
        result = fusion(
            vision_features=vision_features,
            text_features=text_features,
            audio_features=audio_features,
        )

        # Verify results
        self.assertIsNotNone(result["fused_features"])
        self.assertEqual(len(result["modality_names"]), 3)
        self.assertGreater(len(result["consistency_scores"]), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestTableExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiModalFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
