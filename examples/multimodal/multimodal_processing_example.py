"""
Example: Advanced Multi-Modal Invoice Processing.

Demonstrates processing invoices from multiple modalities:
- Video documents (invoice video recordings)
- Audio descriptions (spoken invoice details)
- Complex tables (nested and multi-page)
- Standard PDF/images

This example shows how SAP_LLM handles ultra-enterprise scenarios
with cutting-edge multi-modal capabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from PIL import Image

from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.models.audio_processor import AudioProcessor
from sap_llm.models.table_extractor import TableExtractor
from sap_llm.models.multimodal_fusion import AdvancedMultiModalFusion
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class MultiModalInvoiceProcessor:
    """
    Complete multi-modal invoice processing pipeline.

    Handles:
    - Video invoices (MP4, AVI, MOV)
    - Audio descriptions (WAV, MP3)
    - Complex tables (nested, multi-page)
    - Standard documents (PDF, images)
    """

    def __init__(self, config=None):
        """Initialize multi-modal processor."""
        logger.info("Initializing MultiModalInvoiceProcessor")

        # Initialize components
        self.preprocessing = PreprocessingStage(config)

        # Audio processor (Whisper)
        try:
            self.audio_processor = AudioProcessor(
                model_size="base",
                enable_diarization=False  # Set True for speaker separation
            )
            logger.info("Audio processor initialized")
        except Exception as e:
            logger.warning(f"Audio processor not available: {e}")
            self.audio_processor = None

        # Table extractor
        try:
            self.table_extractor = TableExtractor()
            logger.info("Table extractor initialized")
        except Exception as e:
            logger.warning(f"Table extractor not available: {e}")
            self.table_extractor = None

        # Multi-modal fusion
        self.fusion = AdvancedMultiModalFusion(
            fusion_dim=768,
            num_heads=32,
            num_layers=2,
        )
        logger.info("Multi-modal fusion initialized")

    def process_video_invoice(self, video_path: str):
        """
        Process video invoice.

        Args:
            video_path: Path to video file

        Returns:
            Processing results
        """
        logger.info(f"Processing video invoice: {video_path}")

        # Step 1: Extract keyframes and OCR
        preprocessing_result = self.preprocessing.process({
            "document_path": video_path,
            "document_id": "video_001",
        })

        logger.info(
            f"Extracted {len(preprocessing_result['keyframes'])} keyframes "
            f"with temporal consistency: {preprocessing_result['temporal_consistency']:.2f}"
        )

        # Step 2: Extract tables from each keyframe
        all_tables = []
        if self.table_extractor:
            for page_num, (image, ocr_result) in enumerate(zip(
                preprocessing_result['pages'],
                preprocessing_result['ocr_results']
            )):
                tables = self.table_extractor.extract_tables(
                    image=image,
                    page_number=page_num,
                    ocr_results=ocr_result,
                )
                all_tables.extend(tables)

            logger.info(f"Extracted {len(all_tables)} tables from video")

        # Step 3: Create dummy features for fusion demo
        # In production, use actual vision encoder and language decoder
        batch_size = 1
        seq_len = 50

        vision_features = torch.randn(batch_size, seq_len, 768)
        text_features = torch.randn(batch_size, seq_len, 768)
        video_features = torch.randn(batch_size, len(preprocessing_result['keyframes']), 768)
        frame_indices = torch.tensor([[kf['index'] for kf in preprocessing_result['keyframes']]])

        # Step 4: Multi-modal fusion
        fusion_result = self.fusion(
            vision_features=vision_features,
            text_features=text_features,
            video_features=video_features,
            video_frame_indices=frame_indices,
        )

        logger.info(
            f"Multi-modal fusion complete. "
            f"Modalities: {fusion_result['modality_names']}"
        )

        return {
            "preprocessing": preprocessing_result,
            "tables": all_tables,
            "fusion": fusion_result,
        }

    def process_audio_invoice(self, audio_path: str):
        """
        Process audio invoice description.

        Args:
            audio_path: Path to audio file

        Returns:
            Processing results
        """
        logger.info(f"Processing audio invoice: {audio_path}")

        if not self.audio_processor:
            logger.error("Audio processor not available")
            return None

        # Step 1: Transcribe audio
        audio_result = self.audio_processor.extract_invoice_data(
            audio_path=audio_path,
            language=None  # Auto-detect
        )

        logger.info(
            f"Audio transcription complete:\n"
            f"  Language: {audio_result['language']}\n"
            f"  Duration: {audio_result['duration']:.1f}s\n"
            f"  Confidence: {audio_result['confidence']:.2f}\n"
            f"  Text: {audio_result['transcription'][:100]}..."
        )

        # Step 2: Create features for fusion
        batch_size = 1
        seq_len = 50

        text_features = torch.randn(batch_size, seq_len, 768)
        audio_features = torch.randn(batch_size, seq_len, 768)
        audio_confidence = torch.tensor([[audio_result['confidence']] * seq_len])

        # Step 3: Multi-modal fusion
        fusion_result = self.fusion(
            text_features=text_features,
            audio_features=audio_features,
            audio_confidence=audio_confidence,
        )

        logger.info(
            f"Multi-modal fusion complete. "
            f"Modalities: {fusion_result['modality_names']}"
        )

        return {
            "audio": audio_result,
            "fusion": fusion_result,
        }

    def process_complex_table_invoice(self, image_path: str):
        """
        Process invoice with complex tables.

        Args:
            image_path: Path to image file

        Returns:
            Processing results
        """
        logger.info(f"Processing complex table invoice: {image_path}")

        # Step 1: Preprocessing and OCR
        preprocessing_result = self.preprocessing.process({
            "document_path": image_path,
            "document_id": "table_001",
        })

        # Step 2: Extract complex tables
        all_tables = []
        if self.table_extractor:
            for page_num, (image, ocr_result) in enumerate(zip(
                preprocessing_result['pages'],
                preprocessing_result['ocr_results']
            )):
                tables = self.table_extractor.extract_tables(
                    image=image,
                    page_number=page_num,
                    ocr_results=ocr_result,
                )
                all_tables.extend(tables)

                # Validate tables
                for table in tables:
                    validation = self.table_extractor.validate_table(table)
                    logger.info(
                        f"Table {table.table_id}: "
                        f"{table.rows}x{table.cols}, "
                        f"confidence={table.confidence:.2f}, "
                        f"nested={table.is_nested}, "
                        f"valid={validation['is_valid']}"
                    )

            logger.info(f"Extracted {len(all_tables)} tables")

        return {
            "preprocessing": preprocessing_result,
            "tables": all_tables,
        }

    def process_multimodal_invoice(
        self,
        video_path: str = None,
        audio_path: str = None,
        image_path: str = None,
    ):
        """
        Process invoice with multiple modalities combined.

        Args:
            video_path: Optional video path
            audio_path: Optional audio path
            image_path: Optional image path

        Returns:
            Combined processing results
        """
        logger.info("Processing multi-modal invoice (all modalities)")

        results = {}

        # Collect features from each modality
        vision_features = None
        text_features = None
        audio_features = None
        video_features = None
        audio_confidence = None
        video_frame_indices = None

        # Process video if available
        if video_path:
            video_result = self.process_video_invoice(video_path)
            results['video'] = video_result

            # Extract features (in production, use actual encoders)
            batch_size = 1
            seq_len = 50
            vision_features = torch.randn(batch_size, seq_len, 768)
            text_features = torch.randn(batch_size, seq_len, 768)
            video_features = torch.randn(
                batch_size,
                len(video_result['preprocessing']['keyframes']),
                768
            )
            video_frame_indices = torch.tensor([[
                kf['index']
                for kf in video_result['preprocessing']['keyframes']
            ]])

        # Process audio if available
        if audio_path and self.audio_processor:
            audio_result = self.audio_processor.extract_invoice_data(audio_path)
            results['audio'] = audio_result

            batch_size = 1
            seq_len = 50
            audio_features = torch.randn(batch_size, seq_len, 768)
            audio_confidence = torch.tensor([[audio_result['confidence']] * seq_len])

            # Initialize text features if not already done
            if text_features is None:
                text_features = torch.randn(batch_size, seq_len, 768)

        # Process image if available
        if image_path:
            image_result = self.process_complex_table_invoice(image_path)
            results['image'] = image_result

            # Initialize features if not already done
            if vision_features is None:
                batch_size = 1
                seq_len = 50
                vision_features = torch.randn(batch_size, seq_len, 768)
            if text_features is None:
                text_features = torch.randn(batch_size, seq_len, 768)

        # Perform multi-modal fusion
        fusion_result = self.fusion(
            vision_features=vision_features,
            text_features=text_features,
            audio_features=audio_features,
            video_features=video_features,
            audio_confidence=audio_confidence,
            video_frame_indices=video_frame_indices,
        )

        results['fusion'] = fusion_result

        # Log fusion results
        logger.info(
            f"\nMulti-modal Fusion Results:\n"
            f"  Modalities used: {fusion_result['modality_names']}\n"
            f"  Modality weights: {fusion_result['modality_weights'].squeeze().tolist()}\n"
            f"  Consistency scores: {fusion_result['consistency_scores']}"
        )

        return results


def main():
    """Run multi-modal processing examples."""
    logger.info("=" * 80)
    logger.info("SAP_LLM Multi-Modal Invoice Processing Example")
    logger.info("=" * 80)

    # Initialize processor
    processor = MultiModalInvoiceProcessor()

    # Example 1: Video Invoice
    logger.info("\n" + "=" * 80)
    logger.info("Example 1: Video Invoice Processing")
    logger.info("=" * 80)

    # Note: You need to provide actual video file
    # video_result = processor.process_video_invoice("examples/multimodal/sample_invoice.mp4")

    logger.info(
        "\nTo process video invoices:\n"
        "  processor.process_video_invoice('path/to/invoice.mp4')\n"
        "Supported formats: MP4, AVI, MOV, MKV, WEBM"
    )

    # Example 2: Audio Invoice
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Audio Invoice Processing")
    logger.info("=" * 80)

    # Note: You need to provide actual audio file
    # audio_result = processor.process_audio_invoice("examples/multimodal/sample_invoice.wav")

    logger.info(
        "\nTo process audio invoices:\n"
        "  processor.process_audio_invoice('path/to/invoice.wav')\n"
        "Supported formats: WAV, MP3, M4A, OGG, FLAC\n"
        "Supported languages: en, es, fr, de, it, pt, nl, pl, ru, zh, ja, ko, ar, hi, tr"
    )

    # Example 3: Complex Table Invoice
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Complex Table Extraction")
    logger.info("=" * 80)

    # Note: You need to provide actual image file
    # table_result = processor.process_complex_table_invoice("examples/multimodal/complex_invoice.pdf")

    logger.info(
        "\nTo process complex table invoices:\n"
        "  processor.process_complex_table_invoice('path/to/invoice.pdf')\n"
        "Features:\n"
        "  - Nested table detection\n"
        "  - Merged cell handling\n"
        "  - Multi-page table tracking\n"
        "  - Hierarchical line items\n"
        "  - Total/subtotal validation"
    )

    # Example 4: Combined Multi-Modal
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Combined Multi-Modal Processing")
    logger.info("=" * 80)

    logger.info(
        "\nTo process with all modalities:\n"
        "  processor.process_multimodal_invoice(\n"
        "      video_path='invoice_video.mp4',\n"
        "      audio_path='invoice_audio.wav',\n"
        "      image_path='invoice_document.pdf'\n"
        "  )\n"
        "\n"
        "The system will:\n"
        "  1. Extract keyframes from video\n"
        "  2. Transcribe audio with Whisper\n"
        "  3. Extract complex tables from documents\n"
        "  4. Fuse all modalities with attention-based fusion\n"
        "  5. Perform cross-modal consistency checks\n"
        "  6. Weight modalities by confidence\n"
        "  7. Generate unified invoice extraction\n"
        "\n"
        "Target Performance:\n"
        "  - Video: Extract from 30fps in <5s\n"
        "  - Audio: WER <5%\n"
        "  - Tables: ≥90% accuracy on nested tables\n"
        "  - Fusion: ≥95% accuracy with multiple modalities\n"
        "  - Total latency: <3s per document"
    )

    logger.info("\n" + "=" * 80)
    logger.info("Example Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
