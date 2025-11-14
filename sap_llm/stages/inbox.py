"""
Stage 1: Inbox - Document Ingestion & Routing

Fast classification to determine if document should be processed.
Uses lightweight ResNet-18 + BERT-tiny for quick categorization.
"""

from pathlib import Path
from typing import Any, Dict

from PIL import Image

from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.hash import compute_file_hash
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class InboxStage(BaseStage):
    """
    Inbox stage for initial document triage.

    Performs fast classification to determine:
    - If document is processable
    - Initial category routing
    - Cache lookup for duplicate detection

    Model: ResNet-18 (image) + BERT-tiny (text) = ~50M params
    Latency target: <50ms
    Accuracy target: 99.5%
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

        # Initialize models (lazy loading)
        self.visual_model = None
        self.text_model = None
        self.cache = None  # Redis cache for seen documents

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document through inbox stage.

        Args:
            input_data: {
                "document_path": str,  # Path to PDF/image
                "metadata": dict,       # Optional metadata
            }

        Returns:
            {
                "document_id": str,
                "document_hash": str,
                "category": str,
                "should_process": bool,
                "cached": bool,
                "confidence": float,
            }
        """
        document_path = input_data.get("document_path")
        if not document_path:
            raise ValueError("document_path is required")

        # Compute document hash
        doc_hash = compute_file_hash(document_path)
        logger.debug(f"Document hash: {doc_hash}")

        # Check cache
        cached_result = self._check_cache(doc_hash)
        if cached_result:
            logger.info(f"Document found in cache: {doc_hash}")
            return {
                "document_id": doc_hash,
                "document_hash": doc_hash,
                "category": cached_result["category"],
                "should_process": False,
                "cached": True,
                "confidence": 1.0,
                "cached_result": cached_result,
            }

        # Load document thumbnail
        thumbnail = self._create_thumbnail(document_path)

        # Extract first page text
        first_page_text = self._extract_first_page_text(document_path)

        # Fast classification
        category, confidence = self._classify_fast(thumbnail, first_page_text)

        # Determine if should process
        should_process = self._should_process(category, confidence)

        return {
            "document_id": doc_hash,
            "document_hash": doc_hash,
            "category": category,
            "should_process": should_process,
            "cached": False,
            "confidence": confidence,
            "thumbnail": thumbnail,
            "first_page_text": first_page_text,
        }

    def _check_cache(self, doc_hash: str) -> Dict[str, Any] | None:
        """Check if document already processed."""
        # TODO: Implement Redis cache lookup
        return None

    def _create_thumbnail(self, document_path: str, size: int = 256) -> Image.Image:
        """Create thumbnail of first page."""
        from pdf2image import convert_from_path

        doc_path = Path(document_path)

        if doc_path.suffix.lower() == ".pdf":
            # Convert PDF first page
            images = convert_from_path(
                document_path,
                first_page=1,
                last_page=1,
                dpi=150,
            )
            thumbnail = images[0]
        else:
            # Load image directly
            thumbnail = Image.open(document_path)

        # Resize to thumbnail
        thumbnail.thumbnail((size, size), Image.Resampling.LANCZOS)

        return thumbnail

    def _extract_first_page_text(self, document_path: str, max_chars: int = 500) -> str:
        """Extract text from first page for quick analysis."""
        # TODO: Implement fast OCR or PDF text extraction
        # For now, return empty string
        return ""

    def _classify_fast(self, thumbnail: Image.Image, text: str) -> tuple[str, float]:
        """
        Fast classification using lightweight models.

        Args:
            thumbnail: Document thumbnail
            text: First page text

        Returns:
            (category, confidence)
        """
        # TODO: Implement actual classification
        # For now, return default category
        categories = [
            "INVOICE",
            "PURCHASE_ORDER",
            "SALES_ORDER",
            "RECEIPT",
            "STATEMENT",
            "OTHER",
        ]

        # Placeholder classification
        category = "INVOICE"
        confidence = 0.95

        return category, confidence

    def _should_process(self, category: str, confidence: float) -> bool:
        """Determine if document should be processed."""
        # Skip if category is OTHER or confidence is too low
        if category == "OTHER":
            return False

        threshold = getattr(self.config, "confidence_threshold", 0.90) if self.config else 0.90

        return confidence >= threshold
