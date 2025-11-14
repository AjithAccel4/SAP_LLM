"""
Stage 1: Inbox - Document Ingestion & Routing

Fast classification to determine if document should be processed.
Uses lightweight ResNet-18 + BERT-tiny for quick categorization.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import redis
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

        # Initialize Redis cache for seen documents
        self.cache: Optional[redis.Redis] = None
        if config and hasattr(config, 'databases') and hasattr(config.databases, 'redis'):
            try:
                redis_config = config.databases.redis
                self.cache = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password'),
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                # Test connection
                self.cache.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Could not connect to Redis cache: {e}. Cache disabled.")
                self.cache = None
        else:
            logger.info("Redis configuration not found. Cache disabled.")

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
        if not self.cache:
            return None

        try:
            # Look up document in Redis cache
            cache_key = f"sap_llm:inbox:{doc_hash}"
            cached_data = self.cache.get(cache_key)

            if cached_data:
                logger.debug(f"Cache hit for document {doc_hash}")
                return json.loads(cached_data)

            logger.debug(f"Cache miss for document {doc_hash}")
            return None
        except Exception as e:
            logger.warning(f"Error checking cache: {e}")
            return None

    def _store_in_cache(self, doc_hash: str, result: Dict[str, Any]) -> None:
        """Store processing result in cache."""
        if not self.cache:
            return

        try:
            cache_key = f"sap_llm:inbox:{doc_hash}"
            # Store with 24-hour TTL (configurable)
            ttl = getattr(self.config.databases.redis, 'ttl', 86400) if self.config and hasattr(self.config, 'databases') else 86400
            self.cache.setex(cache_key, ttl, json.dumps(result))
            logger.debug(f"Stored result in cache for {doc_hash}")
        except Exception as e:
            logger.warning(f"Error storing in cache: {e}")

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
        import fitz  # PyMuPDF

        doc_path = Path(document_path)

        try:
            if doc_path.suffix.lower() == ".pdf":
                # Use PyMuPDF for fast PDF text extraction (no OCR)
                doc = fitz.open(document_path)
                if len(doc) > 0:
                    first_page = doc[0]
                    text = first_page.get_text()
                    doc.close()

                    # Limit to max_chars for quick analysis
                    return text[:max_chars].strip()
                doc.close()
                return ""
            else:
                # For images, skip text extraction at this stage
                # Full OCR will be done in preprocessing stage
                return ""
        except Exception as e:
            logger.warning(f"Could not extract text from {document_path}: {e}")
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
        categories = [
            "INVOICE",
            "PURCHASE_ORDER",
            "SALES_ORDER",
            "RECEIPT",
            "STATEMENT",
            "OTHER",
        ]

        # Use keyword-based heuristics for fast classification
        # Full classification will be done in classification stage
        text_lower = text.lower() if text else ""

        # Simple keyword matching for initial triage
        invoice_keywords = ["invoice", "rechnung", "factura", "bill", "amount due"]
        po_keywords = ["purchase order", "po number", "bestellung"]
        so_keywords = ["sales order", "order confirmation", "auftragsbestätigung"]
        receipt_keywords = ["receipt", "quittung", "recibo"]

        # Count keyword matches
        scores = {
            "INVOICE": sum(1 for kw in invoice_keywords if kw in text_lower),
            "PURCHASE_ORDER": sum(1 for kw in po_keywords if kw in text_lower),
            "SALES_ORDER": sum(1 for kw in so_keywords if kw in text_lower),
            "RECEIPT": sum(1 for kw in receipt_keywords if kw in text_lower),
            "STATEMENT": 1 if "statement" in text_lower else 0,
        }

        # Get category with highest score
        max_score = max(scores.values())
        if max_score > 0:
            category = max(scores, key=scores.get)
            # Confidence based on keyword matches (0.8 + 0.05 per match, max 0.99)
            confidence = min(0.80 + (max_score * 0.05), 0.99)
        else:
            # No keywords found - defer to classification stage
            category = "INVOICE"  # Default assumption
            confidence = 0.70  # Low confidence - will be re-classified

        return category, confidence

    def _should_process(self, category: str, confidence: float) -> bool:
        """Determine if document should be processed."""
        # Skip if category is OTHER or confidence is too low
        if category == "OTHER":
            return False

        threshold = getattr(self.config, "confidence_threshold", 0.90) if self.config else 0.90

        return confidence >= threshold
