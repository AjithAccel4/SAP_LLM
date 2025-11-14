"""
Stage 5: Extraction - Field-Level Data Extraction

Uses Language Decoder (LLaMA-2-7B) with constrained decoding
to extract structured ADC JSON from documents.
"""

from typing import Any, Dict

from sap_llm.models.language_decoder import LanguageDecoder
from sap_llm.models.vision_encoder import VisionEncoder
from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ExtractionStage(BaseStage):
    """
    Field extraction stage.

    Extracts 180+ fields into ADC (Adaptive Document Contract) JSON format.
    Uses combined vision-language approach:
    - Vision Encoder: LayoutLMv3 for visual features
    - Language Decoder: LLaMA-2-7B for structured generation

    Accuracy target: ≥92% field-level F1
    Latency: <800ms per document
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

        # Lazy load models
        self.vision_encoder = None
        self.language_decoder = None

        # Load schemas
        self.schemas = self._load_schemas()

    def _load_models(self):
        """Lazy load extraction models."""
        if self.vision_encoder is None:
            logger.info("Loading vision encoder...")
            self.vision_encoder = VisionEncoder(
                device="cuda",
                precision="fp16",
            )

        if self.language_decoder is None:
            logger.info("Loading language decoder...")
            self.language_decoder = LanguageDecoder(
                device="cuda",
                precision="int8",
            )

    def _load_schemas(self) -> Dict[str, Dict]:
        """Load ADC schemas for each document type."""
        # TODO: Load from files
        return {
            "PURCHASE_ORDER": {
                "type": "object",
                "properties": {
                    "po_number": {"type": "string"},
                    "po_date": {"type": "string"},
                    "vendor_id": {"type": "string"},
                    "vendor_name": {"type": "string"},
                    "total_amount": {"type": "number"},
                    "currency": {"type": "string"},
                    "company_code": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line_number": {"type": "integer"},
                                "material_number": {"type": "string"},
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price": {"type": "number"},
                                "total_price": {"type": "number"},
                            },
                        },
                    },
                },
                "required": ["po_number", "vendor_id", "total_amount"],
            },
            "SUPPLIER_INVOICE": {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "invoice_date": {"type": "string"},
                    "supplier_id": {"type": "string"},
                    "supplier_name": {"type": "string"},
                    "po_number": {"type": "string"},
                    "total_amount": {"type": "number"},
                    "tax_amount": {"type": "number"},
                    "subtotal": {"type": "number"},
                    "currency": {"type": "string"},
                    "payment_terms": {"type": "string"},
                    "due_date": {"type": "string"},
                },
                "required": ["invoice_number", "supplier_id", "total_amount"],
            },
        }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract fields from document.

        Args:
            input_data: {
                "doc_type": str,
                "subtype": str,
                "enhanced_images": List[Image],
                "ocr_results": List[Dict],
            }

        Returns:
            {
                "extracted_data": Dict,  # ADC JSON
                "extraction_metadata": Dict,
                "field_confidences": Dict[str, float],
            }
        """
        # Load models
        self._load_models()

        doc_type = input_data["doc_type"]
        image = input_data["enhanced_images"][0]
        ocr_result = input_data["ocr_results"][0]

        # Get schema
        schema = self.schemas.get(doc_type, {})
        if not schema:
            raise ValueError(f"No schema found for {doc_type}")

        # Extract visual features
        visual_features = self.vision_encoder.encode(
            image,
            ocr_result["words"],
            ocr_result["boxes"],
        )

        # Extract fields
        extracted_data = self.language_decoder.extract_fields(
            ocr_result["text"],
            doc_type,
            schema,
            visual_features=visual_features,
        )

        logger.info(f"Extracted {len(extracted_data)} fields")

        return {
            "extracted_data": extracted_data,
            "extraction_metadata": {
                "doc_type": doc_type,
                "schema_version": "1.0",
                "num_fields": len(extracted_data),
            },
            "field_confidences": self._estimate_field_confidences(extracted_data),
        }

    def _estimate_field_confidences(self, data: Dict) -> Dict[str, float]:
        """Estimate confidence for each extracted field."""
        # TODO: Implement actual confidence estimation
        confidences = {}
        for field in data.keys():
            # For now, assign default confidence
            confidences[field] = 0.90

        return confidences
