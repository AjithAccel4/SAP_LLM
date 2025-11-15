"""
Stage 5: Extraction - Field-Level Data Extraction

Uses Language Decoder (LLaMA-2-7B) with constrained decoding
to extract structured ADC JSON from documents.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
        """
        Load ADC schemas for each document type from JSON files.

        Falls back to hardcoded schemas if files don't exist.
        """
        schemas = {}

        # Define schema directory path
        schema_dir = Path(__file__).parent.parent.parent / "data" / "schemas"

        # List of document types to load
        doc_types = ["PURCHASE_ORDER", "SUPPLIER_INVOICE"]

        for doc_type in doc_types:
            schema_file = schema_dir / f"{doc_type}.json"

            try:
                if schema_file.exists():
                    with open(schema_file, 'r') as f:
                        schemas[doc_type] = json.load(f)
                    logger.info(f"Loaded schema for {doc_type} from {schema_file}")
                else:
                    # Fallback to hardcoded schema
                    logger.warning(f"Schema file not found: {schema_file}, using hardcoded schema")
                    schemas[doc_type] = self._get_hardcoded_schema(doc_type)

            except Exception as e:
                logger.error(f"Error loading schema for {doc_type}: {e}, using hardcoded schema")
                schemas[doc_type] = self._get_hardcoded_schema(doc_type)

        return schemas

    def _get_hardcoded_schema(self, doc_type: str) -> Dict[str, Any]:
        """
        Get hardcoded fallback schema for a document type.

        Args:
            doc_type: Document type name

        Returns:
            Schema dictionary
        """
        hardcoded_schemas = {
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

        return hardcoded_schemas.get(doc_type, {})

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

        # Estimate field confidences with OCR data
        field_confidences = self._estimate_field_confidences(
            extracted_data,
            schema,
            ocr_result,
        )

        return {
            "extracted_data": extracted_data,
            "extraction_metadata": {
                "doc_type": doc_type,
                "schema_version": "1.0",
                "num_fields": len(extracted_data),
            },
            "field_confidences": field_confidences,
        }

    def _estimate_field_confidences(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        ocr_result: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Estimate confidence for each extracted field.

        Confidence is calculated based on multiple factors:
        1. Field completeness: Whether required/optional fields are populated
        2. OCR quality scores: Average OCR confidence for field values
        3. Model prediction scores: Estimated from field value characteristics
        4. Schema validation: Whether field matches expected type/format

        Args:
            data: Extracted field data
            schema: ADC schema for the document type
            ocr_result: OCR results with confidence scores

        Returns:
            Dictionary mapping field names to confidence scores (0.0-1.0)
        """
        confidences = {}

        # Get required and optional fields from schema
        required_fields = set(schema.get("required", []))
        all_fields = set(schema.get("properties", {}).keys())
        optional_fields = all_fields - required_fields

        # Calculate average OCR confidence
        ocr_confidences = ocr_result.get("confidences", [])
        avg_ocr_confidence = np.mean(ocr_confidences) if ocr_confidences else 0.85

        # Process each field in extracted data
        for field_name, field_value in data.items():
            confidence_components = []

            # 1. Field completeness score
            completeness_score = self._calculate_completeness_score(
                field_name,
                field_value,
                required_fields,
                optional_fields,
            )
            confidence_components.append(("completeness", completeness_score, 0.25))

            # 2. OCR quality score
            ocr_quality_score = self._calculate_ocr_quality_score(
                field_value,
                ocr_result,
            )
            confidence_components.append(("ocr_quality", ocr_quality_score, 0.30))

            # 3. Model prediction score (based on field characteristics)
            prediction_score = self._calculate_prediction_score(
                field_name,
                field_value,
                schema.get("properties", {}).get(field_name, {}),
            )
            confidence_components.append(("prediction", prediction_score, 0.30))

            # 4. Schema validation score
            validation_score = self._calculate_validation_score(
                field_value,
                schema.get("properties", {}).get(field_name, {}),
            )
            confidence_components.append(("validation", validation_score, 0.15))

            # Calculate weighted average
            total_confidence = sum(score * weight for _, score, weight in confidence_components)

            # Apply penalty for required fields
            if field_name in required_fields and field_value in [None, "", [], {}]:
                total_confidence *= 0.5  # Significant penalty for missing required fields

            # Clip to valid range
            total_confidence = max(0.0, min(1.0, total_confidence))

            confidences[field_name] = round(total_confidence, 3)

            logger.debug(
                f"Field '{field_name}' confidence: {total_confidence:.3f} "
                f"(components: {', '.join(f'{name}={score:.2f}' for name, score, _ in confidence_components)})"
            )

        return confidences

    def _calculate_completeness_score(
        self,
        field_name: str,
        field_value: Any,
        required_fields: set,
        optional_fields: set,
    ) -> float:
        """
        Calculate completeness score based on field presence and content.

        Args:
            field_name: Name of the field
            field_value: Value of the field
            required_fields: Set of required field names
            optional_fields: Set of optional field names

        Returns:
            Completeness score (0.0-1.0)
        """
        # Check if field has meaningful content
        is_empty = field_value in [None, "", [], {}]

        if field_name in required_fields:
            # Required fields: high score if present, low if empty
            return 0.95 if not is_empty else 0.30
        elif field_name in optional_fields:
            # Optional fields: medium-high score if present, medium if empty
            return 0.85 if not is_empty else 0.60
        else:
            # Unknown fields: neutral score
            return 0.70

    def _calculate_ocr_quality_score(
        self,
        field_value: Any,
        ocr_result: Dict[str, Any],
    ) -> float:
        """
        Calculate OCR quality score by matching field value with OCR text.

        Args:
            field_value: Extracted field value
            ocr_result: OCR results with text and confidence scores

        Returns:
            OCR quality score (0.0-1.0)
        """
        # Convert field value to string for matching
        if isinstance(field_value, (list, dict)):
            # For complex types, use structural confidence
            return 0.85

        field_str = str(field_value) if field_value is not None else ""

        if not field_str:
            return 0.50

        # Get OCR data
        ocr_text = ocr_result.get("text", "")
        ocr_words = ocr_result.get("words", [])
        ocr_confidences = ocr_result.get("confidences", [])

        if not ocr_confidences:
            return 0.80  # Default score if no OCR confidence available

        # Try to find field value in OCR results
        matched_confidences = []

        # Split field value into tokens
        field_tokens = field_str.split()

        for token in field_tokens:
            if len(token) < 2:  # Skip very short tokens
                continue

            # Find matching OCR words
            for i, ocr_word in enumerate(ocr_words):
                if token.lower() in ocr_word.lower() or ocr_word.lower() in token.lower():
                    if i < len(ocr_confidences):
                        matched_confidences.append(ocr_confidences[i])

        if matched_confidences:
            # Return average of matched OCR confidences
            return float(np.mean(matched_confidences))
        else:
            # No matches found, return overall OCR confidence
            return float(np.mean(ocr_confidences))

    def _calculate_prediction_score(
        self,
        field_name: str,
        field_value: Any,
        field_schema: Dict[str, Any],
    ) -> float:
        """
        Calculate model prediction score based on field characteristics.

        Args:
            field_name: Name of the field
            field_value: Value of the field
            field_schema: Schema definition for the field

        Returns:
            Prediction score (0.0-1.0)
        """
        if field_value in [None, "", [], {}]:
            return 0.40

        field_type = field_schema.get("type", "string")

        # Type-specific scoring
        if field_type == "number":
            # Numbers: check if valid and reasonable
            try:
                num_val = float(field_value)
                # Reasonable range check (e.g., amounts should be positive)
                if num_val >= 0:
                    return 0.90
                else:
                    return 0.70
            except (ValueError, TypeError):
                return 0.50

        elif field_type == "string":
            # Strings: check length and content
            str_val = str(field_value)
            if len(str_val) == 0:
                return 0.40
            elif len(str_val) < 3:
                return 0.70  # Very short strings are less reliable
            elif len(str_val) > 200:
                return 0.75  # Very long strings might be extraction errors
            else:
                return 0.88

        elif field_type == "array":
            # Arrays: check if items are present
            if isinstance(field_value, list):
                if len(field_value) > 0:
                    return 0.85
                else:
                    return 0.60
            else:
                return 0.50

        elif field_type == "object":
            # Objects: check if has properties
            if isinstance(field_value, dict):
                if len(field_value) > 0:
                    return 0.85
                else:
                    return 0.60
            else:
                return 0.50

        else:
            # Unknown type: neutral score
            return 0.75

    def _calculate_validation_score(
        self,
        field_value: Any,
        field_schema: Dict[str, Any],
    ) -> float:
        """
        Calculate validation score based on schema compliance.

        Args:
            field_value: Value of the field
            field_schema: Schema definition for the field

        Returns:
            Validation score (0.0-1.0)
        """
        if field_value in [None, "", [], {}]:
            return 0.50

        field_type = field_schema.get("type", "string")

        # Type checking
        type_valid = False

        if field_type == "string":
            type_valid = isinstance(field_value, str)
        elif field_type == "number":
            type_valid = isinstance(field_value, (int, float))
        elif field_type == "integer":
            type_valid = isinstance(field_value, int)
        elif field_type == "array":
            type_valid = isinstance(field_value, list)
        elif field_type == "object":
            type_valid = isinstance(field_value, dict)
        else:
            type_valid = True  # Unknown type, assume valid

        # Return high score if type matches, lower if not
        return 0.95 if type_valid else 0.60
