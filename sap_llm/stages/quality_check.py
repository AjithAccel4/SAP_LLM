"""
Stage 6: Quality Check - Confidence Scoring & Self-Correction

Assesses extraction quality and triggers self-correction when needed.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class QualityCheckStage(BaseStage):
    """
    Quality check and self-correction stage.

    Computes overall quality score based on:
    - Field-level confidence
    - Required field completeness
    - Schema compliance
    - Business rule consistency

    Target: ≥90% quality score
    Self-correction: Enabled for scores <90%
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.quality_threshold = (
            getattr(config, "overall_threshold", 0.90) if config else 0.90
        )
        self.self_correction_enabled = (
            getattr(config, "self_correction_enabled", True) if config else True
        )
        self.max_correction_attempts = (
            getattr(config, "max_correction_attempts", 3) if config else 3
        )

        # Schema cache for required fields
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._schemas_loaded = False

        # Load schemas on initialization
        self._load_schemas()

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check extraction quality and self-correct if needed.

        Args:
            input_data: {
                "extracted_data": Dict,
                "extraction_metadata": Dict,
                "field_confidences": Dict[str, float],
                "doc_type": str,
            }

        Returns:
            {
                "quality_score": float,
                "passed": bool,
                "low_confidence_fields": List[Dict],
                "corrected": bool,
                "corrected_data": Dict | None,
            }
        """
        extracted_data = input_data["extracted_data"]
        field_confidences = input_data["field_confidences"]
        metadata = input_data.get("extraction_metadata", {})

        # Ensure doc_type is in metadata for schema validation
        if "doc_type" not in metadata and "doc_type" in input_data:
            metadata["doc_type"] = input_data["doc_type"]

        # Compute quality score
        quality_score = self._compute_quality_score(
            extracted_data,
            field_confidences,
            metadata,
        )

        logger.info(f"Quality score: {quality_score:.4f}")

        passed = quality_score >= self.quality_threshold

        # Identify low confidence fields
        low_conf_fields = self._identify_low_confidence_fields(
            extracted_data,
            field_confidences,
        )

        # Self-correction if needed
        corrected = False
        corrected_data = None

        if not passed and self.self_correction_enabled and low_conf_fields:
            logger.warning("Quality below threshold, attempting self-correction...")
            corrected_data = self._self_correct(
                extracted_data,
                low_conf_fields,
                input_data,
            )
            corrected = True

            # Re-compute quality score
            quality_score = self._compute_quality_score(
                corrected_data,
                field_confidences,
                metadata,
            )
            logger.info(f"Quality score after correction: {quality_score:.4f}")

        return {
            "quality_score": quality_score,
            "passed": quality_score >= self.quality_threshold,
            "low_confidence_fields": low_conf_fields,
            "corrected": corrected,
            "corrected_data": corrected_data if corrected else extracted_data,
        }

    def _compute_quality_score(
        self,
        data: Dict,
        confidences: Dict[str, float],
        metadata: Dict,
    ) -> float:
        """Compute overall quality score."""
        scores = []

        # 1. Field confidence score
        if confidences:
            avg_confidence = sum(confidences.values()) / len(confidences)
            scores.append(avg_confidence * 0.5)

        # 2. Completeness score
        # Get required fields from schema based on document type
        doc_type = metadata.get("doc_type", "UNKNOWN")
        required_fields = self._get_required_fields(doc_type)

        if required_fields:
            completeness = sum(1 for f in required_fields if f in data and data[f]) / max(
                len(required_fields), 1
            )
            scores.append(completeness * 0.3)
        else:
            # If no required fields defined, give neutral score
            scores.append(0.3)

        # 3. Schema compliance score
        # Validate against schema if available
        schema_valid = self._validate_schema_compliance(data, doc_type)
        scores.append(0.2 if schema_valid else 0.0)

        return sum(scores)

    def _identify_low_confidence_fields(
        self,
        data: Dict,
        confidences: Dict[str, float],
        threshold: float = 0.85,
    ) -> List[Dict]:
        """Identify fields with low confidence."""
        low_conf = []

        for field, value in data.items():
            conf = confidences.get(field, 0.0)
            if conf < threshold:
                low_conf.append({
                    "field": field,
                    "value": value,
                    "confidence": conf,
                })

        return low_conf

    def _self_correct(
        self,
        data: Dict,
        low_conf_fields: List[Dict],
        context: Dict,
    ) -> Dict:
        """
        Attempt to correct low-confidence extractions.

        Strategies:
        1. Re-extract from specific regions
        2. Use PMG similar documents
        3. Apply heuristic rules
        """
        corrected_data = data.copy()

        for field_info in low_conf_fields:
            field = field_info["field"]

            # Strategy 1: Heuristic correction
            if field in ["total_amount", "subtotal", "tax_amount"]:
                corrected_value = self._correct_monetary_field(field, corrected_data)
                if corrected_value is not None:
                    corrected_data[field] = corrected_value
                    logger.info(f"Corrected {field}: {corrected_value}")

        return corrected_data

    def _correct_monetary_field(self, field: str, data: Dict) -> float | None:
        """Apply heuristics to correct monetary fields."""
        # Check if total = subtotal + tax
        if field == "total_amount":
            if "subtotal" in data and "tax_amount" in data:
                calculated_total = data["subtotal"] + data["tax_amount"]
                return calculated_total

        return None

    def _load_schemas(self):
        """
        Load document schemas from JSON files.

        Searches for schema files in data/schemas/ directory and loads
        required field definitions for each document type.
        """
        try:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            schemas_dir = project_root / "data" / "schemas"

            if not schemas_dir.exists():
                logger.warning(f"Schemas directory not found: {schemas_dir}")
                self._load_default_schemas()
                return

            # Load all JSON schema files (excluding business_rules subdirectory)
            schema_files = [
                f for f in schemas_dir.glob("*.json")
                if f.is_file()
            ]

            if not schema_files:
                logger.warning("No schema files found in schemas directory")
                self._load_default_schemas()
                return

            loaded_count = 0
            for schema_file in schema_files:
                try:
                    # Document type is the filename without .json extension
                    doc_type = schema_file.stem

                    with open(schema_file, 'r') as f:
                        schema = json.load(f)

                    # Cache the full schema
                    self._schema_cache[doc_type] = schema
                    loaded_count += 1

                    logger.debug(f"Loaded schema for {doc_type}")

                except Exception as e:
                    logger.error(f"Failed to load schema {schema_file}: {e}")
                    continue

            logger.info(f"Loaded {loaded_count} document schemas")

            # Load default schemas for document types without schema files
            self._load_default_schemas()

            self._schemas_loaded = True

        except Exception as e:
            logger.error(f"Failed to load schemas: {e}")
            self._load_default_schemas()

    def _load_default_schemas(self):
        """
        Load default schema definitions for all 35+ SAP document types.

        Provides fallback required fields for document types without
        dedicated schema files.
        """
        # Default required fields for each major document type
        # Based on SAP business requirements and common field expectations
        default_schemas = {
            # Purchase Orders
            "PURCHASE_ORDER": {
                "required": ["po_number", "vendor_id", "total_amount"],
                "optional": ["po_date", "vendor_name", "currency", "company_code", "items"]
            },

            # Invoices
            "SUPPLIER_INVOICE": {
                "required": ["invoice_number", "supplier_id", "total_amount"],
                "optional": ["invoice_date", "supplier_name", "po_number", "tax_amount", "subtotal", "currency", "payment_terms", "due_date"]
            },
            "CUSTOMER_INVOICE": {
                "required": ["invoice_number", "customer_id", "total_amount"],
                "optional": ["invoice_date", "customer_name", "tax_amount", "subtotal", "currency", "payment_terms", "due_date"]
            },

            # Sales Orders
            "SALES_ORDER": {
                "required": ["order_number", "customer_id", "total_amount"],
                "optional": ["order_date", "customer_name", "currency", "items", "delivery_date"]
            },

            # Goods & Delivery
            "GOODS_RECEIPT": {
                "required": ["receipt_number", "po_number"],
                "optional": ["receipt_date", "vendor_id", "items", "quantity"]
            },
            "ADVANCED_SHIPPING_NOTICE": {
                "required": ["asn_number", "po_number"],
                "optional": ["ship_date", "expected_delivery_date", "vendor_id", "items"]
            },
            "DELIVERY_NOTE": {
                "required": ["delivery_number", "order_number"],
                "optional": ["delivery_date", "customer_id", "items", "quantity"]
            },

            # Credit/Debit Notes
            "CREDIT_NOTE": {
                "required": ["credit_note_number", "reference_invoice", "total_amount"],
                "optional": ["issue_date", "supplier_id", "customer_id", "reason", "currency"]
            },
            "DEBIT_NOTE": {
                "required": ["debit_note_number", "reference_invoice", "total_amount"],
                "optional": ["issue_date", "supplier_id", "customer_id", "reason", "currency"]
            },

            # Payment
            "PAYMENT_ADVICE": {
                "required": ["payment_number", "total_amount"],
                "optional": ["payment_date", "invoice_number", "vendor_id", "currency", "payment_method"]
            },
            "REMITTANCE_ADVICE": {
                "required": ["remittance_number", "total_amount"],
                "optional": ["remittance_date", "invoices", "vendor_id", "currency"]
            },

            # Statements & Quotes
            "STATEMENT_OF_ACCOUNT": {
                "required": ["statement_number", "account_id"],
                "optional": ["statement_date", "period_start", "period_end", "balance"]
            },
            "QUOTE": {
                "required": ["quote_number", "total_amount"],
                "optional": ["quote_date", "valid_until", "customer_id", "vendor_id", "items"]
            },

            # Contracts
            "CONTRACT": {
                "required": ["contract_number", "party_a", "party_b"],
                "optional": ["start_date", "end_date", "total_value", "terms"]
            },

            # Other
            "OTHER": {
                "required": ["document_number"],
                "optional": []
            }
        }

        # Add default schemas to cache if not already loaded from files
        for doc_type, schema_def in default_schemas.items():
            if doc_type not in self._schema_cache:
                # Convert to JSON Schema format
                self._schema_cache[doc_type] = {
                    "type": "object",
                    "required": schema_def["required"],
                    "properties": {
                        field: {"type": "string"}
                        for field in schema_def["required"] + schema_def["optional"]
                    }
                }
                logger.debug(f"Loaded default schema for {doc_type}")

    def _get_required_fields(self, doc_type: str) -> List[str]:
        """
        Get required fields for a specific document type.

        Args:
            doc_type: Document type (e.g., "PURCHASE_ORDER", "SUPPLIER_INVOICE")

        Returns:
            List of required field names

        Raises:
            No exceptions - returns empty list if schema not found
        """
        try:
            # Normalize document type to uppercase
            doc_type = doc_type.upper()

            # Check if schema exists in cache
            if doc_type not in self._schema_cache:
                logger.warning(f"No schema found for document type: {doc_type}")
                return []

            schema = self._schema_cache[doc_type]

            # Extract required fields from JSON Schema
            required_fields = schema.get("required", [])

            logger.debug(f"Required fields for {doc_type}: {required_fields}")

            return required_fields

        except Exception as e:
            logger.error(f"Failed to get required fields for {doc_type}: {e}")
            return []

    def _validate_schema_compliance(self, data: Dict, doc_type: str) -> bool:
        """
        Validate extracted data against schema.

        Checks:
        1. Required fields are present and non-empty
        2. Field types match schema expectations (basic validation)

        Args:
            data: Extracted document data
            doc_type: Document type

        Returns:
            True if data complies with schema, False otherwise
        """
        try:
            # Normalize document type
            doc_type = doc_type.upper()

            # Get schema
            if doc_type not in self._schema_cache:
                logger.warning(f"No schema for validation: {doc_type}")
                return True  # Pass validation if no schema defined

            schema = self._schema_cache[doc_type]
            required_fields = schema.get("required", [])
            properties = schema.get("properties", {})

            # Check required fields
            for field in required_fields:
                if field not in data:
                    logger.debug(f"Missing required field: {field}")
                    return False

                # Check if field is empty
                value = data[field]
                if value is None or (isinstance(value, str) and not value.strip()):
                    logger.debug(f"Required field is empty: {field}")
                    return False

            # Basic type validation for present fields
            for field, value in data.items():
                if field in properties:
                    expected_type = properties[field].get("type")

                    if expected_type == "number":
                        # Try to convert to number
                        try:
                            if value is not None and value != "":
                                float(value)
                        except (ValueError, TypeError):
                            logger.debug(f"Field {field} should be number but got: {value}")
                            return False

                    elif expected_type == "array":
                        if not isinstance(value, (list, tuple)):
                            logger.debug(f"Field {field} should be array but got: {type(value)}")
                            return False

                    elif expected_type == "object":
                        if not isinstance(value, dict):
                            logger.debug(f"Field {field} should be object but got: {type(value)}")
                            return False

            logger.debug(f"Schema validation passed for {doc_type}")
            return True

        except Exception as e:
            logger.error(f"Schema validation failed for {doc_type}: {e}")
            return False

    def get_schema_info(self, doc_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get schema information for debugging and inspection.

        Args:
            doc_type: Optional document type. If None, returns info for all schemas.

        Returns:
            Dictionary with schema information
        """
        if doc_type:
            doc_type = doc_type.upper()
            if doc_type in self._schema_cache:
                return {
                    "doc_type": doc_type,
                    "schema": self._schema_cache[doc_type],
                    "required_fields": self._get_required_fields(doc_type)
                }
            else:
                return {"error": f"Schema not found for {doc_type}"}

        # Return info for all schemas
        return {
            "total_schemas": len(self._schema_cache),
            "document_types": list(self._schema_cache.keys()),
            "schemas_loaded": self._schemas_loaded
        }
