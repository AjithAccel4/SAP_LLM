"""
Knowledge Base Query Interface

High-level interface for querying the SAP Knowledge Base.
"""

from typing import Any, Dict, List, Optional

from sap_llm.knowledge_base.storage import KnowledgeBaseStorage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseQuery:
    """
    High-level interface for querying SAP API knowledge.

    Provides convenient methods for:
    - Finding APIs for document types
    - Mapping ADC fields to SAP fields
    - Finding relevant business rules
    - Generating transformation code
    """

    def __init__(self, storage: KnowledgeBaseStorage):
        """
        Initialize query interface.

        Args:
            storage: Knowledge base storage instance
        """
        self.storage = storage
        logger.info("Knowledge Base Query initialized")

    def find_api_for_document(
        self, doc_type: str, doc_subtype: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find SAP APIs for document type.

        Args:
            doc_type: Document type (e.g., "purchase_order")
            doc_subtype: Document subtype (e.g., "standard")

        Returns:
            List of relevant API schemas
        """
        # Build query
        query_parts = [doc_type.replace("_", " ")]
        if doc_subtype:
            query_parts.append(doc_subtype)

        query = " ".join(query_parts)

        logger.debug(f"Finding APIs for: {query}")

        # Search APIs
        apis = self.storage.search_apis(query, k=5)

        logger.info(f"Found {len(apis)} APIs for {doc_type}")

        return apis

    def map_fields_to_sap(
        self, adc_data: Dict[str, Any], doc_type: str
    ) -> Dict[str, Any]:
        """
        Map ADC fields to SAP API fields.

        Args:
            adc_data: ADC dictionary
            doc_type: Document type

        Returns:
            Dictionary with SAP field mappings
        """
        sap_payload = {}
        field_mappings = {}

        for field_name, field_value in adc_data.items():
            # Search for field mapping
            mappings = self.storage.search_field_mappings(field_name, k=1)

            if mappings:
                mapping = mappings[0]
                sap_field = mapping.get("sap_field", field_name)
                field_mappings[field_name] = sap_field

                # Apply transformation if available
                transformed_value = self._apply_field_transformation(
                    field_value, mapping
                )

                sap_payload[sap_field] = transformed_value

            else:
                # No mapping found, use original name
                logger.warning(f"No mapping found for field: {field_name}")
                sap_payload[field_name] = field_value

        return {
            "payload": sap_payload,
            "mappings": field_mappings,
        }

    def _apply_field_transformation(
        self, value: Any, mapping: Dict[str, Any]
    ) -> Any:
        """
        Apply field transformation based on mapping.

        Args:
            value: Original value
            mapping: Field mapping with transformation rules

        Returns:
            Transformed value
        """
        # Check for transformation function
        transform_type = mapping.get("transform")

        if transform_type == "uppercase":
            return str(value).upper()
        elif transform_type == "lowercase":
            return str(value).lower()
        elif transform_type == "remove_spaces":
            return str(value).replace(" ", "")
        elif transform_type == "date_format":
            # TODO: Implement date formatting
            return value
        else:
            return value

    def find_validation_rules(
        self, doc_type: str, field_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find validation rules for document or field.

        Args:
            doc_type: Document type
            field_name: Optional specific field name

        Returns:
            List of applicable business rules
        """
        # Build query
        query_parts = [doc_type.replace("_", " "), "validation"]
        if field_name:
            query_parts.append(field_name)

        query = " ".join(query_parts)

        logger.debug(f"Finding validation rules for: {query}")

        # Search rules
        rules = self.storage.search_business_rules(query, k=10)

        # Filter by type
        validation_rules = [r for r in rules if r.get("type") == "validation"]

        logger.info(f"Found {len(validation_rules)} validation rules")

        return validation_rules

    def find_calculation_rules(
        self, doc_type: str
    ) -> List[Dict[str, Any]]:
        """
        Find calculation rules for document type.

        Args:
            doc_type: Document type

        Returns:
            List of calculation rules
        """
        query = f"{doc_type.replace('_', ' ')} calculation formula"

        logger.debug(f"Finding calculation rules for: {query}")

        # Search rules
        rules = self.storage.search_business_rules(query, k=10)

        # Filter by type
        calc_rules = [r for r in rules if r.get("type") == "calculation"]

        logger.info(f"Found {len(calc_rules)} calculation rules")

        return calc_rules

    def get_endpoint_for_action(
        self, doc_type: str, action: str = "create"
    ) -> Optional[Dict[str, Any]]:
        """
        Get API endpoint for specific action.

        Args:
            doc_type: Document type
            action: Action (create, update, delete)

        Returns:
            Endpoint information or None
        """
        # Find APIs
        apis = self.find_api_for_document(doc_type)

        if not apis:
            logger.warning(f"No APIs found for {doc_type}")
            return None

        # Get best matching API
        api = apis[0]

        # Find endpoint for action
        endpoints = api.get("endpoints", [])

        # Map action to HTTP method
        method_map = {
            "create": "POST",
            "update": ["PUT", "PATCH"],
            "delete": "DELETE",
            "read": "GET",
        }

        target_methods = method_map.get(action, "POST")
        if not isinstance(target_methods, list):
            target_methods = [target_methods]

        # Find matching endpoint
        for endpoint in endpoints:
            if endpoint.get("method") in target_methods:
                return {
                    "api": api.get("title"),
                    "endpoint": endpoint.get("path"),
                    "method": endpoint.get("method"),
                    "description": endpoint.get("description", ""),
                }

        logger.warning(f"No endpoint found for {action} on {doc_type}")
        return None

    def get_example_payload(
        self, doc_type: str, action: str = "create"
    ) -> Optional[Dict[str, Any]]:
        """
        Get example payload for document type.

        Args:
            doc_type: Document type
            action: Action type

        Returns:
            Example payload or None
        """
        # TODO: Implement example retrieval from storage
        logger.debug(f"Getting example payload for {doc_type} {action}")

        return None

    def validate_payload(
        self, payload: Dict[str, Any], doc_type: str
    ) -> Dict[str, Any]:
        """
        Validate payload against business rules.

        Args:
            payload: Payload to validate
            doc_type: Document type

        Returns:
            Validation result with errors
        """
        errors = []
        warnings = []

        # Get validation rules
        rules = self.find_validation_rules(doc_type)

        for rule in rules:
            # Apply rule
            result = self._apply_validation_rule(payload, rule)

            if not result["valid"]:
                errors.append({
                    "rule_id": rule.get("rule_id"),
                    "description": rule.get("description"),
                    "details": result.get("details"),
                })

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _apply_validation_rule(
        self, payload: Dict[str, Any], rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply validation rule to payload.

        Args:
            payload: Payload to validate
            rule: Validation rule

        Returns:
            Validation result
        """
        # TODO: Implement rule execution
        # For now, always return valid
        return {"valid": True}

    def get_transformation_code(
        self, source_format: str, target_format: str
    ) -> Optional[str]:
        """
        Get transformation code between formats.

        Args:
            source_format: Source format (e.g., "ADC")
            target_format: Target format (e.g., "SAP_ODATA")

        Returns:
            Python transformation code or None
        """
        # TODO: Implement transformation code generation
        logger.debug(f"Getting transformation: {source_format} -> {target_format}")

        return None

    def get_similar_documents(
        self, adc_data: Dict[str, Any], k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar processed documents.

        Args:
            adc_data: ADC data
            k: Number of results

        Returns:
            List of similar documents
        """
        # Build query from ADC data
        query_parts = [
            str(v) for k, v in adc_data.items()
            if k in ["document_type", "vendor_name", "total_amount"]
        ]

        query = " ".join(query_parts)

        # Search APIs (as proxy for similar documents)
        results = self.storage.search_apis(query, k=k)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Statistics dictionary
        """
        storage_stats = self.storage.get_stats()

        return {
            "storage": storage_stats,
            "total_items": sum(storage_stats.values()),
        }
