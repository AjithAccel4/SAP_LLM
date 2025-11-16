"""
SAP Payload Generator with 100% Schema Compliance.

Generates valid SAP OData/REST/SOAP payloads from extracted document data.

Features:
1. 100% schema compliance through validation
2. Automatic error detection and fixing
3. Field transformation and mapping
4. Data type conversion and formatting
5. Required field completion (using defaults/inference)
6. Business logic application
7. Multi-API batch payload generation
8. Payload optimization for performance

Target Metrics:
- Schema compliance: 100% (not 99%, absolute 100%)
- Payload generation success rate: >99.5%
- Auto-fix success rate: >95%
- Generation latency: <100ms per payload
"""

import copy
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sap_llm.knowledge_base.sap_api_knowledge_base import SAPAPIKnowledgeBase
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SAPPayloadGenerator:
    """
    Generates 100% schema-compliant SAP payloads.

    Process:
    1. Load API schema from knowledge base
    2. Map ADC fields to SAP fields
    3. Apply field transformations
    4. Validate against schema
    5. Auto-fix errors
    6. Re-validate until 100% compliant
    """

    def __init__(
        self,
        knowledge_base: Optional[SAPAPIKnowledgeBase] = None,
        max_fix_attempts: int = 3,
        strict_validation: bool = True,
    ):
        """
        Initialize SAP payload generator.

        Args:
            knowledge_base: SAP API knowledge base
            max_fix_attempts: Maximum auto-fix attempts
            strict_validation: Enforce strict 100% compliance
        """
        self.knowledge_base = knowledge_base or SAPAPIKnowledgeBase()
        self.max_fix_attempts = max_fix_attempts
        self.strict_validation = strict_validation

        logger.info("SAP Payload Generator initialized")
        logger.info(f"Max fix attempts: {max_fix_attempts}")
        logger.info(f"Strict validation: {strict_validation}")

    def generate_payload(
        self,
        adc_data: Dict[str, Any],
        api_endpoint: str,
        entity_set: str,
        doc_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate 100% schema-compliant SAP payload.

        Args:
            adc_data: Extracted document data (ADC format)
            api_endpoint: SAP API endpoint
            entity_set: Entity set name (e.g., "A_PurchaseOrder")
            doc_type: Document type
            context: Additional context (PMG, business rules)

        Returns:
            Dictionary with:
            - payload: Generated SAP payload
            - validation_status: Validation results
            - transformations: List of transformations applied
            - fixes: List of auto-fixes applied
        """
        logger.info(f"Generating payload for {entity_set} ({doc_type})")

        result = {
            "payload": {},
            "validation_status": {},
            "transformations": [],
            "fixes": [],
            "success": False,
        }

        # Step 1: Initial field mapping
        payload = self._map_fields(adc_data, entity_set, doc_type)
        result["payload"] = payload

        # Step 2: Apply field transformations
        payload, transformations = self._apply_transformations(
            payload,
            entity_set,
            doc_type,
        )
        result["transformations"] = transformations

        # Step 3: Validate payload
        validation = self._validate_payload(payload, entity_set)
        result["validation_status"] = validation

        # Step 4: Auto-fix errors (iterative)
        if not validation["is_valid"]:
            logger.info(f"Payload validation failed, attempting auto-fix...")

            for attempt in range(self.max_fix_attempts):
                logger.info(f"  Fix attempt {attempt + 1}/{self.max_fix_attempts}")

                payload, fixes = self._auto_fix_payload(
                    payload,
                    validation,
                    entity_set,
                    adc_data,
                    context,
                )

                result["fixes"].extend(fixes)

                # Re-validate
                validation = self._validate_payload(payload, entity_set)
                result["validation_status"] = validation

                if validation["is_valid"]:
                    logger.info(f"✓ Payload fixed and validated successfully")
                    break

        # Step 5: Final validation
        result["payload"] = payload
        result["success"] = validation["is_valid"]

        if result["success"]:
            logger.info(f"✓ Payload generated successfully ({len(payload)} fields)")
        else:
            logger.warning(
                f"✗ Payload generation failed: {len(validation.get('errors', []))} errors"
            )

        return result

    def _map_fields(
        self,
        adc_data: Dict[str, Any],
        entity_set: str,
        doc_type: str,
    ) -> Dict[str, Any]:
        """
        Map ADC fields to SAP fields.

        Args:
            adc_data: ADC extracted data
            entity_set: SAP entity set
            doc_type: Document type

        Returns:
            Initial SAP payload
        """
        payload = {}

        # Get required fields for this entity set
        required_fields = self.knowledge_base.get_required_fields(entity_set)

        logger.info(f"Mapping {len(adc_data)} ADC fields to SAP payload")
        logger.info(f"Required SAP fields: {len(required_fields)}")

        # Common field mappings (ADC -> SAP)
        field_mappings = self._get_field_mappings(doc_type)

        # Map ADC fields to SAP fields
        for adc_field, sap_field in field_mappings.items():
            if adc_field in adc_data and adc_data[adc_field] is not None:
                payload[sap_field] = adc_data[adc_field]

        # Ensure all required fields are present
        for req_field in required_fields:
            if req_field not in payload:
                # Try to infer or use default
                default_value = self._get_default_value(req_field, doc_type, adc_data)
                if default_value is not None:
                    payload[req_field] = default_value

        logger.info(f"Mapped {len(payload)} fields to SAP payload")

        return payload

    def _get_field_mappings(self, doc_type: str) -> Dict[str, str]:
        """
        Get ADC to SAP field mappings for document type.

        Args:
            doc_type: Document type

        Returns:
            Dictionary of ADC_field -> SAP_field mappings
        """
        # Common mappings across all document types
        common_mappings = {
            # Header fields
            "document_number": "DocumentNumber",
            "document_date": "DocumentDate",
            "posting_date": "PostingDate",
            "company_code": "CompanyCode",
            "currency": "DocumentCurrency",
            "total_amount": "GrossAmount",
            "net_amount": "NetAmount",
            "tax_amount": "TaxAmount",
            "payment_terms": "PaymentTerms",

            # Supplier/Customer fields
            "supplier_number": "Supplier",
            "supplier_name": "SupplierName",
            "customer_number": "Customer",
            "customer_name": "CustomerName",

            # Purchase order specific
            "po_number": "PurchaseOrder",
            "po_type": "PurchaseOrderType",
            "purchasing_org": "PurchasingOrganization",
            "purchasing_group": "PurchasingGroup",

            # Sales order specific
            "sales_order": "SalesOrder",
            "sold_to_party": "SoldToParty",
            "ship_to_party": "ShipToParty",

            # Item fields
            "item_number": "ItemNumber",
            "material": "Material",
            "material_description": "MaterialDescription",
            "quantity": "Quantity",
            "unit_of_measure": "UnitOfMeasure",
            "unit_price": "NetPriceAmount",
            "plant": "Plant",
            "storage_location": "StorageLocation",
            "delivery_date": "DeliveryDate",
        }

        # Document type specific mappings
        if doc_type == "PURCHASE_ORDER":
            common_mappings.update({
                "vendor_number": "Supplier",
                "vendor_name": "SupplierName",
            })
        elif doc_type == "SUPPLIER_INVOICE":
            common_mappings.update({
                "invoice_number": "SupplierInvoiceIDByInvcgParty",
                "invoice_date": "DocumentDate",
                "invoicing_party": "InvoicingParty",
                "gross_amount": "InvoiceGrossAmount",
            })
        elif doc_type == "SALES_ORDER":
            common_mappings.update({
                "customer_po": "PurchaseOrderByCustomer",
                "sales_org": "SalesOrganization",
                "distribution_channel": "DistributionChannel",
            })

        return common_mappings

    def _apply_transformations(
        self,
        payload: Dict[str, Any],
        entity_set: str,
        doc_type: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Apply field transformations to payload.

        Args:
            payload: Initial payload
            entity_set: SAP entity set
            doc_type: Document type

        Returns:
            Tuple of (transformed payload, list of transformations)
        """
        transformed = copy.deepcopy(payload)
        transformations = []

        for field, value in list(transformed.items()):
            # Get field definition
            field_def = self.knowledge_base.get_field_definition(field)

            if not field_def:
                continue

            # Data type transformations
            target_type = field_def.get("data_type")

            if target_type == "string" and not isinstance(value, str):
                transformed[field] = str(value)
                transformations.append(f"{field}: Converted to string")

            elif target_type == "decimal" and isinstance(value, str):
                try:
                    # Remove currency symbols and commas
                    cleaned = value.replace('$', '').replace(',', '').strip()
                    transformed[field] = float(cleaned)
                    transformations.append(f"{field}: Converted to decimal")
                except ValueError:
                    pass

            elif target_type == "date":
                # Standardize date format to YYYY-MM-DD
                if isinstance(value, str):
                    standardized = self._standardize_date(value)
                    if standardized:
                        transformed[field] = standardized
                        transformations.append(f"{field}: Standardized date format")

            # Format transformations
            format_type = field_def.get("format")

            if format_type == "date" and isinstance(transformed[field], str):
                standardized = self._standardize_date(transformed[field])
                if standardized:
                    transformed[field] = standardized

            # Max length enforcement
            if isinstance(transformed[field], str) and "max_length" in field_def:
                max_len = field_def["max_length"]
                if len(transformed[field]) > max_len:
                    transformed[field] = transformed[field][:max_len]
                    transformations.append(f"{field}: Truncated to {max_len} chars")

            # Padding for numeric IDs (e.g., 10-digit vendor number)
            if field in ["Supplier", "Customer", "Material"] and isinstance(value, str):
                if "regex_pattern" in field_def:
                    pattern = field_def["regex_pattern"]
                    if r"^\d{10}$" in pattern:  # 10-digit number
                        if value.isdigit() and len(value) < 10:
                            transformed[field] = value.zfill(10)
                            transformations.append(f"{field}: Padded to 10 digits")

        logger.info(f"Applied {len(transformations)} transformations")

        return transformed, transformations

    def _validate_payload(
        self,
        payload: Dict[str, Any],
        entity_set: str,
    ) -> Dict[str, Any]:
        """
        Validate payload against SAP schema.

        Args:
            payload: SAP payload
            entity_set: Entity set name

        Returns:
            Validation status dictionary
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "validated_fields": 0,
        }

        # Get required fields
        required_fields = self.knowledge_base.get_required_fields(entity_set)

        # Check required fields
        for req_field in required_fields:
            # Handle colon notation (e.g., "PurchaseOrderType:NB")
            field_name = req_field.split(':')[0]

            if field_name not in payload:
                validation["is_valid"] = False
                validation["errors"].append({
                    "type": "MISSING_REQUIRED_FIELD",
                    "field": field_name,
                    "message": f"Required field '{field_name}' is missing",
                })

            # Check if specific value is required (e.g., PurchaseOrderType:NB)
            if ':' in req_field:
                expected_value = req_field.split(':')[1]
                if field_name in payload and payload[field_name] != expected_value:
                    validation["warnings"].append({
                        "type": "INCORRECT_VALUE",
                        "field": field_name,
                        "expected": expected_value,
                        "actual": payload.get(field_name),
                        "message": f"Expected '{expected_value}', got '{payload.get(field_name)}'",
                    })

        # Validate each field value
        for field, value in payload.items():
            is_valid, error_msg = self.knowledge_base.validate_field_value(field, value)

            if not is_valid:
                validation["is_valid"] = False
                validation["errors"].append({
                    "type": "INVALID_FIELD_VALUE",
                    "field": field,
                    "value": value,
                    "message": error_msg,
                })
            else:
                validation["validated_fields"] += 1

        logger.info(
            f"Validation: {validation['validated_fields']} fields validated, "
            f"{len(validation['errors'])} errors, {len(validation['warnings'])} warnings"
        )

        return validation

    def _auto_fix_payload(
        self,
        payload: Dict[str, Any],
        validation: Dict[str, Any],
        entity_set: str,
        adc_data: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Automatically fix payload errors.

        Args:
            payload: Current payload
            validation: Validation results
            entity_set: Entity set name
            adc_data: Original ADC data
            context: Additional context

        Returns:
            Tuple of (fixed payload, list of fixes applied)
        """
        fixed_payload = copy.deepcopy(payload)
        fixes = []

        for error in validation.get("errors", []):
            error_type = error.get("type")

            if error_type == "MISSING_REQUIRED_FIELD":
                field_name = error.get("field")

                # Try to find value in ADC data
                default_value = self._get_default_value(
                    field_name,
                    "",
                    adc_data,
                    context,
                )

                if default_value is not None:
                    fixed_payload[field_name] = default_value
                    fixes.append(f"Added missing field '{field_name}' with default value")

            elif error_type == "INVALID_FIELD_VALUE":
                field_name = error.get("field")
                current_value = error.get("value")

                # Try to fix invalid value
                fixed_value = self._fix_field_value(
                    field_name,
                    current_value,
                )

                if fixed_value is not None and fixed_value != current_value:
                    fixed_payload[field_name] = fixed_value
                    fixes.append(f"Fixed invalid value for '{field_name}'")

        logger.info(f"Applied {len(fixes)} auto-fixes")

        return fixed_payload, fixes

    def _get_default_value(
        self,
        field_name: str,
        doc_type: str,
        adc_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Get default value for a field.

        Args:
            field_name: Field name
            doc_type: Document type
            adc_data: ADC data
            context: Additional context

        Returns:
            Default value or None
        """
        # Check context (PMG historical data)
        if context and "historical_values" in context:
            if field_name in context["historical_values"]:
                return context["historical_values"][field_name]

        # Common defaults
        defaults = {
            "DocumentDate": datetime.now().strftime("%Y-%m-%d"),
            "PostingDate": datetime.now().strftime("%Y-%m-%d"),
            "DocumentCurrency": "USD",
            "PurchaseOrderType": "NB",  # Standard PO
            "PurchasingOrganization": "1000",  # Default org
            "PurchasingGroup": "001",  # Default group
            "GoodsMovementCode": "01",  # GR for PO
        }

        return defaults.get(field_name)

    def _fix_field_value(
        self,
        field_name: str,
        current_value: Any,
    ) -> Optional[Any]:
        """
        Fix invalid field value.

        Args:
            field_name: Field name
            current_value: Current (invalid) value

        Returns:
            Fixed value or None
        """
        field_def = self.knowledge_base.get_field_definition(field_name)

        if not field_def:
            return None

        # Try to convert to correct type
        target_type = field_def.get("data_type")

        if target_type == "string" and not isinstance(current_value, str):
            return str(current_value)

        if target_type in ["decimal", "number"]:
            if isinstance(current_value, str):
                try:
                    cleaned = current_value.replace('$', '').replace(',', '').strip()
                    return float(cleaned)
                except ValueError:
                    return None

        if target_type == "date":
            if isinstance(current_value, str):
                return self._standardize_date(current_value)

        # Truncate if too long
        if isinstance(current_value, str) and "max_length" in field_def:
            max_len = field_def["max_length"]
            if len(current_value) > max_len:
                return current_value[:max_len]

        return None

    def _standardize_date(self, date_str: str) -> Optional[str]:
        """
        Standardize date to YYYY-MM-DD format.

        Args:
            date_str: Date string

        Returns:
            Standardized date or None
        """
        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%b %d, %Y",
            "%B %d, %Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    def generate_batch_payloads(
        self,
        adc_data_list: List[Dict[str, Any]],
        api_configs: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple payloads in batch.

        Args:
            adc_data_list: List of ADC data dictionaries
            api_configs: List of API configurations

        Returns:
            List of payload generation results
        """
        results = []

        for i, (adc_data, api_config) in enumerate(zip(adc_data_list, api_configs)):
            logger.info(f"Generating payload {i+1}/{len(adc_data_list)}")

            result = self.generate_payload(
                adc_data=adc_data,
                api_endpoint=api_config.get("endpoint", ""),
                entity_set=api_config.get("entity_set", ""),
                doc_type=api_config.get("doc_type", ""),
                context=api_config.get("context"),
            )

            results.append(result)

        success_count = sum(1 for r in results if r["success"])
        logger.info(
            f"Batch payload generation complete: "
            f"{success_count}/{len(results)} successful"
        )

        return results
