"""
Stage 8: Routing - SAP API Endpoint Selection & Payload Generation

Uses Reasoning Engine (Mixtral-8x7B) to autonomously select SAP API endpoints
and generate compliant OData payloads.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.query import PMGQueryEngine
from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class RoutingStage(BaseStage):
    """
    SAP API routing and payload generation stage.

    Uses Reasoning Engine for:
    - Endpoint selection (400+ SAP APIs)
    - Payload transformation (ADC → OData)
    - Decision reasoning and confidence scoring

    Integrates with PMG for similar routing cases.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

        # Lazy load reasoning engine
        self.reasoning_engine = None

        # Initialize PMG for routing context
        self.pmg_client = None
        self.pmg_query_engine = None
        self._init_pmg()

        # Load API schemas and field mappings from knowledge base
        self.api_schemas = self._load_api_schemas()
        self.field_mappings = self._load_field_mappings()

    def _init_pmg(self):
        """Initialize PMG client for routing context."""
        try:
            self.pmg_client = ProcessMemoryGraph()
            self.pmg_query_engine = PMGQueryEngine(self.pmg_client)
            logger.info("PMG initialized for routing decisions")
        except Exception as e:
            logger.warning(f"Failed to initialize PMG: {e}. Routing without PMG context.")
            self.pmg_query_engine = None

    def _load_reasoning_engine(self):
        """Lazy load reasoning engine."""
        if self.reasoning_engine is None:
            logger.info("Loading reasoning engine...")
            self.reasoning_engine = ReasoningEngine(
                device="cuda",
                precision="int8",
            )

    def _load_api_schemas(self) -> List[Dict]:
        """Load SAP API endpoint mappings from knowledge base."""
        try:
            # Get the project root directory
            current_dir = Path(__file__).parent.parent.parent
            schemas_path = current_dir / "data" / "schemas" / "api_endpoints.json"

            if schemas_path.exists():
                with open(schemas_path, 'r') as f:
                    data = json.load(f)
                    api_endpoints = data.get("api_endpoints", [])
                    logger.info(f"Loaded {len(api_endpoints)} API endpoint schemas from knowledge base")
                    return api_endpoints
            else:
                logger.warning(f"API endpoints file not found at {schemas_path}, using fallback")
                return self._get_fallback_api_schemas()

        except Exception as e:
            logger.error(f"Failed to load API schemas: {e}, using fallback")
            return self._get_fallback_api_schemas()

    def _get_fallback_api_schemas(self) -> List[Dict]:
        """Fallback API schemas if knowledge base is unavailable."""
        return [
            {
                "name": "API_PURCHASEORDER_PROCESS_SRV",
                "entity": "A_PurchaseOrder",
                "methods": ["POST", "PATCH"],
                "doc_types": ["PURCHASE_ORDER"],
            },
            {
                "name": "API_SUPPLIERINVOICE_PROCESS_SRV",
                "entity": "A_SupplierInvoice",
                "methods": ["POST", "PATCH"],
                "doc_types": ["SUPPLIER_INVOICE"],
            },
            {
                "name": "API_SALES_ORDER_SRV",
                "entity": "A_SalesOrder",
                "methods": ["POST", "PATCH"],
                "doc_types": ["SALES_ORDER"],
            },
        ]

    def _load_field_mappings(self) -> Dict[str, Dict]:
        """Load field transformation mappings from knowledge base."""
        try:
            # Get the project root directory
            current_dir = Path(__file__).parent.parent.parent
            mappings_path = current_dir / "data" / "schemas" / "field_mappings.json"

            if mappings_path.exists():
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                    logger.info(f"Loaded field mappings for {len(mappings)} document types")
                    return mappings
            else:
                logger.warning(f"Field mappings file not found at {mappings_path}")
                return {}

        except Exception as e:
            logger.error(f"Failed to load field mappings: {e}")
            return {}

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route document to SAP API and generate payload.

        Args:
            input_data: {
                "corrected_data": Dict,
                "doc_type": str,
                "subtype": str,
                "valid": bool,
                "violations": List[Dict],
            }

        Returns:
            {
                "endpoint": str,
                "method": str,
                "payload": Dict,
                "confidence": float,
                "reasoning": str,
                "next_action_hint": str,
            }
        """
        # Load reasoning engine
        self._load_reasoning_engine()

        data = input_data["corrected_data"]
        doc_type = input_data["doc_type"]
        valid = input_data["valid"]

        # If validation failed, route to exception handling
        if not valid:
            return {
                "endpoint": "EXCEPTION_HANDLER",
                "method": "POST",
                "payload": {
                    "data": data,
                    "violations": input_data["violations"],
                },
                "confidence": 1.0,
                "reasoning": "Document has validation violations",
                "next_action_hint": "shwl.exception.handle",
            }

        # Get PMG context (similar routing cases)
        similar_cases = self._get_pmg_routing_context(data, doc_type)

        # Use reasoning engine to make routing decision
        decision = self.reasoning_engine.decide_routing(
            data,
            doc_type,
            self.api_schemas,
            similar_cases,
        )

        # Generate SAP payload
        payload = self._generate_sap_payload(
            data,
            decision["endpoint"],
            doc_type,
        )

        logger.info(
            f"Routing: {decision['endpoint']} "
            f"(confidence: {decision.get('confidence', 0):.4f})"
        )

        return {
            "endpoint": decision["endpoint"],
            "method": decision.get("method", "POST"),
            "payload": payload,
            "confidence": decision.get("confidence", 0.0),
            "reasoning": decision.get("reasoning", ""),
            "next_action_hint": f"router.post.{decision['endpoint']}",
        }

    def _get_pmg_routing_context(
        self,
        data: Dict[str, Any],
        doc_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Query PMG for similar routing decisions.

        Args:
            data: Document data
            doc_type: Document type

        Returns:
            List of similar routing cases
        """
        if self.pmg_query_engine is None:
            logger.debug("PMG not available, skipping routing context")
            return []

        try:
            # Extract context fields
            supplier_id = data.get("supplier_id") or data.get("vendor_id")
            company_code = data.get("company_code")

            # Query PMG for routing context
            routing_context = self.pmg_query_engine.get_routing_context(
                doc_type=doc_type,
                supplier_id=supplier_id,
                company_code=company_code,
            )

            similar_routings = routing_context.get("similar_routings", [])

            if similar_routings:
                logger.info(
                    f"Found {routing_context.get('num_similar', 0)} similar routing cases. "
                    f"Recommended endpoint: {routing_context.get('recommended_endpoint')}"
                )
            else:
                logger.debug("No similar routing cases found in PMG")

            return similar_routings

        except Exception as e:
            logger.warning(f"Failed to query PMG for routing context: {e}")
            return []

    def _generate_sap_payload(
        self,
        data: Dict,
        endpoint: str,
        doc_type: str,
    ) -> Dict:
        """
        Transform ADC data to SAP OData payload using field mappings.

        Args:
            data: Extracted ADC data
            endpoint: SAP API endpoint
            doc_type: Document type

        Returns:
            SAP-compliant OData payload
        """
        # Use field mappings from knowledge base if available
        if doc_type in self.field_mappings:
            logger.debug(f"Using field mappings from knowledge base for {doc_type}")
            return self._transform_with_mappings(data, doc_type)

        # Fallback to hardcoded transformations
        logger.warning(f"No field mappings found for {doc_type}, using fallback transformation")
        if doc_type == "PURCHASE_ORDER":
            return self._transform_purchase_order(data)
        elif doc_type == "SUPPLIER_INVOICE":
            return self._transform_supplier_invoice(data)
        else:
            return data

    def _transform_with_mappings(self, data: Dict, doc_type: str) -> Dict:
        """
        Transform data using field mappings from knowledge base.

        Args:
            data: ADC data
            doc_type: Document type

        Returns:
            Transformed SAP payload
        """
        mappings = self.field_mappings.get(doc_type, {})
        payload = {"d": {}}

        for adc_field, value in data.items():
            if adc_field in mappings:
                mapping = mappings[adc_field]
                sap_field = mapping.get("sap_field", adc_field)
                transform = mapping.get("transform")

                # Apply transformation
                transformed_value = self._apply_transform(value, transform, mapping)

                # Handle nested mappings (like line items)
                if "nested_mappings" in mapping and isinstance(value, list):
                    transformed_value = self._transform_nested_items(
                        value,
                        mapping["nested_mappings"]
                    )

                payload["d"][sap_field] = transformed_value
            else:
                # No mapping found, use as-is
                payload["d"][adc_field] = value

        return payload

    def _apply_transform(
        self,
        value: Any,
        transform: Optional[str],
        mapping: Dict,
    ) -> Any:
        """
        Apply field transformation.

        Args:
            value: Original value
            transform: Transformation type
            mapping: Field mapping configuration

        Returns:
            Transformed value
        """
        if value is None:
            return mapping.get("default", "")

        if transform == "uppercase":
            return str(value).upper()
        elif transform == "lowercase":
            return str(value).lower()
        elif transform == "remove_spaces":
            return str(value).replace(" ", "")
        elif transform == "string":
            return str(value)
        elif transform == "date_format":
            # Keep ISO format for SAP
            return str(value)
        else:
            return value

    def _transform_nested_items(
        self,
        items: List[Dict],
        nested_mappings: Dict,
    ) -> List[Dict]:
        """
        Transform nested items (e.g., line items).

        Args:
            items: List of item dictionaries
            nested_mappings: Field mappings for nested items

        Returns:
            List of transformed items
        """
        transformed_items = []

        for item in items:
            transformed_item = {}
            for adc_field, value in item.items():
                if adc_field in nested_mappings:
                    mapping = nested_mappings[adc_field]
                    sap_field = mapping.get("sap_field", adc_field)
                    transform = mapping.get("transform")

                    transformed_value = self._apply_transform(value, transform, mapping)
                    transformed_item[sap_field] = transformed_value
                else:
                    transformed_item[adc_field] = value

            transformed_items.append(transformed_item)

        return transformed_items

    def _transform_purchase_order(self, data: Dict) -> Dict:
        """Transform PO to SAP format."""
        payload = {
            "d": {
                "PurchaseOrder": data.get("po_number", ""),
                "PurchaseOrderType": "NB",  # Standard PO
                "Supplier": data.get("vendor_id", ""),
                "CompanyCode": data.get("company_code", ""),
                "PurchasingOrganization": data.get("purchasing_org", ""),
                "DocumentCurrency": data.get("currency", "USD"),
                # Add more fields...
            }
        }

        # Add line items
        if "items" in data:
            payload["d"]["to_PurchaseOrderItem"] = []
            for item in data["items"]:
                payload["d"]["to_PurchaseOrderItem"].append({
                    "PurchaseOrderItem": str(item.get("line_number", "10")),
                    "Material": item.get("material_number", ""),
                    "PurchaseOrderItemText": item.get("description", ""),
                    "OrderQuantity": str(item.get("quantity", 0)),
                    "NetPriceAmount": str(item.get("unit_price", 0)),
                })

        return payload

    def _transform_supplier_invoice(self, data: Dict) -> Dict:
        """Transform supplier invoice to SAP format."""
        # Extract fiscal year from invoice date
        fiscal_year = self._extract_fiscal_year(
            data.get("invoice_date") or data.get("posting_date")
        )

        payload = {
            "d": {
                "SupplierInvoice": data.get("invoice_number", ""),
                "FiscalYear": fiscal_year,
                "Supplier": data.get("supplier_id", ""),
                "DocumentDate": data.get("invoice_date", ""),
                "PostingDate": data.get("posting_date", ""),
                "InvoicingParty": data.get("supplier_id", ""),
                "DocumentCurrency": data.get("currency", "USD"),
                "InvoiceGrossAmount": str(data.get("total_amount", 0)),
                "PurchaseOrder": data.get("po_number", ""),
                # Add more fields...
            }
        }

        return payload

    def _extract_fiscal_year(self, date_value: Optional[str]) -> str:
        """
        Extract fiscal year from date string.

        Args:
            date_value: Date string in various formats (ISO, US, EU, etc.)

        Returns:
            Fiscal year as string (e.g., "2025")
        """
        if not date_value:
            # Default to current year if no date provided
            current_year = datetime.now().year
            logger.warning(f"No date provided, using current year: {current_year}")
            return str(current_year)

        try:
            # Try parsing common date formats
            date_formats = [
                "%Y-%m-%d",           # ISO: 2025-11-14
                "%Y/%m/%d",           # 2025/11/14
                "%d.%m.%Y",           # EU: 14.11.2025
                "%d/%m/%Y",           # EU: 14/11/2025
                "%m/%d/%Y",           # US: 11/14/2025
                "%Y%m%d",             # Compact: 20251114
                "%d-%m-%Y",           # 14-11-2025
                "%m-%d-%Y",           # 11-14-2025
                "%Y-%m-%dT%H:%M:%S",  # ISO with time
                "%Y-%m-%dT%H:%M:%SZ", # ISO with time and Z
            ]

            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(str(date_value), date_format)
                    fiscal_year = str(parsed_date.year)
                    logger.debug(f"Extracted fiscal year {fiscal_year} from date {date_value}")
                    return fiscal_year
                except ValueError:
                    continue

            # If no format matches, try to extract 4-digit year
            import re
            year_match = re.search(r'\b(20\d{2})\b', str(date_value))
            if year_match:
                fiscal_year = year_match.group(1)
                logger.debug(f"Extracted fiscal year {fiscal_year} using regex from {date_value}")
                return fiscal_year

            # Fallback to current year
            current_year = datetime.now().year
            logger.warning(
                f"Could not parse date '{date_value}', using current year: {current_year}"
            )
            return str(current_year)

        except Exception as e:
            current_year = datetime.now().year
            logger.error(
                f"Error extracting fiscal year from '{date_value}': {e}. "
                f"Using current year: {current_year}"
            )
            return str(current_year)
