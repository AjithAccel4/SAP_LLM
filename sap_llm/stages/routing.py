"""
Stage 8: Routing - SAP API Endpoint Selection & Payload Generation

Uses Reasoning Engine (Mixtral-8x7B) to autonomously select SAP API endpoints
and generate compliant OData payloads.
"""

from typing import Any, Dict, List

from sap_llm.models.reasoning_engine import ReasoningEngine
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

        # Load API schemas
        self.api_schemas = self._load_api_schemas()

    def _load_reasoning_engine(self):
        """Lazy load reasoning engine."""
        if self.reasoning_engine is None:
            logger.info("Loading reasoning engine...")
            self.reasoning_engine = ReasoningEngine(
                device="cuda",
                precision="int8",
            )

    def _load_api_schemas(self) -> List[Dict]:
        """Load SAP API schemas."""
        # TODO: Load from knowledge base
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
        # TODO: Query PMG
        similar_cases = []

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

    def _generate_sap_payload(
        self,
        data: Dict,
        endpoint: str,
        doc_type: str,
    ) -> Dict:
        """
        Transform ADC data to SAP OData payload.

        Args:
            data: Extracted ADC data
            endpoint: SAP API endpoint
            doc_type: Document type

        Returns:
            SAP-compliant OData payload
        """
        # TODO: Load field mappings from knowledge base
        # For now, create simple mapping

        if doc_type == "PURCHASE_ORDER":
            return self._transform_purchase_order(data)
        elif doc_type == "SUPPLIER_INVOICE":
            return self._transform_supplier_invoice(data)
        else:
            return data

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
        payload = {
            "d": {
                "SupplierInvoice": data.get("invoice_number", ""),
                "FiscalYear": "2025",  # TODO: Extract from date
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
