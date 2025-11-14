"""
Stage 4: Type Identifier - Document Subtype Classification

Hierarchical classification for 35+ document subtypes.
"""

from typing import Any, Dict

from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class TypeIdentifierStage(BaseStage):
    """
    Document subtype identification stage.

    Uses hierarchical classification:
    Level 1: Major category (from Stage 3)
    Level 2: Subtype (35+ total)

    Example PO subtypes: Standard, Blanket, Contract, Service, etc.
    """

    SUBTYPES = {
        "PURCHASE_ORDER": [
            "STANDARD",
            "BLANKET",
            "CONTRACT",
            "SERVICE",
            "SUBCONTRACT",
            "CONSIGNMENT",
            "STOCK_TRANSFER",
            "LIMIT",
            "DROP_SHIP",
            "CAPEX",
        ],
        "SUPPLIER_INVOICE": [
            "STANDARD",
            "CREDIT_MEMO",
            "DEBIT_MEMO",
            "PREPAYMENT",
            "DOWN_PAYMENT",
            "RECURRING",
            "PROFORMA",
            "COMMERCIAL",
        ],
        # Add more subtypes...
    }

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.model = None  # Lazy load

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify document subtype.

        Args:
            input_data: {
                "doc_type": str,
                "enhanced_images": List[Image],
                "ocr_results": List[Dict],
            }

        Returns:
            {
                "subtype": str,
                "confidence": float,
            }
        """
        doc_type = input_data["doc_type"]

        # Get available subtypes for this document type
        available_subtypes = self.SUBTYPES.get(doc_type, ["STANDARD"])

        # For now, use simple keyword matching
        # TODO: Implement actual hierarchical classifier
        ocr_text = input_data["ocr_results"][0]["text"].lower()

        subtype = self._identify_subtype_keywords(ocr_text, available_subtypes)

        logger.info(f"Subtype: {subtype}")

        return {
            "subtype": subtype,
            "confidence": 0.85,
        }

    def _identify_subtype_keywords(self, text: str, subtypes: list) -> str:
        """Simple keyword-based subtype identification."""
        keywords = {
            "BLANKET": ["blanket", "framework"],
            "CONTRACT": ["contract", "agreement"],
            "SERVICE": ["service", "hours"],
            "CONSIGNMENT": ["consignment"],
        }

        for subtype, kw_list in keywords.items():
            if subtype in subtypes:
                if any(kw in text for kw in kw_list):
                    return subtype

        return "STANDARD"
