"""
Document Subtype Classifier.

Classifies documents into subtypes based on OCR text patterns,
keywords, and structural features.
"""

import re
from typing import Dict, List, Optional, Tuple

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SubtypeClassifier:
    """
    Classify document subtypes using rule-based and pattern matching.

    Supports 35+ subtypes across 13 document types:
    - Purchase Orders: Standard, Blanket, Contract, Emergency
    - Invoices: Standard, Credit, Debit, Pro-forma, Recurring
    - etc.
    """

    # Subtype patterns for each document type
    SUBTYPE_PATTERNS = {
        "PURCHASE_ORDER": {
            "BLANKET": [
                r"blanket\s+(?:purchase\s+)?order",
                r"standing\s+order",
                r"framework\s+agreement",
            ],
            "CONTRACT": [
                r"contract\s+(?:purchase\s+)?order",
                r"long[- ]term\s+agreement",
            ],
            "EMERGENCY": [
                r"emergency\s+(?:purchase\s+)?order",
                r"urgent\s+(?:purchase\s+)?order",
                r"rush\s+order",
            ],
            "STANDARD": [],  # Default fallback
        },
        "SUPPLIER_INVOICE": {
            "CREDIT_NOTE": [
                r"credit\s+note",
                r"credit\s+memo",
                r"return\s+invoice",
            ],
            "DEBIT_NOTE": [
                r"debit\s+note",
                r"debit\s+memo",
            ],
            "PRO_FORMA": [
                r"pro[- ]?forma\s+invoice",
                r"proforma\s+invoice",
                r"preliminary\s+invoice",
            ],
            "RECURRING": [
                r"recurring\s+invoice",
                r"subscription\s+invoice",
                r"monthly\s+invoice",
            ],
            "PREPAYMENT": [
                r"prepayment\s+invoice",
                r"advance\s+payment\s+invoice",
                r"down\s+payment\s+invoice",
            ],
            "FINAL": [
                r"final\s+invoice",
                r"closing\s+invoice",
            ],
            "STANDARD": [],
        },
        "SALES_ORDER": {
            "RUSH": [
                r"rush\s+order",
                r"urgent\s+order",
                r"priority\s+order",
            ],
            "DROP_SHIP": [
                r"drop[- ]?ship",
                r"direct\s+ship(?:ment)?",
            ],
            "BLANKET": [
                r"blanket\s+(?:sales\s+)?order",
                r"standing\s+(?:sales\s+)?order",
            ],
            "STANDARD": [],
        },
        "CUSTOMER_INVOICE": {
            "CREDIT_NOTE": [
                r"credit\s+note",
                r"credit\s+memo",
            ],
            "DEBIT_NOTE": [
                r"debit\s+note",
                r"debit\s+memo",
            ],
            "PRO_FORMA": [
                r"pro[- ]?forma",
                r"proforma",
            ],
            "COMMERCIAL": [
                r"commercial\s+invoice",
            ],
            "STANDARD": [],
        },
        "GOODS_RECEIPT": {
            "PARTIAL": [
                r"partial\s+(?:goods\s+)?receipt",
                r"partial\s+delivery",
            ],
            "FINAL": [
                r"final\s+(?:goods\s+)?receipt",
                r"complete\s+delivery",
            ],
            "RETURN": [
                r"return\s+(?:goods\s+)?receipt",
                r"goods\s+return",
            ],
            "STANDARD": [],
        },
        "ADVANCED_SHIPPING_NOTICE": {
            "PARTIAL": [
                r"partial\s+(?:shipment|delivery)",
            ],
            "COMPLETE": [
                r"complete\s+(?:shipment|delivery)",
                r"full\s+(?:shipment|delivery)",
            ],
            "STANDARD": [],
        },
        "DELIVERY_NOTE": {
            "PARTIAL": [
                r"partial\s+delivery",
            ],
            "COMPLETE": [
                r"complete\s+delivery",
                r"full\s+delivery",
            ],
            "RETURN": [
                r"return\s+delivery",
            ],
            "STANDARD": [],
        },
        "CREDIT_NOTE": {
            "FULL_CREDIT": [
                r"full\s+credit",
                r"complete\s+credit",
            ],
            "PARTIAL_CREDIT": [
                r"partial\s+credit",
            ],
            "STANDARD": [],
        },
        "DEBIT_NOTE": {
            "ADJUSTMENT": [
                r"adjustment\s+debit",
                r"price\s+adjustment",
            ],
            "STANDARD": [],
        },
        "PAYMENT_ADVICE": {
            "WIRE_TRANSFER": [
                r"wire\s+transfer",
                r"bank\s+transfer",
                r"electronic\s+transfer",
            ],
            "CHECK": [
                r"check\s+payment",
                r"cheque\s+payment",
            ],
            "ACH": [
                r"ach\s+payment",
                r"automated\s+clearing\s+house",
            ],
            "STANDARD": [],
        },
        "REMITTANCE_ADVICE": {
            "CONSOLIDATED": [
                r"consolidated\s+remittance",
                r"combined\s+remittance",
            ],
            "SINGLE": [
                r"single\s+remittance",
            ],
            "STANDARD": [],
        },
        "STATEMENT_OF_ACCOUNT": {
            "MONTHLY": [
                r"monthly\s+statement",
            ],
            "QUARTERLY": [
                r"quarterly\s+statement",
            ],
            "ANNUAL": [
                r"annual\s+statement",
                r"yearly\s+statement",
            ],
            "STANDARD": [],
        },
        "QUOTE": {
            "FORMAL": [
                r"formal\s+(?:quote|quotation)",
                r"official\s+(?:quote|quotation)",
            ],
            "INFORMAL": [
                r"informal\s+(?:quote|quotation)",
                r"budget\s+(?:quote|quotation)",
            ],
            "STANDARD": [],
        },
        "CONTRACT": {
            "SERVICE": [
                r"service\s+contract",
                r"maintenance\s+contract",
            ],
            "SUPPLY": [
                r"supply\s+contract",
                r"procurement\s+contract",
            ],
            "LEASE": [
                r"lease\s+(?:contract|agreement)",
            ],
            "STANDARD": [],
        },
    }

    def __init__(self):
        """Initialize subtype classifier."""
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Dict[str, List[re.Pattern]]]:
        """Precompile all regex patterns for performance."""
        compiled = {}

        for doc_type, subtypes in self.SUBTYPE_PATTERNS.items():
            compiled[doc_type] = {}
            for subtype, patterns in subtypes.items():
                compiled[doc_type][subtype] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in patterns
                ]

        return compiled

    def classify(
        self,
        doc_type: str,
        ocr_text: str,
        extracted_data: Optional[Dict] = None,
    ) -> Tuple[str, float]:
        """
        Classify document subtype.

        Args:
            doc_type: Document type (e.g., "PURCHASE_ORDER")
            ocr_text: OCR extracted text
            extracted_data: Optional extracted field data for additional context

        Returns:
            Tuple of (subtype, confidence)
        """
        if doc_type not in self.compiled_patterns:
            logger.warning(f"Unknown document type: {doc_type}, returning STANDARD")
            return "STANDARD", 0.5

        # Get patterns for this document type
        subtype_patterns = self.compiled_patterns[doc_type]

        # Score each subtype
        scores = {}
        for subtype, patterns in subtype_patterns.items():
            if not patterns:  # STANDARD has empty patterns
                scores[subtype] = 0.0
                continue

            # Count pattern matches
            match_count = sum(
                1 for pattern in patterns
                if pattern.search(ocr_text)
            )

            # Score is percentage of patterns matched
            scores[subtype] = match_count / len(patterns) if patterns else 0.0

        # Find best match
        if scores and max(scores.values()) > 0:
            best_subtype = max(scores, key=scores.get)
            confidence = scores[best_subtype]

            # Boost confidence if multiple patterns matched
            if confidence > 0.5:
                confidence = min(0.95, confidence + 0.1)

            logger.info(f"Subtype classified: {best_subtype} (confidence: {confidence:.2f})")
            return best_subtype, confidence
        else:
            # No patterns matched, return STANDARD with medium confidence
            logger.info(f"No subtype patterns matched, returning STANDARD")
            return "STANDARD", 0.75

    def get_subtypes(self, doc_type: str) -> List[str]:
        """Get all possible subtypes for a document type."""
        return list(self.SUBTYPE_PATTERNS.get(doc_type, {}).keys())

    def add_pattern(
        self,
        doc_type: str,
        subtype: str,
        pattern: str,
    ) -> None:
        """
        Add a new pattern for subtype detection.

        Args:
            doc_type: Document type
            subtype: Subtype name
            pattern: Regex pattern string
        """
        if doc_type not in self.SUBTYPE_PATTERNS:
            self.SUBTYPE_PATTERNS[doc_type] = {}

        if subtype not in self.SUBTYPE_PATTERNS[doc_type]:
            self.SUBTYPE_PATTERNS[doc_type][subtype] = []

        self.SUBTYPE_PATTERNS[doc_type][subtype].append(pattern)

        # Recompile patterns
        self.compiled_patterns = self._compile_patterns()

        logger.info(f"Added pattern for {doc_type}/{subtype}: {pattern}")
