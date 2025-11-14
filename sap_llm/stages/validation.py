"""
Stage 7: Validation - Business Rules & Tolerance Checks

Validates extracted data against business rules, historical patterns,
and tolerance thresholds using PMG context.
"""

from typing import Any, Dict, List

from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationStage(BaseStage):
    """
    Business rule validation stage.

    Validates:
    - Required fields presence
    - Three-way match (PO vs Invoice vs GR)
    - Price variance tolerances
    - Duplicate detection
    - Date reasonableness
    - Business logic consistency

    Integrates with PMG for historical lookups.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.rules_enabled = (
            getattr(config, "business_rules_enabled", True) if config else True
        )
        self.pmg_enabled = (
            getattr(config, "pmg_lookup_enabled", True) if config else True
        )

        # Load business rules
        self.rules = self._load_business_rules()

    def _load_business_rules(self) -> List[Dict]:
        """Load business rules from configuration."""
        # TODO: Load from files/PMG
        return [
            {
                "rule_id": "VAL_001",
                "name": "Three-Way Match Price Variance",
                "doc_types": ["SUPPLIER_INVOICE"],
                "severity": "MEDIUM",
                "threshold": 0.03,  # 3% tolerance
            },
            {
                "rule_id": "VAL_002",
                "name": "Duplicate Invoice Detection",
                "doc_types": ["SUPPLIER_INVOICE"],
                "severity": "HIGH",
            },
            {
                "rule_id": "VAL_003",
                "name": "Invoice Date Reasonableness",
                "doc_types": ["SUPPLIER_INVOICE"],
                "severity": "LOW",
            },
        ]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data against business rules.

        Args:
            input_data: {
                "corrected_data": Dict,
                "doc_type": str,
                "subtype": str,
            }

        Returns:
            {
                "valid": bool,
                "violations": List[Dict],
                "warnings": List[Dict],
            }
        """
        data = input_data["corrected_data"]
        doc_type = input_data["doc_type"]

        # Get applicable rules
        applicable_rules = [
            rule for rule in self.rules if doc_type in rule["doc_types"]
        ]

        violations = []
        warnings = []

        # Execute rules
        for rule in applicable_rules:
            result = self._execute_rule(rule, data, doc_type)

            if not result["passed"]:
                entry = {
                    "rule_id": rule["rule_id"],
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "message": result["message"],
                    "field": result.get("field"),
                    "expected": result.get("expected"),
                    "actual": result.get("actual"),
                }

                if rule["severity"] in ["HIGH", "MEDIUM"]:
                    violations.append(entry)
                else:
                    warnings.append(entry)

        valid = len(violations) == 0

        logger.info(f"Validation: {len(violations)} violations, {len(warnings)} warnings")

        return {
            "valid": valid,
            "violations": violations,
            "warnings": warnings,
        }

    def _execute_rule(self, rule: Dict, data: Dict, doc_type: str) -> Dict:
        """Execute a single business rule."""
        rule_id = rule["rule_id"]

        if rule_id == "VAL_001":
            # Three-way match
            return self._check_three_way_match(data, rule["threshold"])
        elif rule_id == "VAL_002":
            # Duplicate detection
            return self._check_duplicate(data, doc_type)
        elif rule_id == "VAL_003":
            # Date reasonableness
            return self._check_date_reasonableness(data)
        else:
            return {"passed": True}

    def _check_three_way_match(self, data: Dict, tolerance: float) -> Dict:
        """Check if invoice matches PO within tolerance."""
        if "po_number" not in data:
            return {"passed": True}  # No PO reference

        # TODO: Lookup PO from PMG
        po_data = None  # self.pmg.get_document("PURCHASE_ORDER", data["po_number"])

        if po_data is None:
            return {
                "passed": False,
                "message": "PO not found in system",
                "field": "po_number",
                "actual": data.get("po_number"),
            }

        # For now, pass
        return {"passed": True}

    def _check_duplicate(self, data: Dict, doc_type: str) -> Dict:
        """Check for duplicate documents."""
        # TODO: Check PMG for duplicates
        invoice_number = data.get("invoice_number")
        supplier_id = data.get("supplier_id")

        if not invoice_number or not supplier_id:
            return {"passed": True}

        # TODO: Query PMG
        # duplicates = pmg.find_duplicates(invoice_number, supplier_id)

        return {"passed": True}

    def _check_date_reasonableness(self, data: Dict) -> Dict:
        """Check if dates are reasonable."""
        from datetime import datetime, timedelta

        invoice_date_str = data.get("invoice_date")
        if not invoice_date_str:
            return {"passed": True}

        # TODO: Parse date and validate
        # - Not future dated
        # - Within 365 days past
        # - Before due date

        return {"passed": True}
