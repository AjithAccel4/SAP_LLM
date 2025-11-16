"""
Comprehensive Business Rule Validation Engine.

Validates extracted data against business rules including:
- Required field validation
- Value range validation
- Three-way matching
- Tolerance checking
- Cross-field dependencies
- Custom business logic
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class BusinessRuleValidator:
    """
    Enterprise-grade business rule validation engine.

    Validates extracted document data against configurable business rules:
    1. Required field presence
    2. Value range constraints
    3. Three-way matching (PO, invoice, GR)
    4. Tolerance checks
    5. Date logic
    6. Cross-field dependencies
    7. Custom business rules
    """

    # Default tolerances
    DEFAULT_PRICE_TOLERANCE = 0.03  # 3%
    DEFAULT_QUANTITY_TOLERANCE = 0.05  # 5%
    DEFAULT_PAYMENT_TERMS_DAYS = 30

    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize validator.

        Args:
            custom_rules: Optional custom validation rules by document type
        """
        self.custom_rules = custom_rules or {}

        # Load default rules
        self.default_rules = self._load_default_rules()

    def _load_default_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load default validation rules for each document type."""
        return {
            "PURCHASE_ORDER": [
                {
                    "rule_id": "PO_REQUIRED_FIELDS",
                    "type": "required_fields",
                    "fields": ["po_number", "vendor_id", "total_amount", "line_items"],
                    "severity": "ERROR",
                },
                {
                    "rule_id": "PO_POSITIVE_AMOUNT",
                    "type": "value_range",
                    "field": "total_amount",
                    "min": 0.01,
                    "severity": "ERROR",
                },
                {
                    "rule_id": "PO_LINE_ITEMS_NOT_EMPTY",
                    "type": "array_not_empty",
                    "field": "line_items",
                    "severity": "ERROR",
                },
            ],
            "SUPPLIER_INVOICE": [
                {
                    "rule_id": "INV_REQUIRED_FIELDS",
                    "type": "required_fields",
                    "fields": ["invoice_number", "supplier_id", "invoice_date", "total_amount"],
                    "severity": "ERROR",
                },
                {
                    "rule_id": "INV_POSITIVE_AMOUNT",
                    "type": "value_range",
                    "field": "total_amount",
                    "min": 0.01,
                    "severity": "ERROR",
                },
                {
                    "rule_id": "INV_THREE_WAY_MATCH",
                    "type": "three_way_match",
                    "price_tolerance": DEFAULT_PRICE_TOLERANCE,
                    "quantity_tolerance": DEFAULT_QUANTITY_TOLERANCE,
                    "severity": "WARNING",
                },
                {
                    "rule_id": "INV_TOTALS_MATCH",
                    "type": "totals_consistency",
                    "tolerance": 0.01,  # 1%
                    "severity": "ERROR",
                },
                {
                    "rule_id": "INV_DATE_LOGIC",
                    "type": "date_logic",
                    "check": "due_after_invoice",
                    "severity": "WARNING",
                },
            ],
            "SALES_ORDER": [
                {
                    "rule_id": "SO_REQUIRED_FIELDS",
                    "type": "required_fields",
                    "fields": ["order_number", "customer_id", "total_amount"],
                    "severity": "ERROR",
                },
                {
                    "rule_id": "SO_POSITIVE_AMOUNT",
                    "type": "value_range",
                    "field": "total_amount",
                    "min": 0.01,
                    "severity": "ERROR",
                },
            ],
            "CUSTOMER_INVOICE": [
                {
                    "rule_id": "CINV_REQUIRED_FIELDS",
                    "type": "required_fields",
                    "fields": ["invoice_number", "customer_id", "total_amount"],
                    "severity": "ERROR",
                },
                {
                    "rule_id": "CINV_TOTALS_MATCH",
                    "type": "totals_consistency",
                    "tolerance": 0.01,
                    "severity": "ERROR",
                },
            ],
            "GOODS_RECEIPT": [
                {
                    "rule_id": "GR_REQUIRED_FIELDS",
                    "type": "required_fields",
                    "fields": ["gr_number", "po_number", "line_items"],
                    "severity": "ERROR",
                },
                {
                    "rule_id": "GR_QUANTITY_MATCH",
                    "type": "quantity_match",
                    "tolerance": DEFAULT_QUANTITY_TOLERANCE,
                    "severity": "WARNING",
                },
            ],
        }

    def validate(
        self,
        data: Dict[str, Any],
        doc_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate extracted data against business rules.

        Args:
            data: Extracted document data
            doc_type: Document type
            context: Optional context (PO data, GR data, etc.)

        Returns:
            List of validation violations
        """
        violations = []

        # Get rules for this document type
        rules = self.default_rules.get(doc_type, [])

        # Add custom rules if any
        if doc_type in self.custom_rules:
            rules.extend(self.custom_rules[doc_type])

        # Execute each rule
        for rule in rules:
            rule_violations = self._execute_rule(rule, data, context)
            violations.extend(rule_violations)

        logger.info(
            f"Validation complete: {len(violations)} violations found "
            f"({sum(1 for v in violations if v['severity'] == 'ERROR')} errors, "
            f"{sum(1 for v in violations if v['severity'] == 'WARNING')} warnings)"
        )

        return violations

    def _execute_rule(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute a single validation rule."""
        rule_type = rule.get("type")

        if rule_type == "required_fields":
            return self._validate_required_fields(rule, data)
        elif rule_type == "value_range":
            return self._validate_value_range(rule, data)
        elif rule_type == "array_not_empty":
            return self._validate_array_not_empty(rule, data)
        elif rule_type == "three_way_match":
            return self._validate_three_way_match(rule, data, context)
        elif rule_type == "totals_consistency":
            return self._validate_totals_consistency(rule, data)
        elif rule_type == "date_logic":
            return self._validate_date_logic(rule, data)
        elif rule_type == "quantity_match":
            return self._validate_quantity_match(rule, data, context)
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return []

    def _validate_required_fields(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate required fields are present and non-empty."""
        violations = []
        required_fields = rule.get("fields", [])

        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                violations.append({
                    "rule_id": rule["rule_id"],
                    "type": "REQUIRED_FIELD",
                    "field": field,
                    "severity": rule.get("severity", "ERROR"),
                    "message": f"Required field '{field}' is missing or empty",
                })

        return violations

    def _validate_value_range(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate field value is within acceptable range."""
        violations = []
        field = rule.get("field")

        if field not in data:
            return violations

        try:
            value = float(data[field])
            min_val = rule.get("min")
            max_val = rule.get("max")

            if min_val is not None and value < min_val:
                violations.append({
                    "rule_id": rule["rule_id"],
                    "type": "VALUE_OUT_OF_RANGE",
                    "field": field,
                    "severity": rule.get("severity", "WARNING"),
                    "message": f"Field '{field}' value {value} is below minimum {min_val}",
                    "value": value,
                    "min": min_val,
                })

            if max_val is not None and value > max_val:
                violations.append({
                    "rule_id": rule["rule_id"],
                    "type": "VALUE_OUT_OF_RANGE",
                    "field": field,
                    "severity": rule.get("severity", "WARNING"),
                    "message": f"Field '{field}' value {value} exceeds maximum {max_val}",
                    "value": value,
                    "max": max_val,
                })

        except (ValueError, TypeError):
            violations.append({
                "rule_id": rule["rule_id"],
                "type": "INVALID_VALUE_TYPE",
                "field": field,
                "severity": "ERROR",
                "message": f"Field '{field}' has invalid numeric value",
            })

        return violations

    def _validate_array_not_empty(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate array field is not empty."""
        violations = []
        field = rule.get("field")

        if field not in data:
            violations.append({
                "rule_id": rule["rule_id"],
                "type": "MISSING_FIELD",
                "field": field,
                "severity": rule.get("severity", "ERROR"),
                "message": f"Array field '{field}' is missing",
            })
        elif not isinstance(data[field], list) or len(data[field]) == 0:
            violations.append({
                "rule_id": rule["rule_id"],
                "type": "EMPTY_ARRAY",
                "field": field,
                "severity": rule.get("severity", "ERROR"),
                "message": f"Array field '{field}' is empty",
            })

        return violations

    def _validate_three_way_match(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate three-way match (PO, Invoice, GR)."""
        violations = []

        if not context or "purchase_order" not in context:
            # Cannot perform three-way match without PO
            return violations

        po_data = context["purchase_order"]
        price_tolerance = rule.get("price_tolerance", self.DEFAULT_PRICE_TOLERANCE)
        quantity_tolerance = rule.get("quantity_tolerance", self.DEFAULT_QUANTITY_TOLERANCE)

        # Check total amount
        if "total_amount" in data and "total_amount" in po_data:
            inv_total = float(data["total_amount"])
            po_total = float(po_data["total_amount"])

            price_diff_pct = abs(inv_total - po_total) / po_total if po_total > 0 else 0

            if price_diff_pct > price_tolerance:
                violations.append({
                    "rule_id": rule["rule_id"],
                    "type": "THREE_WAY_MATCH_PRICE",
                    "severity": rule.get("severity", "WARNING"),
                    "message": (
                        f"Invoice total ${inv_total:,.2f} differs from PO total ${po_total:,.2f} "
                        f"by {price_diff_pct*100:.1f}% (tolerance: {price_tolerance*100:.1f}%)"
                    ),
                    "invoice_amount": inv_total,
                    "po_amount": po_total,
                    "difference_pct": price_diff_pct * 100,
                    "tolerance_pct": price_tolerance * 100,
                })

        # Check line item quantities (if available)
        if "line_items" in data and "line_items" in po_data:
            inv_items = data["line_items"]
            po_items = po_data["line_items"]

            # Match by item number or description
            for inv_item in inv_items:
                # Find matching PO item
                po_item = self._find_matching_item(inv_item, po_items)

                if po_item and "quantity" in inv_item and "quantity" in po_item:
                    inv_qty = float(inv_item["quantity"])
                    po_qty = float(po_item["quantity"])

                    qty_diff_pct = abs(inv_qty - po_qty) / po_qty if po_qty > 0 else 0

                    if qty_diff_pct > quantity_tolerance:
                        violations.append({
                            "rule_id": rule["rule_id"],
                            "type": "THREE_WAY_MATCH_QUANTITY",
                            "severity": rule.get("severity", "WARNING"),
                            "message": (
                                f"Line item quantity mismatch: Invoice {inv_qty}, PO {po_qty} "
                                f"({qty_diff_pct*100:.1f}% difference)"
                            ),
                            "item": inv_item.get("item_number", inv_item.get("description")),
                            "invoice_qty": inv_qty,
                            "po_qty": po_qty,
                        })

        return violations

    def _validate_totals_consistency(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate subtotal + tax = total."""
        violations = []
        tolerance = rule.get("tolerance", 0.01)

        # Check if all required fields are present
        if all(k in data for k in ["subtotal", "tax_amount", "total_amount"]):
            try:
                subtotal = float(data["subtotal"])
                tax = float(data["tax_amount"])
                total = float(data["total_amount"])

                calculated_total = subtotal + tax
                diff_pct = abs(calculated_total - total) / total if total > 0 else 0

                if diff_pct > tolerance:
                    violations.append({
                        "rule_id": rule["rule_id"],
                        "type": "TOTALS_MISMATCH",
                        "severity": rule.get("severity", "ERROR"),
                        "message": (
                            f"Totals don't match: Subtotal ${subtotal:,.2f} + Tax ${tax:,.2f} "
                            f"= ${calculated_total:,.2f}, but Total is ${total:,.2f}"
                        ),
                        "subtotal": subtotal,
                        "tax": tax,
                        "calculated_total": calculated_total,
                        "stated_total": total,
                        "difference": abs(calculated_total - total),
                    })
            except (ValueError, TypeError):
                violations.append({
                    "rule_id": rule["rule_id"],
                    "type": "INVALID_NUMERIC_VALUE",
                    "severity": "ERROR",
                    "message": "Cannot validate totals - invalid numeric values",
                })

        return violations

    def _validate_date_logic(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate date logic (e.g., due date after invoice date)."""
        violations = []
        check_type = rule.get("check")

        if check_type == "due_after_invoice":
            if "invoice_date" in data and "due_date" in data:
                try:
                    # Simple string comparison (assumes ISO format)
                    if data["due_date"] < data["invoice_date"]:
                        violations.append({
                            "rule_id": rule["rule_id"],
                            "type": "INVALID_DATE_LOGIC",
                            "severity": rule.get("severity", "WARNING"),
                            "message": "Due date is before invoice date",
                            "invoice_date": data["invoice_date"],
                            "due_date": data["due_date"],
                        })
                except:
                    pass

        return violations

    def _validate_quantity_match(
        self,
        rule: Dict[str, Any],
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate GR quantities match PO."""
        violations = []

        if not context or "purchase_order" not in context:
            return violations

        po_data = context["purchase_order"]
        tolerance = rule.get("tolerance", self.DEFAULT_QUANTITY_TOLERANCE)

        # Match GR line items to PO line items
        if "line_items" in data and "line_items" in po_data:
            gr_items = data["line_items"]
            po_items = po_data["line_items"]

            for gr_item in gr_items:
                po_item = self._find_matching_item(gr_item, po_items)

                if po_item and "quantity" in gr_item and "quantity" in po_item:
                    gr_qty = float(gr_item["quantity"])
                    po_qty = float(po_item["quantity"])

                    qty_diff_pct = abs(gr_qty - po_qty) / po_qty if po_qty > 0 else 0

                    if qty_diff_pct > tolerance:
                        violations.append({
                            "rule_id": rule["rule_id"],
                            "type": "QUANTITY_MISMATCH",
                            "severity": rule.get("severity", "WARNING"),
                            "message": f"GR quantity {gr_qty} exceeds PO quantity {po_qty}",
                            "item": gr_item.get("item_number"),
                            "gr_qty": gr_qty,
                            "po_qty": po_qty,
                        })

        return violations

    def _find_matching_item(
        self,
        item: Dict[str, Any],
        item_list: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find matching item in list by item_number or description."""
        item_number = item.get("item_number") or item.get("line_number")

        if item_number:
            for candidate in item_list:
                if candidate.get("item_number") == item_number or candidate.get("line_number") == item_number:
                    return candidate

        # Fallback: match by description
        description = item.get("description", "").lower()
        if description:
            for candidate in item_list:
                if candidate.get("description", "").lower() == description:
                    return candidate

        return None
