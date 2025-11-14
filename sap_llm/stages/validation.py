"""
Stage 7: Validation - Business Rules & Tolerance Checks

Validates extracted data against business rules, historical patterns,
and tolerance thresholds using PMG context.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.query import PMGQueryEngine
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

        # Initialize PMG
        self.pmg = None
        self.pmg_query = None
        if self.pmg_enabled:
            try:
                self.pmg = ProcessMemoryGraph()
                self.pmg_query = PMGQueryEngine(self.pmg)
                logger.info("PMG integration enabled for validation")
            except Exception as e:
                logger.warning(f"Failed to initialize PMG: {e}. Running without PMG.")
                self.pmg_enabled = False

        # Load business rules
        self.rules = self._load_business_rules()
        self.tolerance_rules = self._load_tolerance_rules()

    def _load_business_rules(self) -> List[Dict]:
        """Load business rules from JSON files."""
        rules = []

        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        rules_dir = project_root / "data" / "schemas" / "business_rules"

        # Load validation rules
        validation_rules_path = rules_dir / "validation_rules.json"

        try:
            if validation_rules_path.exists():
                with open(validation_rules_path, 'r') as f:
                    data = json.load(f)
                    loaded_rules = data.get("validation_rules", [])
                    # Filter only enabled rules
                    rules.extend([r for r in loaded_rules if r.get("enabled", True)])
                    logger.info(f"Loaded {len(loaded_rules)} validation rules from {validation_rules_path}")
            else:
                logger.warning(f"Validation rules file not found: {validation_rules_path}")
                # Fallback to default rules
                rules = self._get_default_rules()
        except Exception as e:
            logger.error(f"Failed to load business rules: {e}. Using default rules.")
            rules = self._get_default_rules()

        return rules

    def _get_default_rules(self) -> List[Dict]:
        """Get default business rules as fallback."""
        return [
            {
                "rule_id": "VAL_001",
                "name": "Three-Way Match Price Variance",
                "doc_types": ["SUPPLIER_INVOICE"],
                "severity": "MEDIUM",
                "threshold": 0.03,
                "parameters": {"tolerance_percentage": 3.0}
            },
            {
                "rule_id": "VAL_002",
                "name": "Duplicate Invoice Detection",
                "doc_types": ["SUPPLIER_INVOICE"],
                "severity": "HIGH",
                "parameters": {"lookback_days": 365}
            },
            {
                "rule_id": "VAL_003",
                "name": "Invoice Date Reasonableness",
                "doc_types": ["SUPPLIER_INVOICE"],
                "severity": "LOW",
                "parameters": {"max_past_days": 365, "max_future_days": 0}
            },
        ]

    def _load_tolerance_rules(self) -> List[Dict]:
        """Load tolerance rules from JSON files."""
        rules = []

        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        rules_dir = project_root / "data" / "schemas" / "business_rules"

        # Load tolerance rules
        tolerance_rules_path = rules_dir / "tolerance_rules.json"

        try:
            if tolerance_rules_path.exists():
                with open(tolerance_rules_path, 'r') as f:
                    data = json.load(f)
                    rules = data.get("tolerance_rules", [])
                    logger.info(f"Loaded {len(rules)} tolerance rules from {tolerance_rules_path}")
            else:
                logger.warning(f"Tolerance rules file not found: {tolerance_rules_path}")
        except Exception as e:
            logger.error(f"Failed to load tolerance rules: {e}")

        return rules

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
            threshold = rule.get("threshold", 0.03)
            return self._check_three_way_match(data, threshold, rule)
        elif rule_id == "VAL_002":
            # Duplicate detection
            return self._check_duplicate(data, doc_type, rule)
        elif rule_id == "VAL_003":
            # Date reasonableness
            return self._check_date_reasonableness(data, rule)
        elif rule_id == "VAL_004":
            # Vendor validation
            return self._check_vendor_validation(data, rule)
        elif rule_id == "VAL_005":
            # Required fields
            return self._check_required_fields(data, rule)
        elif rule_id == "VAL_006":
            # Amount reasonableness
            return self._check_amount_reasonableness(data, rule)
        else:
            return {"passed": True}

    def _check_three_way_match(self, data: Dict, tolerance: float, rule: Dict) -> Dict:
        """Check if invoice matches PO within tolerance."""
        po_number = data.get("po_number")

        if not po_number:
            return {"passed": True}  # No PO reference

        # Lookup PO from PMG
        po_data = None
        if self.pmg_enabled and self.pmg:
            try:
                # Search for PO in PMG by finding documents of type PURCHASE_ORDER
                similar_docs = self.pmg.find_similar_documents(
                    doc_type="PURCHASE_ORDER",
                    limit=100
                )

                # Find matching PO number in the results
                for doc in similar_docs:
                    doc_adc = doc.get("adc_json", "{}")
                    if isinstance(doc_adc, str):
                        try:
                            doc_adc = json.loads(doc_adc)
                        except:
                            continue

                    if doc_adc.get("po_number") == po_number:
                        po_data = doc_adc
                        break

                logger.debug(f"PO lookup result for {po_number}: {'found' if po_data else 'not found'}")

            except Exception as e:
                logger.error(f"Failed to lookup PO from PMG: {e}")

        if po_data is None:
            return {
                "passed": False,
                "message": f"Purchase Order {po_number} not found in system",
                "field": "po_number",
                "actual": po_number,
            }

        # Check price variance
        invoice_amount = float(data.get("total_amount", 0))
        po_amount = float(po_data.get("total_amount", 0))

        if po_amount == 0:
            return {"passed": True}  # Cannot validate without PO amount

        variance = abs(invoice_amount - po_amount) / po_amount

        if variance > tolerance:
            return {
                "passed": False,
                "message": f"Price variance {variance*100:.2f}% exceeds tolerance {tolerance*100:.2f}%",
                "field": "total_amount",
                "expected": po_amount,
                "actual": invoice_amount,
            }

        return {"passed": True}

    def _check_duplicate(self, data: Dict, doc_type: str, rule: Dict) -> Dict:
        """Check for duplicate documents using invoice number, supplier, and document hash."""
        invoice_number = data.get("invoice_number")
        supplier_id = data.get("supplier_id")

        if not invoice_number or not supplier_id:
            return {"passed": True}

        # Check PMG for duplicates
        if self.pmg_enabled and self.pmg:
            try:
                # Calculate document hash for content-based duplicate detection
                doc_hash = self._calculate_document_hash(data)

                # Query PMG for documents with same supplier and invoice number
                similar_docs = self.pmg.find_similar_documents(
                    doc_type=doc_type,
                    supplier_id=supplier_id,
                    limit=100
                )

                # Check for duplicates
                for doc in similar_docs:
                    doc_adc = doc.get("adc_json", "{}")
                    if isinstance(doc_adc, str):
                        try:
                            doc_adc = json.loads(doc_adc)
                        except:
                            continue

                    # Check invoice number match
                    if doc_adc.get("invoice_number") == invoice_number:
                        return {
                            "passed": False,
                            "message": f"Duplicate invoice detected: Invoice {invoice_number} already exists for supplier {supplier_id}",
                            "field": "invoice_number",
                            "actual": invoice_number,
                        }

                    # Check document hash match (content-based duplicate)
                    existing_hash = self._calculate_document_hash(doc_adc)
                    if existing_hash == doc_hash:
                        return {
                            "passed": False,
                            "message": f"Duplicate document content detected for invoice {invoice_number}",
                            "field": "document_hash",
                            "actual": doc_hash,
                        }

                logger.debug(f"No duplicates found for invoice {invoice_number}")

            except Exception as e:
                logger.error(f"Failed to check duplicates in PMG: {e}")

        return {"passed": True}

    def _calculate_document_hash(self, data: Dict) -> str:
        """Calculate hash of document for duplicate detection."""
        # Create a deterministic string representation of key fields
        hash_fields = [
            data.get("invoice_number", ""),
            data.get("supplier_id", ""),
            str(data.get("total_amount", 0)),
            data.get("currency", ""),
            data.get("invoice_date", ""),
        ]

        hash_string = "|".join(hash_fields)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def _check_date_reasonableness(self, data: Dict, rule: Dict) -> Dict:
        """Check if dates are reasonable with multiple format support."""
        invoice_date_str = data.get("invoice_date")
        if not invoice_date_str:
            return {"passed": True}

        # Parse invoice date
        invoice_date = self._parse_date(invoice_date_str)

        if invoice_date is None:
            return {
                "passed": False,
                "message": f"Invalid date format: {invoice_date_str}",
                "field": "invoice_date",
                "actual": invoice_date_str,
            }

        # Get parameters from rule
        params = rule.get("parameters", {})
        max_future_days = params.get("max_future_days", 0)
        max_past_days = params.get("max_past_days", 365)
        check_before_due = params.get("check_before_due_date", True)

        now = datetime.now()

        # Check if future dated
        if invoice_date > now + timedelta(days=max_future_days):
            return {
                "passed": False,
                "message": f"Invoice date {invoice_date_str} is future dated (max {max_future_days} days allowed)",
                "field": "invoice_date",
                "actual": invoice_date_str,
                "expected": f"<= {(now + timedelta(days=max_future_days)).strftime('%Y-%m-%d')}",
            }

        # Check if too old
        if invoice_date < now - timedelta(days=max_past_days):
            return {
                "passed": False,
                "message": f"Invoice date {invoice_date_str} is too old (max {max_past_days} days in past)",
                "field": "invoice_date",
                "actual": invoice_date_str,
                "expected": f">= {(now - timedelta(days=max_past_days)).strftime('%Y-%m-%d')}",
            }

        # Check due date if present
        if check_before_due and "due_date" in data:
            due_date = self._parse_date(data["due_date"])
            if due_date and invoice_date > due_date:
                return {
                    "passed": False,
                    "message": f"Invoice date {invoice_date_str} is after due date {data['due_date']}",
                    "field": "invoice_date",
                    "actual": invoice_date_str,
                    "expected": f"<= {data['due_date']}",
                }

        logger.debug(f"Date validation passed for {invoice_date_str}")
        return {"passed": True}

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string with support for multiple formats."""
        if not date_str:
            return None

        # Common date formats
        date_formats = [
            "%Y-%m-%d",           # 2024-01-15
            "%d/%m/%Y",           # 15/01/2024
            "%m/%d/%Y",           # 01/15/2024
            "%Y/%m/%d",           # 2024/01/15
            "%d.%m.%Y",           # 15.01.2024
            "%d-%m-%Y",           # 15-01-2024
            "%Y%m%d",             # 20240115
            "%b %d, %Y",          # Jan 15, 2024
            "%B %d, %Y",          # January 15, 2024
            "%d %b %Y",           # 15 Jan 2024
            "%d %B %Y",           # 15 January 2024
            "%Y-%m-%dT%H:%M:%S",  # ISO 8601: 2024-01-15T10:30:00
            "%Y-%m-%d %H:%M:%S",  # 2024-01-15 10:30:00
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Unable to parse date: {date_str}")
        return None

    def _check_vendor_validation(self, data: Dict, rule: Dict) -> Dict:
        """Validate vendor against historical data using PMG."""
        supplier_id = data.get("supplier_id")

        if not supplier_id:
            return {
                "passed": False,
                "message": "Supplier ID is missing",
                "field": "supplier_id",
            }

        # Check vendor in PMG
        if self.pmg_enabled and self.pmg_query:
            try:
                # Get supplier history from PMG
                supplier_history = self.pmg_query.get_supplier_history(
                    supplier_id=supplier_id,
                    limit=10
                )

                params = rule.get("parameters", {})
                min_transactions = params.get("min_transaction_history", 1)

                if not supplier_history:
                    # New vendor - might be valid but flag for review
                    return {
                        "passed": True,  # Don't block new vendors
                        "message": f"New vendor {supplier_id} with no transaction history",
                        "field": "supplier_id",
                        "actual": supplier_id,
                    }

                if len(supplier_history) < min_transactions:
                    return {
                        "passed": True,
                        "message": f"Vendor {supplier_id} has limited transaction history ({len(supplier_history)} transactions)",
                        "field": "supplier_id",
                        "actual": supplier_id,
                    }

                logger.debug(f"Vendor validation passed for {supplier_id} ({len(supplier_history)} historical transactions)")

            except Exception as e:
                logger.error(f"Failed to validate vendor in PMG: {e}")

        return {"passed": True}

    def _check_required_fields(self, data: Dict, rule: Dict) -> Dict:
        """Check that all required fields are present."""
        params = rule.get("parameters", {})
        required_fields = params.get("required_fields", [])

        missing_fields = []
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)

        if missing_fields:
            return {
                "passed": False,
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "field": "required_fields",
                "actual": list(data.keys()),
                "expected": required_fields,
            }

        return {"passed": True}

    def _check_amount_reasonableness(self, data: Dict, rule: Dict) -> Dict:
        """Check if amounts are within reasonable ranges."""
        total_amount = data.get("total_amount")

        if total_amount is None:
            return {"passed": True}

        try:
            amount = float(total_amount)
        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": f"Invalid amount format: {total_amount}",
                "field": "total_amount",
                "actual": total_amount,
            }

        params = rule.get("parameters", {})
        min_amount = params.get("min_amount", 0.01)
        max_amount = params.get("max_amount", 10000000)
        check_negative = params.get("check_negative_amounts", True)

        if check_negative and amount < 0:
            return {
                "passed": False,
                "message": f"Negative amount not allowed: {amount}",
                "field": "total_amount",
                "actual": amount,
            }

        if amount < min_amount:
            return {
                "passed": False,
                "message": f"Amount {amount} is below minimum threshold {min_amount}",
                "field": "total_amount",
                "actual": amount,
                "expected": f">= {min_amount}",
            }

        if amount > max_amount:
            return {
                "passed": False,
                "message": f"Amount {amount} exceeds maximum threshold {max_amount}",
                "field": "total_amount",
                "actual": amount,
                "expected": f"<= {max_amount}",
            }

        return {"passed": True}
