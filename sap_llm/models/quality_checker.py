"""
Comprehensive Quality Checker for Extracted Data.

Performs multi-dimensional quality assessment including:
- Field completeness
- Data type validation
- Format validation
- Confidence scoring
- Cross-field consistency
- Anomaly detection
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class QualityChecker:
    """
    Comprehensive quality checker for extracted document data.

    Assesses extraction quality across multiple dimensions:
    1. Completeness: Required fields present and populated
    2. Type Validity: Data types match schema expectations
    3. Format Validity: Values match expected patterns (dates, amounts, etc.)
    4. Confidence: Field-level confidence scores
    5. Consistency: Cross-field logical consistency
    6. Anomaly Detection: Statistical outliers and unusual values
    """

    # Format patterns
    DATE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
    ]

    AMOUNT_PATTERN = r'^-?\d{1,3}(?:,?\d{3})*(?:\.\d{2})?$'
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    PHONE_PATTERN = r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}$'

    def __init__(self, confidence_threshold: float = 0.70):
        """
        Initialize quality checker.

        Args:
            confidence_threshold: Minimum confidence for acceptable quality
        """
        self.confidence_threshold = confidence_threshold

    def check_quality(
        self,
        extracted_data: Dict[str, Any],
        schema: Dict[str, Any],
        field_confidences: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality check.

        Args:
            extracted_data: Extracted field values
            schema: Document schema with field definitions
            field_confidences: Optional confidence scores per field

        Returns:
            Quality assessment with:
            - overall_score: 0-1 quality score
            - completeness_score: Field completeness
            - validity_score: Data validity
            - confidence_score: Average confidence
            - consistency_score: Cross-field consistency
            - issues: List of quality issues found
            - field_scores: Per-field quality scores
        """
        issues = []
        field_scores = {}

        # 1. Completeness check
        completeness_score, completeness_issues = self._check_completeness(
            extracted_data,
            schema,
        )
        issues.extend(completeness_issues)

        # 2. Type validity check
        validity_score, validity_issues, field_validity = self._check_validity(
            extracted_data,
            schema,
        )
        issues.extend(validity_issues)
        field_scores.update(field_validity)

        # 3. Format validation
        format_score, format_issues, field_formats = self._check_formats(
            extracted_data,
            schema,
        )
        issues.extend(format_issues)
        for field, score in field_formats.items():
            if field in field_scores:
                field_scores[field] = (field_scores[field] + score) / 2
            else:
                field_scores[field] = score

        # 4. Confidence check
        confidence_score, confidence_issues = self._check_confidence(
            field_confidences or {},
        )
        issues.extend(confidence_issues)

        # 5. Consistency check
        consistency_score, consistency_issues = self._check_consistency(
            extracted_data,
            schema,
        )
        issues.extend(consistency_issues)

        # 6. Anomaly detection
        anomaly_score, anomaly_issues = self._detect_anomalies(
            extracted_data,
            schema,
        )
        issues.extend(anomaly_issues)

        # Calculate overall score (weighted average)
        overall_score = (
            0.30 * completeness_score +
            0.25 * validity_score +
            0.20 * format_score +
            0.15 * confidence_score +
            0.10 * consistency_score +
            0.00 * anomaly_score  # Anomalies don't reduce score, just flag issues
        )

        return {
            "overall_score": overall_score,
            "completeness_score": completeness_score,
            "validity_score": validity_score,
            "format_score": format_score,
            "confidence_score": confidence_score,
            "consistency_score": consistency_score,
            "anomaly_score": anomaly_score,
            "issues": issues,
            "field_scores": field_scores,
            "requires_review": overall_score < 0.90,
            "requires_correction": overall_score < 0.70,
        }

    def _check_completeness(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Check field completeness."""
        issues = []

        required_fields = schema.get("required", [])
        if not required_fields:
            return 1.0, issues

        missing = []
        empty = []

        for field in required_fields:
            if field not in data:
                missing.append(field)
                issues.append({
                    "type": "MISSING_FIELD",
                    "field": field,
                    "severity": "HIGH",
                    "message": f"Required field '{field}' is missing",
                })
            elif data[field] is None or data[field] == "":
                empty.append(field)
                issues.append({
                    "type": "EMPTY_FIELD",
                    "field": field,
                    "severity": "HIGH",
                    "message": f"Required field '{field}' is empty",
                })

        present = len(required_fields) - len(missing) - len(empty)
        score = present / len(required_fields) if required_fields else 1.0

        return score, issues

    def _check_validity(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[float, List[Dict[str, Any]], Dict[str, float]]:
        """Check data type validity."""
        issues = []
        field_scores = {}

        properties = schema.get("properties", {})
        if not properties:
            return 1.0, issues, field_scores

        valid_count = 0
        total_count = 0

        for field, field_schema in properties.items():
            if field not in data or data[field] is None:
                continue

            total_count += 1
            expected_type = field_schema.get("type", "string")
            value = data[field]

            is_valid = self._validate_type(value, expected_type)

            if is_valid:
                valid_count += 1
                field_scores[field] = 1.0
            else:
                field_scores[field] = 0.0
                issues.append({
                    "type": "INVALID_TYPE",
                    "field": field,
                    "severity": "MEDIUM",
                    "message": f"Field '{field}' has invalid type. Expected {expected_type}, got {type(value).__name__}",
                    "expected": expected_type,
                    "actual": type(value).__name__,
                })

        score = valid_count / total_count if total_count > 0 else 1.0
        return score, issues, field_scores

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value against expected type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_mapping.get(expected_type, str)

        if isinstance(expected_python_type, tuple):
            return isinstance(value, expected_python_type)
        else:
            return isinstance(value, expected_python_type)

    def _check_formats(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[float, List[Dict[str, Any]], Dict[str, float]]:
        """Check format validity (dates, amounts, emails, etc.)."""
        issues = []
        field_scores = {}

        properties = schema.get("properties", {})
        if not properties:
            return 1.0, issues, field_scores

        valid_count = 0
        total_count = 0

        for field, field_schema in properties.items():
            if field not in data or data[field] is None:
                continue

            format_type = field_schema.get("format")
            if not format_type:
                continue

            total_count += 1
            value = str(data[field])

            is_valid = self._validate_format(value, format_type)

            if is_valid:
                valid_count += 1
                field_scores[field] = 1.0
            else:
                field_scores[field] = 0.0
                issues.append({
                    "type": "INVALID_FORMAT",
                    "field": field,
                    "severity": "MEDIUM",
                    "message": f"Field '{field}' has invalid format. Expected {format_type}",
                    "expected_format": format_type,
                    "value": value,
                })

        score = valid_count / total_count if total_count > 0 else 1.0
        return score, issues, field_scores

    def _validate_format(self, value: str, format_type: str) -> bool:
        """Validate value format."""
        if format_type == "date":
            return any(re.match(pattern, value) for pattern in self.DATE_PATTERNS)
        elif format_type == "email":
            return bool(re.match(self.EMAIL_PATTERN, value))
        elif format_type == "phone":
            return bool(re.match(self.PHONE_PATTERN, value))
        elif format_type == "amount" or format_type == "currency":
            return bool(re.match(self.AMOUNT_PATTERN, value.replace(' ', '')))
        else:
            # Unknown format, assume valid
            return True

    def _check_confidence(
        self,
        field_confidences: Dict[str, float],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Check field-level confidence scores."""
        issues = []

        if not field_confidences:
            return 1.0, issues

        low_confidence_fields = [
            (field, conf)
            for field, conf in field_confidences.items()
            if conf < self.confidence_threshold
        ]

        for field, conf in low_confidence_fields:
            issues.append({
                "type": "LOW_CONFIDENCE",
                "field": field,
                "severity": "MEDIUM",
                "message": f"Field '{field}' has low confidence: {conf:.2f}",
                "confidence": conf,
            })

        avg_confidence = sum(field_confidences.values()) / len(field_confidences)
        return avg_confidence, issues

    def _check_consistency(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Check cross-field consistency."""
        issues = []
        checks_passed = 0
        total_checks = 0

        # Check 1: Line item totals should sum to document total
        if "line_items" in data and "total_amount" in data:
            total_checks += 1
            line_items = data["line_items"]

            if isinstance(line_items, list):
                line_total = sum(
                    float(item.get("amount", 0))
                    for item in line_items
                    if isinstance(item, dict)
                )
                doc_total = float(data.get("total_amount", 0))

                # Allow 1% tolerance
                if abs(line_total - doc_total) / max(doc_total, 1) < 0.01:
                    checks_passed += 1
                else:
                    issues.append({
                        "type": "INCONSISTENT_TOTALS",
                        "severity": "HIGH",
                        "message": f"Line items total ({line_total}) doesn't match document total ({doc_total})",
                        "line_total": line_total,
                        "doc_total": doc_total,
                    })

        # Check 2: Due date should be after invoice date
        if "invoice_date" in data and "due_date" in data:
            total_checks += 1
            try:
                # Simple date comparison (assumes ISO format)
                if data["due_date"] >= data["invoice_date"]:
                    checks_passed += 1
                else:
                    issues.append({
                        "type": "INCONSISTENT_DATES",
                        "severity": "MEDIUM",
                        "message": "Due date is before invoice date",
                        "invoice_date": data["invoice_date"],
                        "due_date": data["due_date"],
                    })
            except:
                # Date comparison failed, skip
                pass

        # Check 3: Subtotal + tax should equal total
        if all(k in data for k in ["subtotal", "tax_amount", "total_amount"]):
            total_checks += 1
            calculated = float(data["subtotal"]) + float(data["tax_amount"])
            actual = float(data["total_amount"])

            # Allow 1% tolerance
            if abs(calculated - actual) / max(actual, 1) < 0.01:
                checks_passed += 1
            else:
                issues.append({
                    "type": "INCONSISTENT_CALCULATION",
                    "severity": "MEDIUM",
                    "message": f"Subtotal + tax ({calculated}) doesn't match total ({actual})",
                    "calculated": calculated,
                    "actual": actual,
                })

        score = checks_passed / total_checks if total_checks > 0 else 1.0
        return score, issues

    def _detect_anomalies(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Detect anomalous values."""
        issues = []

        # Check for unusually large amounts
        for field in ["total_amount", "subtotal", "tax_amount"]:
            if field in data:
                try:
                    amount = float(data[field])

                    # Flag amounts over $1M as potentially anomalous
                    if amount > 1_000_000:
                        issues.append({
                            "type": "ANOMALY_LARGE_AMOUNT",
                            "field": field,
                            "severity": "LOW",
                            "message": f"Unusually large amount detected: ${amount:,.2f}",
                            "value": amount,
                        })

                    # Flag negative amounts (except for credit notes)
                    if amount < 0 and data.get("doc_type") != "CREDIT_NOTE":
                        issues.append({
                            "type": "ANOMALY_NEGATIVE_AMOUNT",
                            "field": field,
                            "severity": "MEDIUM",
                            "message": f"Negative amount detected: ${amount:,.2f}",
                            "value": amount,
                        })
                except:
                    pass

        # Always return score of 1.0 (anomalies don't reduce score, just flag)
        return 1.0, issues
