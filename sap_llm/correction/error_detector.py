"""
Comprehensive Error Detection System.

Detects errors using multiple methods:
1. Confidence-based detection
2. Business rule violations
3. Historical inconsistencies
4. Anomalies and outliers
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Error:
    """Represents a detected error."""

    type: str  # "low_confidence", "rule_violation", "inconsistency", "anomaly"
    severity: str  # "low", "medium", "high", "critical"
    fields: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "type": self.type,
            "severity": self.severity,
            "fields": self.fields,
            "violations": self.violations,
            "details": self.details,
            "anomalies": self.anomalies,
            "message": self.message,
        }


@dataclass
class ErrorReport:
    """Report of detected errors."""

    has_errors: bool
    errors: List[Error]
    overall_confidence: float
    needs_correction: bool
    detection_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "has_errors": self.has_errors,
            "errors": [e.to_dict() for e in self.errors],
            "overall_confidence": self.overall_confidence,
            "needs_correction": self.needs_correction,
            "detection_timestamp": self.detection_timestamp.isoformat(),
        }

    def get_highest_severity_error(self) -> Optional[Error]:
        """Get the error with highest severity."""
        if not self.errors:
            return None

        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return max(
            self.errors,
            key=lambda e: severity_order.get(e.severity, 0)
        )


class AnomalyDetector:
    """Detects anomalies in extracted data."""

    def __init__(self):
        """Initialize anomaly detector."""
        self.anomaly_thresholds = {
            "amount_deviation": 3.0,  # Z-score threshold
            "date_range_days": 365,   # Max days from today
        }

    def detect(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in prediction.

        Args:
            prediction: Extracted prediction data

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Check for unrealistic amounts
        if "total_amount" in prediction:
            amount = prediction["total_amount"].get("value", 0)
            if isinstance(amount, (int, float)):
                if amount < 0:
                    anomalies.append({
                        "field": "total_amount",
                        "type": "negative_amount",
                        "value": amount,
                        "message": "Negative amount detected"
                    })
                elif amount > 1000000:  # $1M threshold
                    anomalies.append({
                        "field": "total_amount",
                        "type": "extreme_amount",
                        "value": amount,
                        "message": f"Unusually high amount: ${amount:,.2f}"
                    })

        # Check for future dates
        if "invoice_date" in prediction:
            inv_date = prediction["invoice_date"].get("value")
            if inv_date:
                try:
                    from dateutil import parser
                    date_obj = parser.parse(str(inv_date))
                    if date_obj > datetime.now():
                        anomalies.append({
                            "field": "invoice_date",
                            "type": "future_date",
                            "value": inv_date,
                            "message": "Invoice date is in the future"
                        })
                except Exception as e:
                    logger.debug(f"Could not parse date '{inv_date}': {e}")

        return anomalies


class ErrorDetector:
    """
    Comprehensive error detection for extracted document data.

    Uses multiple detection methods:
    1. Confidence-based detection
    2. Business rule violations
    3. Historical inconsistencies
    4. Anomaly detection
    """

    def __init__(self, pmg=None):
        """
        Initialize error detector.

        Args:
            pmg: Optional ProcessMemoryGraph for historical data
        """
        self.pmg = pmg
        self.anomaly_detector = AnomalyDetector()

        # Confidence thresholds
        self.confidence_thresholds = {
            "critical_fields": 0.85,  # vendor_id, total_amount, invoice_number
            "required_fields": 0.75,  # All required fields
            "optional_fields": 0.70,  # Optional fields
        }

        # Critical fields that require high confidence
        self.critical_fields = {
            "total_amount", "vendor_id", "vendor_name",
            "invoice_number", "invoice_date"
        }

    def detect_errors(
        self,
        prediction: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ErrorReport:
        """
        Detect errors in prediction using multiple methods.

        Args:
            prediction: Extracted prediction data
            context: Context information (document, doc_type, etc.)

        Returns:
            ErrorReport with detected errors
        """
        errors = []

        logger.info("Starting comprehensive error detection")

        # 1. Confidence-based detection
        low_confidence_fields = self._detect_low_confidence(prediction)
        if low_confidence_fields:
            errors.append(Error(
                type="low_confidence",
                fields=low_confidence_fields,
                severity=self._determine_severity(low_confidence_fields),
                message=f"Low confidence detected in {len(low_confidence_fields)} field(s)"
            ))
            logger.warning(f"Low confidence in fields: {low_confidence_fields}")

        # 2. Business rule violations
        rule_violations = self._detect_rule_violations(prediction)
        if rule_violations:
            errors.append(Error(
                type="rule_violation",
                violations=rule_violations,
                severity="high",
                fields=[v.get("field", "") for v in rule_violations],
                message=f"{len(rule_violations)} business rule violation(s) detected"
            ))
            logger.warning(f"Business rule violations: {len(rule_violations)}")

        # 3. Historical inconsistencies (if PMG available)
        if self.pmg:
            inconsistencies = self._detect_historical_inconsistencies(
                prediction,
                context
            )
            if inconsistencies:
                errors.append(Error(
                    type="inconsistency",
                    details=inconsistencies,
                    severity="medium",
                    fields=list(inconsistencies.keys()),
                    message=f"Inconsistencies with historical data detected"
                ))
                logger.info(f"Historical inconsistencies: {len(inconsistencies)}")

        # 4. Anomalies
        anomalies = self.anomaly_detector.detect(prediction)
        if anomalies:
            errors.append(Error(
                type="anomaly",
                anomalies=anomalies,
                severity="low",
                fields=[a["field"] for a in anomalies],
                message=f"{len(anomalies)} anomaly(ies) detected"
            ))
            logger.info(f"Anomalies detected: {len(anomalies)}")

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(prediction)

        # Determine if correction is needed
        needs_correction = (
            len(errors) > 0 or
            overall_confidence < 0.80 or
            any(e.severity in ["high", "critical"] for e in errors)
        )

        logger.info(
            f"Error detection complete: {len(errors)} error type(s), "
            f"confidence={overall_confidence:.2f}, needs_correction={needs_correction}"
        )

        return ErrorReport(
            has_errors=len(errors) > 0,
            errors=errors,
            overall_confidence=overall_confidence,
            needs_correction=needs_correction
        )

    def _detect_low_confidence(self, prediction: Dict[str, Any]) -> List[str]:
        """
        Detect fields with low confidence scores.

        Args:
            prediction: Prediction data

        Returns:
            List of field names with low confidence
        """
        low_conf_fields = []

        for field, data in prediction.items():
            if not isinstance(data, dict):
                continue

            confidence = data.get("confidence", 1.0)

            # Determine threshold based on field type
            if field in self.critical_fields:
                threshold = self.confidence_thresholds["critical_fields"]
            else:
                threshold = self.confidence_thresholds["required_fields"]

            if confidence < threshold:
                low_conf_fields.append(field)
                logger.debug(f"Field '{field}' has low confidence: {confidence:.2f} < {threshold:.2f}")

        return low_conf_fields

    def _detect_rule_violations(
        self,
        prediction: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect business rule violations.

        Args:
            prediction: Prediction data

        Returns:
            List of rule violations
        """
        violations = []

        # Rule 1: Total amount should match sum of line items
        if "total_amount" in prediction and "line_items" in prediction:
            try:
                line_items = prediction["line_items"].get("value", [])
                if isinstance(line_items, list) and line_items:
                    calculated_total = sum(
                        float(item.get("amount", 0))
                        for item in line_items
                        if isinstance(item, dict)
                    )

                    actual_total = float(prediction["total_amount"].get("value", 0))

                    # Allow small rounding differences
                    if abs(calculated_total - actual_total) > 0.01:
                        violations.append({
                            "rule": "total_matches_line_items",
                            "field": "total_amount",
                            "expected": calculated_total,
                            "actual": actual_total,
                            "difference": abs(calculated_total - actual_total),
                            "message": f"Total mismatch: expected ${calculated_total:.2f}, got ${actual_total:.2f}"
                        })
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not validate total_amount rule: {e}")

        # Rule 2: Total = Subtotal + Tax
        if all(k in prediction for k in ["total_amount", "subtotal", "tax_amount"]):
            try:
                subtotal = float(prediction["subtotal"].get("value", 0))
                tax = float(prediction["tax_amount"].get("value", 0))
                total = float(prediction["total_amount"].get("value", 0))

                calculated = subtotal + tax

                if abs(calculated - total) > 0.01:
                    violations.append({
                        "rule": "total_equals_subtotal_plus_tax",
                        "field": "total_amount",
                        "expected": calculated,
                        "actual": total,
                        "difference": abs(calculated - total),
                        "message": f"Total should be ${calculated:.2f} (subtotal + tax)"
                    })
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not validate subtotal+tax rule: {e}")

        # Rule 3: Due date should be after invoice date
        if "invoice_date" in prediction and "due_date" in prediction:
            try:
                from dateutil import parser

                invoice_date = parser.parse(str(prediction["invoice_date"].get("value")))
                due_date = parser.parse(str(prediction["due_date"].get("value")))

                if due_date < invoice_date:
                    violations.append({
                        "rule": "due_date_after_invoice_date",
                        "field": "due_date",
                        "invoice_date": str(invoice_date),
                        "due_date": str(due_date),
                        "message": "Due date cannot be before invoice date"
                    })
            except Exception as e:
                logger.debug(f"Could not validate date rule: {e}")

        # Rule 4: Required fields must be present
        required_fields = ["invoice_number", "vendor_name", "total_amount"]
        for field in required_fields:
            if field not in prediction or not prediction[field].get("value"):
                violations.append({
                    "rule": "required_field_present",
                    "field": field,
                    "message": f"Required field '{field}' is missing or empty"
                })

        return violations

    def _detect_historical_inconsistencies(
        self,
        prediction: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect inconsistencies with historical data from PMG.

        Args:
            prediction: Current prediction
            context: Context with document info

        Returns:
            Dictionary of inconsistencies
        """
        inconsistencies = {}

        if not self.pmg:
            return inconsistencies

        # Get vendor if available
        vendor = prediction.get("vendor_name", {}).get("value")
        if not vendor:
            return inconsistencies

        try:
            # Query PMG for similar documents from same vendor
            similar_docs = self.pmg.query_similar_documents(
                vendor=vendor,
                doc_type=context.get("document_type"),
                limit=10
            )

            if not similar_docs:
                return inconsistencies

            # Check for payment terms consistency
            if "payment_terms" in prediction:
                current_terms = prediction["payment_terms"].get("value")
                historical_terms = [
                    doc.get("payment_terms")
                    for doc in similar_docs
                    if doc.get("payment_terms")
                ]

                if historical_terms:
                    from collections import Counter
                    most_common = Counter(historical_terms).most_common(1)[0][0]

                    if current_terms != most_common:
                        inconsistencies["payment_terms"] = {
                            "current": current_terms,
                            "historical_common": most_common,
                            "message": f"Payment terms differ from historical pattern"
                        }

            # Check for currency consistency
            if "currency" in prediction:
                current_currency = prediction["currency"].get("value")
                historical_currencies = [
                    doc.get("currency")
                    for doc in similar_docs
                    if doc.get("currency")
                ]

                if historical_currencies:
                    from collections import Counter
                    most_common = Counter(historical_currencies).most_common(1)[0][0]

                    if current_currency != most_common:
                        inconsistencies["currency"] = {
                            "current": current_currency,
                            "historical_common": most_common,
                            "message": f"Currency differs from historical pattern for this vendor"
                        }

        except Exception as e:
            logger.error(f"Error checking historical inconsistencies: {e}")

        return inconsistencies

    def _calculate_overall_confidence(
        self,
        prediction: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence score for prediction.

        Args:
            prediction: Prediction data

        Returns:
            Overall confidence score (0-1)
        """
        confidences = []

        for field, data in prediction.items():
            if isinstance(data, dict) and "confidence" in data:
                confidence = data["confidence"]

                # Weight critical fields more heavily
                if field in self.critical_fields:
                    confidences.extend([confidence] * 2)  # Count twice
                else:
                    confidences.append(confidence)

        if not confidences:
            return 1.0  # No confidence scores available

        # Return weighted average
        return sum(confidences) / len(confidences)

    def _determine_severity(self, low_conf_fields: List[str]) -> str:
        """
        Determine severity based on which fields have low confidence.

        Args:
            low_conf_fields: Fields with low confidence

        Returns:
            Severity level
        """
        # Check if any critical fields are affected
        critical_affected = any(
            field in self.critical_fields
            for field in low_conf_fields
        )

        if critical_affected:
            return "high"
        elif len(low_conf_fields) > 3:
            return "medium"
        else:
            return "low"
