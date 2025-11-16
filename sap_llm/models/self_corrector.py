"""
Self-Correction Mechanism for Low-Quality Extractions.

Attempts to automatically correct extraction errors using:
- Re-extraction with different parameters
- OCR re-processing
- Pattern-based fixes
- Confidence-based field replacement
- PMG historical data lookup
"""

from typing import Any, Dict, List, Optional, Tuple

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SelfCorrector:
    """
    Self-correction mechanism for improving extraction quality.

    Strategies:
    1. Re-extract low-confidence fields with adjusted parameters
    2. Apply pattern-based fixes for common errors
    3. Use PMG historical data to fill missing fields
    4. Cross-validate fields for consistency
    5. Apply business logic for inference
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        max_attempts: int = 2,
    ):
        """
        Initialize self-corrector.

        Args:
            confidence_threshold: Minimum confidence to accept without correction
            max_attempts: Maximum correction attempts per field
        """
        self.confidence_threshold = confidence_threshold
        self.max_attempts = max_attempts

    def correct(
        self,
        extracted_data: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        ocr_text: str,
        schema: Dict[str, Any],
        pmg_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Attempt to correct low-quality extraction.

        Args:
            extracted_data: Original extracted data
            quality_assessment: Quality assessment results
            ocr_text: Original OCR text for re-processing
            schema: Document schema
            pmg_context: Optional PMG context for historical lookups

        Returns:
            Tuple of (corrected_data, correction_metadata)
        """
        corrected_data = extracted_data.copy()
        corrections_made = []

        logger.info(f"Starting self-correction (overall quality: {quality_assessment['overall_score']:.2f})")

        # Strategy 1: Fix missing required fields
        missing_fields = [
            issue["field"]
            for issue in quality_assessment.get("issues", [])
            if issue["type"] in ["MISSING_FIELD", "EMPTY_FIELD"]
        ]

        if missing_fields:
            logger.info(f"Attempting to fix {len(missing_fields)} missing fields")
            fixed = self._fix_missing_fields(
                corrected_data,
                missing_fields,
                ocr_text,
                schema,
                pmg_context,
            )
            corrections_made.extend(fixed)

        # Strategy 2: Fix low-confidence fields
        low_confidence_fields = [
            issue["field"]
            for issue in quality_assessment.get("issues", [])
            if issue["type"] == "LOW_CONFIDENCE"
        ]

        if low_confidence_fields:
            logger.info(f"Attempting to fix {len(low_confidence_fields)} low-confidence fields")
            fixed = self._fix_low_confidence_fields(
                corrected_data,
                low_confidence_fields,
                ocr_text,
                schema,
            )
            corrections_made.extend(fixed)

        # Strategy 3: Fix format errors
        format_errors = [
            issue["field"]
            for issue in quality_assessment.get("issues", [])
            if issue["type"] == "INVALID_FORMAT"
        ]

        if format_errors:
            logger.info(f"Attempting to fix {len(format_errors)} format errors")
            fixed = self._fix_format_errors(
                corrected_data,
                format_errors,
                schema,
            )
            corrections_made.extend(fixed)

        # Strategy 4: Fix consistency issues
        consistency_issues = [
            issue
            for issue in quality_assessment.get("issues", [])
            if issue["type"] in ["INCONSISTENT_TOTALS", "INCONSISTENT_CALCULATION"]
        ]

        if consistency_issues:
            logger.info(f"Attempting to fix {len(consistency_issues)} consistency issues")
            fixed = self._fix_consistency_issues(
                corrected_data,
                consistency_issues,
            )
            corrections_made.extend(fixed)

        # Prepare metadata
        correction_metadata = {
            "corrections_attempted": len(corrections_made),
            "corrections_successful": sum(1 for c in corrections_made if c["success"]),
            "corrections": corrections_made,
        }

        logger.info(
            f"Self-correction complete: {correction_metadata['corrections_successful']}/"
            f"{correction_metadata['corrections_attempted']} successful"
        )

        return corrected_data, correction_metadata

    def _fix_missing_fields(
        self,
        data: Dict[str, Any],
        missing_fields: List[str],
        ocr_text: str,
        schema: Dict[str, Any],
        pmg_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fix missing required fields."""
        corrections = []

        for field in missing_fields:
            # Strategy 1: Look up in PMG similar documents
            if pmg_context and "similar_documents" in pmg_context:
                value = self._lookup_field_in_pmg(
                    field,
                    pmg_context["similar_documents"],
                )

                if value:
                    data[field] = value
                    corrections.append({
                        "field": field,
                        "strategy": "pmg_lookup",
                        "success": True,
                        "value": value,
                    })
                    logger.info(f"Fixed missing field '{field}' using PMG lookup")
                    continue

            # Strategy 2: Pattern-based extraction from OCR
            value = self._extract_field_from_ocr(field, ocr_text, schema)

            if value:
                data[field] = value
                corrections.append({
                    "field": field,
                    "strategy": "pattern_extraction",
                    "success": True,
                    "value": value,
                })
                logger.info(f"Fixed missing field '{field}' using pattern extraction")
            else:
                corrections.append({
                    "field": field,
                    "strategy": "none",
                    "success": False,
                })
                logger.warning(f"Could not fix missing field '{field}'")

        return corrections

    def _fix_low_confidence_fields(
        self,
        data: Dict[str, Any],
        low_conf_fields: List[str],
        ocr_text: str,
        schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Re-extract low-confidence fields."""
        corrections = []

        for field in low_conf_fields:
            # Try pattern-based re-extraction
            new_value = self._extract_field_from_ocr(field, ocr_text, schema)

            if new_value and new_value != data.get(field):
                old_value = data.get(field)
                data[field] = new_value
                corrections.append({
                    "field": field,
                    "strategy": "re_extraction",
                    "success": True,
                    "old_value": old_value,
                    "new_value": new_value,
                })
                logger.info(f"Re-extracted field '{field}': '{old_value}' → '{new_value}'")
            else:
                corrections.append({
                    "field": field,
                    "strategy": "re_extraction",
                    "success": False,
                })

        return corrections

    def _fix_format_errors(
        self,
        data: Dict[str, Any],
        format_error_fields: List[str],
        schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Fix format errors (dates, amounts, etc.)."""
        corrections = []

        for field in format_error_fields:
            if field not in data:
                continue

            field_schema = schema.get("properties", {}).get(field, {})
            format_type = field_schema.get("format")

            old_value = data[field]
            new_value = self._fix_format(str(old_value), format_type)

            if new_value and new_value != old_value:
                data[field] = new_value
                corrections.append({
                    "field": field,
                    "strategy": "format_fix",
                    "success": True,
                    "old_value": old_value,
                    "new_value": new_value,
                })
                logger.info(f"Fixed format for '{field}': '{old_value}' → '{new_value}'")
            else:
                corrections.append({
                    "field": field,
                    "strategy": "format_fix",
                    "success": False,
                })

        return corrections

    def _fix_consistency_issues(
        self,
        data: Dict[str, Any],
        consistency_issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fix consistency issues (totals, calculations)."""
        corrections = []

        for issue in consistency_issues:
            if issue["type"] == "INCONSISTENT_TOTALS":
                # Recalculate total from subtotal + tax
                if "subtotal" in data and "tax_amount" in data:
                    calculated = float(data["subtotal"]) + float(data["tax_amount"])
                    old_total = data.get("total_amount")
                    data["total_amount"] = calculated

                    corrections.append({
                        "field": "total_amount",
                        "strategy": "recalculation",
                        "success": True,
                        "old_value": old_total,
                        "new_value": calculated,
                    })
                    logger.info(f"Recalculated total_amount: {old_total} → {calculated}")

            elif issue["type"] == "INCONSISTENT_CALCULATION":
                # Similar logic for other calculations
                corrections.append({
                    "issue_type": issue["type"],
                    "strategy": "recalculation",
                    "success": False,
                    "message": "Manual review required",
                })

        return corrections

    def _lookup_field_in_pmg(
        self,
        field: str,
        similar_docs: List[Dict[str, Any]],
    ) -> Optional[Any]:
        """Look up field value in similar PMG documents."""
        # Get most common value for this field from similar documents
        values = [
            doc.get(field)
            for doc in similar_docs
            if doc.get(field) is not None
        ]

        if not values:
            return None

        # Return most common value
        from collections import Counter
        most_common = Counter(values).most_common(1)
        return most_common[0][0] if most_common else None

    def _extract_field_from_ocr(
        self,
        field: str,
        ocr_text: str,
        schema: Dict[str, Any],
    ) -> Optional[str]:
        """Extract field value from OCR text using patterns."""
        import re

        # Common patterns for field extraction
        patterns = {
            "invoice_number": [
                r'invoice\s*(?:number|#|no\.?)\s*:?\s*(\S+)',
                r'inv\.?\s*(?:number|#|no\.?)\s*:?\s*(\S+)',
            ],
            "po_number": [
                r'(?:purchase\s*order|po)\s*(?:number|#|no\.?)\s*:?\s*(\S+)',
            ],
            "invoice_date": [
                r'invoice\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            "total_amount": [
                r'total\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})',
                r'amount\s*due\s*:?\s*\$?\s*([\d,]+\.?\d{0,2})',
            ],
        }

        field_patterns = patterns.get(field, [])

        for pattern in field_patterns:
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _fix_format(self, value: str, format_type: Optional[str]) -> Optional[str]:
        """Fix format of a value."""
        if not format_type:
            return value

        if format_type == "date":
            # Try to standardize date format
            import re
            # Convert MM/DD/YYYY or DD/MM/YYYY to YYYY-MM-DD
            patterns = [
                (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', r'\3-\1-\2'),
                (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', r'\1-\2-\3'),
            ]

            for pattern, replacement in patterns:
                if re.match(pattern, value):
                    return re.sub(pattern, replacement, value)

        elif format_type in ["amount", "currency"]:
            # Remove currency symbols and normalize
            cleaned = value.replace('$', '').replace(',', '').strip()
            try:
                float(cleaned)
                return cleaned
            except:
                pass

        return value
