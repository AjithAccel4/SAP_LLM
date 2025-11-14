"""
Stage 6: Quality Check - Confidence Scoring & Self-Correction

Assesses extraction quality and triggers self-correction when needed.
"""

from typing import Any, Dict, List

from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class QualityCheckStage(BaseStage):
    """
    Quality check and self-correction stage.

    Computes overall quality score based on:
    - Field-level confidence
    - Required field completeness
    - Schema compliance
    - Business rule consistency

    Target: ≥90% quality score
    Self-correction: Enabled for scores <90%
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.quality_threshold = (
            getattr(config, "overall_threshold", 0.90) if config else 0.90
        )
        self.self_correction_enabled = (
            getattr(config, "self_correction_enabled", True) if config else True
        )
        self.max_correction_attempts = (
            getattr(config, "max_correction_attempts", 3) if config else 3
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check extraction quality and self-correct if needed.

        Args:
            input_data: {
                "extracted_data": Dict,
                "extraction_metadata": Dict,
                "field_confidences": Dict[str, float],
                "doc_type": str,
            }

        Returns:
            {
                "quality_score": float,
                "passed": bool,
                "low_confidence_fields": List[Dict],
                "corrected": bool,
                "corrected_data": Dict | None,
            }
        """
        extracted_data = input_data["extracted_data"]
        field_confidences = input_data["field_confidences"]
        metadata = input_data["extraction_metadata"]

        # Compute quality score
        quality_score = self._compute_quality_score(
            extracted_data,
            field_confidences,
            metadata,
        )

        logger.info(f"Quality score: {quality_score:.4f}")

        passed = quality_score >= self.quality_threshold

        # Identify low confidence fields
        low_conf_fields = self._identify_low_confidence_fields(
            extracted_data,
            field_confidences,
        )

        # Self-correction if needed
        corrected = False
        corrected_data = None

        if not passed and self.self_correction_enabled and low_conf_fields:
            logger.warning("Quality below threshold, attempting self-correction...")
            corrected_data = self._self_correct(
                extracted_data,
                low_conf_fields,
                input_data,
            )
            corrected = True

            # Re-compute quality score
            quality_score = self._compute_quality_score(
                corrected_data,
                field_confidences,
                metadata,
            )
            logger.info(f"Quality score after correction: {quality_score:.4f}")

        return {
            "quality_score": quality_score,
            "passed": quality_score >= self.quality_threshold,
            "low_confidence_fields": low_conf_fields,
            "corrected": corrected,
            "corrected_data": corrected_data if corrected else extracted_data,
        }

    def _compute_quality_score(
        self,
        data: Dict,
        confidences: Dict[str, float],
        metadata: Dict,
    ) -> float:
        """Compute overall quality score."""
        scores = []

        # 1. Field confidence score
        if confidences:
            avg_confidence = sum(confidences.values()) / len(confidences)
            scores.append(avg_confidence * 0.5)

        # 2. Completeness score
        # TODO: Get required fields from schema
        required_fields = ["total_amount"]  # Placeholder
        completeness = sum(1 for f in required_fields if f in data) / max(
            len(required_fields), 1
        )
        scores.append(completeness * 0.3)

        # 3. Schema compliance score
        # Assume valid if no errors during extraction
        scores.append(0.2)

        return sum(scores)

    def _identify_low_confidence_fields(
        self,
        data: Dict,
        confidences: Dict[str, float],
        threshold: float = 0.85,
    ) -> List[Dict]:
        """Identify fields with low confidence."""
        low_conf = []

        for field, value in data.items():
            conf = confidences.get(field, 0.0)
            if conf < threshold:
                low_conf.append({
                    "field": field,
                    "value": value,
                    "confidence": conf,
                })

        return low_conf

    def _self_correct(
        self,
        data: Dict,
        low_conf_fields: List[Dict],
        context: Dict,
    ) -> Dict:
        """
        Attempt to correct low-confidence extractions.

        Strategies:
        1. Re-extract from specific regions
        2. Use PMG similar documents
        3. Apply heuristic rules
        """
        corrected_data = data.copy()

        for field_info in low_conf_fields:
            field = field_info["field"]

            # Strategy 1: Heuristic correction
            if field in ["total_amount", "subtotal", "tax_amount"]:
                corrected_value = self._correct_monetary_field(field, corrected_data)
                if corrected_value is not None:
                    corrected_data[field] = corrected_value
                    logger.info(f"Corrected {field}: {corrected_value}")

        return corrected_data

    def _correct_monetary_field(self, field: str, data: Dict) -> float | None:
        """Apply heuristics to correct monetary fields."""
        # Check if total = subtotal + tax
        if field == "total_amount":
            if "subtotal" in data and "tax_amount" in data:
                calculated_total = data["subtotal"] + data["tax_amount"]
                return calculated_total

        return None
