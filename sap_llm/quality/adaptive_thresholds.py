"""
Adaptive Quality Thresholds

Learn optimal quality thresholds per supplier and document type:
- Baseline threshold: 0.85 (global)
- Supplier-specific adjustments (-0.10 to +0.10)
- Document type adjustments
- Historical accuracy tracking
- Automatic threshold tuning

Benefits:
- Fewer false rejections for trusted suppliers
- Higher touchless rate
- Maintained accuracy standards
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SupplierProfile:
    """Supplier accuracy profile."""
    supplier_id: str
    supplier_name: str

    # Historical accuracy
    total_documents: int
    successful_documents: int
    accuracy_rate: float

    # Quality metrics
    avg_extraction_confidence: float
    avg_validation_success_rate: float

    # Threshold adjustments
    base_threshold: float = 0.85
    threshold_adjustment: float = 0.0
    effective_threshold: float = 0.85

    # Last updated
    last_updated: str = ""


@dataclass
class DocumentTypeProfile:
    """Document type accuracy profile."""
    doc_type: str

    # Historical accuracy
    total_documents: int
    successful_documents: int
    accuracy_rate: float

    # Complexity score (0-1, higher = more complex)
    complexity_score: float

    # Threshold adjustment
    base_threshold: float = 0.85
    threshold_adjustment: float = 0.0
    effective_threshold: float = 0.85


class AdaptiveQualityThresholds:
    """
    Adaptive quality threshold management.

    Learning Strategy:
    1. Track supplier performance over time
    2. Calculate supplier-specific accuracy rates
    3. Adjust thresholds based on historical success
    4. Maintain minimum quality standards
    5. Re-evaluate monthly

    Threshold Rules:
    - Trusted suppliers (>98% accuracy): -0.05 adjustment
    - Standard suppliers (95-98%): No adjustment
    - Problematic suppliers (<95%): +0.05 adjustment
    - Never go below 0.75 or above 0.95
    """

    def __init__(self):
        self.base_threshold = 0.85

        # Supplier profiles
        self.supplier_profiles: Dict[str, SupplierProfile] = {}

        # Document type profiles
        self.doc_type_profiles: Dict[str, DocumentTypeProfile] = {}

        # Adjustment limits
        self.min_threshold = 0.75
        self.max_threshold = 0.95

        logger.info("AdaptiveQualityThresholds initialized")

    def get_threshold(
        self,
        supplier_id: Optional[str] = None,
        doc_type: Optional[str] = None
    ) -> float:
        """
        Get adaptive quality threshold.

        Args:
            supplier_id: Supplier identifier
            doc_type: Document type

        Returns:
            Adjusted quality threshold
        """
        threshold = self.base_threshold

        # Supplier adjustment
        if supplier_id and supplier_id in self.supplier_profiles:
            profile = self.supplier_profiles[supplier_id]
            threshold += profile.threshold_adjustment

        # Document type adjustment
        if doc_type and doc_type in self.doc_type_profiles:
            profile = self.doc_type_profiles[doc_type]
            threshold += profile.threshold_adjustment

        # Enforce limits
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))

        return threshold

    def update_supplier_profile(
        self,
        supplier_id: str,
        supplier_name: str,
        extraction_confidence: float,
        validation_success: bool,
        overall_success: bool
    ):
        """
        Update supplier profile with new data point.

        Args:
            supplier_id: Supplier identifier
            supplier_name: Supplier name
            extraction_confidence: Confidence score
            validation_success: Validation passed
            overall_success: Document processed successfully
        """
        if supplier_id not in self.supplier_profiles:
            # Create new profile
            self.supplier_profiles[supplier_id] = SupplierProfile(
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                total_documents=0,
                successful_documents=0,
                accuracy_rate=0.0,
                avg_extraction_confidence=0.0,
                avg_validation_success_rate=0.0
            )

        profile = self.supplier_profiles[supplier_id]

        # Update counts
        profile.total_documents += 1
        if overall_success:
            profile.successful_documents += 1

        # Update accuracy rate
        profile.accuracy_rate = profile.successful_documents / profile.total_documents

        # Update average confidence (rolling average)
        alpha = 0.1  # Exponential moving average weight
        profile.avg_extraction_confidence = (
            alpha * extraction_confidence +
            (1 - alpha) * profile.avg_extraction_confidence
        )

        # Update validation success rate
        profile.avg_validation_success_rate = (
            alpha * (1.0 if validation_success else 0.0) +
            (1 - alpha) * profile.avg_validation_success_rate
        )

        # Re-calculate threshold adjustment
        profile.threshold_adjustment = self._calculate_supplier_adjustment(profile)
        profile.effective_threshold = self.base_threshold + profile.threshold_adjustment

        profile.last_updated = datetime.now().isoformat()

        logger.debug(
            f"Supplier {supplier_id} updated: "
            f"accuracy={profile.accuracy_rate:.2%}, "
            f"threshold={profile.effective_threshold:.2f}"
        )

    def update_doc_type_profile(
        self,
        doc_type: str,
        success: bool,
        complexity_indicators: Dict[str, Any]
    ):
        """
        Update document type profile.

        Args:
            doc_type: Document type
            success: Processing success
            complexity_indicators: Complexity metrics
        """
        if doc_type not in self.doc_type_profiles:
            self.doc_type_profiles[doc_type] = DocumentTypeProfile(
                doc_type=doc_type,
                total_documents=0,
                successful_documents=0,
                accuracy_rate=0.0,
                complexity_score=0.5
            )

        profile = self.doc_type_profiles[doc_type]

        # Update counts
        profile.total_documents += 1
        if success:
            profile.successful_documents += 1

        # Update accuracy
        profile.accuracy_rate = profile.successful_documents / profile.total_documents

        # Update complexity score
        profile.complexity_score = self._calculate_complexity(complexity_indicators)

        # Re-calculate threshold adjustment
        profile.threshold_adjustment = self._calculate_doc_type_adjustment(profile)
        profile.effective_threshold = self.base_threshold + profile.threshold_adjustment

    def _calculate_supplier_adjustment(self, profile: SupplierProfile) -> float:
        """
        Calculate threshold adjustment for supplier.

        Rules:
        - >98% accuracy: -0.05 (lower threshold, higher touchless)
        - 95-98% accuracy: 0.0 (standard)
        - 90-95% accuracy: +0.03 (slightly higher threshold)
        - <90% accuracy: +0.05 (higher threshold, more scrutiny)
        """
        # Need at least 50 documents for adjustment
        if profile.total_documents < 50:
            return 0.0

        accuracy = profile.accuracy_rate

        if accuracy >= 0.98:
            return -0.05  # Trusted supplier
        elif accuracy >= 0.95:
            return 0.0  # Standard
        elif accuracy >= 0.90:
            return +0.03  # Caution
        else:
            return +0.05  # High scrutiny

    def _calculate_doc_type_adjustment(self, profile: DocumentTypeProfile) -> float:
        """
        Calculate threshold adjustment for document type.

        Factors:
        - Complexity score (higher = stricter)
        - Historical accuracy
        """
        # Need at least 100 documents
        if profile.total_documents < 100:
            return 0.0

        adjustment = 0.0

        # Complexity adjustment
        if profile.complexity_score > 0.7:
            adjustment += 0.03  # Complex docs need higher confidence
        elif profile.complexity_score < 0.3:
            adjustment -= 0.02  # Simple docs can be more lenient

        # Accuracy adjustment
        if profile.accuracy_rate >= 0.98:
            adjustment -= 0.02
        elif profile.accuracy_rate < 0.92:
            adjustment += 0.03

        return adjustment

    def _calculate_complexity(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate document complexity score.

        Factors:
        - Number of fields (more fields = more complex)
        - Line item count
        - Multi-language content
        - Image quality
        - Table complexity
        """
        scores = []

        # Field count complexity
        field_count = indicators.get("field_count", 20)
        if field_count > 40:
            scores.append(0.8)
        elif field_count > 25:
            scores.append(0.5)
        else:
            scores.append(0.3)

        # Line items
        line_items = indicators.get("line_items", 5)
        if line_items > 20:
            scores.append(0.9)
        elif line_items > 10:
            scores.append(0.6)
        else:
            scores.append(0.3)

        # Multi-language
        if indicators.get("multi_language", False):
            scores.append(0.7)

        # Image quality
        quality = indicators.get("image_quality", 0.8)
        if quality < 0.6:
            scores.append(0.8)
        elif quality < 0.8:
            scores.append(0.5)
        else:
            scores.append(0.2)

        # Average complexity
        if scores:
            return statistics.mean(scores)
        else:
            return 0.5

    def get_supplier_profile(self, supplier_id: str) -> Optional[SupplierProfile]:
        """Get supplier profile."""
        return self.supplier_profiles.get(supplier_id)

    def get_trusted_suppliers(self, min_accuracy: float = 0.98) -> List[str]:
        """Get list of trusted suppliers."""
        trusted = []

        for supplier_id, profile in self.supplier_profiles.items():
            if profile.total_documents >= 50 and profile.accuracy_rate >= min_accuracy:
                trusted.append(supplier_id)

        return trusted

    def get_problematic_suppliers(self, max_accuracy: float = 0.90) -> List[str]:
        """Get list of problematic suppliers."""
        problematic = []

        for supplier_id, profile in self.supplier_profiles.items():
            if profile.total_documents >= 50 and profile.accuracy_rate < max_accuracy:
                problematic.append(supplier_id)

        return problematic

    def export_profiles(self) -> Dict[str, Any]:
        """Export all profiles for analysis."""
        return {
            "supplier_profiles": {
                sid: {
                    "name": p.supplier_name,
                    "total_documents": p.total_documents,
                    "accuracy_rate": p.accuracy_rate,
                    "effective_threshold": p.effective_threshold,
                    "threshold_adjustment": p.threshold_adjustment
                }
                for sid, p in self.supplier_profiles.items()
            },
            "doc_type_profiles": {
                dtype: {
                    "total_documents": p.total_documents,
                    "accuracy_rate": p.accuracy_rate,
                    "complexity_score": p.complexity_score,
                    "effective_threshold": p.effective_threshold
                }
                for dtype, p in self.doc_type_profiles.items()
            }
        }


# Singleton instance
_adaptive_thresholds: Optional[AdaptiveQualityThresholds] = None


def get_adaptive_thresholds() -> AdaptiveQualityThresholds:
    """Get singleton adaptive thresholds."""
    global _adaptive_thresholds

    if _adaptive_thresholds is None:
        _adaptive_thresholds = AdaptiveQualityThresholds()

    return _adaptive_thresholds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    thresholds = get_adaptive_thresholds()

    # Simulate supplier learning
    supplier_id = "SUP-001"

    # 100 successful documents from trusted supplier
    for i in range(100):
        thresholds.update_supplier_profile(
            supplier_id=supplier_id,
            supplier_name="Acme Corp",
            extraction_confidence=0.95,
            validation_success=True,
            overall_success=True
        )

    # Get adjusted threshold
    threshold = thresholds.get_threshold(supplier_id=supplier_id)
    profile = thresholds.get_supplier_profile(supplier_id)

    print(f"\nSupplier Profile:")
    print(f"  Supplier: {profile.supplier_name}")
    print(f"  Documents: {profile.total_documents}")
    print(f"  Accuracy: {profile.accuracy_rate:.2%}")
    print(f"  Base Threshold: {profile.base_threshold:.2f}")
    print(f"  Adjustment: {profile.threshold_adjustment:+.2f}")
    print(f"  Effective Threshold: {threshold:.2f}")
    print(f"\nTrusted Suppliers: {thresholds.get_trusted_suppliers()}")
